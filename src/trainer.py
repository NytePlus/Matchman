import numpy as np
import threading

from itertools import count
from src.env import pack_state, unpack_action

# 如果我要拓展writer的写出，那么我就要在下一行新增一个写出 -> 无法在一行完成写出
# 如果我要将两个合并为一个Writer，那么原本没有monitor的writer也要兼容Writer的接口，要么for循环writer要么for循环writer.write -> 无法在一行完成定义
# 所以我要将两个合并为一个Writer，并且新兼容旧Writer的接口
class MultiTargetWriter:
    def __init__(self, targets = []):
        self.targets = targets

    def add_scalar(self, tag, scalar_value, global_step):
        for target in self.targets:
            if hasattr(target, 'add_scalar'):
                target.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)
            else:
                raise ValueError(f'{target} has no add_scalar')
            
    def send_update(self, data):
        _, data = data
        for target in self.targets:
            if hasattr(target, 'send_update'):
                target.send_update(data)
            else:
                pass

class EpsilonScheduler:
    def __init__(self, start_epsilon, end_epsilon, decay_epochs, model, decay_type='linear'):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_epochs = decay_epochs
        self.decay_type = decay_type

        self.model = model
        self.epoch = 0
        self.model.epsilon = self.get_epsilon(self.epoch)
        
    def get_epsilon(self, epoch):
        if epoch >= self.decay_epochs:
            return self.end_epsilon
        
        if self.decay_type == 'linear':
            progress = epoch / self.decay_epochs
            return self.start_epsilon - (self.start_epsilon - self.end_epsilon) * progress
        elif self.decay_type == 'exponential':
            decay_factor = (self.end_epsilon / self.start_epsilon) ** (1 / self.decay_epochs)
            return self.start_epsilon * (decay_factor ** epoch)
        else:
            raise ValueError(f"Unsupported decay type: {self.decay_type}")
        
    def step(self):
        self.epoch += 1
        self.model.epsilon = self.get_epsilon(self.epoch)

class Trainer:
    def __init__(self, env, agent, writer, num_epochs, max_steps_per_epoch):
        self.env = env
        self.agent = agent
        self.writer = writer

        self.num_epochs = num_epochs
        self.max_steps_per_epoch = max_steps_per_epoch

    def train(self, print_interval = 10, save_interval = 100, test_interval = -1):
        epoch_r = 0
        scheduler = EpsilonScheduler(0.8, 0.1, self.num_epochs, self.agent.policy)

        for epoch in range(self.num_epochs):
            state = pack_state(self.env.reset())
            for t in count():
                action = self.agent.select_action(state)

                next_state, reward, done = self.env.step(unpack_action(action))
                next_state = pack_state(next_state)

                if t < 200:
                    state = next_state
                    continue

                epoch_r += reward
                self.agent.replay_buffer.push((state, next_state, action, np.float32(reward), np.float32(done)))

                state = next_state

                print(f'\r>⏳ Epoch: {epoch:4d} | 🕹️ Action: {action[0]:> 3.1f} | 🎯 Reward: {reward:8.2f} | 🏆 Total: {epoch_r:8.2f}<', end='', flush=True)

                if done or t >= self.max_steps_per_epoch:
                    self.writer.add_scalar('epoch_r', epoch_r, epoch)
                    if epoch % print_interval == 0:
                        print(f'\r✅ Epoch: {epoch:4d} | 🏆 Total: {epoch_r:8.2f} | 📈 Steps: {t:4d} ' + ' '*40)
                        epoch_r = 0
                        break

            scheduler.step()
            if (epoch + 1) % save_interval == 0:
                self.agent.save()

            if self.agent.update_check():
                for al, ai, cl, ci in self.agent.update():
                    self.writer.add_scalar('Loss/actor_loss', al, global_step = ai)
                    self.writer.add_scalar('Loss/critic_loss', cl, global_step = ci)

            if test_interval != -1 and epoch % test_interval == 0:
                self.test(1)

    def test(self, test_epochs = 10):
        epoch_r = 0

        for epoch in range(test_epochs):
            state = pack_state(self.env.reset())
            for t in count():
                action = self.agent.select_action(state)

                next_state, reward, done = self.env.step(unpack_action(action), test=True)

                next_state = pack_state(next_state)
                epoch_r += reward
                self.agent.replay_buffer.push((state, next_state, action, np.float32(reward), np.float32(done)))

                state = next_state
                if done or t >= self.max_steps_per_epoch:
                    print(f'Test: {epoch}, Reward: {epoch_r:0.2f}, Step:{t}')
                    epoch_r = 0
                    break