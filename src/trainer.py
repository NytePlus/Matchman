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

class Trainer:
    def __init__(self, env, agent, writer, num_epochs, max_steps_per_epoch):
        self.env = env
        self.agent = agent
        self.writer = writer

        self.num_epochs = num_epochs
        self.max_steps_per_epoch = max_steps_per_epoch

    def train(self, print_interval = 10, save_interval = 100, test_interval = -1):
        epoch_r = 0

        for epoch in range(self.num_epochs):
            state = pack_state(self.env.reset())
            for t in count():
                action = self.agent.select_action(state)

                next_state, reward, done = self.env.step(unpack_action(action))
                next_state = pack_state(next_state)
                epoch_r += reward
                self.agent.replay_buffer.push((state, next_state, action, np.float32(reward), np.float32(done)))

                state = next_state
                
                self.writer.send_update(self.env.get_state())
                self.writer.add_scalar('sum_step_r', epoch_r, t)
                if done or t >= self.max_steps_per_epoch:
                    self.writer.add_scalar('epoch_r', epoch_r, epoch)
                    if epoch % print_interval == 0:
                        print(f'Train Epoch: {epoch}, Reward: {epoch_r:0.2f}, Step:{t}')
                    epoch_r = 0
                    break

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