import numpy as np
import threading

from itertools import count
from src.env import pack_state, unpack_action

class Trainer:
    def __init__(self, env, agent, num_epochs, max_steps_per_epoch):
        self.env = env
        self.agent = agent

        self.num_epochs = num_epochs
        self.max_steps_per_epoch = max_steps_per_epoch

    def train(self):
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

                if done or t >= self.max_steps_per_epoch:
                    self.agent.writer.add_scalar('epoch_r', epoch_r, global_step=epoch)
                    if epoch % 10 == 0:
                        print(f'Epoch: {epoch}, Reward: {epoch_r:0.2f}, Step:{t}')
                    epoch_r = 0
                    break

            if (epoch + 1) % 100 == 0:
                self.agent.save()

            if self.agent.replay_buffer.full():
                self.agent.update()

    def test(self, test_epochs = 10):
        self.agent.load()
        epoch_r = 0

        for epoch in range(test_epochs):
            state = pack_state(self.env.reset())
            for t in count():
                action = self.agent.select_action(state)

                next_state, reward, done = self.env.step(unpack_action(action))

                next_state = pack_state(next_state)
                epoch_r += reward
                self.agent.replay_buffer.push((state, next_state, action, np.float32(reward), np.float32(done)))

                state = next_state
                if done or t >= self.max_steps_per_epoch:
                    print(f'Epoch: {epoch}, Reward: {epoch_r:0.2f}, Step:{t}')
                    epoch_r = 0
                    break