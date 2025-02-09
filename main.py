from itertools import count
from reward import stand_reward
from network import DDPG
from env import *

state_size, action_size, hidden_size = 117, 9, (400, 300)
lr, batch_size = 0.001, 64
num_epochs, max_steps_per_epoch = 100000, 2000
test_epochs = 10
device = 'cpu'

def train(env, agent):
    epoch_r = 0

    for epoch in range(num_epochs):
        state = pack_state(env.reset())
        for t in count():
            action = agent.select_action(state)

            next_state, reward, done = env.step(unpack_action(action))
            next_state = pack_state(next_state)
            epoch_r += reward
            agent.replay_buffer.push((state, next_state, action, np.float32(reward), np.float32(done)))

            state = next_state

            if done or t >= max_steps_per_epoch:
                agent.writer.add_scalar('epoch_r', epoch_r, global_step=epoch)
                if epoch % 1 == 0:
                    print(f'Epoch: {epoch}, Reward: {epoch_r:0.2f}, Step:{t}')
                epoch_r = 0
                break

        if (epoch + 1) % 100 == 0:
            agent.save()

        if agent.replay_buffer.full():
            agent.update()

def test(env, agent):
    agent.load()
    epoch_r = 0

    for epoch in range(test_epochs):
        state = pack_state(env.reset())
        for t in count():
            action = agent.select_action(state)

            next_state, reward, done = env.step(unpack_action(action))
            next_state = pack_state(next_state)
            epoch_r += reward
            agent.replay_buffer.push((state, next_state, action, np.float32(reward), np.float32(done)))

            state = next_state
            if done or t >= max_steps_per_epoch:
                print(f'Epoch: {epoch}, Reward: {epoch_r:0.2f}, Step:{t}')
                epoch_r = 0
                break

if __name__ == "__main__":
    agent = DDPG(state_size, action_size, lr, batch_size, hidden_size, device, noise = 0.01, name = 'cpu 0.01 noise, 0.001 init')
    env = MatchmanEnv([stand_reward])

    train(env, agent)