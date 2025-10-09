import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.algorithms.base import RLAlgorithm, ReplayBuffer

class WeightInitializer:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=self.mean, std=self.std)
            nn.init.zeros_(m.bias)

class MatchmanActor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scale = 1):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], output_size)
        self.scale = scale

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.scale
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.l1 = nn.Linear(state_size + action_size, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1],1)

    def forward(self, x, u):
        x = torch.relu(self.l1(torch.cat([x, u],1)))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x
    
class ReplayBuffer():
    def __init__(self, max_size = 5000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def clear(self):
        self.storage = []

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def full(self):
        return len(self.storage) == self.max_size

    def sample(self, batch_size = -1):
        if batch_size == -1:
            ind = np.arange(0,len(self.storage))
        else:
            ind = np.random.randint(0,len(self.storage),size=batch_size).sort()
        x, y, u, r, d = [],[],[],[],[]

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(X)
            y.append(Y)
            u.append(U)
            r.append(R)
            d.append(D)

        return np.array(x), np.array(y), np.array(u), np.array(r), np.array(d)

class DDPG():
    def __init__(self, state_size, action_size, lr, batch_size, hidden_size, device, update_iteration = 10, noise = 0.1, tau = 0.005, gamma = 0.99):
        super().__init__()

        self.actor = MatchmanActor(state_size, hidden_size, action_size).to(device)
        self.actor.apply(WeightInitializer())
        self.actor_target = MatchmanActor(state_size, hidden_size, action_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr)

        self.critic = Critic(state_size, action_size, hidden_size).to(device)
        self.critic_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr)

        self.replay_buffer = ReplayBuffer()
        
        self.workspace = './'
        self.critic_update_iter = 0
        self.actor_update_iter = 0
        self.num_training = 0
        self.device = device
        self.batch_size = batch_size
        self.tau = tau
        self.noise = noise
        self.update_iteration = update_iteration
        self.gamma = gamma

    def select_action(self, state : np.array):
        inputs = torch.from_numpy(state).float().to(self.device)
        output = self.actor(inputs).cpu().data.numpy().flatten()
        
        # 鼓励探索
        output = (output + np.random.normal(0, self.noise, size=output.shape[0])).clip(-1, 1) 
        return output, None, None

    def update(self):
        for it in range(self.update_iteration):
            items = self.replay_buffer.sample()
            state, action, next_state, done, reward = [torch.FloatTensor(item).to(self.device) for item in items]

            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

            current_Q = self.critic(state, action)

            critic_loss = F.mse_loss(current_Q, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = - self.critic(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param,target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param,target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.actor_update_iter += 1
            self.critic_update_iter += 1
            yield critic_loss.item(), self.critic_update_iter, actor_loss.item(), self.actor_update_iter

    def save(self):
        torch.save(self.actor.state_dict(), self.workspace + 'ckpt/actor.pth')
        torch.save(self.critic.state_dict(), self.workspace + 'ckpt/critic.pth')

    def load(self):
        self.actor.load_state_dict(torch.load(self.workspace + 'ckpt/actor.pth'))
        self.critic.load_state_dict(torch.load(self.workspace + 'ckpt/critic.pth'))