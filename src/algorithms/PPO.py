import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

from torch import nn
from torch import optim
from src.algorithms.base import RLAlgorithm

class NormalInitializer:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=self.mean, std=self.std)
            nn.init.zeros_(m.bias)

class XavierInitializer:
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, m):
        if isinstance(m, nn.Linear):
            # Xavier均匀初始化，专门为tanh设计
            nn.init.xavier_uniform_(m.weight, gain=self.gain)
            nn.init.zeros_(m.bias)

class MatchmanPolicyContinious(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scale = 2):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l_mean = nn.Linear(hidden_size[1], output_size)
        self.l_std = nn.Linear(hidden_size[1], output_size)
        self.scale = scale

    def distribution(self, state):
        x = torch.tanh(self.l1(state))
        x = torch.tanh(self.l2(x))
        mean = torch.tanh(self.l_mean(x)) * self.scale
        std = torch.exp(self.l_std(x)).clamp(max=1.0)

        return dist.Normal(mean, std)
    
    def compute_prob(self, state, action):
        dist = self.distribution(state)
        return torch.exp(dist.log_prob(action).sum(dim=-1))

    def forward(self, state):
        dist = self.distribution(state)
        return dist.sample()
    
class MatchmanPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_strategy=3, epsilon=0.8):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], output_size * num_strategy)
        self.output_size = output_size
        self.num_strategy = num_strategy
        self.epsilon = epsilon

        self.t = 1

    def distribution(self, state):
        x = torch.tanh(self.l1(state))
        x = torch.tanh(self.l2(x))
        logits = self.l3(x).reshape(-1, self.output_size, self.num_strategy)

        return logits.clamp(-2, 1)
    
    def compute_prob(self, state, action):
        logits = self.distribution(state)

        log_all_prob = F.log_softmax(logits, dim=-1)
        action_idx = (action + 1).long()
        log_prob = log_all_prob.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
        
        return torch.exp(log_prob.sum(-1))

    def forward(self, state):
        logits = self.distribution(state)

        all_prob = F.softmax(logits, dim=-1)
        action = all_prob.argmax(dim=-1) - 1

        random_mask = torch.rand(*action.shape) < self.epsilon
        random_action = torch.randint(-1, 2, action.shape)
        action = torch.where(random_mask, random_action, action)
        return action
    
    def forward1(self, state):
        batch_size = state.shape[0] if len(state.shape) == 2 else 1
        action = torch.ones(batch_size, self.output_size)

        if self.t % 100 > 90:
            if self.t % 200 < 100:
                action *= -10
            else:
                action *= 10
        elif self.t % 2 == 0:
            action *= -1
        self.t += 1
        return action

# Critic是个Q函数，Value是值函数
class Value(nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.l1 = nn.Linear(state_size, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], 1)

    def forward(self, state):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x

def MCAdvantage(reward: torch.Tensor, gamma): # (batch_size, 1)
    advantages = []
    G = 0
    for r in reversed(reward):
        G = r + gamma * G
        advantages.insert(0, G)
    advantages = torch.stack(advantages)
    # return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

class PPO(RLAlgorithm):
    def __init__(self, state_size, action_size, lrs, batch_size, hidden_size, device, update_iteration = 10, tau = 1, gamma = 0.99, epsilon = 0.1):
        super().__init__()
        assert tau > 0.95, "参考模型的EMA更新程度必须大于0.97，否则会出现数值稳定性问题"

        self.policy = MatchmanPolicy(state_size, hidden_size, action_size).to(device)
        self.policy.apply(XavierInitializer())
        self.policy_old = MatchmanPolicy(state_size, hidden_size, action_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lrs[0])

        self.value = Value(state_size, hidden_size).to(device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lrs[1])

        self.workspace = './'
        self.policy_update_iter = 0 # 两者更新频率可以不同
        self.value_update_iter = 0
        self.num_training = 0
        self.device = device
        self.batch_size = batch_size
        self.update_iteration = update_iteration
        self.tau = tau # EMA更新，等于1时全量同步
        self.gamma = gamma # 奖励累加的递减参数
        self.epsilon = epsilon # ppo裁剪项

    def select_action(self, state : np.array):
        inputs = torch.from_numpy(state).float().to(self.device)
        output = self.policy(inputs)
        action = output.cpu().data.numpy().flatten()
        
        return action
    
    def update_check(self):
        return True

    def update(self):
        for it in range(self.update_iteration):
            items = self.replay_buffer.sample()
            state, next_state, action, reward, done = [torch.FloatTensor(item).to(self.device) for item in items]

            # 计算policy损失
            prob, old_prob = self.policy.compute_prob(state, action), self.policy_old.compute_prob(state, action)
            value = self.value(state).squeeze(1)
            advantage = MCAdvantage(reward, self.gamma) - value

            ratio = prob / old_prob + 1e-8
            policy_loss = - torch.min(ratio * advantage, torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage).mean()

            # 计算value损失
            value_loss = F.mse_loss(advantage, value).mean()

            # 忽略entropy损失
            loss = policy_loss + 0.5 * value_loss

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()

            self.policy_update_iter += 1
            self.value_update_iter += 1
            yield policy_loss.item(), self.policy_update_iter, value_loss.item(), self.value_update_iter

        for param, old_param in zip(self.policy.parameters(), self.policy_old.parameters()):
            old_param.data.copy_(self.tau * param.data + (1 - self.tau) * old_param.data)
        self.replay_buffer.clear()

    def save(self):
        torch.save(self.policy.state_dict(), self.workspace + 'ckpt/policy.pth')
        torch.save(self.value.state_dict(), self.workspace + 'ckpt/value.pth')

    def load(self):
        self.policy.load_state_dict(torch.load(self.workspace + 'ckpt/policy.pth'))
        self.value.load_state_dict(torch.load(self.workspace + 'ckpt/value.pth'))