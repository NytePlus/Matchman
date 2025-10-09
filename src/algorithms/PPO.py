import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

from torch import nn
from torch import optim
from itertools import count

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

# return = G_t, E(G_t) = V(s_t)
# advantage = A_t = Q(s_t, a_t) - V(s_t) = G_t - V(s_t), E[G_t | s_t, a_t] = Q(s_t, a_t)
class ReturnAdvantageCalculator:
    def __init__(self, gamma):
        self.gamma = gamma

    def MC(self, values, rewards): # (batch_size, 1)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.stack(returns).reshape(-1, 1)
        advantages = returns - values
        return returns, advantages

    def TD(self, values, rewards):
        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]
        returns = rewards + self.gamma * next_values
        advantages = returns - values
        return returns, advantages

    def GAE(self, values, rewards):
        self.lam = 0.1
        values = np.array(values).flatten()
        rewards = np.array(rewards).flatten()
        
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        returns = np.zeros(n_steps)
        
        last_advantage = 0
        
        # 反向计算GAE
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                # 最后一个状态
                next_value = 0  # 如果终止，next_value=0
            else:
                next_value = values[t + 1]
            
            # TD误差
            delta = rewards[t] + self.gamma * next_value - values[t]
            
            # GAE优势
            advantages[t] = delta + self.gamma * self.lam * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return returns.reshape(-1, 1), advantages.reshape(-1, 1)

class RolloutBuffer():
    def __init__(self, max_size = 5000):
        self.storage = dict()
        self.max_size = max_size
        self.clear()

    def clear(self):
        self.storage = {
            "states": np.zeros((self.max_size, 24), dtype=np.float32),
            "actions": np.zeros((self.max_size, 4), dtype=np.float32),
            "values": np.zeros((self.max_size, 1), dtype=np.float32),
            "log_probs": np.zeros((self.max_size, 1), dtype=np.float32),
            "advantages": np.zeros((self.max_size, 1), dtype=np.float32),
            "returns": np.zeros((self.max_size, 1), dtype=np.float32),
        }
        self.dones = np.zeros((self.max_size, 1))
        self.rewards = np.zeros((self.max_size, 1))
        self.start = 0
        self.pos = 0

    def full(self):
        return self.pos == self.max_size

    def push(self, state: np.ndarray, action, reward, done, value, log_prob):
        if self.pos < self.max_size:
            self.storage['states'][self.pos] = np.array(state)
            self.storage['actions'][self.pos] = np.array(action)
            self.storage['values'][self.pos] = np.array(value)
            self.storage['log_probs'][self.pos] = np.array(log_prob)

            self.rewards[self.pos] = np.array(reward)
            self.dones[self.pos] = np.array(done)
            self.pos += 1

    def compute_returns_and_advantages(self, calculator):
        if self.start == self.pos:
            return
        indices = np.arange(self.start, self.pos)
        returns, advantages = calculator(self.storage['values'][indices], self.rewards[indices])

        self.storage['advantages'][self.start: self.pos] = advantages
        self.storage['returns'][self.start: self.pos] = returns
        self.start = self.pos

    def get(self, batch_size):
        indices = np.random.permutation(self.pos)
        for i in range(0, self.pos, batch_size):
            batch_idx = indices[i: min(i + batch_size, self.pos)]
            yield [val[batch_idx] for _, val in self.storage.items()]

class MatchmanPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, log_std_init = 0, scale = 2):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l_mean = nn.Linear(hidden_size[1], output_size)

        # self.l_std = nn.Linear(hidden_size[1], output_size)
        self.log_std = nn.Parameter(torch.ones(output_size) * log_std_init, requires_grad=True)
        self.scale = scale
        self.noise_dist = dist.Normal(0, torch.ones(output_size) * 0.1)

    def proba_distribution(self, state):
        x = torch.tanh(self.l1(state))
        x = torch.tanh(self.l2(x))
        mean = self.l_mean(x)

        # std = torch.exp(self.l_std(x)).clamp(max=1.0)
        std = torch.ones_like(mean) * self.log_std.exp() # 保持在N(1e-1, 1.0)左右最佳

        self.distribution = dist.Normal(mean, std)
    
    def evaluate_action(self, state, action):
        self.proba_distribution(state)
        log_prob = self.distribution.log_prob(action).sum(dim=-1)
        entropy = self.distribution.entropy().sum()
        return log_prob, entropy
    
    def get_noise(self):
        return self.noise_dist.sample()

    def forward(self, state):
        self.proba_distribution(state)
        action = self.distribution.rsample()
        log_prob = self.distribution.log_prob(action).sum(dim=-1)
        return action, log_prob
    
class MatchmanPolicyDiscret(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_strategy=3, epsilon=0):
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
    
    def compute_log_prob(self, state, action):
        logits = self.distribution(state)

        log_all_prob = F.log_softmax(logits, dim=-1)
        action_idx = (action + 1).long()
        log_prob = log_all_prob.gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
        
        return log_prob.sum(-1)
    
    def compute_entropy(self, state):
        logits = self.distribution(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        entropy = - (probs * log_probs).sum(dim=-1)
        return entropy.mean()

    def forward(self, state):
        logits = self.distribution(state)

        all_prob = F.softmax(logits, dim=-1)
        action = all_prob.argmax(dim=-1) - 1

        random_mask = torch.rand(*action.shape) < self.epsilon
        random_action = torch.randint(-1, 2, action.shape)
        action = torch.where(random_mask, random_action, action)
        return action

# Critic是个Q函数，Value是值函数
class Value(nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.l1 = nn.Linear(state_size, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], 1)

    def forward(self, state):
        x = torch.tanh(self.l1(state))
        x = torch.tanh(self.l2(x))
        return self.l3(x).reshape(-1, 1)

class PPO():
    def __init__(self, writer, env, test_env, lrs, batch_size, hidden_size, device, clip_range_vf = 1.0, norm_advantage = False, total_steps = 10000, max_steps_per_round=2000, num_epochs = 10, gamma = 0.99, epsilon = 0.1):
        super().__init__()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        self.policy = MatchmanPolicy(state_size, hidden_size, action_size).to(device)
        self.policy.apply(XavierInitializer())
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lrs[0])

        self.value = Value(state_size, hidden_size).to(device)
        self.value.apply(NormalInitializer(std=0.01))
        self.value_optimizer = optim.Adam(self.value.parameters(), lrs[1])

        self.workspace = './'
        self.num_training = 0
        self.num_steps = 0
        self.epoch = 0

        self.writer = writer
        self.env = env
        self.test_env = test_env
        self.rollout_buffer = RolloutBuffer(2048)

        self.device = device
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.max_steps_per_round = max_steps_per_round
        self.num_epochs = num_epochs
        self.clip_range_vf = clip_range_vf
        self.norm_advantage = norm_advantage
        self.gamma = gamma # 奖励累加的递减参数
        self.epsilon = epsilon # ppo裁剪项

    def select_action(self, state : np.array):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            action, log_prob = self.policy(state)
            value = self.value(state)

            return (
                action.cpu().numpy(),
                value.cpu().numpy(),
                log_prob.cpu().numpy()
            )

    def update(self, test_interval=-1):
        for epoch in range(self.num_epochs):
            all_policy_loss, all_value_loss, all_entropy = [], [], []
            for items in self.rollout_buffer.get(self.batch_size):
                states, actions, old_values, old_log_probs, advantages, returns = [torch.FloatTensor(item).to(self.device) for item in items]

                # 计算policy损失
                log_probs, entropy = self.policy.evaluate_action(states, actions)
                values = self.value(states)
                print(f'[{values.abs().max().item(): .2f}, {log_probs.abs().max().item(): .2f}]', end=' ')

                # 这里是乱序，计算优势只能在rollout的时候计算
                if self.norm_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_probs - old_log_probs)
                policy_loss = - torch.min(ratio * advantages, torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages).mean()

                # 计算value损失
                if self.clip_range_vf is not None:
                    values = old_values + torch.clip(values - old_values, -self.clip_range_vf, self.clip_range_vf)
                value_loss = F.mse_loss(returns, values).mean()
                # print(f'[{returns.abs().max().item(): .2f}, {values.abs().max().item(): .2f}]', end=' ')

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                all_policy_loss.append(policy_loss.item())
                all_value_loss.append(value_loss.item())
                all_entropy.append(0)# entropy.item())

            self.epoch += 1
            print(f'\r✅ Epoch: {self.epoch:4d} | Policy Loss: {np.mean(all_policy_loss):4.2f} | Value Loss: {np.mean(all_value_loss):4.2f} | Entropy Loss: {np.mean(all_entropy):4.2f}')
            self.writer.add_scalar('Loss/policy_loss', np.mean(all_policy_loss), global_step = self.epoch)
            self.writer.add_scalar('Loss/value_loss', np.mean(all_value_loss), global_step = self.epoch)

            if test_interval != -1 and self.epoch % test_interval == 0:
                self.test(1)

    def save(self):
        torch.save(self.policy.state_dict(), self.workspace + 'ckpt/policy.pth')
        torch.save(self.value.state_dict(), self.workspace + 'ckpt/value.pth')

    def load(self):
        self.policy.load_state_dict(torch.load(self.workspace + 'ckpt/policy.pth'))
        self.value.load_state_dict(torch.load(self.workspace + 'ckpt/value.pth'))

    def collect_rollout(self, print_rollout = True):
        for round in count():
            epoch_r = 0
            state, _ = self.env.reset()
            for t in count():
                action, value, log_prob = self.select_action(state) 
                # log_prob和value既可以更新时利用ref模型重新计算，也可以存下来
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                # state控制在1.0附近最佳，action均匀分布(0, 3）最佳

                next_state, reward, done, _, _ = self.env.step(action)

                epoch_r += reward

                self.rollout_buffer.push(
                    state, action, reward, done, value, log_prob)

                state = next_state
                if print_rollout:
                    print(f'\r>⏳ Round: {round:4d} | 🕹️ Action: {action[0]:> 3.1f} | 🎯 Reward: {reward:8.2f} | 🏆 Total: {epoch_r:8.2f}<', end='', flush=True)

                self.num_steps += 1
                if done or t >= self.max_steps_per_round:
                    break

            calc = ReturnAdvantageCalculator(self.gamma).GAE
            self.rollout_buffer.compute_returns_and_advantages(calc)

            if print_rollout:
                print(f'\r📦 Round: {round:4d} | 🏆 Total: {epoch_r:8.2f} | 📈 Steps: {t:4d} ' + ' '*40)
                
            if self.rollout_buffer.full():
                return

    def train(self, print_rollout = True, test_interval = -1):
        # scheduler = EpsilonScheduler(0.8, 0.1, self.total_steps / self.rollout_buffer.max_size, self.policy)

        while self.num_steps < self.total_steps:
            self.collect_rollout(print_rollout)

            self.update(test_interval)
            self.rollout_buffer.clear()

            self.save()
            # scheduler.step()

    def test(self, test_round = 10):
        for round in range(test_round):
            state, _ = self.test_env.reset()
            round_r = 0
            for t in count():
                action, _, _ = self.select_action(state)
                state, reward, done, _, _ = self.test_env.step(action)

                round_r += reward
                if done or t >= self.max_steps_per_round:
                    break
            print(f'\r🔍 Test: {round:4d} | 🏆 Total: {round_r:8.2f} | 📈 Steps: {t:4d} ' + ' '*40)