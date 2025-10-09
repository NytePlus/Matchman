import torch
import torch.nn as nn
import torch.distributions as dist
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class MatchmanPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scale=4):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l_mean = nn.Linear(hidden_size[1], output_size)
        self.l_std = nn.Linear(hidden_size[1], output_size)
        self.scale = scale

    def forward(self, x):
        # 这个forward用于SB3框架
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        mean = torch.tanh(self.l_mean(x)) * self.scale
        log_std = self.l_std(x)
        std = torch.exp(log_std).clamp(max=1.0)

        d = dist.Normal(mean, std)
        return d.sample()
    
    def distribution(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std).clamp(max=1.0)
        return dist.Normal(mean, std)
    
    def compute_log_prob(self, state, action):
        distribution = self.distribution(state)
        return distribution.log_prob(action).sum(dim=-1)
    
    def compute_entropy(self, state):
        distribution = self.distribution(state)
        return distribution.entropy().mean()

class Value(nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.l1 = nn.Linear(state_size, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], 1)

    def forward(self, state):
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        return self.l3(x)

# 自定义策略类
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # 从kwargs中获取自定义参数
        policy_hidden_size = kwargs.pop('policy_hidden_size', [64, 64])
        value_hidden_size = kwargs.pop('value_hidden_size', [64, 64])
        scale = kwargs.pop('scale', 2)
        
        super().__init__(*args, **kwargs)
        self.action_dim = self.action_space.shape[0]
        
        # 保存你的自定义模型实例
        self.custom_policy = MatchmanPolicy(
            input_size=self.mlp_extractor.latent_dim_pi,
            hidden_size=policy_hidden_size,
            output_size=self.action_dim,
            scale=scale
        )
        
        self.custom_value = Value(
            state_size=self.mlp_extractor.latent_dim_vf,
            hidden_size=value_hidden_size
        )
        
        # 替换原有的网络
        self.action_net = self.custom_policy
        self.value_net = self.custom_value

    def _build(self, lr_schedule):
        # 覆盖_build方法，使用自定义网络
        super()._build(lr_schedule)
        
        # 重新设置优化器，包含自定义网络的参数
        self.optimizer = self.optimizer_class(
            self.parameters(),  # 这会包含自定义网络的参数
            lr=lr_schedule(1),  # 初始学习率
            **self.optimizer_kwargs,
        )
