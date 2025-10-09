import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist
from typing import Tuple, Optional

class FlattenExtractor(nn.Module):
    """特征扁平化提取器"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations)

class MlpExtractor(nn.Module):
    """MLP提取器，包含策略和价值网络"""
    def __init__(self, feature_dim: int, net_arch: list = [64, 64]):
        super().__init__()
        net_arch = [64, 64]
        # 策略网络
        policy_layers = []
        prev_dim = feature_dim
        for dim in net_arch:
            policy_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.Tanh()
            ])
            prev_dim = dim
        self.policy_net = nn.Sequential(*policy_layers)
        
        # 价值网络  
        value_layers = []
        prev_dim = feature_dim
        for dim in net_arch:
            value_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.Tanh()
            ])
            prev_dim = dim
        self.value_net = nn.Sequential(*value_layers)
        
        self.latent_dim_pi = net_arch[-1] if net_arch else feature_dim
        self.latent_dim_vf = net_arch[-1] if net_arch else feature_dim

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)
    
    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

class DiagGaussianDistribution:
    """对角高斯分布"""
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None
        
    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor):
        self.mean_actions = mean_actions
        self.log_std = log_std
        self.std = torch.exp(log_std)
        return self
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # 计算对数概率
        return dist.Normal(self.mean_actions, self.std).log_prob(actions).sum(dim=-1)
    
    def entropy(self) -> torch.Tensor:
        return dist.Normal(self.mean_actions, self.std).entropy().sum(dim=-1)
    
    def sample(self) -> torch.Tensor:
        return dist.Normal(self.mean_actions, self.std).rsample()
    
    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.mean_actions
        return self.sample()

class CustomActorCriticPolicy(nn.Module):
    """完整的Actor-Critic策略网络，与SB3结构相同"""
    def __init__(
        self, 
        observation_space, 
        action_space,
        net_arch: list = [64, 64],
        share_features_extractor: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.share_features_extractor = share_features_extractor

        self.squash_output = False
        
        # 特征提取器
        self.features_extractor = FlattenExtractor()
        feature_dim = observation_space.shape[0]  # 假设状态是1D
        
        if not share_features_extractor:
            self.pi_features_extractor = FlattenExtractor()
            self.vf_features_extractor = FlattenExtractor()
        
        # MLP提取器
        self.mlp_extractor = MlpExtractor(feature_dim, net_arch)
        
        # 动作网络和价值网络
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_space.shape[0])
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        
        # 可学习的log_std参数
        self.log_std = nn.Parameter(torch.ones(action_space.shape[0]) * -0.5)
        
        # 分布对象
        self.action_dist = DiagGaussianDistribution(action_space.shape[0])

    def obs_to_tensor(self, observation):
        """将观测值转换为tensor - SB3需要的接口方法"""
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        elif not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        
        # 确保有batch维度
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
            
        return observation

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        if self.share_features_extractor:
            return self.features_extractor(obs)
        else:
            return self.pi_features_extractor(obs), self.vf_features_extractor(obs)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        """从潜在变量获取动作分布"""
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 提取特征
        features = self.extract_features(obs)
        
        # MLP提取
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        # 计算价值
        values = self.value_net(latent_vf)
        
        # 获取动作分布
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # 采样动作
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        # 调整动作形状
        actions = actions.reshape((-1, *self.action_space.shape))
        
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估给定动作的价值和对数概率"""
        features = self.extract_features(obs)
        
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """仅预测状态价值"""
        features = self.extract_features(obs)
        
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
        else:
            _, vf_features = features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        return self.value_net(latent_vf)

    def set_training_mode(self, mode: bool):
        """设置训练模式 - SB3需要的接口方法"""
        self.train(mode)