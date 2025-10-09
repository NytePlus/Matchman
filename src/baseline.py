import torch
from itertools import count

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.env import MatchmanEnv
from src.reward import stand_reward

# --- baseline train ---
# env = MatchmanEnv([stand_reward], False)

import gymnasium as gym
env = gym.make("BipedalWalker-v3")
test_env = gym.make("BipedalWalker-v3", render_mode="human")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gae_lambda=0.0, # 0==TD, 1==MC
    gamma=0.99,
    # ent_coef=0.1,           # 显著增加熵系数
    # vf_coef=0.8,            # 增加价值函数权重
    # normalize_advantage=True,
    use_sde=False,
    target_kl=None,
    policy_kwargs=dict(
        share_features_extractor=False,
    ),
)

model.learn(total_timesteps=100000)
model.save("ckpt/official_ppo.ckpt")

# --- baseline test ---
# env = MatchmanEnv([stand_reward], True)
env = test_env
observation, info = env.reset()

for i in count():
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode finished after {i} steps")
        observation, info = env.reset()
        break

env.close()