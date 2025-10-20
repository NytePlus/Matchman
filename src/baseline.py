import torch
from itertools import count

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from env.pymunk import MatchmanEnv
from src.reward import stand_reward, stand_reward_box2d, action_penalty

# --- baseline train ---
env = MatchmanEnv([stand_reward], True)
test_env = MatchmanEnv([stand_reward], True)

# env = MatchmanEnv([stand_reward_box2d, action_penalty], False)
# test_env = MatchmanEnv([stand_reward_box2d, action_penalty], True)

# import gymnasium as gym
# env = gym.make("BipedalWalker-v3")
# test_env = gym.make("BipedalWalker-v3", render_mode="human")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gae_lambda=0.95, # 0==TD, 1==MC
    gamma=0.99,
    # ent_coef=0.1,           # 显著增加熵系数
    # vf_coef=0.8,            # 增加价值函数权重
    # normalize_advantage=True,
    # use_sde=False,
    # target_kl=None,
    policy_kwargs=dict(
        share_features_extractor=False,
    ),
)

eval_callback = EvalCallback(
    test_env,
    best_model_save_path="ckpt/best_model/",
    log_path="ckpt/logs/",
    eval_freq=50000,
    n_eval_episodes=1,
    deterministic=True,
    render=False,
    verbose=1
)

model.learn(total_timesteps=100000, callback=eval_callback)

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