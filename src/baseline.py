import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.env import MatchmanEnv
from src.reward import stand_reward

# --- baseline train ---
env = MatchmanEnv([stand_reward], False)

policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=[128, 128]
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)

model.learn(total_timesteps=100000)
model.save("ckpt/official_ppo.ckpt")

# --- baseline test ---
env = MatchmanEnv([stand_reward], True)
observation, info = env.reset()

for i in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode finished after {i} steps")
        observation, info = env.reset()
        break

env.close()