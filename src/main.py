import argparse

from src.reward import stand_reward
from src.trainer import Trainer
from src.network import DDPG
from src.env import *

state_size, action_size, hidden_size = 9 * 9, 9, (400, 300)
lr, batch_size = 0.001, 64
num_epochs, max_steps_per_epoch = 100000, 2000
device = 'cpu'

def parse_arguments():
    parser = argparse.ArgumentParser(description='火柴人强化学习')
    
    parser.add_argument(
        '--draw',
        action='store_true',
        help='启动屏幕可视化'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    agent = DDPG(state_size, action_size, lr, batch_size, hidden_size, device, noise = 0.01, name = 'cpu 0.01 noise, 0.001 init')
    env = MatchmanEnv([stand_reward], args.draw)

    trainer = Trainer(env, agent, num_epochs, max_steps_per_epoch)

    trainer.train()