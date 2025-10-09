import argparse

from tensorboardX import SummaryWriter
from threading import Thread

from src.reward import stand_reward
from src.trainer import MultiTargetWriter
from algorithms.PPO import PPO
from src.env import *

hidden_size = (400, 300)
lr, batch_size = 0.0001, 64
total_steps, max_steps_per_round, num_epochs = 100000, 2000, 10
device = 'cpu'
tensorboard_host, tensorboard_port = '127.0.0.1', '6006'

def parse_arguments():
    parser = argparse.ArgumentParser(description='火柴人强化学习')
    
    parser.add_argument(
        '--draw',
        action='store_true',
        help='启动屏幕可视化'
    )
    
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='启动看板'
    )
    
    return parser.parse_args()

class TensorboardDaemon(Thread):
    def __init__(self, log_dir, lanch = True):
        super().__init__()
        self.log_dir = log_dir
        self.daemon = True
        self.lanch = lanch
    
    def run(self):
        if not self.lanch:
            return
        from tensorboard import program
        
        tb = program.TensorBoard()
        tb.configure(argv=[
            None, 
            '--logdir', self.log_dir, 
            '--port', tensorboard_port,
            '--host', tensorboard_host,
            '--reload_interval', '5',
        ])
        url = tb.launch()
        print(f'TensorBoard started at {url}')

if __name__ == "__main__":
    args = parse_arguments()
    # env = MatchmanEnv([stand_reward], args.draw)
    env = gym.make("BipedalWalker-v3")
    test_env = gym.make("BipedalWalker-v3", render_mode="human")
    writer = MultiTargetWriter([SummaryWriter('logs/' + 'ppo3')])
    agent = PPO(
        writer,
        env,
        test_env,
        [lr, lr], 
        batch_size, 
        hidden_size, 
        device, 
        total_steps = total_steps,
        max_steps_per_round = max_steps_per_round,
        num_epochs=num_epochs,
        norm_advantage=True,
        clip_range_vf=None
    )

    TensorboardDaemon(agent.workspace + 'logs', args.tensorboard).start()
    agent.train(print_rollout=False, test_interval = 20)