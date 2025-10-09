import argparse

from tensorboardX import SummaryWriter
from threading import Thread

from src.reward import stand_reward
from src.trainer import Trainer, MultiTargetWriter
from algorithms.PPO import PPO
from src.env import *

state_size, action_size, hidden_size = 9 * 9, 9, (400, 300)
lr, batch_size = 0.001, 64
num_epochs, max_steps_per_epoch = 100000, 2000
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
    agent = PPO(state_size, action_size, [lr, 0.01], batch_size, hidden_size, device)
    env = MatchmanEnv([stand_reward], args.draw)
    
    writer = MultiTargetWriter([SummaryWriter('logs/' + 'ppo3')])
    trainer = Trainer(env, agent, writer, num_epochs, max_steps_per_epoch)

    TensorboardDaemon(agent.workspace + 'logs', args.tensorboard).start()
    trainer.train(print_interval=1)