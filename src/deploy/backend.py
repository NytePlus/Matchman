import time

from flask import Flask, jsonify, request, Response
from flask_socketio import SocketIO
from flask_cors import CORS
from threading import Lock, Thread

from src.deploy.repository import TensorboardRepository
from src.main import *
from src.trainer import Trainer, MultiTargetWriter

name = 'test'

class Backend:
    def __init__(self, host='127.0.0.1', port=5000, debug=False, db=None):
        self.app = Flask(__name__)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins=["*"])
        
        self.host = host
        self.port = port
        self.debug = debug
        self.db = db

        self._setup_health_check()
        self._setup_socket_events()
        self._set_up_tensorboard_static()

    def _setup_health_check(self):
        @self.app.route('/')
        @self.app.route('/healthz')
        def home():
            return jsonify({
                "status": "success", 
                "message": "RL Training Backend is running",
                "service": "Reinforcement Learning Training Monitor"
            })
    
    def _setup_socket_events(self):
        """设置SocketIO事件处理器"""
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')

    def _set_up_tensorboard_static(self):
        print('由于静态路径无法正常转发，前端不能显示，不能直接转发tensorboard前端')

        @self.app.route('/data')
        def data():
            run = request.args.get('run', '')
            tag = request.args.get('tag', '')
            offset = int(request.args.get('offset', 0))
            limit = int(request.args.get('limit', 1000))
            
            data = self.db.get_scalar_data(run, tag, offset, limit)
            return jsonify(data)

    def run(self):
        """运行后端服务器"""
        self.socketio.run(self.app, allow_unsafe_werkzeug=True, host=self.host, port=self.port, debug=self.debug)

class TrainingMonitor:
    def __init__(self, target, max_fps=60):
        self.clients = []
        self.max_fps = max_fps
        self.min_interval = 1.0 / max_fps  # 最小发送间隔
        self.last_send_time = 0
        self.target = target
    
    def send_update(self, data):
        """发送训练数据到所有连接的客户端，带有限速"""
        current_time = time.time()
        elapsed = current_time - self.last_send_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        try:
            self.last_send_time = current_time
            self.target.socketio.emit('training_update', data)
            return True
        except Exception as e:
            print(f"发送训练数据失败: {e}")
            return False
        
    def add_scalar(self, tag, scalar_value, global_step):
        if tag not in ['sum_step_r', 'epoch_r']:
            return False
        try:
            self.target.socketio.emit('training_statics', {
                'tag': tag, 
                'value': scalar_value, 
                'timestamp': global_step
            })
            return True
        except Exception as e:
            print(f"发送训练指标失败: {e}")
            return False

if __name__ == '__main__':
    static = TensorboardRepository(log_dir='logs', runs = [name], tags = ['epoch_r'])
    backend = Backend(host='127.0.0.1', debug=False, port=5000, db=static)
    monitor = TrainingMonitor(backend, max_fps=30)

    agent = DDPG(state_size, action_size, lr, batch_size, hidden_size, device, noise = 0.01)
    env = MatchmanEnv([stand_reward], draw=False)
    writer = MultiTargetWriter([SummaryWriter('logs/' + 'test'), monitor])
    trainer = Trainer(env, agent, writer, num_epochs, max_steps_per_epoch)

    print('Starting backend server...')
    Thread(target=trainer.train).start()
    static.start()
    backend.run()