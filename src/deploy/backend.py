import time
import requests

from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from threading import Lock, Thread
from src.main import *

class Backend:
    def __init__(self, host='127.0.0.1', port=5000, debug=False):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins=["*"])
        self.lock = Lock()
        self.lock.acquire()  # 初始锁定，等待客户端连接
        self.tensorboard_url = 'http://127.0.0.1:6006'
        
        self._setup_socket_events()
        self._setup_tensorboard_proxy()
        self.host = host
        self.port = port
        self.debug = debug
    
    def _setup_socket_events(self):
        """设置SocketIO事件处理器"""
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            self.lock.release()  # 客户端连接后释放锁
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')
            self.lock.acquire()  # 客户端断开后重新锁定

    def _setup_tensorboard_proxy(self):
        """设置 TensorBoard 数据代理路由"""
        
        @self.app.route('/tensorboard/tags')
        def get_tensorboard_tags():
            """获取 TensorBoard 标量标签"""
            try:
                response = requests.get(f"{self.tensorboard_url}/data/plugin/scalars/tags", timeout=5)
                response.raise_for_status()
                data = response.json()
                return jsonify({'tags': list(data.keys())})
            except Exception as e:
                print(f"获取 TensorBoard 标签失败: {e}")
                return jsonify({'tags': []})
        
        @self.app.route('/tensorboard/scalars')
        def get_tensorboard_scalars():
            """获取 TensorBoard 标量数据"""
            try:
                tag = request.args.get('tag', '')
                if not tag:
                    return jsonify({'error': '缺少 tag 参数'}), 400
                
                url = f"{self.tensorboard_url}/data/plugin/scalars/scalars"
                params = {
                    'tag': tag,
                    'run': '.',
                    'format': 'json'
                }
                
                response = requests.get(url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                
                return jsonify(data)
            except Exception as e:
                print(f"获取 TensorBoard 标量数据失败: {e}")
                return jsonify({'error': str(e)}), 500
    
    def wait_for_client(self):
        """等待客户端连接"""
        print("Waiting for client connection...")
        self.lock.acquire()  # 阻塞直到客户端连接
        self.lock.release()  # 立即释放以便后续使用
        print("Client connected, starting training...")
    
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
        
        # 如果距离上次发送时间太短，则跳过
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        with self.target.lock:
            self.last_send_time = current_time
            self.target.socketio.emit('training_update', data)
            return True
        
class TensorboardDaemon(Thread):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.daemon = True
    
    def run(self):
        from tensorboard import program
        
        tb = program.TensorBoard()
        tb.configure(argv=[
            None, 
            '--logdir', self.log_dir, 
            '--port', '6006',
            '--host', '127.0.0.1',
            '--reload_interval', '5',
        ])
        url = tb.launch()
        print(f'TensorBoard started at {url}')

if __name__ == '__main__':
    backend = Backend(debug=False, port=5000)
    monitor = TrainingMonitor(backend, max_fps=30)

    agent = DDPG(state_size, action_size, lr, batch_size, hidden_size, device, noise = 0.01, name = 'cpu 0.01 noise, 0.001 init')
    env = MatchmanEnv([stand_reward], draw=False, monitor=monitor)
    trainer = Trainer(env, agent, num_epochs, max_steps_per_epoch)

    print('Starting backend server...')
    Thread(target=trainer.train).start()
    TensorboardDaemon(log_dir=agent.workspace + 'logs').start()
    backend.run()