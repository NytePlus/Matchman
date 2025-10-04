class TrainingVisualizer {
    constructor() {
        this.canvas = document.getElementById('trainingCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.socket = null;
        this.frameCount = 0;
        this.lastFrameTime = 0;
        this.currentFPS = 0;
        
        // 身体部位颜色配置
        this.bodyPartColors = {
            'head': 'black',
        };
        
        this.init();
    }

    init() {
        this.setupSocket();
        this.setupEventListeners();
        this.startAnimation();
    }

    setupSocket() {
        this.socket = io('https://matchman.onrender.com', {
            reconnection: true,
            reconnectionAttempts: Infinity,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000
        });

        this.socket.on('connect', () => {
            console.log('连接到服务器');
            document.getElementById('connectionStatus').textContent = '已连接';
            document.getElementById('connectionStatus').style.color = '#4CAF50';
        });

        this.socket.on('disconnect', () => {
            console.log('与服务器断开连接');
            document.getElementById('connectionStatus').textContent = '未连接';
            document.getElementById('connectionStatus').style.color = '#f44336';
        });

        this.socket.on('training_update', (data) => {
            this.handleTrainingData(data);
        });

        this.socket.on('connect_error', (error) => {
            console.error('连接错误:', error);
        });
    }

    setupEventListeners() {
        // 窗口大小变化时调整canvas
        window.addEventListener('resize', () => {
            this.adjustCanvasSize();
        });
        
        // 初始调整大小
        this.adjustCanvasSize();
    }

    adjustCanvasSize() {
        const container = this.canvas.parentElement;
        const maxWidth = Math.min(800, container.clientWidth - 40);
        const scale = maxWidth / this.canvas.width;
        
        this.canvas.style.width = maxWidth + 'px';
        this.canvas.style.height = (this.canvas.height * scale) + 'px';
    }

    handleTrainingData(data) {
        // 绘制身体部位
        this.drawBodyParts(data);
    }

    drawBodyParts(data) {
        // 清除画布
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 绘制每个身体部位的线段
        for (const [bodyPart, partData] of Object.entries(data)) {
            // 跳过非身体部位的数据（如 timestamp, frame_count 等）
            if (typeof partData !== 'object' || !partData.position_a || !partData.position_b) {
                continue;
            }
            
            this.drawBodyPart(bodyPart, partData);
        }
    }

    drawBodyPart(bodyPart, partData) {
        const positionA = {'x': partData.position_a[0], 'y': partData.position_a[1]};
        const positionB = {'x': partData.position_b[0], 'y': partData.position_b[1]};

        if (!this.isValidPosition(positionA) || !this.isValidPosition(positionB)) {
            console.warn(`无效的坐标: ${bodyPart}`, positionA, positionB);
            return;
        }
        
        // 获取颜色，如果没有配置则使用默认颜色
        const color = this.bodyPartColors[bodyPart] || 'black';
        
        if (bodyPart === 'head') {
            this.drawHeadCircle(positionA, positionB, color);
        } else {
            this.drawBodySegment(positionA, positionB, color, bodyPart);
        }
    }

    drawHeadCircle(positionA, positionB, color) {
        // 计算线段长度（直径）
        const dx = positionB.x - positionA.x;
        const dy = positionB.y - positionA.y;
        const diameter = Math.sqrt(dx * dx + dy * dy);
        
        // 圆心为线段中点
        const centerX = (positionA.x + positionB.x) / 2;
        const centerY = (positionA.y + positionB.y) / 2;
        const radius = diameter / 2;
        
        // 绘制圆形
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // 可选：填充圆形（半透明）
        // this.ctx.fillStyle = this.addAlpha(color, 0.2);
        this.ctx.fill();
    }

    drawBodySegment(positionA, positionB, color) {
        // 设置线条样式
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.lineCap = 'round';
        
        // 开始绘制线段
        this.ctx.beginPath();
        this.ctx.moveTo(positionA.x, positionA.y);
        this.ctx.lineTo(positionB.x, positionB.y);
        this.ctx.stroke();
        
        // 绘制端点
        this.drawEndpoint(positionA, color);
        this.drawEndpoint(positionB, color);
    }

    isValidPosition(pos) {
        return pos && 
            typeof pos.x === 'number' && 
            typeof pos.y === 'number' &&
            !isNaN(pos.x) && 
            !isNaN(pos.y);
    }

    drawEndpoint(position, color) {
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.arc(position.x, position.y, 4, 0, Math.PI * 2);
        this.ctx.fill();
    }

    startAnimation() {
        const animate = () => {
            // 这里可以添加一些持续的动画效果
            // 比如背景网格、参考线等
            this.ctx.lineWidth = 1;
            this.ctx.fillStyle = "black";
            this.ctx.beginPath();
            this.ctx.moveTo(-100, 500);
            this.ctx.lineTo(1000, 500);
            this.ctx.stroke();
            
            requestAnimationFrame(animate);
        };
        
        animate();
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    new TrainingVisualizer();
});