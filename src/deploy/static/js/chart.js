export const BACKEND_URL = "https://matchman.onrender.com"

class DynamicChart {
    constructor(tag, chartName, xName, run) {
        this.containerId = tag;
        this.chartName = chartName;
        this.xName = xName
        this.loadUrl = `${BACKEND_URL}/data?run=${run}&tag=${tag}`;

        this.data = [];
        this.chart = null;
        this.intervalId = null;
        this.isPlaying = false;
        this.lastT = 0
        
        this.initChart();
    }
    
    initChart() {
        const dom = document.getElementById(this.containerId);
        if (!dom) {
            console.error(`Container ${this.containerId} not found`);
            return;
        }
        
        this.chart = echarts.init(dom, null, {
            renderer: 'canvas',
            useDirtyRect: false
        });
        
        this.setOption();
    }
    
    setOption() {
        const option = {
            title: {
                text: this.chartName,
                left: 'center'
            },
            tooltip: {
                trigger: 'axis',
                formatter: (params) => {
                    params = params[0];
                    return `${params.name} : ${params.value[1]}`;
                },
                axisPointer: {
                    animation: false
                }
            },
            xAxis: {
                type: 'value',
                splitLine: {
                    show: false
                }
            },
            yAxis: {
                type: 'value',
                boundaryGap: [0, '100%'],
                splitLine: {
                    show: false
                }
            },
            series: [{
                name: this.chartName,
                type: 'line',
                showSymbol: false,
                data: this.data
            }]
        };
        
        if (this.chart) {
            this.chart.setOption(option);
        }
    }
    
    // 添加新数据点
    addDataPoint(value, timestamp = this.data.length) {
        if(timestamp < this.lastT){
          this.data = []
        }
        this.lastT = timestamp
        const dataPoint = {
            name: `${this.xName} ${timestamp}`,
            value: [
                timestamp,
                Math.round(value)
            ]
        };
        
        this.data.push(dataPoint);
    }
    
    // 更新图表
    updateChart() {
        if (this.chart) {
            this.chart.setOption({
                series: [{
                    data: this.data
                }]
            });
        }
    }
    
    // 开始
    startUpdate(interval = 1000) {
        if (this.isPlaying) return;
        
        this.isPlaying = true;
        this.intervalId = setInterval(async () => {
            const response = await fetch(this.loadUrl + `&offset=${this.data.length}`);
            const data = await response.json();

            if (data) {
                for (const item of data) {
                    this.addDataPoint(item["value"]);
                }
                this.updateChart();
            }
        }, interval);
    }
    
    // 停止
    stopUpdate() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
            this.isPlaying = false;
        }
    }
    
    // 从后端加载数据
    async loadDataFromBackend() {
        try {
            const response = await fetch(this.loadUrl);
            const data = await response.json();
            
            console.log(data)
            if (data) {
                this.data = data.map(item => ({
                  name: `${this.xName} ${item["timestamp"]}`,
                  value: [
                      item["timestamp"],
                      Math.round(item["value"])
                  ]
                }));
                this.updateChart();
            }
        } catch (error) {
            console.error('Failed to load data from backend:', error);
        }
    }
    
    // 调整大小
    resize() {
        if (this.chart) {
            this.chart.resize();
        }
    }
    
    // 销毁图表
    destroy() {
        this.stopSimulation();
        if (this.chart) {
            this.chart.dispose();
            this.chart = null;
        }
    }
}

const epochRwardChart = new DynamicChart('epoch_r', 'Epoch Reward', 'epoch', 'test');
const stepRwardChart = new DynamicChart('sum_step_r', 'Step Reward', 'step', 'test');

export const charts = {
  'epoch_r': epochRwardChart,
  'sum_step_r': stepRwardChart
}

window.addEventListener('resize', () => {
    epochRwardChart.resize();
    stepRwardChart.resize();
});

window.addEventListener('load', () => {
    epochRwardChart.loadDataFromBackend();
    stepRwardChart.loadDataFromBackend(); 
});