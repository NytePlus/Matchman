import os
import time
from threading import Thread, Lock
from tensorboard.backend.event_processing import event_accumulator
from flask import Flask, jsonify, request
import json

class TensorboardRepository(Thread):
    def __init__(self, log_dir, runs=[], tags=[], poll_interval=-1):
        super().__init__()
        self.log_dir = log_dir
        self.poll_interval = poll_interval
        self.runs = runs
        self.tags = tags

        self.daemon = True
        self._running = False
        self.last_reload = time.time()
        self.is_async = poll_interval != -1
        
        self.accumulators = {}
        self.data = {}
        self.data_lock = Lock()
        for run in self.runs:
            self.data[run] = {}
            for tag in self.tags:
                self.data[run][tag] = []

        for run_name in self.runs:
            run_path = os.path.join(self.log_dir, run_name)
            if os.path.exists(run_path) and os.path.isdir(run_path):
                self._create_accumulator(run_name, run_path)
    
    def _create_accumulator(self, run_name, run_path):
        """为单个run创建accumulator"""
        try:
            accumulator = event_accumulator.EventAccumulator(
                run_path,
                size_guidance={
                    event_accumulator.SCALARS: 100000,
                    event_accumulator.IMAGES: 0,
                    event_accumulator.HISTOGRAMS: 0,
                    event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                }
            )
            accumulator.Reload()
            self.accumulators[run_name] = accumulator
        except Exception as e:
            print(f"Failed to create accumulator for {run_name}: {e}")
    
    def run(self):
        """后台线程：定期重载数据"""
        if self.is_async:
            self._running = True
            while self._running:
                try:
                    self._reload_all()
                    time.sleep(self.poll_interval)
                except Exception as e:
                    print(f"Reload error: {e}")
                    time.sleep(self.poll_interval)
    
    def _reload_all(self):
        """重载所有accumulator的数据"""
        for run_name, accumulator in self.accumulators.items():
            try:
                accumulator.Reload()
                for tag in self.tags:
                    scalars = accumulator.Scalars(tag)
                    step_groups = {}
                    for s in scalars:
                        if (s.step not in step_groups or 
                            s.wall_time > step_groups[s.step].wall_time):
                            step_groups[s.step] = s
                    
                    with self.data_lock:
                        self.data[run_name][tag] = [
                            {
                                'timestamp': step,
                                'wall_time': s.wall_time,
                                'value': s.value
                            }
                            for step, s in sorted(step_groups.items())
                        ]
            except Exception as e:
                print(f"Reload failed for {run_name}: {e}")
    
    def stop(self):
        self._running = False
    
    def get_scalar_data(self, run, tag, offset=0, limit=1000):
        """获取标量数据"""
        now = time.time()
        if not self.is_async and now - self.last_reload > 3:
            self._reload_all()
            self.last_reload = now
            
        with self.data_lock:
            data = self.data.get(run, {}).get(tag, [])
            begin = min(offset, len(data))
            end = min(offset + limit, len(data))
            return data[begin: end]