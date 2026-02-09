import time
import psutil
import os
import functools
import json
import pandas as pd
from datetime import datetime
#completed
def measure_resource_usage(func):
    """Decorator to measure execution time and memory"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        # Memory before
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Execution time
        start_time = time.time()
        start_perf = time.perf_counter()
        
        # Function execution
        result = func(*args, **kwargs)
        
        end_perf = time.perf_counter()
        end_time = time.time()
        
        # Memory after
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        
        return {
            "exec_time_ms": round((end_perf - start_perf) * 1000, 3),
            "wall_time_ms": round((end_time - start_time) * 1000, 3),
            "mem_used_MB": round(mem_after - mem_before, 4),
            "mem_total_MB": round(mem_after, 2),
            "result": result
        }
    return wrapper

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / (1024 * 1024)
        self.usage_data = []
        
        # Create output directory
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.output_dir = os.path.join(BASE_DIR, "data", "mcu", "outputs", "resource_monitoring")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_current_usage(self):
        """Returns current resource usage"""
        memory = self.process.memory_info().rss / (1024 * 1024)
        cpu = self.process.cpu_percent()
        
        usage_data = {
            "timestamp": datetime.now().isoformat(),
            "memory_MB": round(memory, 2),
            "memory_delta_MB": round(memory - self.initial_memory, 2),
            "cpu_percent": round(cpu, 1)
        }
        
        self.usage_data.append(usage_data)
        
        # Save periodically
        if len(self.usage_data) % 10 == 0:
            self.save_usage_data()
        
        return usage_data
    
    def save_usage_data(self):
        """Saves resource usage data"""
        try:
            # Save to CSV
            csv_path = os.path.join(self.output_dir, "cpu_memory_usage.csv")
            df = pd.DataFrame(self.usage_data)
            df.to_csv(csv_path, index=False)
            
            # Save JSON report
            report_path = os.path.join(self.output_dir, "performance_report.json")
            if len(self.usage_data) > 0:
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "samples_collected": len(self.usage_data),
                    "average_memory_MB": round(df["memory_MB"].mean(), 2),
                    "max_memory_MB": round(df["memory_MB"].max(), 2),
                    "average_cpu_percent": round(df["cpu_percent"].mean(), 1),
                    "max_cpu_percent": round(df["cpu_percent"].max(), 1)
                }
                
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                    
        except Exception as e:
            print(f"Resource data save error: {e}")