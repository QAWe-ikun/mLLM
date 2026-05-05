"""日志和实验跟踪工具"""
import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path


class Logger:
    """统一的日志管理器"""
    
    def __init__(self, name: str, log_file: Optional[str] = None, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # 文件handler（可选）
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)


class MetricsTracker:
    """训练指标跟踪器"""
    
    def __init__(self, log_dir: str, use_tensorboard: bool = False, use_wandb: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("Warning: tensorboard not available, install with: pip install tensorboard")
                self.use_tensorboard = False
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not available, install with: pip install wandb")
                self.use_wandb = False
        
        self.current_step = 0
        self.metrics_history = []
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """记录指标
        
        Args:
            metrics: 指标字典，如{'loss': 0.5, 'reward': 10.0}
            step: 训练步数，不指定则自动递增
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        self.metrics_history.append({'step': self.current_step, **metrics})
        
        # TensorBoard
        if self.use_tensorboard and self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.current_step)
        
        # WandB
        if self.use_wandb:
            self.wandb.log(metrics, step=self.current_step)
    
    def save_metrics(self, filename: str = 'metrics.json'):
        """保存指标到JSON文件"""
        import json
        filepath = self.log_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def close(self):
        if self.use_tensorboard and self.writer:
            self.writer.close()
        if self.use_wandb:
            self.wandb.finish()
