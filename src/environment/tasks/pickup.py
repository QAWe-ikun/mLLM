"""Pickup交互任务定义"""
import numpy as np
from typing import Dict, Any, Optional


class PickupTask:
    """拾取交互任务
    
    在到达目标物体后，执行拾取动作
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        env_config = config['environment']
        
        # Pickup最大步数
        self.max_steps = env_config['tasks'][1]['max_steps']
        
        # 拾取前需要先导航
        self.requires_nav = env_config['tasks'][1]['requires_nav']
    
    def check_pickup_success(self, held_object: Optional[str], target_object: str) -> bool:
        """检查拾取是否成功
        
        Args:
            held_object: 当前持有的物体
            target_object: 目标物体
            
        Returns:
            是否成功拾取
        """
        return held_object == target_object
    
    def generate_instruction(self, target: str) -> str:
        """生成拾取指令"""
        templates = [
            "Pick up the {target}",
            "Grab the {target}",
        ]
        return np.random.choice(templates).format(target=target)
