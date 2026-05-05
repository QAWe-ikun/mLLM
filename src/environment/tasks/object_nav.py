"""ObjectNav任务定义"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import yaml
from pathlib import Path


class ObjectNavTask:
    """目标导航任务
    
    给定目标物体类别，Agent需导航至该物体可视范围内（距离<=success_distance）
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        env_config = config['environment']
        
        # 目标物体列表
        self.target_objects = env_config['tasks'][0]['target_objects']
        
        # 成功判定距离
        self.success_distance = env_config['episode']['success_distance']
        
        # 最大步数
        self.max_steps = env_config['tasks'][0]['max_steps']
    
    def sample_target(self) -> str:
        """随机采样目标物体"""
        return np.random.choice(self.target_objects)
    
    def check_success(self, distance: float, is_visible: bool) -> bool:
        """检查导航是否成功
        
        Args:
            distance: Agent到目标的距离
            is_visible: 目标是否可见
            
        Returns:
            是否成功
        """
        return is_visible and distance <= self.success_distance
    
    def compute_spl(self, success: bool, optimal_length: float, episode_length: float) -> float:
        """计算SPL (Success weighted by Path Length)
        
        SPL = (1/N) * sum(S_i * l_i* / max(l_i*, l_i))
        
        Args:
            success: 是否成功
            optimal_length: 最短路径长度
            episode_length: 实际路径长度
            
        Returns:
            SPL值
        """
        if not success:
            return 0.0
        
        if optimal_length == 0:
            return 1.0 if episode_length == 0 else 0.0
        
        return success * (optimal_length / max(optimal_length, episode_length))
    
    def get_instruction_templates(self) -> List[str]:
        """获取导航指令模板"""
        return [
            "Go to the {target}",
            "Find the {target}",
            "Navigate to the {target}",
            "Walk over to the {target}",
        ]
    
    def generate_instruction(self, target: str) -> str:
        """生成随机指令"""
        templates = self.get_instruction_templates()
        template = np.random.choice(templates)
        return template.format(target=target)
