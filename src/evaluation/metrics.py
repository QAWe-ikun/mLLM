"""评估指标计算 - SR/SPL"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class EpisodeResult:
    """单个episode的结果"""
    success: bool
    steps: int
    optimal_length: float  # 最短路径长度
    actual_length: float   # 实际路径长度
    final_distance: float  # 最终到目标距离
    rewards: List[float]
    scene: str
    target_object: str


class MetricsCalculator:
    """评估指标计算器
    
    主要指标:
    - Success Rate (SR): 成功率
    - SPL: Success weighted by Path Length
    """
    
    def __init__(self):
        self.results: List[EpisodeResult] = []
    
    def add_result(self, result: EpisodeResult):
        """添加episode结果"""
        self.results.append(result)
    
    def add_batch(self, results: List[EpisodeResult]):
        """批量添加"""
        self.results.extend(results)
    
    def clear(self):
        """清空结果"""
        self.results = []
    
    def compute_success_rate(self) -> float:
        """计算成功率
        
        SR = N_success / N_total
        """
        if not self.results:
            return 0.0
        
        n_success = sum(1 for r in self.results if r.success)
        return n_success / len(self.results)
    
    def compute_spl(self) -> float:
        """计算SPL (Success weighted by Path Length)
        
        SPL = (1/N) * sum(S_i * l_i* / max(l_i*, l_i))
        
        其中:
        - S_i: 第i个episode是否成功 (0/1)
        - l_i*: 最短路径长度
        - l_i: 实际路径长度
        """
        if not self.results:
            return 0.0
        
        spl_values = []
        for r in self.results:
            if not r.success:
                spl_values.append(0.0)
            else:
                if r.optimal_length == 0:
                    spl = 1.0 if r.actual_length == 0 else 0.0
                else:
                    spl = r.optimal_length / max(r.optimal_length, r.actual_length)
                spl_values.append(spl)
        
        return np.mean(spl_values)
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """计算所有指标"""
        return {
            'success_rate': self.compute_success_rate(),
            'spl': self.compute_spl(),
            'avg_steps': self.compute_avg_steps(),
            'path_efficiency': self.compute_path_efficiency(),
            'avg_final_distance': self.compute_avg_final_distance(),
            'avg_total_reward': self.compute_avg_total_reward(),
        }
    
    def compute_avg_steps(self) -> float:
        """平均步数（仅成功episode）"""
        success_results = [r for r in self.results if r.success]
        if not success_results:
            return 0.0
        return np.mean([r.steps for r in success_results])
    
    def compute_path_efficiency(self) -> float:
        """路径效率 (l*/l)"""
        success_results = [r for r in self.results if r.success and r.optimal_length > 0]
        if not success_results:
            return 0.0
        
        efficiencies = [r.optimal_length / r.actual_length for r in success_results]
        return np.mean(efficiencies)
    
    def compute_avg_final_distance(self) -> float:
        """平均最终距离"""
        if not self.results:
            return 0.0
        return np.mean([r.final_distance for r in self.results])
    
    def compute_avg_total_reward(self) -> float:
        """平均总奖励"""
        if not self.results:
            return 0.0
        return np.mean([sum(r.rewards) for r in self.results])
    
    def compute_by_scene(self) -> Dict[str, Dict[str, float]]:
        """按场景分组计算指标"""
        scene_results = {}
        for r in self.results:
            if r.scene not in scene_results:
                scene_results[r.scene] = []
            scene_results[r.scene].append(r)
        
        scene_metrics = {}
        for scene, results in scene_results.items():
            calc = MetricsCalculator()
            calc.add_batch(results)
            scene_metrics[scene] = calc.compute_all_metrics()
        
        return scene_metrics
    
    def compute_by_object(self) -> Dict[str, Dict[str, float]]:
        """按目标物体分组计算指标"""
        obj_results = {}
        for r in self.results:
            if r.target_object not in obj_results:
                obj_results[r.target_object] = []
            obj_results[r.target_object].append(r)
        
        obj_metrics = {}
        for obj, results in obj_results.items():
            calc = MetricsCalculator()
            calc.add_batch(results)
            obj_metrics[obj] = calc.compute_all_metrics()
        
        return obj_metrics
    
    def save_results(self, filepath: str):
        """保存结果到JSON"""
        import json
        from dataclasses import asdict
        
        data = {
            'summary': self.compute_all_metrics(),
            'episodes': [asdict(r) for r in self.results],
            'by_scene': self.compute_by_scene(),
            'by_object': self.compute_by_object(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filepath}")
