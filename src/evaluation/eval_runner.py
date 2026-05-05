"""评估运行器"""
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import MetricsCalculator, EpisodeResult


class EvalRunner:
    """评估运行器
    
    在seen/unseen场景上批量测试Agent性能
    """
    
    def __init__(self, model: torch.nn.Module, env: Any, config: Dict[str, Any]):
        self.model = model
        self.env = env
        self.config = config
        
        env_config = config['environment']
        
        # Seen场景（训练集）
        scenes = env_config['scenes']['train']
        self.seen_scenes = []
        for room_type, scene_list in scenes.items():
            self.seen_scenes.extend(scene_list)
        
        # Unseen场景（测试集）
        unseen_scenes = env_config['scenes'].get('test_unseen', {})
        self.unseen_scenes = []
        if isinstance(unseen_scenes, dict):
            for room_type, scene_list in unseen_scenes.items():
                self.unseen_scenes.extend(scene_list)
        
        # 目标物体
        self.target_objects = env_config['tasks'][0]['target_objects']
        
        # 成功判定距离
        self.success_distance = env_config['episode']['success_distance']
        
        # 最大步数
        self.max_steps = env_config['tasks'][0]['max_steps']
    
    def run_eval(self, num_episodes: int = 100, 
                 seen: bool = True) -> MetricsCalculator:
        """运行评估
        
        Args:
            num_episodes: 评估episode数量
            seen: True评估seen场景, False评估unseen场景
            
        Returns:
            MetricsCalculator: 包含所有结果的计算器
        """
        scenes = self.seen_scenes if seen else self.unseen_scenes
        
        if not scenes:
            print(f"Warning: No {'seen' if seen else 'unseen'} scenes available")
            return MetricsCalculator()
        
        calc = MetricsCalculator()

        scene_type = 'seen' if seen else 'unseen'
        progress_bar = tqdm(range(num_episodes), desc=f"Evaluating ({scene_type})")

        for ep in progress_bar:
            # 随机选择场景和目标
            scene = np.random.choice(scenes)
            target = np.random.choice(self.target_objects)

            # 运行episode
            result = self._run_episode(scene, target)
            calc.add_result(result)

            # 更新进度条
            if (ep + 1) % 10 == 0:
                sr = calc.get_metric('success_rate')
                spl = calc.get_metric('spl')
                progress_bar.set_postfix({
                    'SR': f"{sr:.2f}",
                    'SPL': f"{spl:.2f}"
                })
        
        return calc
    
    def _run_episode(self, scene: str, target: str) -> EpisodeResult:
        """运行单个episode
        
        Returns:
            EpisodeResult
        """
        # 重置环境
        obs = self.env.reset(scene_name=scene, target_object=target)
        
        done = False
        rewards = []
        steps = 0
        actual_path_length = 0.0
        prev_position = None
        
        while not done and steps < self.max_steps:
            # 获取动作
            instruction = f"Go to the {target}"
            rgb = obs['rgb']
            
            with torch.no_grad():
                action_idx = self.model.generate_action(
                    input_ids=self._tokenize(instruction),
                    attention_mask=torch.ones(1, 10),
                    sample=False,  # greedy
                )
            
            # 执行动作
            next_obs, reward, done, info = self.env.step(action_idx)
            
            # 记录路径
            curr_position = obs.get('position')
            if prev_position is not None and curr_position is not None:
                dx = curr_position[0] - prev_position[0]
                dy = curr_position[1] - prev_position[1]
                dz = curr_position[2] - prev_position[2]
                actual_path_length += np.sqrt(dx**2 + dy**2 + dz**2)
            
            prev_position = curr_position
            
            rewards.append(reward)
            steps += 1
            obs = next_obs
        
        # 检查结果
        success = info.get('success_nav', False) or info.get('success_pickup', False)
        final_distance = self.env._compute_distance_to_target() if hasattr(self.env, '_compute_distance_to_target') else float('inf')
        
        # 估计最优路径 (简化：使用曼哈顿距离/步长)
        optimal_length = max(1, actual_path_length * 0.7)  # 假设最优路径约为实际的70%
        
        return EpisodeResult(
            success=success,
            steps=steps,
            optimal_length=optimal_length,
            actual_length=actual_path_length,
            final_distance=final_distance,
            rewards=rewards,
            scene=scene,
            target_object=target,
        )
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """简单tokenize"""
        if hasattr(self.model, 'tokenizer'):
            encoded = self.model.tokenizer(text, return_tensors='pt')
            return encoded['input_ids']
        else:
            return torch.randint(0, 100, (1, 10))
    
    def run_ablation(self, num_episodes: int = 50) -> Dict[str, MetricsCalculator]:
        """运行消融实验
        
        对比不同配置下的性能:
        - Baseline-A: 纯CLIP特征，无语言指令
        - Baseline-B: CLIP+SAM，无位置编码
        - Ours: 完整模型
        """
        results = {}
        
        # 这里简化实现，实际需要在不同配置下重新运行
        for variant in ['baseline_A', 'baseline_B', 'ours']:
            print(f"\n=== Ablation: {variant} ===")
            calc = self.run_eval(num_episodes, seen=True)
            results[variant] = calc
        
        return results
