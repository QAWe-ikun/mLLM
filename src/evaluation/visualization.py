"""轨迹可视化与失败分析"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json


class TrajectoryVisualizer:
    """Agent轨迹可视化工具"""
    
    def __init__(self, output_dir: str = 'results/trajectories'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_trajectory_2d(self, positions: List[Tuple[float, float, float]],
                           target_pos: Tuple[float, float, float],
                           scene_name: str,
                           target_object: str,
                           success: bool,
                           save_name: Optional[str] = None) -> str:
        """绘制2D俯视图轨迹
        
        Args:
            positions: Agent位置列表 [(x, y, z), ...]
            target_pos: 目标物体位置
            scene_name: 场景名称
            target_object: 目标物体
            success: 是否成功
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 提取x, z坐标 (俯视图)
        x_coords = [p[0] for p in positions]
        z_coords = [p[2] for p in positions]
        
        # 绘制轨迹
        ax.plot(x_coords, z_coords, 'b-', linewidth=2, alpha=0.6, label='Agent Path')
        
        # 绘制起点
        ax.scatter(x_coords[0], z_coords[0], c='green', s=150, marker='o', 
                  label='Start', edgecolors='darkgreen', linewidths=2, zorder=5)
        
        # 绘制终点
        ax.scatter(x_coords[-1], z_coords[-1], c='red' if not success else 'lime', 
                  s=150, marker='X', label='End', edgecolors='darkred', linewidths=2, zorder=5)
        
        # 绘制目标位置
        ax.scatter(target_pos[0], target_pos[2], c='orange', s=200, marker='*',
                  label=f'Target: {target_object}', edgecolors='darkorange', linewidths=2, zorder=5)
        
        # 绘制成功区域 (圆形)
        success_circle = plt.Circle((target_pos[0], target_pos[2]), 1.5, 
                                   fill=False, linestyle='--', color='orange', alpha=0.5)
        ax.add_patch(success_circle)
        
        # 设置标题
        status = 'SUCCESS' if success else 'FAILED'
        ax.set_title(f'{scene_name} - {target_object}\n'
                    f'Status: {status} | Steps: {len(positions)}',
                    fontsize=14, fontweight='bold')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Z Position (m)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 保存
        if save_name is None:
            save_name = f'traj_{scene_name}_{target_object}.png'
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_multiple_trajectories(self, trajectories: List[Dict],
                                   save_name: str = 'all_trajectories.png') -> str:
        """绘制多个轨迹对比
        
        Args:
            trajectories: 列表，每个元素包含:
                - positions: [(x,y,z), ...]
                - target_pos: (x,y,z)
                - scene: str
                - target: str
                - success: bool
            save_name: 保存文件名
        """
        n = min(len(trajectories), 6)  # 最多6个
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, traj in enumerate(trajectories[:n]):
            ax = axes[i]
            
            positions = traj['positions']
            target_pos = traj['target_pos']
            success = traj['success']
            
            x_coords = [p[0] for p in positions]
            z_coords = [p[2] for p in positions]
            
            ax.plot(x_coords, z_coords, 'b-', linewidth=1.5, alpha=0.6)
            ax.scatter(x_coords[0], z_coords[0], c='green', s=100, marker='o', zorder=5)
            ax.scatter(x_coords[-1], z_coords[-1], c='red' if not success else 'lime', 
                      s=100, marker='X', zorder=5)
            ax.scatter(target_pos[0], target_pos[2], c='orange', s=150, marker='*', zorder=5)
            
            status = '✓' if success else '✗'
            ax.set_title(f'{traj["scene"]}\n{traj["target"]} {status}', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # 隐藏多余子图
        for i in range(n, 6):
            axes[i].axis('off')
        
        plt.suptitle('Navigation Trajectories Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_reward_curve(self, rewards_per_episode: List[List[float]],
                         save_name: str = 'reward_curve.png') -> str:
        """绘制每episode奖励曲线"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        total_rewards = [sum(r) for r in rewards_per_episode]
        ax.plot(total_rewards, 'b-', alpha=0.7)
        
        # 移动平均
        window = 10
        if len(total_rewards) > window:
            ma = np.convolve(total_rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(total_rewards)), ma, 'r-', linewidth=2, 
                   label=f'Moving Average ({window})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)


class FailureAnalyzer:
    """失败案例分析工具"""
    
    def __init__(self, output_dir: str = 'results/failure_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.failure_cases = []
    
    def add_failure_case(self, episode_result, observations: Optional[List] = None):
        """添加失败案例"""
        case = {
            'scene': episode_result.scene,
            'target': episode_result.target_object,
            'steps': episode_result.steps,
            'final_distance': episode_result.final_distance,
            'total_reward': sum(episode_result.rewards),
            'failure_type': self._classify_failure(episode_result),
        }
        self.failure_cases.append(case)
    
    def _classify_failure(self, episode_result) -> str:
        """分类失败类型"""
        if episode_result.final_distance > 5.0:
            return 'navigation_failed'  # 根本没靠近
        elif episode_result.final_distance > 1.5:
            return 'partial_navigation'  # 部分导航，但没到达
        else:
            return 'pickup_failed'  # 到达了但没拾取
    
    def generate_report(self, save_name: str = 'failure_report.json') -> str:
        """生成失败分析报告"""
        if not self.failure_cases:
            return "No failure cases to analyze."
        
        # 统计
        failure_types = {}
        for case in self.failure_cases:
            ft = case['failure_type']
            if ft not in failure_types:
                failure_types[ft] = 0
            failure_types[ft] += 1
        
        # 按场景统计
        scene_failures = {}
        for case in self.failure_cases:
            scene = case['scene']
            if scene not in scene_failures:
                scene_failures[scene] = 0
            scene_failures[scene] += 1
        
        # 按目标物体统计
        object_failures = {}
        for case in self.failure_cases:
            obj = case['target']
            if obj not in object_failures:
                object_failures[obj] = 0
            object_failures[obj] += 1
        
        report = {
            'total_failures': len(self.failure_cases),
            'failure_type_distribution': failure_types,
            'failure_by_scene': scene_failures,
            'failure_by_object': object_failures,
            'avg_final_distance': np.mean([c['final_distance'] for c in self.failure_cases]),
            'avg_steps': np.mean([c['steps'] for c in self.failure_cases]),
            'top_failure_cases': sorted(self.failure_cases, 
                                       key=lambda x: x['final_distance'], 
                                       reverse=True)[:5],
        }
        
        save_path = self.output_dir / save_name
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 打印摘要
        print(f"\n=== Failure Analysis Report ===")
        print(f"Total failures: {len(self.failure_cases)}")
        print(f"Failure types: {failure_types}")
        print(f"Most failing scenes: {sorted(scene_failures.items(), key=lambda x: -x[1])[:3]}")
        print(f"Most failing objects: {sorted(object_failures.items(), key=lambda x: -x[1])[:3]}")
        print(f"Report saved to: {save_path}")
        
        return str(save_path)
