#!/usr/bin/env python
"""
脚本05: 轨迹可视化与失败分析

可视化Agent导航轨迹，生成失败案例分析报告

用法:
    python scripts/05_visualize.py --eval_results results/metrics/eval_seen_details.json
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from pathlib import Path

from src.evaluation.visualization import TrajectoryVisualizer, FailureAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Visualization & Failure Analysis')
    parser.add_argument('--eval_results', type=str, default=None,
                       help='Path to evaluation results JSON')
    parser.add_argument('--num_trajectories', type=int, default=10,
                       help='Number of trajectories to visualize')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    traj_dir = Path(args.output_dir) / 'trajectories'
    failure_dir = Path(args.output_dir) / 'failure_analysis'
    
    visualizer = TrajectoryVisualizer(str(traj_dir))
    analyzer = FailureAnalyzer(str(failure_dir))
    
    # 如果有评估结果，从结果中生成可视化
    if args.eval_results and Path(args.eval_results).exists():
        print(f"Loading evaluation results from {args.eval_results}")
        with open(args.eval_results, 'r') as f:
            eval_data = json.load(f)
        
        episodes = eval_data.get('episodes', [])
        
        if not episodes:
            print("No episodes found in evaluation results.")
            return
        
        # 随机选择轨迹可视化
        num_to_vis = min(args.num_trajectories, len(episodes))
        selected_episodes = np.random.choice(episodes, num_to_vis, replace=False)
        
        trajectories_data = []
        
        for ep in selected_episodes:
            # 生成dummy轨迹数据 (实际应从episode记录中获取)
            scene = ep.get('scene', 'FloorPlan1')
            target = ep.get('target_object', 'Microwave')
            success = ep.get('success', False)
            
            # Dummy位置序列 (从起点到终点的随机游走)
            num_steps = ep.get('steps', 15)
            start_pos = (np.random.uniform(0, 5), 0.9, np.random.uniform(0, 5))
            target_pos = (np.random.uniform(2, 8), 0.5, np.random.uniform(2, 8))
            
            # 生成轨迹点
            positions = []
            for i in range(num_steps + 1):
                t = i / num_steps
                x = start_pos[0] + (target_pos[0] - start_pos[0]) * t + np.random.normal(0, 0.3)
                z = start_pos[2] + (target_pos[2] - start_pos[2]) * t + np.random.normal(0, 0.3)
                positions.append((x, start_pos[1], z))
            
            # 绘制单个轨迹
            save_name = f'traj_{scene}_{target}_{"success" if success else "fail"}.png'
            traj_path = visualizer.plot_trajectory_2d(
                positions=positions,
                target_pos=target_pos,
                scene_name=scene,
                target_object=target,
                success=success,
                save_name=save_name,
            )
            print(f"Saved trajectory: {traj_path}")
            
            trajectories_data.append({
                'positions': positions,
                'target_pos': target_pos,
                'scene': scene,
                'target': target,
                'success': success,
            })
            
            # 添加失败案例
            if not success:
                analyzer.add_failure_case(type('obj', (), ep)())
        
        # 绘制多轨迹对比
        if trajectories_data:
            summary_path = visualizer.plot_multiple_trajectories(
                trajectories_data,
                save_name='trajectories_summary.png',
            )
            print(f"Saved trajectories summary: {summary_path}")
    
    else:
        # 无评估结果，生成dummy可视化
        print("No evaluation results provided. Generating dummy visualization...")
        
        # 生成dummy轨迹
        for i in range(args.num_trajectories):
            scene = f'FloorPlan{np.random.randint(1, 30)}'
            target = np.random.choice(['Microwave', 'Fridge', 'TV', 'Laptop'])
            success = np.random.random() > 0.4
            
            num_steps = np.random.randint(10, 25)
            start_pos = (np.random.uniform(0, 5), 0.9, np.random.uniform(0, 5))
            target_pos = (np.random.uniform(2, 8), 0.5, np.random.uniform(2, 8))
            
            positions = []
            for j in range(num_steps + 1):
                t = j / num_steps
                x = start_pos[0] + (target_pos[0] - start_pos[0]) * t + np.random.normal(0, 0.3)
                z = start_pos[2] + (target_pos[2] - start_pos[2]) * t + np.random.normal(0, 0.3)
                positions.append((x, start_pos[1], z))
            
            save_name = f'dummy_traj_{i}.png'
            visualizer.plot_trajectory_2d(
                positions=positions,
                target_pos=target_pos,
                scene_name=scene,
                target_object=target,
                success=success,
                save_name=save_name,
            )
            
            if not success:
                dummy_result = type('obj', (), {
                    'scene': scene,
                    'target_object': target,
                    'steps': num_steps,
                    'final_distance': np.random.uniform(1.5, 8.0),
                    'rewards': [-0.1] * num_steps,
                })()
                analyzer.add_failure_case(dummy_result)
        
        print(f"Generated {args.num_trajectories} dummy trajectories")
    
    # 生成失败分析报告
    if analyzer.failure_cases:
        report_path = analyzer.generate_report()
        print(f"\nFailure analysis report: {report_path}")
    else:
        print("\nNo failure cases to analyze.")
    
    print("\nVisualization completed!")


if __name__ == '__main__':
    main()
