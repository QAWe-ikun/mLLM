#!/usr/bin/env python
"""
脚本01: 生成SFT演示数据

使用A*启发式策略生成专家演示数据
- 导航: A*最短路径
- 拾取: 距离<=1.5m时执行Pickup
- 数据增强: 随机偏离1-2步

用法:
    python scripts/01_generate_sft_data.py --output data/sft_trajectories.json --num 5000
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def generate_dummy_trajectories(num_trajectories: int, output_path: str):
    """生成dummy演示数据（无AI2-THOR环境时）"""
    print(f"Generating {num_trajectories} dummy trajectories...")
    
    objects = ['Microwave', 'Fridge', 'TV', 'Laptop', 'Sofa', 'Chair', 'Bed', 'Sink', 'Toilet', 'Lamp']
    scenes = [f'FloorPlan{i}' for i in list(range(1, 6)) + list(range(201, 206)) + list(range(301, 306))]
    instructions = ['Go to the {obj}', 'Find the {obj}', 'Navigate to the {obj}', 'Walk over to the {obj}']
    actions = ['MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown', 'Pickup']
    
    trajectories = []
    
    for traj_idx in tqdm(range(num_trajectories)):
        scene = np.random.choice(scenes)
        target = np.random.choice(objects)
        instruction = np.random.choice(instructions).format(obj=target)
        
        # 生成随机轨迹 (10-30步)
        num_steps = np.random.randint(10, 31)
        trajectory = {
            'scene': scene,
            'target': target,
            'instruction': instruction,
            'steps': [],
        }
        
        for step_idx in range(num_steps):
            # 前90%步为导航动作，最后1步为Pickup
            if step_idx < num_steps - 1:
                action = np.random.choice(actions[:3], p=[0.6, 0.2, 0.2])  # MoveAhead更频繁
            else:
                action = 'Pickup'
            
            trajectory['steps'].append({
                'step': step_idx,
                'action': action,
                'action_idx': actions.index(action),
            })
        
        trajectories.append(trajectory)
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(trajectories, f, indent=2)
    
    print(f"Saved {len(trajectories)} trajectories to {output_path}")
    
    # 统计
    total_steps = sum(len(t['steps']) for t in trajectories)
    print(f"Total steps: {total_steps}")
    print(f"Avg steps per trajectory: {total_steps / len(trajectories):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Generate SFT demonstration data')
    parser.add_argument('--output', type=str, default='data/sft_trajectories.json',
                       help='Output file path')
    parser.add_argument('--num', type=int, default=5000,
                       help='Number of trajectories to generate')
    parser.add_argument('--with_env', action='store_true',
                       help='Use AI2-THOR environment for real data generation')
    
    args = parser.parse_args()
    
    if args.with_env:
        print("Warning: Real AI2-THOR data generation not yet implemented.")
        print("Generating dummy data instead.")
    
    generate_dummy_trajectories(args.num, args.output)


if __name__ == '__main__':
    main()
