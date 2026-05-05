#!/usr/bin/env python
"""
脚本04: 模型评估

在seen/unseen场景上评估Agent性能，计算SR/SPL指标

用法:
    python scripts/04_evaluate.py --model results/ppo_checkpoints/final --episodes 100
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import json

from src.utils.config import load_all_configs
from src.utils.logger import Logger
from src.models.vla_backbone import VLABackbone
from src.environment.ai2thor_wrapper import AI2THORWrapper
from src.evaluation.eval_runner import EvalRunner


def main():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--model', type=str, default='results/ppo_checkpoints/final',
                       help='Path to trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--seen', action='store_true',
                       help='Evaluate on seen scenes (default: unseen)')
    parser.add_argument('--output', type=str, default='results/metrics',
                       help='Output directory for metrics')
    
    args = parser.parse_args()
    
    # 加载配置
    configs = load_all_configs()
    logger = Logger('Eval_Script')
    
    logger.info("=" * 60)
    logger.info("Starting Evaluation")
    logger.info("=" * 60)
    
    # 初始化环境
    logger.info("Initializing AI2-THOR environment...")
    try:
        env = AI2THORWrapper(configs)
        logger.info("Environment initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        return
    
    # 初始化模型
    logger.info(f"Loading model from {args.model}")
    model = VLABackbone(configs['model'])
    
    if Path(args.model).exists():
        logger.info("Loading trained weights...")
        if hasattr(model, 'load_model'):
            model.load_model(args.model)
    
    # 初始化评估运行器
    eval_runner = EvalRunner(model, env, configs)
    
    # 运行评估
    scene_type = 'seen' if args.seen else 'unseen'
    logger.info(f"Evaluating on {scene_type} scenes, {args.episodes} episodes")
    
    metrics_calc = eval_runner.run_eval(
        num_episodes=args.episodes,
        seen=args.seen,
    )
    
    # 计算指标
    metrics = metrics_calc.compute_all_metrics()
    
    logger.info(f"\n=== Evaluation Results ({scene_type}) ===")
    logger.info(f"Success Rate: {metrics['success_rate']:.4f}")
    logger.info(f"SPL:          {metrics['spl']:.4f}")
    logger.info(f"Avg Steps:    {metrics['avg_steps']:.2f}")
    logger.info(f"Path Eff.:    {metrics['path_efficiency']:.4f}")
    logger.info(f"Avg Final Dist: {metrics['avg_final_distance']:.2f}m")
    logger.info(f"Avg Reward:   {metrics['avg_total_reward']:.2f}")
    
    # 保存指标
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_dir / f'eval_{scene_type}.json'
    with open(metrics_file, 'w') as f:
        json.dump({
            'scene_type': scene_type,
            'num_episodes': args.episodes,
            'metrics': metrics,
            'by_scene': metrics_calc.compute_by_scene(),
            'by_object': metrics_calc.compute_by_object(),
        }, f, indent=2)
    
    logger.info(f"\nMetrics saved to {metrics_file}")
    
    # 保存详细结果
    details_file = output_dir / f'eval_{scene_type}_details.json'
    metrics_calc.save_results(str(details_file))
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
