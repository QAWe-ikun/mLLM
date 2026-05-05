#!/usr/bin/env python
"""
脚本03: PPO强化学习训练

在SFT基础上继续训练，优化长期回报

用法:
    python scripts/03_train_ppo.py --sft_ckpt results/sft_checkpoints/final --steps 50000
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

from src.utils.logger import Logger
from src.agent.ppo_trainer import PPOTrainer
from src.utils.config import load_all_configs
from src.models.vla_backbone import VLABackbone
from src.environment.ai2thor_wrapper import AI2THORWrapper


def main():
    parser = argparse.ArgumentParser(description='PPO Training')
    parser.add_argument('--sft_ckpt', type=str, default='results/sft_checkpoints/final',
                       help='Path to SFT checkpoint')
    parser.add_argument('--steps', type=int, default=None,
                       help='Override total PPO steps')
    parser.add_argument('--output', type=str, default='results/ppo_checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--eval_interval', type=int, default=None,
                       help='Override evaluation interval')
    
    args = parser.parse_args()
    
    # 加载配置
    configs = load_all_configs()
    logger = Logger('PPO_Script')
    
    logger.info("=" * 60)
    logger.info("Starting PPO Training")
    logger.info("=" * 60)
    
    # 覆盖配置
    if args.steps:
        configs['train']['ppo']['training']['total_steps'] = args.steps
    if args.eval_interval:
        configs['train']['ppo']['evaluation']['eval_interval'] = args.eval_interval
    
    # 初始化环境
    logger.info("Initializing AI2-THOR environment...")
    try:
        env = AI2THORWrapper(configs)
        logger.info("Environment initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize AI2-THOR: {e}")
        logger.warning("Using dummy environment for testing")
        env = None
    
    if env is None:
        print("Please ensure AI2-THOR is installed and running on WSL/Linux")
        return
    
    # 初始化模型 (VLABackbone 内部会取 configs['model'])
    logger.info("Loading VLA backbone from SFT checkpoint...")
    model = VLABackbone(configs)
    
    # 加载SFT权重 (如果有)
    if Path(args.sft_ckpt).exists():
        logger.info(f"Loading SFT weights from {args.sft_ckpt}")
        if hasattr(model, 'load_model'):
            model.load_model(args.sft_ckpt)
    
    # 初始化PPO训练器
    trainer = PPOTrainer(
        model=model,
        env=env,
        config=configs,
        output_dir=args.output,
    )
    
    # 开始训练
    logger.info("Starting PPO training loop...")
    trainer.train()
    
    logger.info("PPO Training completed!")


if __name__ == '__main__':
    main()
