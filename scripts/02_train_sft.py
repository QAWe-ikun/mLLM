#!/usr/bin/env python
"""
脚本02: SFT监督微调训练

使用专家演示数据交叉熵损失微调VLA模型

用法:
    python scripts/02_train_sft.py --data data/sft_trajectories.json --epochs 3
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import torch

from src.utils.config import load_all_configs
from src.utils.logger import Logger
from src.models.vla_backbone import VLABackbone
from src.agent.sft_trainer import SFTTrainer


def main():
    parser = argparse.ArgumentParser(description='SFT Training')
    parser.add_argument('--data', type=str, default='data/sft_trajectories.json',
                       help='Path to demonstration data')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override training epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--output', type=str, default='results/sft_checkpoints',
                       help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # 加载配置
    configs = load_all_configs()
    logger = Logger('SFT_Script')
    
    logger.info("=" * 60)
    logger.info("Starting SFT Training")
    logger.info("=" * 60)
    
    # 覆盖配置
    if args.epochs:
        configs['train']['sft']['training']['epochs'] = args.epochs
    if args.lr:
        configs['train']['sft']['training']['learning_rate'] = args.lr
    
    # 初始化模型 (VLABackbone 内部会取 configs['model'])
    logger.info("Loading VLA backbone...")
    model = VLABackbone(configs)
    
    # 初始化训练器
    trainer = SFTTrainer(
        model=model,
        config=configs,
        output_dir=args.output,
    )
    
    # 开始训练
    logger.info(f"Training data: {args.data}")
    trainer.train(data_path=args.data)
    
    logger.info("SFT Training completed!")


if __name__ == '__main__':
    main()
