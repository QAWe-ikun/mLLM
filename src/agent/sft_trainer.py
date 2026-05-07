"""SFT监督微调训练器"""
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from typing import Dict, Any, List
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import Logger, MetricsTracker

class SFTDataset(Dataset):
    """SFT训练数据集
    
    加载演示数据：(instruction, visual_features, expert_action)
    """
    
    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # 加载数据
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """加载演示数据并展平为独立样本"""
        import json
        import os

        flattened_data = []
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                if self.data_path.endswith('.jsonl'):
                    lines = f.readlines()
                    episodes = [json.loads(line) for line in lines]
                else:
                    episodes = json.load(f)
            
            # 展平轨迹: episodes -> steps
            for episode in episodes:
                instruction = episode.get('instruction', '')
                for step in episode.get('steps', []):
                    flattened_data.append({
                        'instruction': instruction,
                        'action': step.get('action_idx', 0),
                        'scene': episode.get('scene', ''),
                    })
        else:
            print(f"Warning: Data file not found: {self.data_path}")
            print("Generating dummy data for testing")
            flattened_data = self._generate_dummy_data()

        return flattened_data
    
    def _generate_dummy_data(self, num_samples: int = 1000) -> List[Dict]:
        """生成dummy演示数据（用于测试）"""
        data = []
        objects = ['Microwave', 'Fridge', 'TV', 'Laptop', 'Sofa', 'Chair', 'Bed', 'Sink']
        instructions = ['Go to the {obj}', 'Find the {obj}', 'Navigate to the {obj}']
        
        for i in range(num_samples):
            obj = np.random.choice(objects)
            instr = np.random.choice(instructions).format(obj=obj)
            action = np.random.randint(0, 6)  # 6个动作
            
            data.append({
                'instruction': instr,
                'action': action,
                'scene': f'FloorPlan{np.random.randint(1, 30)}',
            })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        
        # Tokenize指令
        instruction = sample['instruction']
        if self.tokenizer:
            encoded = self.tokenizer(
                instruction,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
        else:
            # Dummy
            input_ids = torch.randint(0, 100, (self.max_seq_len,))
            attention_mask = torch.ones(self.max_seq_len)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'action': torch.tensor(sample['action'], dtype=torch.long),
            'instruction': instruction,
        }


class SFTTrainer:
    """SFT监督微调训练器
    
    使用专家演示数据交叉熵损失微调VLA模型
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 output_dir: str = 'results/sft_checkpoints'):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        sft_config = config['train']['sft']
        
        # 训练参数
        self.epochs = sft_config['training']['epochs']
        self.batch_size = sft_config['training']['per_device_train_batch_size']
        self.gradient_accumulation = sft_config['training']['gradient_accumulation_steps']
        self.learning_rate = sft_config['training']['learning_rate']
        self.warmup_ratio = sft_config['training']['warmup_ratio']
        self.max_grad_norm = sft_config['training']['max_grad_norm']
        self.logging_steps = sft_config['training']['logging_steps']
        self.save_steps = sft_config['training']['save_steps']
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # 日志
        self.logger = Logger('SFT', log_file=str(self.output_dir / 'sft.log'))
        self.metrics = MetricsTracker(
            log_dir=str(self.output_dir),
            use_tensorboard=True,
        )
        
        self.global_step = 0
    
    def train(self, data_path: str):
        """执行SFT训练
        
        Args:
            data_path: 演示数据路径 (JSON/JSONL)
        """
        self.logger.info(f"Starting SFT training, data: {data_path}")
        self.logger.info(f"Epochs: {self.epochs}, Batch size: {self.batch_size}")
        
        # 创建数据集和数据加载器
        dataset = SFTDataset(data_path, getattr(self.model, 'tokenizer', None))
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        total_steps = self.epochs * len(dataloader)
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        self.logger.info(f"Total samples: {len(dataset)}, Total steps: {total_steps}")
        
        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            self._train_epoch(dataloader, epoch, total_steps)
        
        # 保存最终模型
        self.save_model(self.output_dir / 'final')
        self.logger.info("SFT training completed!")
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int, total_steps: int):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # 前向传播
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            actions = batch['action'].to(self.device)
            
            # 模型输出
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 交叉熵损失
            loss = F.cross_entropy(logits, actions)
            loss = loss / self.gradient_accumulation  # 梯度累积
            
            # 反向传播
            loss.backward()
            
            # 梯度累积更新
            if (step + 1) % self.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 保持 lm_head 和 embed_tokens 的新增token参数同步
                # (PEFT 的 trainable_token_indices 只训练embedding，
                #  但 tied weights 需要手动同步以确保 lm_head 也更新)
                if hasattr(self.model, 'tie_weights'):
                    self.model.tie_weights()

                self.global_step += 1
            
            # 记录
            epoch_loss += loss.item() * self.gradient_accumulation
            
            # 更新进度条显示的 Loss，避免 logger 打印导致进度条刷屏
            if self.global_step % self.logging_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                self.metrics.log({'loss': avg_loss}, step=self.global_step)
        
        # Epoch结束
        avg_epoch_loss = epoch_loss / len(dataloader)
        self.logger.info(f"Epoch {epoch+1} completed, avg loss: {avg_epoch_loss:.4f}")
    
    def save_model(self, save_path: Path):
        """保存模型"""
        save_path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(str(save_path))
        else:
            torch.save(self.model.state_dict(), save_path / 'model.pt')
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: Path):
        """加载模型"""
        if hasattr(self.model, 'from_pretrained'):
            self.model = self.model.from_pretrained(str(load_path), self.config)
        else:
            self.model.load_state_dict(torch.load(load_path / 'model.pt'))
        self.logger.info(f"Model loaded from {load_path}")
