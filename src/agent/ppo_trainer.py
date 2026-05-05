"""PPO强化学习训练器"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agent.rollout_buffer import RolloutBuffer, Transition
from src.utils.logger import Logger, MetricsTracker
from src.utils.config import load_config


class ValueHead(nn.Module):
    """价值网络头 (Critic)"""
    
    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """预测状态价值
        
        Args:
            hidden_states: [B, hidden_dim] 或 [B, seq_len, hidden_dim]
            
        Returns:
            values: [B, 1]
        """
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]  # 取最后一个token
        
        return self.value_net(hidden_states)


class PPOTrainer:
    """PPO强化学习训练器
    
    算法: PPO-Clip (稳定版)
    - 折扣因子 γ=0.99
    - GAE λ=0.95
    - Clip ε=0.2
    """
    
    def __init__(self, model: nn.Module, env: Any, config: Dict[str, Any],
                 output_dir: str = 'results/ppo_checkpoints'):
        self.model = model
        self.env = env
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        ppo_config = config['train']['ppo']
        
        # PPO算法参数
        self.clip_epsilon = ppo_config['algorithm']['clip_epsilon']
        self.gamma = ppo_config['algorithm']['gamma']
        self.gae_lambda = ppo_config['algorithm']['gae_lambda']
        self.value_loss_coef = ppo_config['algorithm']['value_loss_coef']
        self.entropy_coef = ppo_config['algorithm']['entropy_coef']
        
        # 训练参数
        self.total_steps = ppo_config['training']['total_steps']
        self.rollout_steps = ppo_config['training']['rollout_steps']
        self.epochs_per_update = ppo_config['training']['epochs_per_update']
        self.batch_size = ppo_config['training']['per_device_train_batch_size']
        self.gradient_accumulation = ppo_config['training']['gradient_accumulation_steps']
        self.learning_rate = ppo_config['training']['learning_rate']
        self.max_grad_norm = ppo_config['training']['max_grad_norm']
        
        # 评估参数
        self.eval_episodes = ppo_config['evaluation']['eval_episodes']
        self.eval_interval = ppo_config['evaluation']['eval_interval']
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Value head (critic)
        self.value_head = ValueHead().to(self.device)
        
        # 优化器 (同时优化policy和value)
        all_params = list(self.model.parameters()) + list(self.value_head.parameters())
        self.optimizer = torch.optim.AdamW(all_params, lr=self.learning_rate)
        
        # Rollout缓冲区
        self.buffer = RolloutBuffer(max_size=self.rollout_steps)
        
        # 日志
        self.logger = Logger('PPO', log_file=str(self.output_dir / 'ppo.log'))
        self.metrics = MetricsTracker(
            log_dir=str(self.output_dir),
            use_tensorboard=True,
        )
        
        self.global_step = 0
    
    def collect_rollout(self) -> RolloutBuffer:
        """收集rollout数据
        
        与环境交互收集self.rollout_steps步数据
        """
        self.buffer.clear()
        
        # 重置环境
        obs = self.env.reset()
        done = False
        episode_steps = 0
        
        while len(self.buffer) < self.rollout_steps:
            if done:
                obs = self.env.reset()
                episode_steps = 0
            
            # 准备模型输入
            instruction = obs.get('target', 'Go to the object')
            rgb_image = obs['rgb']  # [H, W, 3]
            
            # 构建输入
            input_data = self._prepare_input(instruction, rgb_image)
            
            # 获取动作概率
            with torch.no_grad():
                action_probs = self._get_action_probs(input_data)
                # Value估计
                value = self.value_head(input_data['hidden']).item()
            
            # 采样动作
            action_idx, log_prob = self.model.generate_action(
                input_data['input_ids'],
                input_data['attention_mask'],
                input_data.get('visual_features'),
                sample=True,
            )
            
            # 执行动作
            next_obs, reward, done, info = self.env.step(action_idx)
            episode_steps += 1
            
            # 存储到buffer
            transition = Transition(
                observation=obs,
                instruction=instruction,
                action=action_idx,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
                next_observation=next_obs,
            )
            self.buffer.add(transition)
            
            obs = next_obs
        
        return self.buffer
    
    def _prepare_input(self, instruction: str, rgb_image: np.ndarray) -> Dict:
        """准备模型输入数据"""
        # Tokenize指令
        if hasattr(self.model, 'tokenizer'):
            encoded = self.model.tokenizer(instruction, return_tensors='pt')
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
        else:
            input_ids = torch.randint(0, 100, (1, 10), device=self.device)
            attention_mask = torch.ones(1, 10, device=self.device)
        
        # 获取隐藏状态 (简化)
        hidden = torch.randn(1, 4096, device=self.device)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'hidden': hidden,
            'rgb': rgb_image,
        }
    
    def _get_action_probs(self, input_data: Dict) -> torch.Tensor:
        """获取动作概率分布"""
        logits = self.model(
            input_ids=input_data['input_ids'],
            attention_mask=input_data['attention_mask'],
        )
        return F.softmax(logits, dim=-1)
    
    def train(self):
        """执行PPO训练主循环"""
        self.logger.info(f"Starting PPO training, total steps: {self.total_steps}")
        self.logger.info(f"Rollout steps: {self.rollout_steps}, "
                        f"Clip epsilon: {self.clip_epsilon}")

        progress_bar = tqdm(range(self.total_steps), desc="PPO Training")
        progress_bar.update(self.global_step)

        while self.global_step < self.total_steps:
            # 1. 收集rollout
            progress_bar.set_description(f"Step {self.global_step}: Collecting rollout")
            self.collect_rollout()

            # 2. 计算GAE
            self.buffer.compute_gae(gamma=self.gamma, gae_lambda=self.gae_lambda)

            # 3. PPO更新
            for epoch in range(self.epochs_per_update):
                self._ppo_update()
                progress_bar.update(1)

            # 4. 评估（不打印日志，仅记录）
            if self.global_step % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                self.metrics.log(eval_metrics, step=self.global_step)
                progress_bar.set_postfix({
                    'success': f"{eval_metrics.get('eval_success_rate', 0):.2f}",
                    'reward': f"{eval_metrics.get('eval_avg_reward', 0):.2f}"
                })

                # 保存检查点
                self.save_model(self.output_dir / f'checkpoint_{self.global_step}')

        # 保存最终模型
        self.save_model(self.output_dir / 'final')
        progress_bar.close()
        self.logger.info("PPO training completed!")
    
    def _ppo_update(self):
        """执行一次PPO策略更新"""
        self.model.train()
        self.value_head.train()
        
        # 采样批次
        batch_data = self.buffer.to_batch(device=self.device)
        
        actions = batch_data['actions']
        old_log_probs = batch_data['log_probs']
        advantages = batch_data['advantages']
        returns = batch_data['returns']
        
        # 标准化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0.0
        
        for _ in range(self.epochs_per_update):
            # 重新计算当前策略的log probs和values
            # (简化实现：使用旧值)
            
            # PPO-Clip loss
            ratio = torch.exp(batch_data['log_probs'] - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = self.value_head(batch_data.get('hidden', torch.randn(len(actions), 4096, device=self.device)))
            values = values.squeeze(-1)
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            action_probs = self._get_action_probs({'input_ids': torch.zeros(1, 10, device=self.device),
                                                    'attention_mask': torch.ones(1, 10, device=self.device)})
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1).mean()
            
            # 总损失
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.value_head.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
        
        # 记录
        avg_loss = total_loss / self.epochs_per_update
        self.metrics.log({
            'ppo_loss': avg_loss,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }, step=self.global_step)
    
    def evaluate(self, num_episodes: Optional[int] = None) -> Dict[str, float]:
        """评估当前策略"""
        if num_episodes is None:
            num_episodes = self.eval_episodes
        
        self.logger.info(f"Evaluating for {num_episodes} episodes")
        
        successes = 0
        total_rewards = []
        total_steps = []
        
        for ep in range(num_episodes):
            obs = self.env.reset()
            done = False
            ep_reward = 0
            ep_steps = 0
            
            while not done:
                instruction = obs.get('target', 'Go to the object')
                rgb = obs['rgb']
                
                input_data = self._prepare_input(instruction, rgb)
                
                with torch.no_grad():
                    action_idx = self.model.generate_action(
                        input_data['input_ids'],
                        input_data['attention_mask'],
                        sample=False,  # greedy
                    )
                
                obs, reward, done, info = self.env.step(action_idx)
                ep_reward += reward
                ep_steps += 1
            
            if info.get('success_nav', False) or info.get('success_pickup', False):
                successes += 1
            
            total_rewards.append(ep_reward)
            total_steps.append(ep_steps)
        
        metrics = {
            'eval_success_rate': successes / num_episodes,
            'eval_avg_reward': np.mean(total_rewards),
            'eval_avg_steps': np.mean(total_steps),
        }
        
        return metrics
    
    def save_model(self, save_path: Path):
        """保存模型"""
        save_path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(str(save_path))
        torch.save(self.value_head.state_dict(), save_path / 'value_head.pt')
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: Path):
        """加载模型"""
        if hasattr(self.model, 'from_pretrained'):
            self.model = self.model.from_pretrained(str(load_path), self.config)
        if (load_path / 'value_head.pt').exists():
            self.value_head.load_state_dict(torch.load(load_path / 'value_head.pt'))
        self.logger.info(f"Model loaded from {load_path}")
