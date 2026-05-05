"""动作输出头"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ActionHead(nn.Module):
    """动作策略头
    
    从VLM输出中提取动作概率分布，支持argmax/采样选择
    """
    
    def __init__(self, hidden_dim: int = 4096, num_actions: int = 6):
        super().__init__()
        
        self.num_actions = num_actions
        
        # 动作分类器
        self.action_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_actions),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算动作概率分布
        
        Args:
            hidden_states: VLM最后隐藏状态 [B, seq_len, hidden_dim]
                           或 [B, hidden_dim]
            
        Returns:
            action_probs: 动作概率 [B, num_actions]
        """
        # 取最后一个token的状态
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]  # [B, hidden_dim]
        
        logits = self.action_classifier(hidden_states)  # [B, num_actions]
        action_probs = F.softmax(logits, dim=-1)
        
        return action_probs
    
    def select_action(self, action_probs: torch.Tensor,
                     temperature: float = 1.0,
                     sample: bool = True) -> Tuple[int, float]:
        """选择动作
        
        Args:
            action_probs: 动作概率 [B, num_actions] 或 [num_actions]
            temperature: 采样温度
            sample: 是否采样（False则argmax）
            
        Returns:
            action_idx: 动作索引
            log_prob: 动作的对数概率
        """
        if action_probs.ndim == 1:
            action_probs = action_probs.unsqueeze(0)  # [1, num_actions]
        
        # 应用温度
        logits = torch.log(action_probs + 1e-8) / temperature
        scaled_probs = F.softmax(logits, dim=-1)
        
        if sample:
            # 多项式采样
            action_idx = torch.multinomial(scaled_probs, num_samples=1)
        else:
            # Argmax
            action_idx = torch.argmax(scaled_probs, dim=-1, keepdim=True)
        
        # 计算log概率
        log_prob = torch.log(scaled_probs.gather(1, action_idx) + 1e-8)
        
        return action_idx.item(), log_prob.squeeze().item()
    
    def compute_entropy(self, action_probs: torch.Tensor) -> torch.Tensor:
        """计算策略熵 (用于PPO的entropy bonus)
        
        Args:
            action_probs: [B, num_actions]
            
        Returns:
            entropy: [B]
        """
        # H = -sum(p * log(p))
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
        return entropy
