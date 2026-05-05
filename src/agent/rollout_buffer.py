"""Rollout经验回放缓冲区"""
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class Transition:
    """单步转移数据"""
    observation: Dict[str, Any]
    instruction: str
    action: int
    reward: float
    done: bool
    value: float
    log_prob: float
    next_observation: Optional[Dict[str, Any]] = None


class RolloutBuffer:
    """PPO经验回放缓冲区
    
    存储rollout轨迹数据，支持GAE计算和批量采样
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.transitions: List[Transition] = []
    
    def add(self, transition: Transition):
        """添加转移"""
        self.transitions.append(transition)
        if len(self.transitions) > self.max_size:
            self.transitions.pop(0)
    
    def add_batch(self, transitions: List[Transition]):
        """批量添加"""
        self.transitions.extend(transitions)
    
    def clear(self):
        """清空缓冲区"""
        self.transitions = []
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def get_all(self) -> List[Transition]:
        """获取所有转移"""
        return self.transitions
    
    def sample_batch(self, batch_size: int) -> List[Transition]:
        """随机采样"""
        indices = np.random.choice(len(self.transitions), batch_size, replace=False)
        return [self.transitions[i] for i in indices]
    
    def compute_gae(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> List[Dict]:
        """计算Generalized Advantage Estimation (GAE)
        
        Args:
            gamma: 折扣因子
            gae_lambda: GAE参数
            
        Returns:
            包含advantages和returns的字典列表
        """
        transitions = self.transitions
        
        # 从后向前计算
        last_gae = 0.0
        advantages = []
        
        for t in reversed(range(len(transitions))):
            if t == len(transitions) - 1 or transitions[t + 1].done:
                # 最后一步或done状态
                next_value = 0.0
            else:
                next_value = transitions[t + 1].value
            
            # TD error: delta = r + gamma * V(s') - V(s)
            delta = transitions[t].reward + gamma * next_value * (1 - transitions[t].done) - transitions[t].value
            
            # GAE:加权累积和
            last_gae = delta + gamma * gae_lambda * (1 - transitions[t].done) * last_gae
            advantages.insert(0, last_gae)
        
        # 计算returns = advantages + values
        returns = []
        for t in range(len(transitions)):
            returns.append(advantages[t] + transitions[t].value)
        
        # 附加到transitions
        for t, (adv, ret) in enumerate(zip(advantages, returns)):
            transitions[t].advantage = adv
            transitions[t].returns = ret
        
        return transitions
    
    def to_batch(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """转换为批次张量
        
        Returns:
            字典: {observations, actions, rewards, dones, values, log_probs, advantages, returns}
        """
        trans = self.transitions
        
        actions = torch.tensor([t.action for t in trans], dtype=torch.long, device=device)
        rewards = torch.tensor([t.reward for t in trans], dtype=torch.float32, device=device)
        dones = torch.tensor([t.done for t in trans], dtype=torch.float32, device=device)
        values = torch.tensor([t.value for t in trans], dtype=torch.float32, device=device)
        log_probs = torch.tensor([t.log_prob for t in trans], dtype=torch.float32, device=device)
        
        # advantages和returns (需要先用GAE计算)
        advantages = torch.tensor([getattr(t, 'advantage', 0.0) for t in trans], 
                                 dtype=torch.float32, device=device)
        returns = torch.tensor([getattr(t, 'returns', 0.0) for t in trans],
                              dtype=torch.float32, device=device)
        
        return {
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'log_probs': log_probs,
            'advantages': advantages,
            'returns': returns,
        }
