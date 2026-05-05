"""位置编码器 - Agent 3D状态编码"""
import torch
import torch.nn as nn
import math
from typing import Optional


class PositionEncoder(nn.Module):
    """Agent状态位置编码器
    
    输入: [x, y, z, rotation] 4维连续值
    处理: 归一化 -> Sinusoidal PE -> Linear(256->4096)
    输出: [1, 4096] 状态特征向量
    """
    
    def __init__(self, input_dim: int = 4,
                 hidden_dim: int = 256,
                 output_dim: int = 4096,
                 max_coord: float = 10.0,  # 最大坐标范围(m)
                 device: Optional[str] = None):
        super().__init__()
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        self.input_dim = input_dim
        self.max_coord = max_coord
        
        # Sinusoidal位置编码维度
        self.pe_dim = hidden_dim  # 256
        
        # Linear投影层
        self.projection = nn.Sequential(
            nn.Linear(self.pe_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        self.to(self.device)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """编码Agent状态
        
        Args:
            state: [x, y, z, rotation] 形状 [B, 4] 或 [4]
            
        Returns:
            位置特征 [B, 4096] 或 [1, 4096]
        """
        if state.ndim == 1:
            state = state.unsqueeze(0)  # [4] -> [1, 4]
        
        batch_size = state.shape[0]
        
        # 归一化到 [0, 1]
        state = state / self.max_coord
        state = torch.clamp(state, 0, 1)
        
        # Sinusoidal位置编码
        pe = self._sinusoidal_encoding(state)  # [B, 4*pe_dim/4] = [B, 256]
        
        # 投影到4096维
        features = self.projection(pe)  # [B, 4096]
        
        return features
    
    def _sinusoidal_encoding(self, state: torch.Tensor) -> torch.Tensor:
        """Sinusoidal位置编码 (类似Transformer)
        
        对每个输入维度应用不同频率的正弦/余弦函数
        """
        batch_size = state.shape[0]
        
        # 生成频率
        # 对于4维输入，每维分配 pe_dim/4 = 64 维
        dim_per_input = self.pe_dim // self.input_dim  # 64
        
        pe = torch.zeros(batch_size, self.pe_dim, device=self.device)
        
        for i in range(self.input_dim):
            for j in range(dim_per_input):
                freq = 1.0 / (10000 ** (2 * j / dim_per_input))
                angle = state[:, i] * freq
                
                if j % 2 == 0:
                    pe[:, i * dim_per_input + j] = torch.sin(angle)
                else:
                    pe[:, i * dim_per_input + j] = torch.cos(angle)
        
        return pe
    
    def get_output_dim(self) -> int:
        """返回输出维度"""
        return 4096
