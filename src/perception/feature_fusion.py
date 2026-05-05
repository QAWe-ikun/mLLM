"""多模态特征融合模块"""
import torch
import torch.nn as nn
from typing import Dict, Optional, List


class FeatureFusion(nn.Module):
    """多模态特征融合器
    
    融合CLIP全局特征、SAM目标位置、Agent位置编码
    输出: [seq_len, 4096] 对齐到LLM输入维度
    """
    
    def __init__(self, clip_dim: int = 4096,
                 sam_dim: int = 2,
                 pos_dim: int = 4096,
                 projected_dim: int = 4096,
                 fusion_method: str = "concat"):
        super().__init__()
        
        self.fusion_method = fusion_method  # concat | additive
        self.projected_dim = projected_dim
        
        # SAM位置特征投影 (2D -> 4096)
        self.sam_projection = nn.Sequential(
            nn.Linear(sam_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projected_dim),
            nn.LayerNorm(projected_dim),
        )
        
        # 融合后的投影（concat模式）
        if fusion_method == "concat":
            # CLIP(4096) + SAM(4096) + Pos(4096) = 12288
            concat_dim = projected_dim * 3
            self.fusion_projection = nn.Sequential(
                nn.Linear(concat_dim, projected_dim),
                nn.LayerNorm(projected_dim),
            )
        
        self.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, clip_features: torch.Tensor,
                sam_features: torch.Tensor,
                pos_features: torch.Tensor) -> torch.Tensor:
        """融合多模态特征
        
        Args:
            clip_features: CLIP全局特征 [B, 4096]
            sam_features: SAM目标位置 [B, 2] (center_x, center_y)
            pos_features: Agent位置编码 [B, 4096]
            
        Returns:
            融合特征 [B, seq_len, 4096] 或 [B, 4096]
        """
        # 投影SAM特征
        sam_proj = self.sam_projection(sam_features)  # [B, 4096]
        
        if self.fusion_method == "concat":
            # 拼接所有特征
            fused = torch.cat([clip_features, sam_proj, pos_features], dim=-1)  # [B, 12288]
            fused = self.fusion_projection(fused)  # [B, 4096]
            # 添加sequence维度 [B, 1, 4096]
            return fused.unsqueeze(1)
        
        elif self.fusion_method == "additive":
            # 逐元素相加（需要相同维度）
            fused = clip_features + sam_proj + pos_features  # [B, 4096]
            return fused.unsqueeze(1)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def create_visual_tokens(self, clip_features: torch.Tensor,
                              sam_bbox: torch.Tensor,
                              num_patches: int = 64) -> torch.Tensor:
        """创建视觉token序列 (用于VLM输入)
        
        将CLIP全局特征扩展为patch级别的视觉token序列
        
        Args:
            clip_features: [B, 4096]
            sam_bbox: [B, 4] (x_min, y_min, x_max, y_max)
            num_patches: 视觉token数量
            
        Returns:
            视觉token序列 [B, num_patches, 4096]
        """
        batch_size = clip_features.shape[0]
        
        # 将全局特征复制为patch序列
        visual_tokens = clip_features.unsqueeze(1).expand(-1, num_patches, -1)  # [B, N, 4096]
        
        # 在SAM bbox对应的patch位置添加位置偏置
        # 简化处理：将bbox信息编码为位置偏置加到对应token
        bbox_centers = (sam_bbox[:, :2] + sam_bbox[:, 2:]) / 2  # [B, 2]
        
        # 生成位置偏置 (简化版)
        for i in range(batch_size):
            cx, cy = bbox_centers[i]
            # 将bbox中心映射到patch索引
            patch_idx_x = int(cx * (num_patches ** 0.5))
            patch_idx_y = int(cy * (num_patches ** 0.5))
            
            if patch_idx_x < int(num_patches ** 0.5) and patch_idx_y < int(num_patches ** 0.5):
                patch_idx = patch_idx_y * int(num_patches ** 0.5) + patch_idx_x
                if patch_idx < num_patches:
                    # 添加目标位置偏置
                    visual_tokens[i, patch_idx] += 0.1  # 简单偏置
        
        return visual_tokens
