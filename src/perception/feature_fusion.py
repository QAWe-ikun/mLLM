"""多模态特征融合模块

设计说明:
- CLIP 仅用于 MobileSAM 的 zero-shot 目标定位 (不在这里做特征融合)
- Qwen3-VL 的 NaViT 视觉编码器已能捕获全局语义
- 融合层仅包含: SAM目标位置 + Agent位置编码
- 如需消融实验对比，可在此重新加入 CLIP 全局特征
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, List


class FeatureFusion(nn.Module):
    """多模态特征融合器

    融合SAM目标位置、Agent位置编码
    输出: [seq_len, 4096] 对齐到LLM输入维度
    """

    def __init__(self,
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
            # SAM(4096) + Pos(4096) = 8192
            concat_dim = projected_dim * 2
            self.fusion_projection = nn.Sequential(
                nn.Linear(concat_dim, projected_dim),
                nn.LayerNorm(projected_dim),
            )

        self.to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, sam_features: torch.Tensor,
                pos_features: torch.Tensor) -> torch.Tensor:
        """融合多模态特征

        Args:
            sam_features: SAM目标位置 [B, 2] (center_x, center_y)
            pos_features: Agent位置编码 [B, 4096]

        Returns:
            融合特征 [B, seq_len, 4096] 或 [B, 4096]
        """
        # 投影SAM特征
        sam_proj = self.sam_projection(sam_features)  # [B, 4096]

        if self.fusion_method == "concat":
            # 拼接所有特征
            fused = torch.cat([sam_proj, pos_features], dim=-1)  # [B, 8192]
            fused = self.fusion_projection(fused)  # [B, 4096]
            # 添加sequence维度 [B, 1, 4096]
            return fused.unsqueeze(1)

        elif self.fusion_method == "additive":
            # 逐元素相加（需要相同维度）
            fused = sam_proj + pos_features  # [B, 4096]
            return fused.unsqueeze(1)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
