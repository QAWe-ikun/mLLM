"""CLIP ViT-B/32特征提取器"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import numpy as np
from PIL import Image


class CLIPEncoder(nn.Module):
    """CLIP ViT-B/32全局场景语义特征提取
    
    输入: 300x300 RGB图像
    输出: 512维全局特征向量
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 output_dim: int = 512,
                 projected_dim: int = 4096,
                 device: Optional[str] = None):
        super().__init__()
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # 加载CLIP模型
        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor
            self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
            self.clip_model = CLIPVisionModel.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: Failed to load CLIP from HuggingFace: {e}")
            print("Using dummy CLIP encoder for testing")
            self._init_dummy(output_dim)
            return
        
        # 投影层: 512 -> 4096 (对齐到LLM hidden dim)
        self.clip_output_dim = self.clip_model.config.hidden_size  # 768 for ViT-B/32
        self.projection = nn.Sequential(
            nn.Linear(self.clip_output_dim, projected_dim),
            nn.LayerNorm(projected_dim),
        )
        
        self.clip_model = self.clip_model.to(self.device)
        self.projection = self.projection.to(self.device)
        
        # 冻结CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def _init_dummy(self, output_dim: int):
        """初始化dummy CLIP（用于测试）"""
        self.clip_output_dim = 512
        self.projection = nn.Sequential(
            nn.Linear(self.clip_output_dim, 4096),
            nn.LayerNorm(4096),
        ).to(self.device)
        self.is_dummy = True
    
    @torch.no_grad()
    def forward(self, image: Union[torch.Tensor, np.ndarray, Image.Image], 
                project: bool = True) -> torch.Tensor:
        """提取CLIP特征
        
        Args:
            image: RGB图像 [H, W, 3] 或 [B, H, W, 3] 或 PIL Image
            project: 是否投影到4096维
            
        Returns:
            特征向量 [B, 512] 或 [B, 4096]
        """
        # 预处理图像
        if isinstance(image, Image.Image):
            inputs = self.image_processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = np.expand_dims(image, 0)  # [1, H, W, 3]
            # numpy -> PIL
            images = [Image.fromarray((img * 255).astype(np.uint8)) for img in image]
            inputs = self.image_processor(images=images, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)  # [1, H, W, 3]
            # tensor -> PIL -> processor
            images = [Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)) 
                     for img in image]
            inputs = self.image_processor(images=images, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # 提取特征
        outputs = self.clip_model(pixel_values=pixel_values)
        # 使用pooler_output (CLS token)
        features = outputs.pooler_output  # [B, 768]
        
        # 投影
        if project:
            features = self.projection(features)  # [B, 4096]
        
        return features
    
    def get_output_dim(self) -> int:
        """返回输出维度"""
        return 4096
