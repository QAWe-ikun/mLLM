"""MobileSAM目标检测器"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple
from PIL import Image


class MobileSAMDetector(nn.Module):
    """MobileSAM轻量版目标检测与分割
    
    输入: RGB图像 + 目标类别提示
    输出: 目标物体的bbox + mask + 中心点坐标(归一化)
    """
    
    def __init__(self, model_name: str = "dhkim281/mobilesam",
                 device: Optional[str] = None):
        super().__init__()
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        self.image_size = 300
        
        # 加载MobileSAM (使用mobile_sam库或dummy)
        try:
            from mobile_sam import SamModel, SamPredictor, sam_model_registry
            # 这里使用简化的实现
            self._init_mobile_sam()
        except ImportError:
            print("Warning: mobile_sam not available, using dummy detector")
            self._init_dummy()
    
    def _init_mobile_sam(self):
        """初始化MobileSAM"""
        try:
            from mobile_sam import sam_model_registry, SamPredictor
            checkpoint_url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            
            # 简化：使用预训练权重
            self.is_dummy = False
            self.predictor = None
        except:
            self._init_dummy()
    
    def _init_dummy(self):
        """初始化dummy检测器（用于测试）"""
        self.is_dummy = True
        # 随机目标位置生成器（模拟）
        self.dummy_positions = {}
    
    def forward(self, image: np.ndarray, target_class: str) -> Dict[str, Any]:
        """检测目标物体
        
        Args:
            image: RGB图像 [H, W, 3], [0, 1]
            target_class: 目标类别 (如"microwave")
            
        Returns:
            检测结果字典:
                - bbox: [x_min, y_min, x_max, y_max] 归一化坐标
                - center: [x_center, y_center] 归一化中心点
                - mask: [H, W] 二值mask (dummy模式下为全0)
                - confidence: 置信度 (dummy模式下固定0.5)
        """
        if self.is_dummy:
            return self._dummy_detect(image, target_class)
        else:
            return self._sam_detect(image, target_class)
    
    def _sam_detect(self, image: np.ndarray, target_class: str) -> Dict[str, Any]:
        """真实MobileSAM检测"""
        # 实际实现需要：
        # 1. 使用CLIP或grounding模型找到目标类别的prompt
        # 2. 将prompt输入SAM生成mask
        # 3. 从mask提取bbox和center
        # 这里提供框架接口
        raise NotImplementedError("Real SAM detection needs implementation")
    
    def _dummy_detect(self, image: np.ndarray, target_class: str) -> Dict[str, Any]:
        """Dummy检测：生成随机位置（用于测试流程）"""
        h, w = image.shape[:2]
        
        # 根据target_class生成确定性随机位置
        seed = hash(target_class) % 10000
        rng = np.random.RandomState(seed)
        
        # 随机生成bbox (在图像中心区域)
        cx = rng.uniform(0.3, 0.7)
        cy = rng.uniform(0.3, 0.7)
        bw = rng.uniform(0.1, 0.25)
        bh = rng.uniform(0.1, 0.25)
        
        x_min = max(0, cx - bw / 2)
        y_min = max(0, cy - bh / 2)
        x_max = min(1, cx + bw / 2)
        y_max = min(1, cy + bh / 2)
        
        # 生成dummy mask
        mask = np.zeros((h, w), dtype=np.float32)
        ix_min = int(x_min * w)
        iy_min = int(y_min * h)
        ix_max = int(x_max * w)
        iy_max = int(y_max * h)
        mask[iy_min:iy_max, ix_min:ix_max] = 1.0
        
        return {
            'bbox': np.array([x_min, y_min, x_max, y_max], dtype=np.float32),
            'center': np.array([cx, cy], dtype=np.float32),
            'mask': mask,
            'confidence': 0.5,
        }
    
    def get_output_dim(self) -> int:
        """返回位置编码维度 (center 2D)"""
        return 2
