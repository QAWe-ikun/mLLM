"""MobileSAM目标检测器 + CLIP zero-shot定位"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple
from PIL import Image


class MobileSAMDetector(nn.Module):
    """MobileSAM轻量版目标检测与分割

    使用CLIP进行zero-shot区域定位 (文本→框坐标)
    再将框坐标传给SAM进行精确分割

    输入: RGB图像 + 目标类别文本提示 (如'microwave')
    输出: 目标物体的bbox + mask + 中心点坐标(归一化)
    """

    def __init__(self, model_name: str = "dhkim281/mobilesam",
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 device: Optional[str] = None):
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.image_size = 300
        self.grid_size = 3  # 3x3 网格

        # 加载CLIP用于zero-shot定位
        self.clip_processor = None
        self.clip_model = None
        self._init_clip(clip_model_name)

        # 加载MobileSAM (使用mobile_sam库或dummy)
        try:
            from mobile_sam import SamModel, SamPredictor, sam_model_registry
            self._init_mobile_sam(model_name)
        except ImportError:
            print("Warning: mobile_sam not available, using dummy detector")
            self._init_dummy()

    def _init_clip(self, model_name: str):
        """初始化CLIP用于zero-shot目标定位"""
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            print(f"CLIP loaded for zero-shot localization: {model_name}")
        except Exception as e:
            print(f"Warning: Failed to load CLIP: {e}")
            self.clip_processor = None
            self.clip_model = None

    def _init_mobile_sam(self, model_name: str):
        """初始化MobileSAM"""
        try:
            from mobile_sam import sam_model_registry, SamPredictor
            import tempfile
            import os

            # 简化：加载预训练权重 (需要本地路径或下载)
            self.is_dummy = False
            self.sam_model = None
            self.predictor = None
            print("MobileSAM framework loaded (weights pending)")
        except Exception as e:
            print(f"Warning: Failed to load MobileSAM: {e}")
            self._init_dummy()

    def _init_dummy(self):
        """初始化dummy检测器（用于测试）"""
        self.is_dummy = True
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
        """真实MobileSAM检测 + CLIP zero-shot定位

        流程:
        1. 用CLIP对图像网格做zero-shot文本匹配，找到最可能的目标区域
        2. 将CLIP定位的框坐标传给SAM进行精确分割
        3. 从mask提取bbox和center
        """
        h, w = image.shape[:2]

        # Step 1: CLIP zero-shot定位
        clip_bbox = self._clip_localize(image, target_class)

        # Step 2: 如果有SAM模型，用SAM做精确分割
        if self.predictor is not None and not self.is_dummy:
            # 使用SAM对CLIP定位的区域做精确分割
            mask = self._sam_segment(image, clip_bbox)
            # 从mask提取bbox
            y_ids, x_ids = np.where(mask > 0.5)
            if len(y_ids) > 0:
                x_min, x_max = x_ids.min() / w, x_ids.max() / w
                y_min, y_max = y_ids.min() / h, y_ids.max() / h
                bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
                confidence = float(mask.mean())
            else:
                bbox = clip_bbox
                center = np.array([(clip_bbox[0] + clip_bbox[2]) / 2,
                                   (clip_bbox[1] + clip_bbox[3]) / 2], dtype=np.float32)
                confidence = 0.5
        else:
            # 没有SAM权重，直接使用CLIP定位结果
            bbox = clip_bbox
            center = np.array([(clip_bbox[0] + clip_bbox[2]) / 2,
                               (clip_bbox[1] + clip_bbox[3]) / 2], dtype=np.float32)
            confidence = 0.5
            # 生成粗略mask
            mask = np.zeros((h, w), dtype=np.float32)
            ix_min = int(bbox[0] * w)
            iy_min = int(bbox[1] * h)
            ix_max = int(bbox[2] * w)
            iy_max = int(bbox[3] * h)
            mask[iy_min:iy_max, ix_min:ix_max] = 1.0

        return {
            'bbox': bbox,
            'center': center,
            'mask': mask,
            'confidence': confidence,
        }

    def _clip_localize(self, image: np.ndarray, target_class: str) -> np.ndarray:
        """使用CLIP进行zero-shot区域定位

        将图像划分为 grid_size x grid_size 的网格，
        计算每个区域与目标文本的CLIP相似度，
        返回最高相似度区域的归一化bbox [x_min, y_min, x_max, y_max]

        Args:
            image: RGB图像 [H, W, 3], [0, 1]
            target_class: 目标类别 (如"microwave")

        Returns:
            bbox: 归一化坐标 [x_min, y_min, x_max, y_max]
        """
        if self.clip_model is None or self.clip_processor is None:
            # CLIP未加载，返回中心区域
            return np.array([0.25, 0.25, 0.75, 0.75], dtype=np.float32)

        h, w = image.shape[:2]
        gs = self.grid_size

        # 准备文本prompt
        prompts = [f"a photo of a {target_class}", f"this is {target_class}"]

        best_score = -float('inf')
        best_bbox = np.array([1.0 / gs, 1.0 / gs, (gs - 1.0) / gs, (gs - 1.0) / gs], dtype=np.float32)

        # 遍历所有网格
        for row in range(gs):
            for col in range(gs):
                # 裁剪区域
                y_start = int(row * h / gs)
                y_end = int((row + 1) * h / gs)
                x_start = int(col * w / gs)
                x_end = int((col + 1) * w / gs)

                crop = image[y_start:y_end, x_start:x_end]
                crop_pil = Image.fromarray((crop * 255).astype(np.uint8))

                # CLIP相似度计算
                inputs = self.clip_processor(
                    text=prompts,
                    images=crop_pil,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    # 取第一个prompt的相似度
                    score = outputs.logits_per_image[0, 0].item()

                if score > best_score:
                    best_score = score
                    best_bbox = np.array([
                        x_start / w, y_start / h,
                        x_end / w, y_end / h
                    ], dtype=np.float32)

        return best_bbox

    def _sam_segment(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """使用SAM对给定bbox做精确分割

        Args:
            image: RGB图像 [H, W, 3]
            bbox: 归一化bbox [x_min, y_min, x_max, y_max]

        Returns:
            mask: 二值分割mask [H, W]
        """
        if self.predictor is None:
            return np.zeros(image.shape[:2], dtype=np.float32)

        h, w = image.shape[:2]
        self.predictor.set_image(image)

        # 转换bbox为像素坐标
        x_min = int(bbox[0] * w)
        y_min = int(bbox[1] * h)
        x_max = int(bbox[2] * w)
        y_max = int(bbox[3] * h)
        box = np.array([x_min, y_min, x_max, y_max])

        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )

        return masks[0]
    
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
