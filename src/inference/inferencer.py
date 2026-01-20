import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Union

from src.models import DeepLabV3Plus
from src.data import get_transforms
from src.utils import load_config, get_device


class Inferencer:
    """Semantic Segmentation Inferencer"""

    def __init__(
        self,
        checkpoint: str = "checkpoints/best.pt",
        config_path: str = "configs/config.yaml"
    ):
        self.config = load_config(config_path)
        self.device = get_device()
        self.checkpoint_path = Path(checkpoint)

        self.data_cfg = self.config.get("data", {})
        self.model_cfg = self.config.get("model", {})

        self.img_size = tuple(self.data_cfg.get("img_size", [1024, 768]))
        self.num_classes = self.data_cfg.get("num_classes", 62)

        self.transform = get_transforms(self.img_size)
        self.model = self.load_model()

    def load_model(self) -> torch.nn.Module:
        """모델 로드"""
        model = DeepLabV3Plus(
            num_classes=self.num_classes,
            backbone=self.model_cfg.get("name", "deeplabv3_resnet50"),
            pretrained=False
        ).to(self.device)

        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded checkpoint: {self.checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found: {self.checkpoint_path}")

        model.eval()
        return model

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """이미지 전처리"""
        if image.shape[2] == 4:  # RGBA -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:  # Grayscale -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        transformed = self.transform(image=image)
        img_tensor = transformed["image"].unsqueeze(0)  # (1, C, H, W)
        return img_tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        이미지 추론

        Args:
            image: 이미지 경로 또는 numpy array (H, W, C) RGB

        Returns:
            mask: 예측 마스크 (H, W) - 원본 크기로 리사이즈됨
        """
        # 이미지 로드
        if isinstance(image, str):
            image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        # 전처리 및 추론
        img_tensor = self.preprocess(image)
        output = self.model(img_tensor)["out"]

        # 후처리
        pred = output.argmax(dim=1).squeeze().cpu().numpy()

        # 원본 크기로 리사이즈
        mask = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return mask

    def predict_batch(self, images: list) -> list:
        """배치 추론"""
        return [self.predict(img) for img in images]
