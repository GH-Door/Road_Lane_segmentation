"""Lane detection inference module"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class LaneDetector:
    """Lane detection inference class"""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        use_channels_last: bool = True
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_channels_last = use_channels_last

        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = self.checkpoint.get("config", {})

        self._setup_model()
        self._setup_amp()

    def _setup_model(self):
        """Load model from checkpoint"""
        if smp is None:
            raise ImportError("segmentation_models_pytorch not installed")

        model_cfg = self.config.get("model", {})
        encoder = model_cfg.get("encoder", "resnet50")

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        ).to(self.device)

        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.eval()

        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

    def _setup_amp(self):
        """Setup automatic mixed precision"""
        self.amp_dtype = None
        if self.device.type == "cuda":
            major, _ = torch.cuda.get_device_capability()
            self.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16

    def preprocess(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (768, 1024),
        divisor: int = 32
    ) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preprocess image for inference

        Args:
            image: (H, W, 3) BGR or RGB numpy array
            target_size: (height, width) to resize to
            divisor: padding divisor

        Returns:
            tensor, original_hw, resized_hw
        """
        orig_h, orig_w = image.shape[:2]
        h, w = target_size

        img_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        x = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

        return x.unsqueeze(0), (orig_h, orig_w), (h, w)

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (768, 1024),
        threshold: float = 0.5,
        return_prob: bool = False
    ) -> Dict[str, Any]:
        """
        Run lane detection on image

        Args:
            image: (H, W, 3) numpy array
            target_size: resize target (h, w)
            threshold: binarization threshold
            return_prob: whether to return probability map

        Returns:
            dict with 'mask' (binary), optionally 'prob'
        """
        x, orig_hw, resized_hw = self.preprocess(image, target_size)
        x = x.to(self.device)

        if self.use_channels_last:
            x = x.to(memory_format=torch.channels_last)

        if self.device.type == "cuda" and self.amp_dtype:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                logits = self.model(x)
        else:
            logits = self.model(x)

        h, w = resized_hw
        logits_cropped = logits[:, :, :h, :w]

        logits_up = F.interpolate(
            logits_cropped, size=orig_hw, mode="bilinear", align_corners=False
        )

        prob = torch.sigmoid(logits_up.float())[0, 0].cpu().numpy()
        mask = (prob >= threshold).astype(np.uint8)

        result = {"mask": mask}
        if return_prob:
            result["prob"] = prob

        return result

    def predict_batch(
        self,
        images: list,
        target_size: Tuple[int, int] = (768, 1024),
        threshold: float = 0.5
    ) -> list:
        """Run prediction on multiple images"""
        return [self.predict(img, target_size, threshold) for img in images]
