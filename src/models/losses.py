import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Optional


# Loss 레지스트리
LOSSES = {
    "ce": nn.CrossEntropyLoss,
    "dice": smp.losses.DiceLoss,
    "focal": smp.losses.FocalLoss,
    "jaccard": smp.losses.JaccardLoss,
    "lovasz": smp.losses.LovaszLoss,
}


class SegmentationLoss(nn.Module):
    """Combined loss for semantic segmentation"""

    def __init__(
        self,
        loss_type: str = "ce",
        num_classes: int = 62,
        weights: Optional[Dict[str, float]] = None,
        ignore_index: int = 255
    ):
        """
        Args:
            loss_type: 'ce', 'dice', 'focal', 'jaccard', 'lovasz', 'ce+dice' 등
            num_classes: 클래스 수
            weights: 조합 시 가중치 {'ce': 0.5, 'dice': 0.5}
            ignore_index: 무시할 클래스 인덱스
        """
        super().__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.weights = weights or {}
        self.losses = nn.ModuleDict()

        # 단일 loss
        if "+" not in loss_type:
            self.losses[loss_type] = self._create_loss(loss_type, ignore_index)
            self.weights[loss_type] = 1.0
        # 조합 loss (예: ce+dice)
        else:
            loss_names = loss_type.split("+")
            default_weight = 1.0 / len(loss_names)
            for name in loss_names:
                name = name.strip()
                self.losses[name] = self._create_loss(name, ignore_index)
                if name not in self.weights:
                    self.weights[name] = default_weight

    def _create_loss(self, name: str, ignore_index: int) -> nn.Module:
        """개별 loss 생성"""
        if name == "ce":
            return nn.CrossEntropyLoss(ignore_index=ignore_index)
        elif name == "dice":
            return smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)
        elif name == "focal":
            return smp.losses.FocalLoss(mode="multiclass", ignore_index=ignore_index)
        elif name == "jaccard":
            return smp.losses.JaccardLoss(mode="multiclass")
        elif name == "lovasz":
            return smp.losses.LovaszLoss(mode="multiclass", ignore_index=ignore_index)
        else:
            raise ValueError(f"Unknown loss: {name}. Available: {list(LOSSES.keys())}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for name, loss_fn in self.losses.items():
            total_loss += self.weights[name] * loss_fn(pred, target)
        return total_loss


def get_loss(config: Dict, num_classes: int) -> SegmentationLoss:
    """config에서 loss 생성"""
    loss_cfg = config.get("loss", {})
    return SegmentationLoss(
        loss_type=loss_cfg.get("type", "ce"),
        num_classes=num_classes,
        weights=loss_cfg.get("weights"),
        ignore_index=loss_cfg.get("ignore_index", 255)
    )
