import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification

    논문: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    수식: FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, C, H, W) logits
            target: (N, H, W) labels
        """
        ce_loss = F.cross_entropy(
            pred, target,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MemoryEfficientDiceLoss(nn.Module):
    """
    Memory-Efficient Dice Loss

    smp.losses.DiceLoss 대비 장점:
    - One-hot encoding 없음 → VRAM 대폭 절감
    - for 루프로 클래스별 순차 계산

    수식: Dice = 2 * intersection / (pred + target)
    Loss = 1 - Dice
    """

    def __init__(
        self,
        num_classes: int = 62,
        ignore_index: int = 255,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, C, H, W) logits
            target: (N, H, W) labels

        Returns:
            Dice loss (scalar)
        """
        # Softmax 적용
        pred_soft = F.softmax(pred, dim=1)

        # Valid mask (ignore_index 제외)
        valid_mask = (target != self.ignore_index)

        dice_sum = 0.0
        valid_classes = 0

        # 클래스별 순차 계산 (메모리 효율적)
        for cls in range(self.num_classes):
            # 해당 클래스의 예측 확률
            pred_cls = pred_soft[:, cls]  # (N, H, W)

            # 해당 클래스의 타겟 마스크
            target_cls = (target == cls).float()  # (N, H, W)

            # Valid mask 적용
            pred_cls = pred_cls * valid_mask
            target_cls = target_cls * valid_mask

            # Dice 계산
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()

            # 해당 클래스가 존재하는 경우만 계산
            if union > 0:
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_sum += dice
                valid_classes += 1

        # 평균 Dice Loss
        if valid_classes > 0:
            mean_dice = dice_sum / valid_classes
        else:
            mean_dice = torch.tensor(1.0, device=pred.device)

        return 1.0 - mean_dice


class SegmentationLoss(nn.Module):
    """
    Combined Segmentation Loss

    지원 조합:
    - 단일: "ce", "dice", "focal"
    - 조합: "ce+dice", "focal+dice"
    """

    def __init__(
        self,
        loss_type: str = "ce",
        num_classes: int = 62,
        weights: Optional[Dict[str, float]] = None,
        ignore_index: int = 255,
        class_weight: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            loss_type: 'ce', 'dice', 'focal', 'ce+dice', 'focal+dice' 등
            num_classes: 클래스 수
            weights: 조합 시 가중치 {'ce': 0.5, 'dice': 0.5}
            ignore_index: 무시할 클래스 인덱스
            class_weight: 클래스별 가중치 tensor [num_classes]
            focal_gamma: Focal Loss의 gamma 값
        """
        super().__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.weights = weights or {}
        self.losses = nn.ModuleDict()

        # 단일 loss
        if "+" not in loss_type:
            self.losses[loss_type] = self._create_loss(
                loss_type, num_classes, ignore_index, class_weight, focal_gamma
            )
            self.weights[loss_type] = 1.0
        # 조합 loss (예: ce+dice)
        else:
            loss_names = loss_type.split("+")
            default_weight = 1.0 / len(loss_names)
            for name in loss_names:
                name = name.strip()
                self.losses[name] = self._create_loss(
                    name, num_classes, ignore_index, class_weight, focal_gamma
                )
                if name not in self.weights:
                    self.weights[name] = default_weight

    def _create_loss(
        self,
        name: str,
        num_classes: int,
        ignore_index: int,
        class_weight: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
    ) -> nn.Module:
        """개별 loss 생성"""
        if name == "ce":
            return nn.CrossEntropyLoss(
                weight=class_weight,
                ignore_index=ignore_index
            )
        elif name == "dice":
            # Memory-efficient 버전 사용!
            return MemoryEfficientDiceLoss(
                num_classes=num_classes,
                ignore_index=ignore_index
            )
        elif name == "focal":
            return FocalLoss(
                gamma=focal_gamma,
                alpha=class_weight,
                ignore_index=ignore_index
            )
        else:
            raise ValueError(f"Unknown loss: {name}. Available: ce, dice, focal")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for name, loss_fn in self.losses.items():
            total_loss = total_loss + self.weights[name] * loss_fn(pred, target)
        return total_loss


def get_loss(
    config: Dict,
    num_classes: int,
    class_weight: Optional[torch.Tensor] = None
) -> SegmentationLoss:
    """config에서 loss 생성"""
    loss_cfg = config.get("loss", {})
    return SegmentationLoss(
        loss_type=loss_cfg.get("type", "ce"),
        num_classes=num_classes,
        weights=loss_cfg.get("weights"),
        ignore_index=loss_cfg.get("ignore_index", 255),
        class_weight=class_weight,
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
    )


try:
    import segmentation_models_pytorch as smp

    # smp losses를 직접 사용하고 싶은 경우를 위해 alias 제공
    SMPDiceLoss = smp.losses.DiceLoss
    SMPFocalLoss = smp.losses.FocalLoss
    SMPJaccardLoss = smp.losses.JaccardLoss
    SMPLovaszLoss = smp.losses.LovaszLoss
except ImportError:
    pass
