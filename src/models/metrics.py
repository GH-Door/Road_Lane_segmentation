"""
Memory-Efficient Segmentation Metrics

핵심: torchmetrics 사용 + Dice는 for 루프로 클래스별 순차 계산
→ One-hot encoding 없이 메모리 효율적

사용법:
    metrics = SegmentationMetrics(num_classes=62, device='cuda')

    for images, masks in dataloader:
        outputs = model(images)
        metrics.update(outputs, masks)

    results = metrics.compute()
    # {'miou': 0.45, 'accuracy': 0.82, 'dice': 0.52}
"""

import torch
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from typing import Dict, Optional


class SegmentationMetrics:
    """
    Memory-Efficient Segmentation Metrics

    torchmetrics 기반 mIoU, Accuracy + 수동 Dice 계산
    Dice는 for 루프로 클래스별 계산하여 VRAM 절약
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        device: str = 'cuda'
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device

        # torchmetrics - mIoU
        self.miou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='macro'
        ).to(device)

        # torchmetrics - Accuracy
        self.accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average='micro'
        ).to(device)

        # Dice 계산을 위한 누적 변수
        self.dice_intersection = torch.zeros(num_classes, device=device)
        self.dice_union = torch.zeros(num_classes, device=device)

    def reset(self):
        """메트릭 초기화"""
        self.miou.reset()
        self.accuracy.reset()
        self.dice_intersection = torch.zeros(self.num_classes, device=self.device)
        self.dice_union = torch.zeros(self.num_classes, device=self.device)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        배치 단위 업데이트

        Args:
            pred: (N, C, H, W) logits
            target: (N, H, W) labels
        """
        # argmax로 클래스 예측 (VRAM 절약의 핵심!)
        pred_classes = pred.argmax(dim=1)

        # torchmetrics 업데이트
        self.miou.update(pred_classes, target)
        self.accuracy.update(pred_classes, target)

        # Dice 수동 계산 (for 루프로 클래스별 - 메모리 효율적)
        for cls in range(self.num_classes):
            pred_mask = (pred_classes == cls)
            target_mask = (target == cls)

            # ignore_index 처리
            valid_mask = (target != self.ignore_index)
            pred_mask = pred_mask & valid_mask
            target_mask = target_mask & valid_mask

            intersection = (pred_mask & target_mask).sum().float()
            union = pred_mask.sum().float() + target_mask.sum().float()

            self.dice_intersection[cls] += intersection
            self.dice_union[cls] += union

    def compute(self) -> Dict[str, float]:
        """
        최종 메트릭 계산

        Returns:
            {'miou': float, 'accuracy': float, 'dice': float}
        """
        # Dice 계산: 2 * intersection / union
        dice_per_class = 2 * self.dice_intersection / (self.dice_union + 1e-8)

        # 유효한 클래스만 평균 (union > 0인 경우)
        valid_classes = self.dice_union > 0
        if valid_classes.sum() > 0:
            mean_dice = dice_per_class[valid_classes].mean().item()
        else:
            mean_dice = 0.0

        return {
            'miou': self.miou.compute().item(),
            'accuracy': self.accuracy.compute().item(),
            'dice': mean_dice,
        }

    def compute_per_class(self) -> Dict[str, torch.Tensor]:
        """
        클래스별 메트릭 계산 (디버깅/분석용)

        Returns:
            {'dice_per_class': Tensor[num_classes]}
        """
        dice_per_class = 2 * self.dice_intersection / (self.dice_union + 1e-8)
        return {
            'dice_per_class': dice_per_class,
        }


# ============================================================
# Legacy functions (backward compatibility)
# ============================================================
def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """mIoU 계산 (torchmetrics)"""
    metric = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
    pred_labels = pred.argmax(dim=1)
    miou = metric(pred_labels.cpu(), target.cpu()).item()
    return {"miou": miou}


def calculate_accuracy(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """Accuracy 계산 (torchmetrics)"""
    metric = MulticlassAccuracy(num_classes=num_classes, average="macro")
    pred_labels = pred.argmax(dim=1)
    acc = metric(pred_labels.cpu(), target.cpu()).item()
    return {"accuracy": acc}


def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """모든 메트릭 계산 (mIoU + Accuracy)"""
    pred_labels = pred.argmax(dim=1)

    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average="macro")

    return {
        "miou": miou_metric(pred_labels.cpu(), target.cpu()).item(),
        "accuracy": acc_metric(pred_labels.cpu(), target.cpu()).item(),
    }
