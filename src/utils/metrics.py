import torch
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from typing import Dict


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """mIoU 계산 (torchmetrics)"""
    metric = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
    pred_labels = pred.argmax(dim=1)
    miou = metric(pred_labels, target).item()
    return {"miou": miou}


def calculate_accuracy(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """Accuracy 계산 (torchmetrics)"""
    metric = MulticlassAccuracy(num_classes=num_classes, average="macro")
    pred_labels = pred.argmax(dim=1)
    acc = metric(pred_labels, target).item()
    return {"accuracy": acc}


def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """모든 메트릭 계산 (mIoU + Accuracy)"""
    pred_labels = pred.argmax(dim=1)

    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average="macro")

    return {
        "miou": miou_metric(pred_labels, target).item(),
        "accuracy": acc_metric(pred_labels, target).item(),
    }
