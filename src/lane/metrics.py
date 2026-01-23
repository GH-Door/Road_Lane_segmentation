"""Lane segmentation metrics: Dice, IoU, Boundary F1"""
import numpy as np
import cv2
from typing import Dict, Tuple, List


def dice_iou(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    """Compute Dice and IoU scores"""
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    p = pred.sum()
    g = gt.sum()

    dice = (2.0 * inter + eps) / (p + g + eps)
    iou = (inter + eps) / (p + g - inter + eps)

    return float(dice), float(iou)


def boundary_map(bin_mask: np.ndarray) -> np.ndarray:
    """Extract boundary from binary mask using morphological gradient"""
    kernel = np.ones((3, 3), np.uint8)
    b = cv2.morphologyEx(bin_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    return (b > 0).astype(np.uint8)


def boundary_f1(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    tol_px: int = 3
) -> Tuple[float, float, float]:
    """Compute Boundary F1 score with tolerance"""
    pred_b = boundary_map(pred_bin)
    gt_b = boundary_map(gt_bin)

    pred_pts = pred_b.sum()
    gt_pts = gt_b.sum()

    if pred_pts == 0 and gt_pts == 0:
        return 1.0, 1.0, 1.0
    if pred_pts == 0 or gt_pts == 0:
        return 0.0, 0.0, 0.0

    dt_gt = cv2.distanceTransform((1 - gt_b).astype(np.uint8), cv2.DIST_L2, 3)
    dt_pred = cv2.distanceTransform((1 - pred_b).astype(np.uint8), cv2.DIST_L2, 3)

    pred_match = (pred_b == 1) & (dt_gt <= tol_px)
    gt_match = (gt_b == 1) & (dt_pred <= tol_px)

    precision = pred_match.sum() / (pred_pts + 1e-6)
    recall = gt_match.sum() / (gt_pts + 1e-6)
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)

    return float(precision), float(recall), float(f1)


class LaneMetrics:
    """Accumulator for lane segmentation metrics"""

    def __init__(self, tols: Tuple[int, ...] = (2, 4, 8)):
        self.tols = tols
        self.reset()

    def reset(self):
        self.dices = []
        self.ious = []
        self.bf1 = {t: [] for t in self.tols}

    def update(self, pred_bin: np.ndarray, gt_bin: np.ndarray):
        """Add a single sample's metrics"""
        d, j = dice_iou(pred_bin, gt_bin)
        self.dices.append(d)
        self.ious.append(j)

        for t in self.tols:
            _, _, f1 = boundary_f1(pred_bin, gt_bin, tol_px=t)
            self.bf1[t].append(f1)

    def compute(self) -> Dict[str, float]:
        """Compute aggregated metrics"""
        out = {
            "dice": float(np.mean(self.dices)) if self.dices else 0.0,
            "iou": float(np.mean(self.ious)) if self.ious else 0.0,
        }
        for t in self.tols:
            out[f"bf1@{t}px"] = float(np.mean(self.bf1[t])) if self.bf1[t] else 0.0
        return out
