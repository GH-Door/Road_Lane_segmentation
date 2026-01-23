"""Steering controller for lane-based autonomous driving"""
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SteeringConfig:
    """Configuration for steering controller"""
    max_angle: float = 45.0
    offset_gain: float = 0.8
    curvature_gain: float = 0.3
    smoothing: float = 0.3
    roi_top: float = 0.5
    roi_bottom: float = 0.95


class SteeringController:
    """Controller that computes steering from lane masks"""

    def __init__(self, config: Optional[SteeringConfig] = None):
        self.cfg = config or SteeringConfig()
        self._prev_angle = 0.0

    def reset(self):
        """Reset controller state"""
        self._prev_angle = 0.0

    def _extract_center_line(
        self,
        mask: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract center points of lane from mask"""
        h, w = mask.shape
        top = int(h * self.cfg.roi_top)
        bottom = int(h * self.cfg.roi_bottom)

        ys, xs = [], []
        for y in range(top, bottom):
            row = mask[y, :]
            pixels = np.where(row > 0)[0]
            if len(pixels) > 3:
                ys.append(y)
                xs.append(pixels.mean())

        if len(ys) < 3:
            return None, None

        return np.array(ys), np.array(xs)

    def _fit_curve(self, ys: np.ndarray, xs: np.ndarray) -> Tuple[float, float]:
        """Fit polynomial to get offset and curvature"""
        h = ys.max()
        bottom_idx = np.argmax(ys)
        w = xs.max() - xs.min() + 1e-6

        center_x = (xs.min() + xs.max()) / 2
        offset = (xs[bottom_idx] - center_x) / (w / 2)

        if len(ys) >= 5:
            coeffs = np.polyfit(ys, xs, 2)
            curvature = coeffs[0] * 1000
        else:
            curvature = 0.0

        return float(offset), float(curvature)

    def compute(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Compute steering angle from lane mask

        Args:
            mask: (H, W) binary lane mask

        Returns:
            dict with angle, offset, curvature, valid
        """
        ys, xs = self._extract_center_line(mask)

        if ys is None:
            return {
                "angle": self._prev_angle,
                "offset": 0.0,
                "curvature": 0.0,
                "valid": False
            }

        offset, curvature = self._fit_curve(ys, xs)

        raw = self.cfg.offset_gain * offset + self.cfg.curvature_gain * curvature
        raw = np.clip(raw, -1.0, 1.0)
        angle = raw * self.cfg.max_angle

        angle = self.cfg.smoothing * self._prev_angle + (1 - self.cfg.smoothing) * angle
        self._prev_angle = angle

        return {
            "angle": float(angle),
            "offset": float(offset),
            "curvature": float(curvature),
            "valid": True
        }
