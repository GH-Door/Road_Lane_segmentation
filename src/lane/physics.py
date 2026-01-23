"""Lane physics calculations for steering"""
import numpy as np
import cv2
from typing import Tuple, Dict, Optional


class LanePhysics:
    """Physics calculations for lane-based steering"""

    def __init__(
        self,
        img_height: int = 720,
        img_width: int = 1280,
        roi_top_ratio: float = 0.5,
        roi_bottom_ratio: float = 0.95,
        max_steer_angle: float = 45.0
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.roi_top_ratio = roi_top_ratio
        self.roi_bottom_ratio = roi_bottom_ratio
        self.max_steer_angle = max_steer_angle

        self.center_x = img_width // 2

    def extract_lane_center(self, mask: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract lane center line from binary mask

        Args:
            mask: (H, W) binary lane mask

        Returns:
            (offset_ratio, curvature) or (None, None) if no lane
        """
        h, w = mask.shape
        roi_top = int(h * self.roi_top_ratio)
        roi_bottom = int(h * self.roi_bottom_ratio)

        roi = mask[roi_top:roi_bottom, :]

        center_points = []
        for row_idx in range(roi.shape[0]):
            row = roi[row_idx, :]
            lane_pixels = np.where(row > 0)[0]

            if len(lane_pixels) > 5:
                center = lane_pixels.mean()
                center_points.append((row_idx + roi_top, center))

        if len(center_points) < 3:
            return None, None

        ys, xs = zip(*center_points)
        ys = np.array(ys)
        xs = np.array(xs)

        bottom_idx = np.argmax(ys)
        offset = (xs[bottom_idx] - w / 2) / (w / 2)

        if len(xs) >= 5:
            coeffs = np.polyfit(ys, xs, 2)
            curvature = 2 * coeffs[0]
        else:
            curvature = 0.0

        return float(offset), float(curvature)

    def compute_steering(
        self,
        mask: np.ndarray,
        offset_gain: float = 0.8,
        curvature_gain: float = 0.3
    ) -> Dict[str, float]:
        """
        Compute steering angle from lane mask

        Args:
            mask: (H, W) binary lane mask
            offset_gain: weight for lateral offset
            curvature_gain: weight for curvature

        Returns:
            dict with steering_angle, offset, curvature
        """
        offset, curvature = self.extract_lane_center(mask)

        if offset is None:
            return {
                "steering_angle": 0.0,
                "offset": 0.0,
                "curvature": 0.0,
                "valid": False
            }

        steer = offset_gain * offset + curvature_gain * curvature
        steer = np.clip(steer, -1.0, 1.0)
        angle = steer * self.max_steer_angle

        return {
            "steering_angle": float(angle),
            "offset": float(offset),
            "curvature": float(curvature),
            "valid": True
        }
