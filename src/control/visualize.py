"""Visualization utilities for steering and lane detection"""
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class VisConfig:
    """Configuration for visualization"""
    wheel_size: int = 150
    wheel_pos: Tuple[int, int] = (50, 50)
    wheel_color: Tuple[int, int, int] = (200, 200, 200)
    spoke_color: Tuple[int, int, int] = (50, 50, 50)
    mask_color: Tuple[int, int, int] = (0, 255, 0)
    mask_alpha: float = 0.4
    text_color: Tuple[int, int, int] = (255, 255, 255)
    font_scale: float = 0.7


class SteeringVisualizer:
    """Visualizer for steering wheel and lane overlay"""

    def __init__(self, config: Optional[VisConfig] = None):
        self.cfg = config or VisConfig()

    def draw_wheel(
        self,
        canvas: np.ndarray,
        angle: float,
        pos: Optional[Tuple[int, int]] = None,
        size: Optional[int] = None
    ) -> np.ndarray:
        """
        Draw steering wheel on canvas

        Args:
            canvas: (H, W, 3) image
            angle: steering angle in degrees
            pos: (x, y) top-left position
            size: wheel diameter

        Returns:
            canvas with wheel drawn
        """
        pos = pos or self.cfg.wheel_pos
        size = size or self.cfg.wheel_size
        x, y = pos
        r = size // 2
        cx, cy = x + r, y + r

        cv2.circle(canvas, (cx, cy), r, self.cfg.wheel_color, 3)
        cv2.circle(canvas, (cx, cy), r - 20, self.cfg.wheel_color, 2)

        rad = np.radians(-angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        for spoke_angle in [0, 120, 240]:
            base = np.radians(spoke_angle)
            dx = int((r - 25) * np.cos(base + rad))
            dy = int((r - 25) * np.sin(base + rad))
            cv2.line(canvas, (cx, cy), (cx + dx, cy + dy), self.cfg.spoke_color, 3)

        cv2.circle(canvas, (cx, cy), 15, self.cfg.spoke_color, -1)

        cv2.putText(
            canvas, f"{angle:+.1f}",
            (cx - 25, cy + r + 25),
            cv2.FONT_HERSHEY_SIMPLEX, self.cfg.font_scale,
            self.cfg.text_color, 2
        )

        return canvas

    def overlay_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Optional[Tuple[int, int, int]] = None,
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Overlay lane mask on image

        Args:
            image: (H, W, 3) BGR image
            mask: (H, W) binary mask
            color: BGR color for overlay
            alpha: blend factor

        Returns:
            blended image
        """
        color = color or self.cfg.mask_color
        alpha = alpha if alpha is not None else self.cfg.mask_alpha

        overlay = image.copy()
        mask_bool = mask > 0

        overlay[mask_bool] = (
            (1 - alpha) * overlay[mask_bool] +
            alpha * np.array(color)
        ).astype(np.uint8)

        return overlay

    def compose_frame(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        angle: float,
        show_mask: bool = True,
        show_wheel: bool = True
    ) -> np.ndarray:
        """
        Compose full visualization frame

        Args:
            image: (H, W, 3) original image
            mask: (H, W) lane mask
            angle: steering angle in degrees
            show_mask: whether to overlay mask
            show_wheel: whether to draw wheel

        Returns:
            composed frame
        """
        canvas = image.copy()

        if show_mask:
            canvas = self.overlay_mask(canvas, mask)

        if show_wheel:
            canvas = self.draw_wheel(canvas, angle)

        return canvas


class VideoGenerator:
    """Generate video from frames with steering visualization"""

    def __init__(
        self,
        output_path: str,
        fps: int = 30,
        vis_config: Optional[VisConfig] = None
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.visualizer = SteeringVisualizer(vis_config)
        self._writer = None
        self._frame_size = None

    def _init_writer(self, frame: np.ndarray):
        """Initialize video writer with first frame"""
        h, w = frame.shape[:2]
        self._frame_size = (w, h)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, self._frame_size
        )

    def add_frame(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        angle: float
    ):
        """Add a frame to video"""
        frame = self.visualizer.compose_frame(image, mask, angle)

        if self._writer is None:
            self._init_writer(frame)

        self._writer.write(frame)

    def close(self):
        """Close video writer"""
        if self._writer:
            self._writer.release()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class LanePilot:
    """Complete pipeline: detection + steering + visualization"""

    def __init__(
        self,
        detector,
        controller,
        visualizer: Optional[SteeringVisualizer] = None
    ):
        """
        Args:
            detector: LaneDetector instance
            controller: SteeringController instance
            visualizer: optional SteeringVisualizer
        """
        self.detector = detector
        self.controller = controller
        self.visualizer = visualizer or SteeringVisualizer()

    def process_frame(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (768, 1024)
    ) -> dict:
        """
        Process single frame

        Args:
            image: (H, W, 3) BGR image
            target_size: detection target size

        Returns:
            dict with mask, steering, vis_frame
        """
        result = self.detector.predict(image, target_size)
        mask = result["mask"]

        steering = self.controller.compute(mask)

        vis = self.visualizer.compose_frame(image, mask, steering["angle"])

        return {
            "mask": mask,
            "steering": steering,
            "vis_frame": vis
        }

    def process_video(
        self,
        input_path: str,
        output_path: str,
        target_size: Tuple[int, int] = (768, 1024),
        max_frames: Optional[int] = None
    ):
        """
        Process video file

        Args:
            input_path: input video path
            output_path: output video path
            target_size: detection target size
            max_frames: limit number of frames
        """
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        with VideoGenerator(output_path, fps) as gen:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_idx >= max_frames:
                    break

                result = self.process_frame(frame, target_size)
                gen.add_frame(frame, result["mask"], result["steering"]["angle"])
                frame_idx += 1

        cap.release()
        return frame_idx
