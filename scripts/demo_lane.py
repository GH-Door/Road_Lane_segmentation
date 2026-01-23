#!/usr/bin/env python
"""Lane Detection Demo with Steering Visualization"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.lane import LaneDetector
from src.control import SteeringController, LanePilot, SteeringVisualizer


def demo_image(checkpoint: str, image_path: str, output_path: str = None):
    """Run demo on single image"""
    detector = LaneDetector(checkpoint)
    controller = SteeringController()
    visualizer = SteeringVisualizer()

    pilot = LanePilot(detector, controller, visualizer)

    image = cv2.imread(image_path)
    result = pilot.process_frame(image)

    print(f"Steering: {result['steering']}")

    out = output_path or "demo_output.png"
    cv2.imwrite(out, result["vis_frame"])
    print(f"Saved to {out}")


def demo_video(checkpoint: str, video_path: str, output_path: str = None):
    """Run demo on video"""
    detector = LaneDetector(checkpoint)
    controller = SteeringController()
    pilot = LanePilot(detector, controller)

    out = output_path or "demo_output.mp4"
    n_frames = pilot.process_video(video_path, out)
    print(f"Processed {n_frames} frames -> {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lane detection demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--video", type=str, help="Input video path")
    parser.add_argument("--output", type=str, help="Output path")
    args = parser.parse_args()

    if args.image:
        demo_image(args.checkpoint, args.image, args.output)
    elif args.video:
        demo_video(args.checkpoint, args.video, args.output)
    else:
        print("Please provide --image or --video")
