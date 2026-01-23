#!/usr/bin/env python
"""Lane Segmentation Training CLI"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.lane import LaneTrainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train lane segmentation model")
    parser.add_argument("--config", type=str, default="configs/lane_config.yaml")
    parser.add_argument("--train-root", type=str, default=None)
    parser.add_argument("--val-root", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    trainer = LaneTrainer(
        config_path=args.config,
        train_root=args.train_root,
        val_root=args.val_root,
        run_dir=args.run_dir
    )
    trainer.train()
