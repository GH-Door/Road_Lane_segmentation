#!/usr/bin/env python
"""Evaluation CLI - Semantic Segmentation"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import Evaluator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Data root path override"
    )
    parser.add_argument(
        "--class_info_path",
        type=str,
        default=None,
        help="Class info CSV path"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to evaluate"
    )
    args = parser.parse_args()

    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_root=args.data_root,
        class_info_path=args.class_info_path,
        split=args.split
    )
    metrics = evaluator.evaluate()
