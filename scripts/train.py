#!/usr/bin/env python
"""Training CLI - Semantic Segmentation"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training import Trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Config file path"
    )
    args = parser.parse_args()

    trainer = Trainer(config_path=args.config)
    trainer.train()
