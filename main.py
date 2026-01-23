#!/usr/bin/env python
"""
Main Pipeline - Train & Evaluate Semantic Segmentation

Usage:
    # Terminal
    python main.py --config configs/config.yaml
    python main.py --config configs/config.yaml --skip-train  # 평가만
    python main.py --config configs/config.yaml --skip-eval   # 학습만

    # Jupyter
    from main import run_pipeline
    results = run_pipeline(config_path="configs/config.yaml")
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training import Trainer
from src.evaluation import Evaluator
from src.utils import setup_logger


def run_pipeline(
    config_path: str = "configs/config.yaml",
    skip_train: bool = False,
    skip_eval: bool = False,
    checkpoint_path: Optional[str] = None,
    eval_split: str = "val"
) -> Dict[str, Any]:
    """
    Train & Evaluate 파이프라인 실행

    Args:
        config_path: config YAML 경로
        skip_train: True면 학습 스킵
        skip_eval: True면 평가 스킵
        checkpoint_path: 평가용 체크포인트 (None이면 학습 후 best.pt 사용)
        eval_split: 평가 데이터셋 split ("val" or "test")

    Returns:
        {"train": trainer_instance, "eval": metrics_dict}
    """
    logger = setup_logger(name="Pipeline", log_dir="logs", log_prefix="main")
    results = {"train": None, "eval": None}

    # ========== Training ==========
    trainer = None
    if not skip_train:
        logger.info("=" * 50)
        logger.info("Starting Training...")
        logger.info("=" * 50)

        trainer = Trainer(config_path=config_path)
        trainer.train()
        results["train"] = trainer

        logger.info(f"Training complete. Checkpoint: {trainer.checkpoint_dir / 'best.pt'}")

    # ========== Evaluation ==========
    if not skip_eval:
        logger.info("=" * 50)
        logger.info("Starting Evaluation...")
        logger.info("=" * 50)

        # 체크포인트 경로 결정
        if checkpoint_path:
            ckpt = checkpoint_path
        elif trainer:
            ckpt = str(trainer.checkpoint_dir / "best.pt")
        else:
            raise ValueError("checkpoint_path required when skip_train=True")

        evaluator = Evaluator(
            checkpoint_path=ckpt,
            config_path=config_path,
            split=eval_split
        )
        metrics = evaluator.evaluate()
        results["eval"] = metrics

        logger.info("=" * 50)
        logger.info("Evaluation Results:")
        logger.info(f"  mIoU: {metrics['miou']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Dice: {metrics['dice']:.4f}")
        logger.info("=" * 50)

    return results


# Jupyter용 편의 함수
def train_only(config_path: str = "configs/config.yaml") -> Trainer:
    """학습만 실행"""
    result = run_pipeline(config_path=config_path, skip_eval=True)
    return result["train"]


def eval_only(
    checkpoint_path: str,
    config_path: str = "configs/config.yaml",
    split: str = "val"
) -> Dict[str, float]:
    """평가만 실행"""
    result = run_pipeline(
        config_path=config_path,
        skip_train=True,
        checkpoint_path=checkpoint_path,
        eval_split=split
    )
    return result["eval"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train & Evaluate Semantic Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config configs/config.yaml
  python main.py --config configs/config.yaml --skip-train --checkpoint checkpoints/best.pt
  python main.py --config configs/config.yaml --skip-eval
        """
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Config YAML path"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training, run evaluation only"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation, run training only"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint path for evaluation (required if --skip-train)"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test"],
        help="Evaluation split"
    )
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        checkpoint_path=args.checkpoint,
        eval_split=args.split
    )
