import sys
import torch
from pathlib import Path
from typing import Optional, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

# project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data import DatasetLoader, get_transforms
from src.models import get_model_from_config
from src.models.metrics import SegmentationMetrics
from src.utils import load_config, get_device, setup_logger


class Evaluator:
    """Semantic Segmentation Evaluator"""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/config.yaml",
        data_root: Optional[str] = None,
        class_info_path: Optional[str] = None,
        split: str = "val",
    ):
        self.logger = setup_logger(
            name="Evaluator",
            log_dir=str(PROJECT_ROOT / "logs"),
            log_prefix="eval"
        )

        self.device = get_device()
        self.checkpoint_path = checkpoint_path
        self.split = split
        self.class_info_path = class_info_path

        # 체크포인트 로드
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False) # weights_only=False

        # Config 로드
        self.config = load_config(config_path)
        self.data_cfg = self.config.get("data", {})
        self.model_cfg = self.config.get("model", {})
        self.train_cfg = self.config.get("training", {})

        # Override data_root if provided
        if data_root:
            self.data_cfg["root"] = data_root

        self.setup_data()
        self.setup_model()

    def setup_data(self):
        """데이터로더 초기화"""
        img_size = tuple(self.data_cfg.get("img_size", [1024, 768]))
        transform = get_transforms(img_size=img_size)

        self.dataset = DatasetLoader(
            data_root=self.data_cfg["root"],
            split=self.split,
            transform=transform,
            class_info_path=self.class_info_path
        )

        self.num_classes = self.dataset.num_classes
        batch_size = self.train_cfg.get("batch_size", 4)
        num_workers = self.train_cfg.get("num_workers", 4)

        # MPS에서는 pin_memory 지원 안 됨
        use_pin_memory = self.device.type == "cuda"

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory
        )

        self.logger.info(f"Loaded {len(self.dataset)} samples from {self.split} split")
        self.logger.info(f"Number of classes: {self.num_classes}")

    def setup_model(self):
        """모델 로드"""
        # Config에 num_classes 추가
        self.config["data"]["num_classes"] = self.num_classes
        # pretrained=False로 설정 (체크포인트에서 로드하므로)
        self.config["model"]["pretrained"] = False

        self.model = get_model_from_config(self.config).to(self.device)

        self.model.load_state_dict(self.checkpoint["model"])
        self.model.eval()

        # 모델 정보 로깅
        model_info = self.model.get_info()
        self.logger.info(
            f"Model: {model_info['model_name']} + {model_info['encoder_name']} "
            f"({model_info['total_params_M']}M params)"
        )
        self.logger.info(f"Checkpoint: {self.checkpoint_path}")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """모델 평가 (Memory-Efficient SegmentationMetrics 사용)"""
        self.model.eval()

        # Memory-Efficient Metrics (for 루프로 클래스별 계산)
        ignore_index = self.config.get("loss", {}).get("ignore_index", 255)
        metrics_calculator = SegmentationMetrics(
            num_classes=self.num_classes,
            ignore_index=ignore_index,
            device=self.device
        )

        for images, masks in tqdm(self.dataloader, desc="Evaluating"):
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)["out"]

            # Memory-efficient 메트릭 업데이트
            metrics_calculator.update(outputs, masks)

        # 최종 메트릭 계산
        metrics = metrics_calculator.compute()

        self.logger.info(
            f"Evaluation Results | "
            f"mIoU: {metrics['miou']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Dice: {metrics['dice']:.4f}"
        )

        return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="체크포인트 경로")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config 경로")
    parser.add_argument("--data_root", type=str, default=None, help="데이터 경로 override")
    parser.add_argument("--class_info_path", type=str, default=None, help="클래스 정보 CSV 경로")
    parser.add_argument("--split", type=str, default="val", help="평가할 split")
    args = parser.parse_args()

    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_root=args.data_root,
        class_info_path=args.class_info_path,
        split=args.split
    )
    metrics = evaluator.evaluate()
