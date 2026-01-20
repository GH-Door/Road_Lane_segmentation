import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models import DeepLabV3Plus
from src.data import TestDatasetLoader, get_transforms
from src.utils import load_config, get_device
from src.utils.metrics import calculate_all_metrics


class Evaluator:
    """Test Dataset Evaluator"""

    def __init__(
        self,
        checkpoint: str = "checkpoints/best.pt",
        config_path: str = "configs/config.yaml"
    ):
        self.config = load_config(config_path)
        self.device = get_device()
        self.checkpoint_path = Path(checkpoint)

        self.data_cfg = self.config.get("data", {})
        self.model_cfg = self.config.get("model", {})

        self.img_size = tuple(self.data_cfg.get("img_size", [1024, 768]))
        self.num_classes = self.data_cfg.get("num_classes", 62)

        self.model = self.load_model()

    def load_model(self) -> torch.nn.Module:
        """모델 로드"""
        model = DeepLabV3Plus(
            num_classes=self.num_classes,
            backbone=self.model_cfg.get("name", "deeplabv3_resnet50"),
            pretrained=False
        ).to(self.device)

        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded checkpoint: {self.checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        model.eval()
        return model

    def evaluate(
        self,
        test_data_root: str,
        batch_size: int = 4,
        num_workers: int = 4
    ) -> Dict[str, float]:
        """
        테스트 데이터셋 평가

        Args:
            test_data_root: Mono Camera 테스트 데이터셋 경로
            batch_size: 배치 크기
            num_workers: 데이터 로더 워커 수

        Returns:
            metrics: 평가 메트릭 딕셔너리
        """
        # 테스트 데이터셋 로드
        transform = get_transforms(self.img_size)
        test_dataset = TestDatasetLoader(
            data_root=test_data_root,
            transform=transform
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # 평가
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)["out"]

                all_outputs.append(outputs.cpu())
                all_targets.append(masks.cpu())

        # 전체 메트릭 계산
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = calculate_all_metrics(
            all_outputs,
            all_targets,
            self.num_classes
        )

        # 결과 출력
        print("\n" + "=" * 50)
        print("Test Evaluation Results")
        print("=" * 50)
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print("=" * 50)

        return metrics

    @torch.no_grad()
    def evaluate_per_sample(
        self,
        test_data_root: str
    ) -> list:
        """샘플별 평가 결과 반환"""
        transform = get_transforms(self.img_size)
        test_dataset = TestDatasetLoader(
            data_root=test_data_root,
            transform=transform
        )

        results = []
        for idx in tqdm(range(len(test_dataset)), desc="Per-sample evaluation"):
            img, mask = test_dataset[idx]
            img = img.unsqueeze(0).to(self.device)

            output = self.model(img)["out"]

            metrics = calculate_all_metrics(
                output.cpu(),
                mask.unsqueeze(0),
                self.num_classes
            )

            sample_info = test_dataset.get_sample_info(idx)
            results.append({
                "filename": sample_info["filename"],
                **metrics
            })

        return results
