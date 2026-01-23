import torch
from pathlib import Path
from typing import Optional, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

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
            log_dir="logs",
            log_prefix="eval"
        )

        self.device = get_device()
        self.checkpoint_path = checkpoint_path
        self.split = split
        self.class_info_path = class_info_path

        # load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # load & merge config (checkpoint has priority)
        config = load_config(config_path)
        if "config" in self.checkpoint:
            config.update(self.checkpoint["config"])
        self.config = config

        self.data_cfg = self.config.get("data", {})
        self.model_cfg = self.config.get("model", {})
        self.train_cfg = self.config.get("training", {})

        self.trained_num_classes = self.checkpoint["config"]["data"]["num_classes"]
        self.trained_class_info_path = self.checkpoint["config"]["data"].get("class_info")

        if data_root:
            self.data_cfg["root"] = data_root

        if class_info_path:
            self.final_class_info_path = class_info_path
        elif self.trained_class_info_path:
            self.final_class_info_path = self.trained_class_info_path
        else:
            self.final_class_info_path = self.data_cfg.get("class_info")

        self.setup_data()
        self.setup_model()

    def setup_data(self):
        """build dataloader"""
        img_size = tuple(self.data_cfg.get("img_size", [1024, 768]))
        transform = get_transforms(img_size=img_size)

        group_config_path = self.data_cfg.get("group_config")
        group_cfg = {}
        class_grouping = None
        num_grouped_classes = None

        if group_config_path:
            group_cfg = load_config(group_config_path)
            if group_cfg.get("enabled", False):
                class_grouping = group_cfg.get("mapping")
                num_grouped_classes = group_cfg.get("num_classes")

        self.dataset = DatasetLoader(
            data_root=self.data_cfg["root"],
            split=self.split,
            transform=transform,
            class_info_path=self.final_class_info_path,
            class_grouping=class_grouping,
            num_grouped_classes=num_grouped_classes
        )

        self.num_classes = self.trained_num_classes
        if num_grouped_classes is not None:
            assert self.num_classes == num_grouped_classes, (
                f"Mismatch num_classes: checkpoint({self.num_classes}) "
                f"!= group_config({num_grouped_classes})"
            )
        batch_size = self.train_cfg["eval_batch_size"]
        num_workers = self.train_cfg.get("num_workers", 4)
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
        """load trained model"""
        self.config["data"]["num_classes"] = self.num_classes
        # disable pretrained (load checkpoint)
        self.config["model"]["pretrained"] = False

        self.model = get_model_from_config(self.config).to(self.device)

        self.model.load_state_dict(self.checkpoint["model"])
        self.model.eval()

        model_info = self.model.get_info()
        self.logger.info(
            f"Model: {model_info['model_name']} + {model_info['encoder_name']} "
            f"({model_info['total_params_M']}M params)"
        )
        self.logger.info(f"Checkpoint: {self.checkpoint_path}")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """run evaluation"""
        self.model.eval()

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
            metrics_calculator.update(outputs, masks)

        metrics = metrics_calculator.compute()

        self.logger.info(
            f"Evaluation Results | "
            f"mIoU: {metrics['miou']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Dice: {metrics['dice']:.4f}"
        )

        return metrics
