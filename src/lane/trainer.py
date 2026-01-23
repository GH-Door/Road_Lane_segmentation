"""Lane segmentation trainer with W&B logging"""
import os
import json
import time
import random
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

try:
    import wandb
except ImportError:
    wandb = None

from .dataset import SDLaneDataset, build_lane_items, train_collate_fn, val_collate_fn
from .metrics import LaneMetrics
from .losses import LaneLoss
from src.utils import load_config, get_device, setup_logger


class LaneTrainer:
    """Trainer for binary lane segmentation"""

    def __init__(
        self,
        config_path: str = "configs/lane_config.yaml",
        train_root: Optional[str] = None,
        val_root: Optional[str] = None,
        run_dir: Optional[str] = None
    ):
        self.config = load_config(config_path)
        self.device = get_device()

        self.data_cfg = self.config.get("data", {})
        self.model_cfg = self.config.get("model", {})
        self.train_cfg = self.config.get("training", {})
        self.loss_cfg = self.config.get("loss", {})

        if train_root:
            self.data_cfg["train_root"] = train_root
        if val_root:
            self.data_cfg["val_root"] = val_root

        self._setup_run_dir(run_dir)
        self._setup_logger()
        self._setup_seed()
        self._setup_speed_opts()
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_wandb()

    def _setup_run_dir(self, run_dir: Optional[str]):
        """Setup checkpoint and logging directories"""
        if run_dir:
            self.run_dir = Path(run_dir)
        else:
            runs_root = self.train_cfg.get("runs_root", "runs/lane")
            run_name = datetime.now().strftime("lane_%Y%m%d_%H%M%S")
            self.run_dir = Path(runs_root) / run_name

        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

    def _setup_logger(self):
        """Setup logging"""
        self.logger = setup_logger(
            name="LaneTrainer",
            log_dir=str(self.run_dir / "logs"),
            log_prefix="lane_train"
        )
        self.logger.info(f"Run dir: {self.run_dir}")

        self.history = []
        self.history_path = self.run_dir / "history.jsonl"

    def _setup_seed(self):
        """Set random seeds for reproducibility"""
        seed = self.train_cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cv2.setNumThreads(0)

    def _setup_speed_opts(self):
        """Configure speed optimizations"""
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

        self.use_compile = self.train_cfg.get("use_compile", True)
        self.use_channels_last = self.train_cfg.get("use_channels_last", True)
        self.grad_accum_steps = self.train_cfg.get("grad_accum_steps", 1)

        self.amp_dtype = None
        self.use_scaler = False
        if self.device.type == "cuda":
            major, _ = torch.cuda.get_device_capability()
            self.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
            self.use_scaler = (self.amp_dtype == torch.float16)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)
        self.logger.info(f"AMP dtype: {self.amp_dtype}, GradScaler: {self.use_scaler}")

    def _setup_data(self):
        """Setup datasets and dataloaders"""
        train_root = self.data_cfg["train_root"]
        val_root = self.data_cfg["val_root"]
        exclude_folder = self.data_cfg.get("exclude_folder")

        train_items = build_lane_items(train_root, exclude_folder)
        val_items = build_lane_items(val_root, None)

        max_train = self.data_cfg.get("max_train_samples")
        max_val = self.data_cfg.get("max_val_samples", 500)
        if max_train:
            train_items = train_items[:max_train]
        if max_val:
            val_items = val_items[:max_val]

        shortest = self.data_cfg.get("resize_shortest", 1024)
        longest = self.data_cfg.get("resize_longest", 1920)
        divisor = self.data_cfg.get("size_divisor", 32)
        line_width = self.data_cfg.get("line_width", 6)
        use_aa = self.data_cfg.get("use_aa", False)
        use_crop = self.data_cfg.get("use_train_crop", False)
        crop_size = self.data_cfg.get("crop_size", 1024)

        train_ds = SDLaneDataset(
            train_items, shortest, longest, divisor,
            line_width, use_aa, return_meta=False,
            use_crop=use_crop, crop_size=crop_size
        )
        val_ds = SDLaneDataset(
            val_items, shortest, longest, divisor,
            line_width, use_aa, return_meta=True,
            use_crop=False
        )

        batch_size = self.train_cfg.get("batch_size", 16)
        num_workers = self.train_cfg.get("num_workers", min(8, os.cpu_count() or 2))

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=train_collate_fn,
            drop_last=True, pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None
        )

        self.val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=max(1, num_workers // 2),
            collate_fn=val_collate_fn, pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2 if num_workers > 0 else None
        )

        self.logger.info(f"Train: {len(train_items)} samples, Val: {len(val_items)} samples")

    def _setup_model(self):
        """Setup segmentation model"""
        if smp is None:
            raise ImportError("segmentation_models_pytorch not installed")

        encoder = self.model_cfg.get("encoder", "resnet50")
        encoder_weights = self.model_cfg.get("encoder_weights", "imagenet")

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None
        ).to(self.device)

        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.use_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self.logger.info("torch.compile enabled")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}")

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        self.logger.info(f"Model: DeepLabV3+ ({encoder}), {total_params:.1f}M params")

    def _setup_loss(self):
        """Setup loss function"""
        pos_weight = self.loss_cfg.get("pos_weight", 8.0)
        dice_weight = self.loss_cfg.get("dice_weight", 0.5)
        self.criterion = LaneLoss(pos_weight, dice_weight)

    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        lr = self.train_cfg.get("lr", 6e-4)
        weight_decay = self.train_cfg.get("weight_decay", 1e-4)
        epochs = self.train_cfg.get("epochs", 13)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        self.epochs = epochs
        self.best_score = -1.0
        self.global_step = 0

    def _setup_wandb(self):
        """Initialize W&B if available"""
        self.use_wandb = self.train_cfg.get("use_wandb", False) and wandb is not None

        if self.use_wandb:
            project = self.train_cfg.get("wandb_project", "lane-segmentation")
            wandb.init(
                project=project,
                name=self.run_dir.name,
                config=self.config,
                dir=str(self.run_dir)
            )
            self.logger.info(f"W&B initialized: {project}/{self.run_dir.name}")

    def _append_history(self, row: Dict[str, Any]):
        """Append to training history"""
        self.history.append(row)
        with open(self.history_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _save_checkpoint(
        self,
        tag: str,
        epoch: int,
        metrics: Optional[Dict] = None,
        extra: Optional[Dict] = None
    ):
        """Save model checkpoint"""
        pt_path = self.ckpt_dir / f"{tag}.pt"
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "sched_state": self.scheduler.state_dict(),
            "metrics": metrics or {},
            "config": self.config,
            "best_score": float(self.best_score),
            "extra": extra or {},
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(payload, pt_path)

    def _train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_loss, epoch_bce, epoch_dice = 0.0, 0.0, 0.0
        n_steps = 0

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}", leave=True)
        self.optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(pbar):
            x = batch["pixel_values"].to(self.device, non_blocking=True)
            y = batch["labels"].to(self.device, non_blocking=True)

            if self.use_channels_last:
                x = x.to(memory_format=torch.channels_last)

            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                    logits = self.model(x)
                    loss, bce_v, dice_v = self.criterion(logits, y)
                    loss_scaled = loss / self.grad_accum_steps
            else:
                logits = self.model(x)
                loss, bce_v, dice_v = self.criterion(logits, y)
                loss_scaled = loss / self.grad_accum_steps

            if self.use_scaler:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (step_idx + 1) % self.grad_accum_steps == 0:
                if self.use_scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.use_scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.detach().cpu())
            epoch_bce += bce_v
            epoch_dice += dice_v
            n_steps += 1
            self.global_step += 1

            pbar.set_postfix({"loss": f"{epoch_loss/n_steps:.4f}"})

        self.scheduler.step()
        return epoch_loss / n_steps, epoch_bce / n_steps, epoch_dice / n_steps

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        tols = self.train_cfg.get("boundary_tols", (2, 4, 8))
        metrics_calc = LaneMetrics(tols)

        for batch in tqdm(self.val_loader, desc="Val", leave=False):
            x = batch["pixel_values"].to(self.device, non_blocking=True)
            if self.use_channels_last:
                x = x.to(memory_format=torch.channels_last)

            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                    logits = self.model(x)
            else:
                logits = self.model(x)

            newH, newW = batch["resized_hw"]
            H, W = batch["orig_hw"]
            gt_orig = batch["gt_orig"].numpy().astype(np.uint8)

            logits_rs = logits[:, :, :newH, :newW]
            pred_up = F.interpolate(logits_rs, size=(H, W), mode="bilinear", align_corners=False)
            pred_prob = torch.sigmoid(pred_up.float())[0, 0].detach().cpu().numpy()
            pred_bin = (pred_prob >= 0.5).astype(np.uint8)

            metrics_calc.update(pred_bin, gt_orig)

        return metrics_calc.compute()

    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(1, self.epochs + 1):
            try:
                avg_loss, avg_bce, avg_dice = self._train_epoch(epoch)

                self._save_checkpoint(
                    f"epoch_{epoch:03d}", epoch,
                    extra={"train_loss": avg_loss}
                )

                metrics = self.evaluate()
                sel_score = float(metrics.get("bf1@4px", metrics["dice"]))

                log_msg = (
                    f"[Epoch {epoch}] loss={avg_loss:.4f} (bce={avg_bce:.4f}, dice={avg_dice:.4f}) | "
                    f"val dice={metrics['dice']:.4f} iou={metrics['iou']:.4f} "
                    f"bf1@4={metrics.get('bf1@4px', 0):.4f}"
                )
                self.logger.info(log_msg)

                row = {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "train_loss": avg_loss,
                    "train_bce": avg_bce,
                    "train_dice": avg_dice,
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    **metrics,
                    "sel_score": sel_score,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self._append_history(row)

                if self.use_wandb:
                    wandb.log(row, step=self.global_step)

                if sel_score > self.best_score:
                    self.best_score = sel_score
                    self._save_checkpoint("best", epoch, metrics)
                    self.logger.info(f"  -> New best (bf1@4px={self.best_score:.4f})")

            except Exception as e:
                self.logger.error(f"Error at epoch {epoch}: {e}")
                self._save_checkpoint("crash_last", epoch, extra={"error": str(e)})
                raise

        self._save_checkpoint("final", self.epochs)

        if self.use_wandb:
            wandb.finish()

        self.logger.info(f"Training complete. Best score: {self.best_score:.4f}")
        return self.best_score
