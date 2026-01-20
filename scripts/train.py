import os
import sys
import torch
import wandb
from pathlib import Path
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data import DatasetLoader, get_transforms
from src.models import get_model_from_config
from src.models.losses import get_loss
from src.models.class_weights import get_class_weights
from src.models.metrics import SegmentationMetrics
from src.utils import load_config, get_device, setup_logger
from src.utils.wandbs import login_wandb, log_validation_predictions


class Trainer:
    """Semantic Segmentation Trainer"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        # 로거 초기화 (프로젝트 루트/logs에 저장)
        self.logger = setup_logger(
            name="Trainer",
            log_dir=str(PROJECT_ROOT / "logs"),
            log_prefix="train"
        )

        self.config = load_config(config_path)
        self.device = get_device()

        # Config 파라미터
        self.data_cfg = self.config.get("data", {})
        self.model_cfg = self.config.get("model", {})
        self.train_cfg = self.config.get("training", {})
        self.wandb_cfg = self.config.get("wandb", {})
        self.wandb_num_samples = self.wandb_cfg.get("num_samples", 2)
        self.ckpt_cfg = self.config.get("checkpoint", {})

        # 체크포인트 경로 설정: {dir}/{model_name}_{encoder}_{exp_name}/
        model_name = self.model_cfg.get("name", "DeepLabV3Plus")
        encoder_name = self.model_cfg.get("encoder", "resnet101")
        exp_name = self.ckpt_cfg.get("exp_name", "")
        ckpt_dir = self.ckpt_cfg.get("dir", "checkpoints")
        folder_name = f"{model_name}_{encoder_name}_{exp_name}" if exp_name else f"{model_name}_{encoder_name}"
        self.checkpoint_dir = Path(ckpt_dir) / folder_name

        self.accumulation_steps = self.train_cfg.get("accumulation_steps", 1)
        if self.accumulation_steps > 1:
            self.logger.info(f"Gradient accumulation enabled with {self.accumulation_steps} steps.")

        # 초기화
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_wandb()

    def setup_data(self):
        """데이터로더 초기화"""
        data_root = self.data_cfg.get("root")
        img_size = tuple(self.data_cfg.get("img_size", [1024, 768]))
        batch_size = self.train_cfg.get("batch_size", 4)
        num_workers = self.train_cfg.get("num_workers", 4)

        transform = get_transforms(img_size)

        self.train_dataset = DatasetLoader(
            data_root=data_root,
            split="train",
            transform=transform
        )
        self.val_dataset = DatasetLoader(
            data_root=data_root,
            split="val",
            transform=transform
        )

        self.num_classes = self.train_dataset.num_classes

        # MPS에서는 pin_memory 지원 안 됨
        use_pin_memory = self.device.type == "cuda"

        self.train_loader = TorchDataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory
        )
        self.val_loader = TorchDataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory
        )

    def setup_model(self):
        """모델 초기화"""
        # Config에 num_classes 추가 (get_model_from_config에서 사용)
        self.config["data"]["num_classes"] = self.num_classes

        self.model = get_model_from_config(self.config).to(self.device)

        # 모델 정보 로깅
        model_info = self.model.get_info()
        self.logger.info(
            f"Model: {model_info['model_name']} + {model_info['encoder_name']} "
            f"({model_info['encoder_type']}, {model_info['total_params_M']}M params)"
        )

        # Class weights 로드 (config에서 enabled=true인 경우)
        class_weight = get_class_weights(
            config=self.config,
            num_classes=self.num_classes,
            data_root=self.data_cfg.get("root")
        )
        if class_weight is not None:
            class_weight = class_weight.to(self.device)
            self.logger.info(f"Class weights enabled (method: {self.config['loss']['class_weights']['method']})")

        # config에서 loss 설정 로드
        self.criterion = get_loss(self.config, self.num_classes, class_weight)
        self.logger.info(f"Loss: {self.criterion.loss_type}")

    def setup_optimizer(self):
        """옵티마이저 및 스케줄러 초기화"""
        lr = self.train_cfg.get("lr", 0.001)
        weight_decay = self.train_cfg.get("weight_decay", 0.0001)
        self.epochs = self.train_cfg.get("epochs", 100)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # LR Scheduler 설정
        scheduler_cfg = self.train_cfg.get("scheduler", {})
        scheduler_enabled = scheduler_cfg.get("enabled", False)
        scheduler_type = scheduler_cfg.get("type", "cosine")

        self.scheduler = None
        self.scheduler_name = None

        if scheduler_enabled:
            if scheduler_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_cfg.get("T_max", self.epochs),
                    eta_min=scheduler_cfg.get("min_lr", 1e-6)
                )
                self.scheduler_name = f"CosineAnnealingLR(min_lr={scheduler_cfg.get('min_lr', 1e-6)})"
            elif scheduler_type == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_cfg.get("step_size", 30),
                    gamma=scheduler_cfg.get("gamma", 0.1)
                )
                self.scheduler_name = f"StepLR(step={scheduler_cfg.get('step_size', 30)}, gamma={scheduler_cfg.get('gamma', 0.1)})"

            if self.scheduler:
                self.logger.info(f"Scheduler: {self.scheduler_name}")

    def setup_wandb(self):
        """W&B 초기화"""
        self.use_wandb = self.wandb_cfg.get("enabled", False)

        if not self.use_wandb:
            return

        login_wandb()

        # run_name: {model_name}_{encoder}-{run_name}
        model_name = self.model_cfg.get("name", "DeepLabV3Plus")
        encoder_name = self.model_cfg.get("encoder", "resnet101")
        suffix = self.wandb_cfg.get("run_name", "exp")
        run_name = f"{model_name}_{encoder_name}-{suffix}"

        # 모델 파라미터 수 계산 (백만 단위)
        total_params = round(sum(p.numel() for p in self.model.parameters()) / 1e6, 2)

        # Class weights 설정 정보
        loss_cfg = self.config.get("loss", {})
        cw_cfg = loss_cfg.get("class_weights", {})
        class_weights_info = cw_cfg.get("method", "none") if cw_cfg.get("enabled", False) else "none"

        wandb.init(
            project=self.wandb_cfg.get("project", "Road_Lane_Segmentation"),
            name=run_name,
            tags=self.wandb_cfg.get("tags", []),
            config={
                # 모델
                "model_name": model_name,
                "encoder_name": encoder_name,
                "pretrained": self.model_cfg.get("pretrained", True),
                "num_classes": self.num_classes,
                "total_params_M": total_params,
                # 학습 하이퍼파라미터
                "epochs": self.train_cfg.get("epochs"),
                "batch_size": self.train_cfg.get("batch_size"),
                "accumulation_steps": self.accumulation_steps,
                "effective_batch_size": self.train_cfg.get("batch_size") * self.accumulation_steps,
                "lr": self.train_cfg.get("lr"),
                "weight_decay": self.train_cfg.get("weight_decay"),
                "dropout": self.train_cfg.get("dropout", 0.0),
                "patience": self.train_cfg.get("early_stop", 0),
                # Scheduler
                "scheduler": self.scheduler_name if self.scheduler_name else "none",
                # Loss
                "loss_type": self.criterion.loss_type,
                "class_weights": class_weights_info,
                # 데이터
                "img_size": self.data_cfg.get("img_size"),
                # 환경
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            }
        )

    def log_wandb(self, metrics: dict, step: int):
        """W&B에 메트릭 기록"""
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def train_epoch(self, epoch: int) -> float:
        """1 epoch 학습"""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad() # 에포크 시작 시 그래디언트 초기화

        pbar = tqdm(enumerate(self.train_loader), desc=f"Epoch {epoch}", total=len(self.train_loader))
        for i, (images, masks) in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)["out"]
            loss = self.criterion(outputs, masks)

            # Accumulation을 위해 loss 정규화
            if self.accumulation_steps > 1:
                loss = loss / self.accumulation_steps

            loss.backward()

            # accumulation_steps 마다 optimizer step 및 그래디언트 초기화
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            # 로깅을 위해 원래 loss 스케일로 복원
            batch_loss = loss.item()
            if self.accumulation_steps > 1:
                batch_loss *= self.accumulation_steps
            
            total_loss += batch_loss
            pbar.set_postfix(loss=batch_loss)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> dict:
        """검증 (Memory-Efficient SegmentationMetrics 사용)"""
        self.model.eval()
        total_loss = 0.0

        # Memory-Efficient Metrics (for 루프로 클래스별 계산)
        ignore_index = self.config.get("loss", {}).get("ignore_index", 255)
        metrics_calculator = SegmentationMetrics(
            num_classes=self.num_classes,
            ignore_index=ignore_index,
            device=self.device
        )

        # 시각화를 위한 첫 배치 저장
        first_batch = None

        for i, (images, masks) in enumerate(tqdm(self.val_loader, desc="Validating")):
            if i == 0:
                first_batch = (images, masks)

            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)["out"]
            loss = self.criterion(outputs, masks)
            total_loss += loss.item()

            # Memory-efficient 메트릭 업데이트
            metrics_calculator.update(outputs, masks)

        # 최종 메트릭 계산
        metrics = metrics_calculator.compute()

        return {
            "val_loss": total_loss / len(self.val_loader),
            "first_batch": first_batch,
            **metrics
        }

    def train(self):
        """전체 학습 루프"""
        best_miou = 0.0
        patience = self.train_cfg.get("early_stop", 0)
        no_improve = 0
        stopped_epoch = self.epochs
        early_stopped = False

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            val_loss = val_metrics.pop("val_loss")
            val_images, val_masks = val_metrics.pop("first_batch")

            # 현재 learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            self.logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"mIoU: {val_metrics['miou']:.4f} | "
                f"Accuracy: {val_metrics['accuracy']:.4f} | "
                f"Dice: {val_metrics['dice']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Scheduler step (epoch 단위)
            if self.scheduler is not None:
                self.scheduler.step()

            # W&B 로깅
            log_data = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                **val_metrics
            }
            
            # W&B에 검증 결과 시각화 (매 10 에포크마다 + 첫 에포크)
            if self.use_wandb and (epoch == 1 or epoch % 10 == 0):
                self.logger.info(f"Logging validation images to W&B for epoch {epoch}...")
                with torch.no_grad():
                    self.model.eval()
                    val_preds = self.model(val_images.to(self.device))["out"]
                
                log_data.update(log_validation_predictions(
                    val_images, val_preds.cpu(), val_masks, self.num_classes, num_samples=self.wandb_num_samples
                ))

            self.log_wandb(log_data, step=epoch)

            # Best 모델 저장
            if val_metrics["miou"] > best_miou:
                best_miou = val_metrics["miou"]
                no_improve = 0
                self.save("best.pt")
                self.logger.info("mIoU improved. Early stopping counter reset.")
            else:
                no_improve += 1
                if patience > 0:
                    self.logger.info(f"EarlyStopping counter: {no_improve}/{patience}")

            # Early stopping
            if patience > 0 and no_improve >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                stopped_epoch = epoch
                early_stopped = True
                break

        # W&B 종료
        if self.use_wandb:
            wandb.run.summary["best_miou"] = best_miou
            wandb.run.summary["stopped_epoch"] = stopped_epoch
            if early_stopped:
                wandb.run.summary["early_stopped"] = True
            wandb.finish()

        self.logger.info(f"Training complete. Best mIoU: {best_miou:.4f}")

    def save(self, filename: str = "model.pt"):
        """모델 저장"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.checkpoint_dir / filename
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config
        }, save_path)
        self.logger.info(f"Saved: {save_path}")


if __name__ == "__main__":
    trainer = Trainer("configs/config.yaml")
    trainer.train()