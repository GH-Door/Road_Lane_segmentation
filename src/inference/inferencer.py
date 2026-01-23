import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Union

from src.models import get_model
from src.data import get_transforms
from src.utils import load_config, get_device


class Inferencer:
    """Semantic Segmentation Inferencer"""
    def __init__(self, checkpoint: str, config_path: str):
        # config load
        self.config = load_config(config_path)
        self.data_cfg = self.config.get("data", {})
        self.model_cfg = self.config.get("model", {}) 
        self.img_size = tuple(self.data_cfg.get("img_size", [1024, 768]))
        self.device = get_device()
        self.checkpoint_path = Path(checkpoint)

        # num_classes resolve
        group_config_path = self.data_cfg.get("group_config")
        num_classes_found = False

        if group_config_path:
            group_config = load_config(group_config_path)
            if group_config.get("enabled", False):
                self.num_classes = group_config.get("num_classes")
                if self.num_classes is not None:
                    num_classes_found = True

        if not num_classes_found:
            self.num_classes = self.data_cfg.get("num_classes", 20)

        # inject num_classes
        self.config["data"]["num_classes"] = self.num_classes

        # init
        self.transform = get_transforms(self.img_size)
        self.model = self.load_model()

    def load_model(self) -> torch.nn.Module:
        # model config
        model_name = self.model_cfg.get("name", "DeepLabV3Plus")
        encoder_name = self.model_cfg.get("encoder", "resnet50")

        # model build
        model = get_model(
            model_name=model_name,
            encoder_name=encoder_name,
            num_classes=self.num_classes,
            pretrained=False
        ).to(self.device)

        if self.checkpoint_path.exists():
            # checkpoint load
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)

            if 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
                print(f"Loaded model weights from checkpoint: {self.checkpoint_path}")
            else:
                model.load_state_dict(ckpt)
                print(f"Loaded state dict directly from checkpoint: {self.checkpoint_path}")
        else:
            print(
                f"Warning: Checkpoint not found at {self.checkpoint_path}. "
                "Using a model with random weights."
            )

        # eval mode
        model.eval()
        return model

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        # channel normalize
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # transform
        transformed = self.transform(image=image)
        img_tensor = transformed["image"].unsqueeze(0)
        return img_tensor.to(self.device)


    @torch.no_grad()
    def predict(self, image: Union[str, np.ndarray]) -> np.ndarray:
        # image load
        if isinstance(image, str):
            img_bytes = np.fromfile(image, dtype=np.uint8)
            image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot load image from path: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        # inference
        img_tensor = self.preprocess(image)
        output = self.model(img_tensor)["out"]

        # postprocess
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
        mask = cv2.resize(
            pred.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )
        return mask


    def predict_batch(self, images: list) -> list:
        # batch loop
        return [self.predict(img) for img in images]