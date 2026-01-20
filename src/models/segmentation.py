import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Optional, List


MODELS = {
    "DeepLabV3Plus": smp.DeepLabV3Plus,
    "UnetPlusPlus": smp.UnetPlusPlus,
    "MAnet": smp.MAnet,
    "PAN": smp.PAN,
    "PSPNet": smp.PSPNet,
}

MODEL_DESCRIPTIONS = {
    "DeepLabV3Plus": "ASPP + Decoder, 멀티스케일 처리, 정확한 경계 (추천)",
    "UnetPlusPlus": "Dense skip connections, 가는 객체(차선) 검출에 강함",
    "MAnet": "Multi-scale Attention, 다양한 크기의 객체 처리",
    "PAN": "Pyramid Attention Network, 효율적인 멀티스케일 융합",
    "PSPNet": "Pyramid Pooling Module, global context 파악",
}

ENCODERS = {
    # CNN 계열
    "resnet50": {"type": "CNN", "params": "25M"},
    "resnet101": {"type": "CNN", "params": "44M"},
    "efficientnet-b0": {"type": "CNN", "params": "5M"},
    "efficientnet-b4": {"type": "CNN", "params": "19M"},
    "efficientnet-b7": {"type": "CNN", "params": "66M"},
    "timm-regnetx_032": {"type": "CNN", "params": "15M"},
    "timm-regnety_032": {"type": "CNN", "params": "19M"},
    # Transformer 계열
    "mit_b0": {"type": "Transformer", "params": "4M"},
    "mit_b1": {"type": "Transformer", "params": "14M"},
    "mit_b2": {"type": "Transformer", "params": "25M"},
    "mit_b3": {"type": "Transformer", "params": "45M"},
    "mit_b4": {"type": "Transformer", "params": "62M"},
    "mit_b5": {"type": "Transformer", "params": "82M"},
    "tu-swin_tiny_patch4_window7_224": {"type": "Transformer", "params": "28M"},
    "tu-swin_small_patch4_window7_224": {"type": "Transformer", "params": "50M"},
    "tu-swin_base_patch4_window7_224": {"type": "Transformer", "params": "88M"},
}

RECOMMENDED_CONFIGS = {
    "fast": {"model": "PAN", "encoder": "efficientnet-b0"},
    "balanced": {"model": "DeepLabV3Plus", "encoder": "resnet50"},
    "transformer": {"model": "DeepLabV3Plus", "encoder": "mit_b2"},
    "transformer_high": {"model": "UnetPlusPlus", "encoder": "mit_b3"},
    "high_accuracy": {"model": "DeepLabV3Plus", "encoder": "resnet101"},
    "swin": {"model": "DeepLabV3Plus", "encoder": "tu-swin_small_patch4_window7_224"},
}


class SegmentationModel(nn.Module):
    """SMP 기반 Segmentation 모델 래퍼"""

    def __init__(
        self,
        model_name: str = "DeepLabV3Plus",
        encoder_name: str = "resnet101",
        num_classes: int = 62,
        encoder_weights: Optional[str] = "imagenet",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.model_name = model_name
        self.encoder_name = encoder_name
        self.num_classes = num_classes

        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

        # 모델 생성
        model_kwargs = {
            "encoder_name": encoder_name,
            "encoder_weights": encoder_weights,
            "in_channels": 3,
            "classes": num_classes,
        }

        if model_name in ["DeepLabV3Plus", "UnetPlusPlus", "MAnet"]:
            model_kwargs["decoder_use_batchnorm"] = True

        if model_name == "PSPNet":
            model_kwargs["psp_out_channels"] = 512

        self.model = MODELS[model_name](**model_kwargs)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.model(x)
        if self.dropout is not None and self.training:
            out = self.dropout(out)
        return {"out": out}

    def get_encoder_params(self):
        """Encoder 파라미터 반환 (fine-tuning용)"""
        return self.model.encoder.parameters()

    def get_decoder_params(self):
        """Decoder + Head 파라미터 반환"""
        params = list(self.model.decoder.parameters())
        params += list(self.model.segmentation_head.parameters())
        return params

    def freeze_encoder(self):
        """Encoder freeze"""
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Encoder unfreeze"""
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def get_info(self) -> Dict:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_type = ENCODERS.get(self.encoder_name, {}).get("type", "Unknown")

        return {
            "model_name": self.model_name,
            "encoder_name": self.encoder_name,
            "encoder_type": encoder_type,
            "num_classes": self.num_classes,
            "total_params": total_params,
            "total_params_M": round(total_params / 1e6, 1),
            "trainable_params": trainable_params,
        }


def get_model(
    model_name: str = "DeepLabV3Plus",
    encoder_name: str = "resnet101",
    num_classes: int = 62,
    pretrained: bool = True,
    dropout: float = 0.1,
) -> SegmentationModel:
    """모델 생성 헬퍼 함수"""
    encoder_weights = "imagenet" if pretrained else None

    return SegmentationModel(
        model_name=model_name,
        encoder_name=encoder_name,
        num_classes=num_classes,
        encoder_weights=encoder_weights,
        dropout=dropout,
    )


def get_model_from_config(config: dict, verbose: bool = False) -> SegmentationModel:
    """
    Config dict에서 모델 생성

    Backward compatibility: 기존 "deeplabv3_resnet101" 형식도 지원
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    model_name = model_cfg.get("name", "DeepLabV3Plus")
    encoder_name = model_cfg.get("encoder", None)

    # Backward compatibility: 기존 형식 지원
    if encoder_name is None:
        legacy_mappings = {
            "deeplabv3_resnet50": ("DeepLabV3Plus", "resnet50"),
            "deeplabv3_resnet101": ("DeepLabV3Plus", "resnet101"),
            "deeplabv3plus_resnet50": ("DeepLabV3Plus", "resnet50"),
            "deeplabv3plus_resnet101": ("DeepLabV3Plus", "resnet101"),
        }

        matched = False
        for key, (new_model, new_encoder) in legacy_mappings.items():
            if model_name.lower() == key.lower():
                if verbose:
                    print(f"[Backward Compatibility] '{model_name}' -> {new_model} + {new_encoder}")
                model_name = new_model
                encoder_name = new_encoder
                matched = True
                break

        if not matched:
            encoder_name = "resnet101"

    return get_model(
        model_name=model_name,
        encoder_name=encoder_name,
        num_classes=data_cfg.get("num_classes", 62),
        pretrained=model_cfg.get("pretrained", True),
        dropout=train_cfg.get("dropout", 0.1),
    )


def get_recommended_model(
    preset: str = "balanced",
    num_classes: int = 62,
    pretrained: bool = True,
    dropout: float = 0.1,
) -> SegmentationModel:
    """추천 프리셋으로 모델 생성"""
    if preset not in RECOMMENDED_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(RECOMMENDED_CONFIGS.keys())}")

    config = RECOMMENDED_CONFIGS[preset]
    return get_model(
        model_name=config["model"],
        encoder_name=config["encoder"],
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )


def list_encoders_by_type(encoder_type: str = "all") -> List[str]:
    """타입별 encoder 목록 반환"""
    if encoder_type == "all":
        return list(ENCODERS.keys())
    return [name for name, info in ENCODERS.items() if info["type"] == encoder_type]


# Backward compatibility alias
DeepLabV3Plus = SegmentationModel
