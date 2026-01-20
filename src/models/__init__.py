from .deeplabv3 import DeepLabV3Plus
from .losses import SegmentationLoss

MODELS = {
    "deeplabv3_resnet50": DeepLabV3Plus,
    "deeplabv3_resnet101": DeepLabV3Plus,
    # 추후 모델 추가 시:
    # "segformer": SegFormer,
    # "unet": UNet,
}


def get_model(name: str, num_classes: int, **kwargs):
    """모델 이름으로 모델 인스턴스 반환"""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    return MODELS[name](num_classes=num_classes, backbone=name, **kwargs)
