"""
[DEPRECATED] Legacy DeepLabV3Plus implementation using torchvision

이 파일은 더 이상 사용되지 않습니다.
새로운 구현은 segmentation.py를 사용하세요.

Migration Guide:
    # Old (이 파일)
    from src.models.deeplabv3 import DeepLabV3Plus
    model = DeepLabV3Plus(num_classes=62, backbone="deeplabv3_resnet101")

    # New (segmentation.py)
    from src.models import get_model
    model = get_model(
        model_name="DeepLabV3Plus",
        encoder_name="resnet101",
        num_classes=62
    )
"""

# =============================================================================
# LEGACY CODE - COMMENTED OUT
# 아래 코드는 참고용으로 남겨둡니다.
# 새 프로젝트에서는 segmentation.py의 SegmentationModel을 사용하세요.
# =============================================================================

# import torch
# import torch.nn as nn
# from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
# from torchvision.models.segmentation.deeplabv3 import DeepLabHead
# from typing import Dict
#
#
# class DeepLabV3Plus(nn.Module):
#     """
#     DeepLabV3 wrapper using torchvision pretrained models
#
#     Args:
#         num_classes: Number of output classes
#         backbone: 'deeplabv3_resnet50' or 'deeplabv3_resnet101'
#         pretrained: Whether to use pretrained weights
#         dropout: Dropout rate (0.0 ~ 1.0)
#     """
#
#     def __init__(
#         self,
#         num_classes: int,
#         backbone: str = "deeplabv3_resnet50",
#         pretrained: bool = True,
#         dropout: float = 0.0
#     ):
#         super().__init__()
#
#         self.num_classes = num_classes
#         self.dropout_rate = dropout
#
#         # Load pretrained model from torchvision
#         if backbone == "deeplabv3_resnet50":
#             weights = "COCO_WITH_VOC_LABELS_V1" if pretrained else None
#             self.model = deeplabv3_resnet50(weights=weights)
#         elif backbone == "deeplabv3_resnet101":
#             weights = "COCO_WITH_VOC_LABELS_V1" if pretrained else None
#             self.model = deeplabv3_resnet101(weights=weights)
#         else:
#             raise ValueError(f"Unsupported backbone: {backbone}")
#
#         # Replace classifier head for custom num_classes
#         in_channels = self.model.classifier[0].convs[0][0].in_channels
#         self.model.classifier = DeepLabHead(in_channels, num_classes)
#
#         # Dropout layer
#         self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
#
#         # Remove auxiliary classifier
#         self.model.aux_classifier = None
#
#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         result = self.model(x)
#         if self.dropout is not None and self.training:
#             result["out"] = self.dropout(result["out"])
#         return result


# =============================================================================
# BACKWARD COMPATIBILITY - Import from new module
# =============================================================================
from .segmentation import SegmentationModel, get_model, get_model_from_config

# Alias for backward compatibility
DeepLabV3Plus = SegmentationModel

__all__ = ["DeepLabV3Plus", "SegmentationModel", "get_model", "get_model_from_config"]
