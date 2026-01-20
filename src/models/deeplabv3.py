import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from typing import Dict


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3 wrapper using torchvision pretrained models

    Args:
        num_classes: Number of output classes
        backbone: 'deeplabv3_resnet50' or 'deeplabv3_resnet101'
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate (0.0 ~ 1.0)
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "deeplabv3_resnet50",
        pretrained: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Load pretrained model from torchvision
        if backbone == "deeplabv3_resnet50":
            weights = "COCO_WITH_VOC_LABELS_V1" if pretrained else None
            self.model = deeplabv3_resnet50(weights=weights)
        elif backbone == "deeplabv3_resnet101":
            weights = "COCO_WITH_VOC_LABELS_V1" if pretrained else None
            self.model = deeplabv3_resnet101(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace classifier head for custom num_classes
        in_channels = self.model.classifier[0].convs[0][0].in_channels
        self.model.classifier = DeepLabHead(in_channels, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

        # Remove auxiliary classifier
        self.model.aux_classifier = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = self.model(x)
        if self.dropout is not None and self.training:
            result["out"] = self.dropout(result["out"])
        return result
