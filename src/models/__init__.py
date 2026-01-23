# SMP-based models
from .segmentation import (
    SegmentationModel,
    get_model,
    get_model_from_config,
    MODELS,
    ENCODERS,
)
from .losses import SegmentationLoss, get_loss
from .metrics import SegmentationMetrics

# Backward compatibility alias
DeepLabV3Plus = SegmentationModel

__all__ = [
    # Models
    "SegmentationModel",
    "DeepLabV3Plus",
    # Functions
    "get_model",
    "get_model_from_config",
    # Loss
    "SegmentationLoss",
    "get_loss",
    # Metrics
    "SegmentationMetrics",
    # Registry
    "MODELS",
    "ENCODERS",
]
