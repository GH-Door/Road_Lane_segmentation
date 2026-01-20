# Legacy import (backward compatibility)
from .deeplabv3 import DeepLabV3Plus as DeepLabV3PlusLegacy

# New SMP-based models
from .segmentation import (
    SegmentationModel,
    get_model,
    get_model_from_config,
    MODELS,
    ENCODERS,
)
from .losses import SegmentationLoss, get_loss

# Backward compatibility alias
DeepLabV3Plus = SegmentationModel

__all__ = [
    # Models
    "SegmentationModel",
    "DeepLabV3Plus",
    "DeepLabV3PlusLegacy",
    # Functions
    "get_model",
    "get_model_from_config",
    # Loss
    "SegmentationLoss",
    "get_loss",
    # Registry
    "MODELS",
    "ENCODERS",
]
