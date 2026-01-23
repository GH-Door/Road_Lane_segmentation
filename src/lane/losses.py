"""Loss functions for lane segmentation"""
import torch
import torch.nn as nn
from typing import Tuple


class LaneLoss(nn.Module):
    """BCE + Dice loss for binary lane segmentation"""

    def __init__(self, pos_weight: float = 8.0, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        self.dice_weight = dice_weight
        self._pos_weight = pos_weight

    def soft_dice_loss(self, logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Soft dice loss for binary segmentation"""
        probs = torch.sigmoid(logits)
        tgt = targets.unsqueeze(1).float()

        num = 2 * (probs * tgt).sum(dim=(2, 3)) + eps
        den = (probs + tgt).sum(dim=(2, 3)) + eps

        return 1.0 - (num / den).mean()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Args:
            logits: (B, 1, H, W) raw predictions
            targets: (B, H, W) binary masks

        Returns:
            total_loss, bce_value, dice_value
        """
        self.bce.pos_weight = self.bce.pos_weight.to(logits.device)

        bce_loss = self.bce(logits, targets.unsqueeze(1).float())
        dice_loss = self.soft_dice_loss(logits, targets)

        total = bce_loss + self.dice_weight * dice_loss

        return total, float(bce_loss.detach().cpu()), float(dice_loss.detach().cpu())
