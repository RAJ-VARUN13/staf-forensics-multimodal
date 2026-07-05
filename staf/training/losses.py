"""
STAF Training Module: Loss Functions.

Implements loss functions for multimodal deepfake detection, including
standard BCEWithLogitsLoss (with optional class weights) and Focal Loss.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from staf.configs.schema import TrainingConfig
from staf.utils.logging import get_logger

logger = get_logger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks.

    Focal Loss dynamically scales the loss based on prediction confidence,
    focusing the model on hard, misclassified samples and down-weighting easy ones.

    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Output logits of shape (B, 1) or (B,).
            targets: Binary targets of shape (B, 1) or (B,).

        Returns:
            Computed scalar focal loss.
        """
        # Ensure flat tensors
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Compute standard binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Compute probability of predicting the correct class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal modulation factor
        focal_weight = (1 - p_t) ** self.gamma

        # Apply loss scaling
        loss = focal_weight * bce_loss

        # Apply class weighting (alpha)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()


def get_loss_function(cfg: TrainingConfig, pos_weight: torch.Tensor | None = None) -> nn.Module:
    """
    Factory function returning the appropriate PyTorch loss module based on configuration.

    Args:
        cfg: Training configuration section.
        pos_weight: Optional tensor of shape (1,) weighting positive (fake) samples.

    Returns:
        nn.Module representing the loss function.
    """
    loss_name = cfg.loss_function.lower()
    logger.info(f"Setting up loss function: {loss_name}")

    if loss_name == "bce_with_logits":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    elif loss_name == "focal":
        gamma_val = cfg.focal_gamma if hasattr(cfg, "focal_gamma") else 2.0
        # If class weights are provided, use the pos_weight ratio to configure alpha
        alpha_val = 0.25
        if pos_weight is not None:
            # Approximate alpha based on class weight ratio
            alpha_val = float(1.0 / (1.0 + pos_weight.item()))
        return FocalLoss(alpha=alpha_val, gamma=gamma_val)
    
    else:
        logger.warning(f"Unknown loss function '{loss_name}'. Falling back to BCEWithLogitsLoss.")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
