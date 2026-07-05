"""
STAF Baseline Model: Visual Encoder.

Extracts spatial features frame-by-frame from cropped face sequences using EfficientNet-B0.
Folds the batch and temporal dimensions during processing for high throughput.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from staf.configs.schema import VisualConfig
from staf.utils.logging import get_logger

logger = get_logger(__name__)


class VisualEncoder(nn.Module):
    """
    Extracts visual features from sequences of face crops.

    Input shape: (B, T, C, H, W)
        B: Batch size
        T: Number of frames
        C: Channels (3)
        H, W: Height, Width (e.g., 224, 224)

    Output shape: (B, T, D_vis)
        D_vis: Output feature dimension (e.g., 1280 for EfficientNet-B0)
    """

    def __init__(self, cfg: VisualConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone_name = self.cfg.backbone
        
        logger.info(f"Initializing VisualEncoder with backbone: {self.backbone_name}")
        
        # Initialize timm model. Setting num_classes=0 returns the pooled features
        # (Global Average Pooling output) instead of classification logits.
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=self.cfg.pretrained,
            num_classes=0
        )

        # Freeze backbone parameters if configured
        if self.cfg.frozen:
            logger.info(f"Freezing VisualEncoder backbone: {self.backbone_name}")
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Expose output feature dimension
        self.output_dim = self.cfg.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, C, H, W).

        Returns:
            Features tensor of shape (B, T, D_vis).
        """
        # x shape: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        # Fold batch and time dimensions: [B * T, C, H, W]
        x_folded = x.view(b * t, c, h, w)

        # Extract features
        # Output shape: [B * T, D_vis]
        features_folded = self.backbone(x_folded)

        # Unfold back to sequence: [B, T, D_vis]
        features = features_folded.view(b, t, -1)

        return features
