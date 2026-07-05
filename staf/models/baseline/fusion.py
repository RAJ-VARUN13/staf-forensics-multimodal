"""
STAF Baseline Model: Multimodal Fusion.

Combines visual and audio feature representations.
Implements late concatenation followed by LayerNorm stabilization.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import torch
import torch.nn as nn

from staf.configs.schema import FusionConfig
from staf.utils.logging import get_logger

logger = get_logger(__name__)


class ConcatFusion(nn.Module):
    """
    Fuses two modality representations via concatenation and LayerNorm.

    Inputs:
        x_vis: Visual tensor of shape (B, D_vis)
        x_aud: Audio tensor of shape (B, D_aud)

    Output shape: (B, D_vis + D_aud)
    """

    def __init__(self, cfg: FusionConfig, vis_dim: int, aud_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.vis_dim = vis_dim
        self.aud_dim = aud_dim
        
        self.output_dim = self.vis_dim + self.aud_dim
        logger.info(f"Initializing ConcatFusion. Combined dimension: {self.output_dim}")

        # LayerNorm stabilizes feature ranges before passing to classifier
        if self.cfg.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.output_dim)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, x_vis: torch.Tensor, x_aud: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_vis: Visual representation of shape (B, D_vis).
            x_aud: Audio representation of shape (B, D_aud).

        Returns:
            Fused multimodal tensor of shape (B, D_vis + D_aud).
        """
        # Ensure dimensions match
        assert x_vis.dim() == 2, f"Expected 2D visual tensor [B, D_vis], got {x_vis.shape}"
        assert x_aud.dim() == 2, f"Expected 2D audio tensor [B, D_aud], got {x_aud.shape}"
        
        # Concatenate along feature dimension
        # Output shape: [B, D_vis + D_aud]
        fused = torch.cat([x_vis, x_aud], dim=1)

        # Apply LayerNorm stabilization
        fused = self.layer_norm(fused)

        return fused
