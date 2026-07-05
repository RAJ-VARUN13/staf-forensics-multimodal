"""
STAF Baseline Model: MLP Classifier.

Processes fused multimodal features through a Multi-Layer Perceptron (MLP)
to produce binary classification logits (Real vs. Fake).

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from staf.configs.schema import FusionConfig
from staf.utils.logging import get_logger

logger = get_logger(__name__)


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron classifier head.

    Input shape: (B, D_fused)
        D_fused: Combined visual and audio dimension

    Output shape: (B, num_outputs)
        num_outputs: 1 for binary classification (BCE logits), or num_classes
    """

    def __init__(
        self,
        cfg: FusionConfig,
        input_dim: int,
        num_classes: int = 2,
        loss_function: str = "bce_with_logits"
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.loss_function = loss_function

        # Determine output dimensions: binary logit vs multiclass logits
        if self.loss_function == "bce_with_logits" or self.num_classes == 2:
            self.output_dim = 1
        else:
            self.output_dim = self.num_classes

        logger.info(f"Initializing MLPClassifier: input_dim={self.input_dim}, output_dim={self.output_dim}")

        # Build hidden layers
        layers: List[nn.Module] = []
        curr_dim = self.input_dim

        # Map activation function string
        act_fn = nn.ReLU()
        if self.cfg.activation.lower() == "gelu":
            act_fn = nn.GELU()
        elif self.cfg.activation.lower() == "tanh":
            act_fn = nn.Tanh()

        # Construct hidden layers
        for h_dim in self.cfg.hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            # Optional layer normalization inside MLP
            if self.cfg.use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(p=self.cfg.dropout))
            curr_dim = h_dim

        # Final projection layer to logits
        layers.append(nn.Linear(curr_dim, self.output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Fused features tensor of shape (B, D_fused).

        Returns:
            Logits tensor of shape (B, output_dim).
        """
        # Output shape: [B, output_dim]
        return self.mlp(x)
