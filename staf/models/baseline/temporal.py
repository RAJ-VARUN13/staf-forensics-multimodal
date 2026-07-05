"""
STAF Baseline Model: Temporal Modeling.

Aggregates frame-level feature sequences into a single clip-level representation.
Supports Bidirectional LSTM (BiLSTM), mean pooling, and self-attention pooling.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import torch
import torch.nn as nn

from staf.configs.schema import TemporalConfig
from staf.utils.logging import get_logger

logger = get_logger(__name__)


class AttentionPooling(nn.Module):
    """
    Learns attention weights to pool a sequence of vectors.

    Input shape: (B, T, D_in)
    Output shape: (B, D_in)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sequence tensor of shape (B, T, D_in).

        Returns:
            Weighted sequence representation of shape (B, D_in).
        """
        # x shape: [B, T, D_in]
        
        # Compute raw attention scores: [B, T, 1]
        attn_logits = self.query(x)
        
        # Softmax over time dimension: [B, T, 1]
        attn_weights = torch.softmax(attn_logits, dim=1)
        
        # Compute weighted sum: [B, D_in]
        pooled = torch.sum(x * attn_weights, dim=1)
        
        return pooled


class TemporalModel(nn.Module):
    """
    Applies temporal sequence modeling and pooling to extract visual sequence features.

    Input shape: (B, T, D_in)
        D_in: Input feature dimension (e.g. 1280 for EfficientNet-B0)

    Output shape: (B, D_out)
        D_out: Output feature dimension.
               - For BiLSTM: 2 * hidden_dim (e.g., 512)
               - For Mean/Attention: D_in (e.g., 1280)
    """

    def __init__(self, cfg: TemporalConfig, input_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.pooling_strategy = self.cfg.pooling.lower()
        
        logger.info(f"Initializing TemporalModel with pooling: {self.pooling_strategy}")

        if self.pooling_strategy == "lstm":
            # Bidirectional LSTM configuration
            # Default hidden size is 256, giving a combined output dimension of 512
            self.hidden_dim = 256
            self.lstm = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.output_dim = self.hidden_dim * 2
            
        elif self.pooling_strategy == "attention":
            self.attention_pool = AttentionPooling(self.input_dim)
            self.output_dim = self.input_dim
            
        elif self.pooling_strategy == "mean_attention":
            self.attention_pool = AttentionPooling(self.input_dim)
            self.output_dim = self.input_dim * 2
            
        else:  # "mean" / fallback
            self.pooling_strategy = "mean"
            self.output_dim = self.input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Visual frame features tensor of shape (B, T, D_in).

        Returns:
            Pooled clip-level representation of shape (B, D_out).
        """
        # x shape: [B, T, D_in]

        if self.pooling_strategy == "lstm":
            # Pass sequence through BiLSTM
            # lstm_out shape: [B, T, 2 * hidden_dim]
            lstm_out, _ = self.lstm(x)
            
            # Mean pool over the BiLSTM outputs to get a robust clip summary
            # Output shape: [B, D_out]
            pooled = torch.mean(lstm_out, dim=1)
            return pooled

        elif self.pooling_strategy == "attention":
            # Learnable attention weighted sum
            # Output shape: [B, D_out]
            return self.attention_pool(x)

        elif self.pooling_strategy == "mean_attention":
            # Concatenate mean-pooled and attention-pooled features
            mean_pooled = torch.mean(x, dim=1)
            attn_pooled = self.attention_pool(x)
            pooled = torch.cat([mean_pooled, attn_pooled], dim=-1)
            return pooled

        else:  # "mean"
            # Simple average pooling over frames
            # Output shape: [B, D_out] (D_out == D_in)
            pooled = torch.mean(x, dim=1)
            return pooled
