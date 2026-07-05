"""
STAF Baseline Model: Audio Encoder.

Extracts speech representations from raw audio waveforms using Wav2Vec 2.0.
Supports freezing backbone weights and maps custom schema identifiers to HuggingFace checkpoints.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from staf.configs.schema import AudioConfig
from staf.utils.logging import get_logger

logger = get_logger(__name__)


class AudioEncoder(nn.Module):
    """
    Extracts features from raw audio waveforms using Wav2Vec 2.0.

    Input shape: (B, A_samples)
        B: Batch size
        A_samples: Raw waveform samples (e.g. 160000 for 10s at 16kHz)

    Output shape: (B, A_steps, D_aud)
        A_steps: Temporal speech steps downsampled by the convolutional feature extractor
        D_aud: Feature dimension (e.g., 768 for Wav2Vec2 Base)
    """

    # Map schema enum/keys to actual HuggingFace Hub repository IDs
    BACKBONE_MAP = {
        "wav2vec2_base": "facebook/wav2vec2-base-960h",
        "wav2vec2_large": "facebook/wav2vec2-large-960h",
        "hubert_base": "facebook/hubert-base-ls960",
    }

    def __init__(self, cfg: AudioConfig) -> None:
        super().__init__()
        self.cfg = cfg
        
        # Resolve backbone name (allow direct HuggingFace repo names as well)
        self.repo_id = self.BACKBONE_MAP.get(self.cfg.backbone, self.cfg.backbone)
        
        logger.info(f"Initializing AudioEncoder with HF checkpoint: {self.repo_id}")
        
        # Load configuration or model
        if self.cfg.pretrained:
            self.backbone = Wav2Vec2Model.from_pretrained(self.repo_id)
        else:
            from transformers import Wav2Vec2Config
            # Download config only (fast/lightweight) and initialize randomly
            config = Wav2Vec2Config.from_pretrained(self.repo_id)
            self.backbone = Wav2Vec2Model(config)

        # Freeze backbone parameters if configured
        if self.cfg.frozen:
            logger.info(f"Freezing AudioEncoder backbone: {self.repo_id}")
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Expose output feature dimension
        self.output_dim = self.cfg.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Raw input audio waveform tensor of shape (B, A_samples).

        Returns:
            Speech hidden states tensor of shape (B, A_steps, D_aud).
        """
        # x shape: [B, A_samples]
        
        if self.cfg.frozen:
            # Under frozen setup, run in no_grad mode to save memory
            with torch.no_grad():
                outputs = self.backbone(x)
        else:
            outputs = self.backbone(x)

        # Retrieve the sequence of hidden states from the last layer
        # Shape: [B, A_steps, D_aud] (e.g. [B, 499, 768] for 10s audio)
        features = outputs.last_hidden_state

        return features
