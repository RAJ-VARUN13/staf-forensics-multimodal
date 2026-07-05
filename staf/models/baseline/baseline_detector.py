"""
STAF Baseline Model: Complete Multimodal Detector.

Assembles the complete visual and audio branches, temporal aggregation,
concatenation fusion, and MLP classifier into a single PyTorch Module.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import torch
import torch.nn as nn

from staf.configs.schema import ModelConfig
from staf.models.baseline.visual_encoder import VisualEncoder
from staf.models.baseline.audio_encoder import AudioEncoder
from staf.models.baseline.temporal import TemporalModel
from staf.models.baseline.fusion import ConcatFusion
from staf.models.baseline.classifier import MLPClassifier
from staf.utils.logging import get_logger

logger = get_logger(__name__)


class BaselineDetector(nn.Module):
    """
    Complete end-to-end baseline multimodal deepfake detector.

    Inputs:
        faces: Visual sequence tensor of shape (B, T, C, H, W)
        audio: Raw audio waveform tensor of shape (B, A_samples)

    Output shape: (B, output_dim)
        Returns classification logits.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        loss_function: str = "bce_with_logits"
    ) -> None:
        """
        Args:
            cfg: The ModelConfig section of the configuration tree.
            loss_function: Loss function identifier (determines classifier output dim).
        """
        super().__init__()
        self.cfg = cfg
        self.loss_function = loss_function

        logger.info("Assembling BaselineDetector end-to-end model...")

        # 1. Visual Branch
        self.visual_encoder = VisualEncoder(self.cfg.visual)
        self.temporal_model = TemporalModel(
            cfg=self.cfg.temporal,
            input_dim=self.visual_encoder.output_dim
        )

        # 2. Audio Branch
        self.audio_encoder = AudioEncoder(self.cfg.audio)

        # 3. Fusion Layer
        # Audio feature is mean-pooled over time steps before fusion: shape [B, D_aud]
        self.fusion = ConcatFusion(
            cfg=self.cfg.fusion,
            vis_dim=self.temporal_model.output_dim,
            aud_dim=self.audio_encoder.output_dim
        )

        # 4. Classification Head
        self.classifier = MLPClassifier(
            cfg=self.cfg.fusion,
            input_dim=self.fusion.output_dim,
            num_classes=self.cfg.num_classes,
            loss_function=self.loss_function
        )

        logger.info("BaselineDetector successfully assembled.")

    def forward(self, faces: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            faces: Facial sequences of shape (B, T, 3, image_size, image_size).
            audio: Raw waveforms of shape (B, A_samples).

        Returns:
            Logits tensor of shape (B, output_dim).
        """
        # --- 1. Visual Encoding & Temporal Pooling ---
        # Input faces: [B, T, 3, H, W]
        # v_feats: [B, T, D_vis] (e.g. [B, 16, 1280])
        v_feats = self.visual_encoder(faces)
        
        # v_pooled: [B, D_vis_temporal] (e.g. [B, 512] for BiLSTM)
        v_pooled = self.temporal_model(v_feats)

        # --- 2. Audio Encoding & Temporal Pooling ---
        # Input audio: [B, A_samples]
        # a_feats: [B, A_steps, D_aud] (e.g. [B, 499, 768])
        a_feats = self.audio_encoder(audio)
        
        # Mean pooling over speech sequence time dimension: [B, D_aud] (e.g. [B, 768])
        a_pooled = torch.mean(a_feats, dim=1)

        # --- 3. Multimodal Fusion ---
        # fused: [B, D_vis_temporal + D_aud] (e.g. [B, 512 + 768 = 1280] or [B, 1280 + 768 = 2048])
        fused = self.fusion(v_pooled, a_pooled)

        # --- 4. MLP Classification Head ---
        # logits: [B, output_dim] (e.g. [B, 1] for binary logits)
        logits = self.classifier(fused)

        return logits
