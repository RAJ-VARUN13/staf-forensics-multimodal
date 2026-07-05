"""
Unit tests for STAF Baseline Model Architecture.

Verifies correct output tensor shapes, configurations, parameter freezing,
and end-to-end forward passes for all submodules.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import pytest
import torch

from staf.configs.schema import (
    AudioConfig,
    FusionConfig,
    ModelConfig,
    TemporalConfig,
    VisualConfig,
)
from staf.models.baseline import (
    AudioEncoder,
    BaselineDetector,
    ConcatFusion,
    MLPClassifier,
    TemporalModel,
    VisualEncoder,
)


def test_visual_encoder_shapes_and_freezing() -> None:
    """Verifies that VisualEncoder processes 5D frame sequences and outputs correct dimensions."""
    cfg = VisualConfig(
        backbone="efficientnet_b0",
        pretrained=False,  # Skip downloading weights for fast unit testing
        frozen=True,
        output_dim=1280
    )
    
    encoder = VisualEncoder(cfg)
    
    # Input tensor shape: [B, T, C, H, W]
    dummy_input = torch.randn(2, 4, 3, 224, 224)
    
    output = encoder(dummy_input)
    
    # Output tensor shape: [B, T, D_vis]
    assert output.shape == (2, 4, 1280)
    assert output.dtype == torch.float32

    # Verify parameters are frozen (requires_grad is False)
    for param in encoder.parameters():
        assert not param.requires_grad


def test_audio_encoder_shapes_and_freezing() -> None:
    """Verifies that AudioEncoder extracts sequence features and respects freezing config."""
    cfg = AudioConfig(
        backbone="facebook/wav2vec2-base-960h",  # Use direct HF name for local tests
        pretrained=False,  # Transformer loads default skeleton structure
        frozen=True,
        output_dim=768
    )
    
    encoder = AudioEncoder(cfg)
    
    # Input waveform: [B, A_samples] (2 seconds of audio at 16kHz)
    dummy_input = torch.randn(2, 32000)
    
    output = encoder(dummy_input)
    
    # Output shape: [B, A_steps, D_aud]
    # Wav2Vec2 CNN downsamples 16000Hz by factor of 320, so 32000 samples -> ~99 steps
    assert output.dim() == 3
    assert output.shape[0] == 2
    assert output.shape[2] == 768

    # Verify parameters are frozen
    for param in encoder.parameters():
        assert not param.requires_grad


def test_temporal_model_poolings() -> None:
    """Verifies all temporal modeling pooling strategies and output dimensions."""
    dummy_sequence = torch.randn(2, 8, 1280)
    
    # 1. Mean Pooling
    cfg_mean = TemporalConfig(pooling="mean")
    model_mean = TemporalModel(cfg_mean, input_dim=1280)
    assert model_mean.output_dim == 1280
    out_mean = model_mean(dummy_sequence)
    assert out_mean.shape == (2, 1280)

    # 2. BiLSTM Temporal Pooling
    cfg_lstm = TemporalConfig(pooling="lstm")
    model_lstm = TemporalModel(cfg_lstm, input_dim=1280)
    # Output dim is 2 * hidden_dim = 2 * 256 = 512
    assert model_lstm.output_dim == 512
    out_lstm = model_lstm(dummy_sequence)
    assert out_lstm.shape == (2, 512)

    # 3. Attention Pooling
    cfg_attn = TemporalConfig(pooling="attention")
    model_attn = TemporalModel(cfg_attn, input_dim=1280)
    assert model_attn.output_dim == 1280
    out_attn = model_attn(dummy_sequence)
    assert out_attn.shape == (2, 1280)

    # 4. Mean-Attention Concatenation
    cfg_ma = TemporalConfig(pooling="mean_attention")
    model_ma = TemporalModel(cfg_ma, input_dim=1280)
    assert model_ma.output_dim == 2560
    out_ma = model_ma(dummy_sequence)
    assert out_ma.shape == (2, 2560)


def test_concat_fusion() -> None:
    """Verifies modal concatenation and LayerNorm logic."""
    cfg_ln = FusionConfig(use_layer_norm=True)
    fusion_ln = ConcatFusion(cfg_ln, vis_dim=512, aud_dim=768)
    
    assert fusion_ln.output_dim == 1280
    
    x_vis = torch.randn(2, 512)
    x_aud = torch.randn(2, 768)
    
    fused = fusion_ln(x_vis, x_aud)
    assert fused.shape == (2, 1280)
    assert fused.dtype == torch.float32

    # Test without LayerNorm
    cfg_no_ln = FusionConfig(use_layer_norm=False)
    fusion_no_ln = ConcatFusion(cfg_no_ln, vis_dim=512, aud_dim=768)
    fused_no_ln = fusion_no_ln(x_vis, x_aud)
    assert fused_no_ln.shape == (2, 1280)


def test_mlp_classifier() -> None:
    """Verifies output dimensions of MLPClassifier for both binary and multiclass setups."""
    cfg = FusionConfig(hidden_dims=[256, 128], dropout=0.1, use_layer_norm=True)
    
    # Binary setup (1 output neuron)
    clf_bin = MLPClassifier(cfg, input_dim=1024, num_classes=2, loss_function="bce_with_logits")
    x = torch.randn(2, 1024)
    out_bin = clf_bin(x)
    assert out_bin.shape == (2, 1)

    # Multiclass setup (4 output neurons)
    clf_multi = MLPClassifier(cfg, input_dim=1024, num_classes=4, loss_function="cross_entropy")
    out_multi = clf_multi(x)
    assert out_multi.shape == (2, 4)


def test_baseline_detector_end_to_end() -> None:
    """Verifies end-to-end forward pass on BaselineDetector with dummy visual and audio tensors."""
    model_cfg = ModelConfig(
        visual=VisualConfig(backbone="efficientnet_b0", pretrained=False, frozen=True, output_dim=1280),
        audio=AudioConfig(backbone="facebook/wav2vec2-base-960h", pretrained=False, frozen=True, output_dim=768),
        temporal=TemporalConfig(pooling="lstm"),
        fusion=FusionConfig(use_layer_norm=True, hidden_dims=[128])
    )
    
    detector = BaselineDetector(model_cfg, loss_function="bce_with_logits")
    
    # Mock inputs: Batch of 2, 4 frames, 2 seconds waveform
    dummy_faces = torch.randn(2, 4, 3, 224, 224)
    dummy_audio = torch.randn(2, 32000)
    
    logits = detector(dummy_faces, dummy_audio)
    
    # Binary outputs shape: [B, 1]
    assert logits.shape == (2, 1)
    assert logits.dtype == torch.float32
