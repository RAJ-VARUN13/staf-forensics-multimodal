"""
STAF Configuration Schema — Typed dataclass definitions for all framework configs.

Uses OmegaConf structured configs for type-safe YAML loading with validation.
Every configurable parameter in the framework is defined here, with defaults,
documentation, and type constraints.

Design Decision:
    We use Python dataclasses + OmegaConf instead of raw dicts or argparse because:
    1. Type safety catches config errors before training starts
    2. YAML serialization gives human-readable, diffable experiment configs
    3. Hierarchical composition allows overriding subsets without rewriting
    4. Structured configs document every parameter in one place

Usage:
    # Load from YAML file:
    cfg = load_config("configs/baseline.yaml")

    # Load with overrides:
    cfg = load_config("configs/baseline.yaml", overrides=["training.batch_size=32"])

    # Programmatic construction:
    cfg = STAFConfig(
        model=ModelConfig(visual=VisualConfig(backbone="efficientnet_b0")),
        training=TrainingConfig(max_epochs=30),
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import DictConfig, OmegaConf, MISSING


# =============================================================================
# Enums for constrained choices
# =============================================================================

class VisualBackbone(str, Enum):
    """Supported visual feature extraction backbones."""
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B4 = "efficientnet_b4"
    RESNET50 = "resnet50"
    XCEPTION = "xception"


class AudioBackbone(str, Enum):
    """Supported audio feature extraction backbones."""
    WAV2VEC2_BASE = "wav2vec2_base"
    WAV2VEC2_LARGE = "wav2vec2_large"
    HUBERT_BASE = "hubert_base"


class FusionStrategy(str, Enum):
    """Supported multimodal fusion strategies."""
    LATE_CONCAT = "late_concat"
    CROSS_ATTENTION = "cross_attention"
    TENSOR_FUSION = "tensor_fusion"


class TemporalPooling(str, Enum):
    """Supported temporal aggregation strategies."""
    MEAN = "mean"
    ATTENTION = "attention"
    MEAN_ATTENTION = "mean_attention"
    LSTM = "lstm"


class FaceDetector(str, Enum):
    """Supported face detection backends."""
    MTCNN = "mtcnn"
    RETINAFACE = "retinaface"


# =============================================================================
# Data Configuration
# =============================================================================

@dataclass
class DataPathConfig:
    """Paths to raw datasets. Resolved at runtime based on environment."""

    # FakeAVCeleb dataset
    fakeavceleb_root: str = MISSING
    fakeavceleb_metadata_csv: str = ""

    # Professor's dataset
    professor_root: str = ""
    professor_metadata_csv: str = ""

    # Processed / cached data directory
    processed_dir: str = ""

    # Train/val/test split file paths (generated during preprocessing)
    splits_dir: str = ""


@dataclass
class AugmentationConfig:
    """
    Data augmentation configuration (reserved for future implementation).

    These parameters define the augmentation strategy applied during training.
    Currently unused — set ``enabled: false`` in YAML. Implementing in Phase 3+.
    """

    enabled: bool = False

    # Geometric
    rotation_degrees: float = 10.0
    horizontal_flip_prob: float = 0.5

    # Photometric
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    saturation_range: float = 0.1

    # Noise & compression
    gaussian_noise_std: float = 0.01
    jpeg_quality_range_low: int = 70
    jpeg_quality_range_high: int = 100
    gaussian_blur_kernel: int = 3
    gaussian_blur_prob: float = 0.1

    # Video-specific
    compression_level: int = 23  # CRF for re-encoding attacks


@dataclass
class DataConfig:
    """Complete data pipeline configuration."""

    paths: DataPathConfig = field(default_factory=DataPathConfig)

    # Frame sampling
    num_frames: int = 16
    frame_sampling_strategy: str = "uniform"  # "uniform", "random", "center"

    # Visual preprocessing
    face_detector: str = FaceDetector.RETINAFACE.value
    image_size: int = 224  # Height and width of cropped face
    face_margin: float = 0.3  # Margin around detected face (fraction)
    face_alignment_enabled: bool = True  # Five-point eye alignment before crop
    preprocessing_version: str = "v1.0"

    # Audio preprocessing
    audio_sample_rate: int = 16000
    audio_max_duration_sec: float = 10.0  # Max audio clip length
    audio_normalize: bool = True

    # Data augmentation (reserved — not implemented yet)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    subject_independent: bool = True  # Split by subject, not by video

    # DataLoader
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class VisualConfig:
    """Configuration for the visual feature extraction stream."""

    backbone: str = VisualBackbone.EFFICIENTNET_B0.value
    pretrained: bool = True
    frozen: bool = True
    freeze_until_layer: str = ""  # If non-empty, freeze up to this layer name
    output_dim: int = 1280  # EfficientNet-B0 feature dim


@dataclass
class AudioConfig:
    """Configuration for the audio feature extraction stream."""

    backbone: str = AudioBackbone.WAV2VEC2_BASE.value
    pretrained: bool = True
    frozen: bool = True
    freeze_until_layer: str = ""
    output_dim: int = 768  # Wav2Vec2 Base hidden dim


@dataclass
class TemporalConfig:
    """Configuration for temporal aggregation module."""

    pooling: str = TemporalPooling.ATTENTION.value
    attention_heads: int = 4
    attention_dropout: float = 0.1


@dataclass
class FusionConfig:
    """Configuration for multimodal fusion module."""

    strategy: str = FusionStrategy.LATE_CONCAT.value
    hidden_dims: List[int] = field(default_factory=lambda: [512])
    dropout: float = 0.3
    activation: str = "relu"
    use_layer_norm: bool = True

    # Cross-attention specific (used when strategy == "cross_attention")
    cross_attention_heads: int = 8
    cross_attention_layers: int = 2


class ModelType(str, Enum):
    """Supported model type configurations."""
    BASELINE = "baseline"
    STAF = "staf"


@dataclass
class ModelConfig:
    """Complete model architecture configuration."""

    type: str = ModelType.BASELINE.value
    visual: VisualConfig = field(default_factory=VisualConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Classification head
    num_classes: int = 2  # Binary: Real vs Fake
    classifier_dropout: float = 0.2


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: str = "adamw"  # "adam", "adamw", "sgd"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    momentum: float = 0.9  # For SGD only


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    name: str = "cosine"  # "cosine", "step", "plateau", "one_cycle"
    warmup_epochs: int = 2
    min_lr: float = 1e-7

    # StepLR specific
    step_size: int = 10
    gamma: float = 0.1

    # ReduceOnPlateau specific
    patience: int = 3
    factor: float = 0.5


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    max_epochs: int = 30
    min_epochs: int = 5
    early_stopping_patience: int = 7
    early_stopping_metric: str = "val_auc"
    early_stopping_mode: str = "max"

    # Mixed precision
    use_amp: bool = True

    # Gradient management
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1

    # Checkpointing
    save_top_k: int = 3
    checkpoint_metric: str = "val_auc"
    checkpoint_mode: str = "max"

    # Loss
    loss_function: str = "bce_with_logits"  # "bce_with_logits", "focal", "cross_entropy"
    class_weights: Optional[List[float]] = None  # Auto-computed if None
    focal_gamma: float = 2.0  # For focal loss

    # Reproducibility
    seed: int = 42
    deterministic: bool = True


# =============================================================================
# Evaluation Configuration
# =============================================================================

@dataclass
class EvaluationConfig:
    """Evaluation and metrics configuration."""

    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "confusion_matrix", "roc_curve", "pr_curve"
    ])
    binary_threshold: float = 0.5
    save_predictions: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 150


# =============================================================================
# Logging & Experiment Tracking Configuration
# =============================================================================

@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = True
    project: str = "staf-deepfake-detection"
    entity: str = ""  # W&B username or team
    run_name: str = ""  # Auto-generated if empty
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    log_model: bool = True  # Upload checkpoints to W&B


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_dir: str = "logs"
    log_to_file: bool = True
    log_to_console: bool = True
    wandb: WandbConfig = field(default_factory=WandbConfig)


# =============================================================================
# Top-Level Configuration
# =============================================================================

@dataclass
class STAFConfig:
    """
    Root configuration for the entire STAF framework.

    All framework behavior is controlled through this single config tree.
    Load from YAML, override via CLI, and snapshot for reproducibility.
    """

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Experiment metadata
    experiment_name: str = "staf_baseline_v0.1"
    description: str = ""
    output_dir: str = "experiments"

    # Hardware
    device: str = "auto"  # "auto", "cuda", "cpu", "cuda:0"
    num_gpus: int = 1


# =============================================================================
# Configuration I/O
# =============================================================================

def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> STAFConfig:
    """
    Load a STAF configuration from a YAML file with optional CLI overrides.

    Args:
        config_path: Path to a YAML configuration file. If None, returns defaults.
        overrides: List of dotted key=value override strings, e.g.,
                   ["training.batch_size=32", "model.visual.frozen=false"]

    Returns:
        A fully resolved STAFConfig instance.

    Examples:
        >>> cfg = load_config("configs/baseline.yaml")
        >>> cfg = load_config(overrides=["training.max_epochs=50"])
    """
    # Start with structured defaults
    schema = OmegaConf.structured(STAFConfig)

    if config_path is not None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        file_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(schema, file_cfg)
    else:
        cfg = schema

    # Apply CLI overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Resolve interpolations
    OmegaConf.resolve(cfg)

    # Convert to typed dataclass
    return OmegaConf.to_object(cfg)


def save_config(cfg: STAFConfig, save_path: str) -> None:
    """
    Save a STAF configuration to a YAML file.

    Args:
        cfg: The configuration to save.
        save_path: Destination file path (will create parent directories).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    omega_cfg = OmegaConf.structured(cfg)
    with open(save_path, "w") as f:
        OmegaConf.save(omega_cfg, f)


def config_to_dict(cfg: STAFConfig) -> dict:
    """Convert a STAFConfig to a plain dictionary (for W&B logging etc.)."""
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
