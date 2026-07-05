"""Baseline multimodal deepfake detector — EfficientNet-B0 + Wav2Vec2 + late fusion MLP."""

from staf.models.baseline.visual_encoder import VisualEncoder
from staf.models.baseline.audio_encoder import AudioEncoder
from staf.models.baseline.temporal import TemporalModel
from staf.models.baseline.fusion import ConcatFusion
from staf.models.baseline.classifier import MLPClassifier
from staf.models.baseline.baseline_detector import BaselineDetector

__all__ = [
    "VisualEncoder",
    "AudioEncoder",
    "TemporalModel",
    "ConcatFusion",
    "MLPClassifier",
    "BaselineDetector",
]
