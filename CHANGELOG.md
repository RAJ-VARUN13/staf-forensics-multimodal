# Changelog

All notable changes to the Spatio-Temporal Attention Framework (STAF) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-05

### Added
- **Repository Scaffold**: Completed baseline package layout with proper submodules and configs.
- **Typed Configuration System**: Added `schema.py` and `baseline.yaml` using OmegaConf for validation and CLI overrides.
- **Logging & Reproducibility**: Added structured logger, W&B workspace integration, and deterministic seed locking.
- **Stage 1 (Frame Extraction)**: Multi-process video frame decoding with checkpoint resume.
- **Stage 2 (Face Detection)**: Pluggable face detection interface support (RetinaFace integration, SCRFD/YOLO stubs) generating structured detection manifests.
- **Stage 2a (Manifest Validation)**: 8 integrity checks for face coordinates, bounds, JSON formats, and timestamp monotonicity.
- **Stage 2b (Detection Visualization)**: Bounding box/landmark overlays and MP4 verification compiler.
- **Stage 3 (Face Crop & Align)**: Five-point eye alignment (tilt correction), configurable margin crop, and resize.
- **Stage 4 (Audio Extraction)**: 16kHz mono PCM WAV extraction using ffmpeg subprocess pool.
- **Stage 5 (Sync & Split Builder)**: Synchronizes face and audio modalities, assigns subject-independent train/val/test splits.
- **Stage 6 (Dataset Class)**: PyTorch `FakeAVCelebDataset` supporting frame sequence sampling, ImageNet standardization, and waveform padding.
- **Baseline Models**: Completed visual encoder (EfficientNet-B0), audio encoder (Wav2Vec2), temporal aggregator (BiLSTM/Attention/Mean), ConcatFusion, MLP classifier, and end-to-end `BaselineDetector`.
- **Test Suite**: 34 unit tests verifying the data pipeline and model architectures.
