# STAF (Spatio-Temporal Attention Framework)

An open-source PyTorch framework for multimodal deepfake detection using joint audio-visual reasoning.

## Current Status

* **✓ Modular preprocessing pipeline**: Multi-threaded frame extraction, audio extraction, face detection (RetinaFace), face alignment, and modality synchronization.
* **✓ FakeAVCeleb support**: Standard loaders, data generators, and subject-independent splits.
* **✓ Baseline multimodal detector**: Spatial-temporal visual and audio branches fused via late concatenation.
* **✓ Training and evaluation framework**: Fully verified training/validation loops with AMP, TensorBoard support, and evaluator generating ROC/PR curves and confusion matrices.

## Roadmap

* **GPU training**: Port verified pipeline to Google Colab / Kaggle for full baseline training.
* **Cross-dataset evaluation**: Testing baseline on out-of-domain benchmarks.
* **STAF architecture**: Design and implement the spatio-temporal attention modules.
* **Explainability**: Integrate attention rollout and activation map visualization.

## Project Structure

STAF is organized as a structured, importable Python package:

```
staf/
│
├── configs/          # YAML configuration files and schema validations
├── datasets/         # PyTorch Dataset classes (FakeAVCeleb, custom datasets)
├── preprocessing/    # Audio-visual pipelines (face detection, audio extraction)
├── models/           # Dual-stream backbones, temporal pooling, fusion encoders
│   ├── audio/        # Audio encoders (e.g. Wav2Vec 2.0)
│   ├── visual/       # Visual encoders (e.g. EfficientNet-B0)
│   ├── fusion/       # Fusion strategy layers (late fusion, cross-attention)
│   ├── temporal/     # Temporal pooling (attention pooling)
│   └── explainability/ # Attention rollout & visualization modules
├── training/         # Clean, explicit PyTorch training loops
├── evaluation/       # Performance evaluation (ROC, F1, confusion matrices)
├── inference/        # CLI-ready inference scripts
├── utils/            # Logging, reproducibility seeding, and system utilities
├── notebooks/        # Diagnostic/experimental Jupyter notebooks
├── docs/             # Technical documentation & architecture specs
└── tests/            # Code quality unit and integration tests
```

---

## Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Verify that GPU resources are available:

```python
import torch
print(torch.cuda.is_available())
```

### 2. Dataset Setup

Update the configuration in `staf/configs/baseline.yaml` or pass them via CLI overrides to point to your datasets:
- **FakeAVCeleb**: Set `data.paths.fakeavceleb_root`
- **Professor's Dataset**: Set `data.paths.professor_root`

### 3. Usage & Configurations

STAF uses OmegaConf structured configuration trees. You can modify parameters inside `staf/configs/baseline.yaml` or override them dynamically from the CLI:

```bash
python -m staf.training.train --config staf/configs/baseline.yaml \
    training.batch_size=32 \
    training.optimizer.learning_rate=5e-4
```

---

## Core Philosophy

- **No Placeholders**: Every file must contain fully realized, working code.
- **Data-Driven Decoupling**: Configuration is strictly separated from architecture and logic.
- **Reproducibility**: All random processes (PyTorch, NumPy, Python) are locked via a central seed module (`staf/utils/reproducibility.py`).
- **Standardized Logging**: Unified logs write to console, rotational files, and Weights & Biases (if enabled).

---

## License

This project is licensed under the [MIT License](LICENSE).
