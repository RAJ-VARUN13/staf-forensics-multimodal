# Contributing to STAF

Welcome to the Spatio-Temporal Attention Framework (STAF) codebase. Below are guidelines and standards to ensure high-quality, reproducible research and production-grade engineering.

## Core Philosophies

1. **Clean Architecture Over Speed**: We do not write quick, messy hacks. Everything must fit inside configured, modular schemas.
2. **Reproducibility**: Always enforce deterministic seeding via `staf.utils.reproducibility` and document pre-processing versions.
3. **Pluggable Backends**: Rely on interfaces/base classes (e.g. `BaseFaceDetector`) so we can easily swap detectors, feature extractors, and fusion layers.

## Development Workflow

1. **Virtual Environment**: Keep all dependencies isolated inside the `.venv`. Always use `.venv/bin/python` or `.venv/bin/pip` on Unix, or `.\.venv\Scripts\python.exe` on Windows.
2. **Adding Dependencies**: Update `requirements.txt` using `.venv/bin/pip freeze > requirements.txt` after installing new modules.
3. **Code Style**:
   - Use strict type annotations (`typing` and Python 3.9+ type hints).
   - Document tensor shapes in module forward passes.
   - Use double-quotes for docstrings.
4. **Testing**: Run pytest regularly to ensure no regressions:
   ```bash
   .\.venv\Scripts\python -m pytest staf/tests/ -v
   ```
