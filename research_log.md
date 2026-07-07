# STAF Research Log

This log documents daily milestones, observations, and experimental results in the development of the Spatio-Temporal Attention-Based Framework (STAF) for multimodal deepfake detection.

---

## 2026-07-05
* **Milestone**: Achieved first successful end-to-end training and evaluation run verification on a CPU subset.
* **Engineering Fixes**:
  * Fixed imports and parameters inside `train.py` and `evaluate.py` (incorrect `get_device`, `config_to_dict` module, and `log_file` parameters).
  * Solved Windows checkpoint deletion crash (`WinError 32 PermissionError`) inside `trainer.py` by wrapping standard checkpoint `unlink()` calls in a `try-except` guard.
  * Corrected config loading in `evaluate.py` by merging with the structured `STAFConfig` schema.
  * Enabled sequential numbered experiment logging (`results/001_...`, `results/002_...`) to prevent overwriting results.
  * Integrated local TensorBoard metrics logging into the `Trainer` module.
* **Observations**:
  * Measured RetinaFace single-frame inference time on CPU: **~17.4 seconds per frame**.
  * Total FakeAVCeleb dataset contains **21,544 videos**.
  * **Conclusion**: CPU preprocessing the full dataset is mathematically prohibitive (~1.2 years). Bypassed CPU bottleneck on verification subset by linking frame extraction folders directly to dataset crop paths.
  * Verified forward pass, backward pass, BCE loss calculation, validation metrics computation, checkpointing, and evaluation pipeline on CPU.
  * Verified loss decreases over training epochs (**0.5484 → 0.1377**, a **74.9%** decrease).
* **Next Step**: Push verified framework to GitHub, mount dataset in Google Colab / Kaggle GPU environment, run GPU-accelerated preprocessing, train baseline on full dataset, and conduct error analysis of baseline predictions before proposing STAF modifications.

---

## 2026-07-07
* **Milestone**: Resolved repository import inconsistency by restoring the `staf/datasets/fakeavceleb.py` module.
* **Root Cause & Fix**:
  * Found that case-insensitive `.gitignore` matching on Windows (`staf/datasets/FakeAVCeleb*`) was silently ignoring the `fakeavceleb.py` module.
  * Corrected `.gitignore` patterns and added `!staf/datasets/*.py` to prevent code source ignoring.
  * Staged, committed, and pushed the restored dataset loader to the remote repository.
* **Verification**:
  * Executed the complete test suite (`pytest staf/tests/`) in the virtual environment. **All 38 tests passed successfully**.
  * Verified that running `python train.py --help` starts correctly and prints options without any import issues.

