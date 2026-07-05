# STAF Engineering Decisions

This document details the architectural and engineering decisions made during the design and verification phases of the Spatio-Temporal Attention-Based Framework (STAF) for multimodal deepfake detection.

---

## 1. Modality-Specific Feature Extractors

### Visual Stream: EfficientNet-B0
* **Decision**: Use `efficientnet_b0` (via `timm`) as the spatial feature extractor.
* **Rationale**: 
  * **Efficiency & Parameter Count**: EfficientNet-B0 provides an optimal trade-off between feature representation quality and computational footprint (5.3M parameters).
  * **Frozen Backbone**: The visual backbone is frozen to prevent overfitting on the FakeAVCeleb subject dataset and to minimize memory footprint. 
  * **Feature Dimension**: Output dimension is 1,280, providing a rich spatial representation to feed into temporal modeling.

### Audio Stream: Wav2Vec 2.0 Base
* **Decision**: Use `facebook/wav2vec2-base-960h` (via `transformers`) as the speech encoder.
* **Rationale**:
  * **Speech Representation**: Pre-trained on 960 hours of speech, Wav2Vec 2.0 captures fine-grained phonetic, acoustic, and temporal speech patterns.
  * **Temporal Alignment**: Output is a sequence of hidden states of shape `(B, A_steps, 768)` which aligns well with the visual frame sequences.
  * **Frozen Backbone**: Frozen weights leverage general speech representations without training stability issues on the smaller dataset.

---

## 2. Preprocessing & Alignment

### Face Detection & Bounding Boxes: RetinaFace
* **Decision**: Use `RetinaFace` as the primary face detector backend.
* **Rationale**:
  * **High Recall**: RetinaFace detects faces at varying scales, angles, and occlusions, minimizing skipped frames.
  * **Five-Point Landmarks**: Provides eye, nose, and mouth corner coordinates which are required to perform five-point eye alignment.
  * **Pluggability**: The registry-factory pattern permits swapping with lighter models (e.g. SCRFD or YOLOFace) in the future.

### Five-Point Eye Alignment
* **Decision**: Align detected faces horizontally based on eye center points before cropping.
* **Rationale**:
  * **Spatial Normalization**: Reduces rotation-variance, allowing the spatial encoder (EfficientNet) to focus on micro-textures and synthesis artifacts instead of head tilt.

---

## 3. Training & Dataset Pipeline

### Split Strategy: Subject-Independent Splitting
* **Decision**: Partition the train, validation, and test sets such that no subject's identity overlaps between splits.
* **Rationale**:
  * **No Identity Leakage**: Deepfake detectors are prone to memorizing the faces of subjects (identity recognition) rather than detecting manipulation artifacts. Subject-independent splitting guarantees generalization to unseen faces.

### Loss Function: BCEWithLogitsLoss
* **Decision**: Use Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) as the baseline default, reserving `FocalLoss` for imbalance.
* **Rationale**:
  * **Stability**: BCE is highly stable and standard for binary classification tasks.
  * **Imbalance mitigation**: Positive class weights are computed dynamically and fed into the BCE loss to adjust for ratio discrepancies between real and fake samples.

### Optimization: Mixed Precision (AMP)
* **Decision**: Enable PyTorch's Automatic Mixed Precision (`torch.amp`) when running on CUDA.
* **Rationale**:
  * **Performance & Memory**: Half-precision floating-point format (`float16`/`bfloat16`) speeds up forward/backward passes and halves VRAM requirements, supporting larger batch sizes.

---

## 4. Hardware Bottlenecks & Future GPU Preprocessing

### CPU Bottlenecks
* **Observation**: Running RetinaFace (via TensorFlow/Keras backend) on CPU takes **~17.4 seconds per frame**. 
* **Conclusion**: CPU-only preprocessing for the entire 21,544 video dataset (~2M frames) would take **~1.2 years**. CPU is strictly reserved for pipeline verification (via a 2-video subset).

### Future GPU Preprocessing
* Preprocessing must be offloaded to a GPU environment (Colab/Kaggle/server).
* Using a CUDA-enabled GPU and batched inference will decrease face detection times from 17.4s to under **0.02s per frame**, reducing the full preprocessing stage to several hours.
