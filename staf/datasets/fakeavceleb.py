"""
STAF PyTorch Dataset: FakeAVCeleb Loader.

Loads synchronized facial frame crops and audio waveforms for training and evaluating
multimodal deepfake detection models. Performs frame sampling, resizing, normalization,
and audio padding/truncation.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from staf.configs.schema import DataConfig
from staf.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Image Transforms & Normalization
# =============================================================================

def preprocess_face_crop(
    image_path: Path,
    image_size: int = 224,
) -> torch.Tensor:
    """
    Loads, resizes, and normalizes a cropped face image to ImageNet standards.

    Args:
        image_path: Path to the crop JPEG.
        image_size: Target height/width.

    Returns:
        Tensor of shape (3, image_size, image_size).
    """
    # Read image using OpenCV (BGR format)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if not matching image_size
    if img.shape[0] != image_size or img.shape[1] != image_size:
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # ImageNet mean & std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # Transpose to Channel-First (3, H, W)
    tensor = torch.from_numpy(img.transpose(2, 0, 1))
    return tensor


# =============================================================================
# FakeAVCeleb PyTorch Dataset
# =============================================================================

class FakeAVCelebDataset(Dataset):
    """
    PyTorch Dataset for the FakeAVCeleb dataset.

    Given a split CSV, it loads a sequence of facial frame crops and the corresponding
    WAV audio track.
    """

    def __init__(
        self,
        split_csv_path: Union[str, Path],
        data_config: DataConfig,
        transform: Optional[Any] = None,
    ) -> None:
        """
        Args:
            split_csv_path: Path to the train.csv, val.csv, or test.csv split file.
            data_config: Preprocessing/data configuration section.
            transform: Optional additional PyTorch transforms (reserved for augmentation).
        """
        self.split_csv_path = Path(split_csv_path)
        self.cfg = data_config
        self.transform = transform
        
        # Base directory for resolving relative paths in the CSV
        # Relies on the fact that face_dir and audio_path are saved relative to the
        # parent of the processed directory (the workspace/data folder)
        self.base_data_dir = Path(self.cfg.paths.processed_dir).parent

        self.samples: List[Dict[str, str]] = []
        self._load_split_csv()

    def _load_split_csv(self) -> None:
        """Reads split CSV file and validates entries."""
        if not self.split_csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv_path}")

        with open(self.split_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

        logger.info(f"Loaded {len(self.samples)} samples for split: {self.split_csv_path.name}")

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        """
        Returns indices of frames to sample based on the configuration strategy.

        Args:
            total_frames: Total number of frames available in the crop folder.

        Returns:
            List of frame indices of size `self.cfg.num_frames`.
        """
        num_target = self.cfg.num_frames

        if total_frames <= 0:
            return [0] * num_target

        if self.cfg.frame_sampling_strategy == "uniform":
            # Select equally spaced indices
            indices = np.linspace(0, total_frames - 1, num_target, dtype=int)
            return indices.tolist()

        elif self.cfg.frame_sampling_strategy == "center":
            # Extract sequence from middle
            if total_frames >= num_target:
                start = (total_frames - num_target) // 2
                return list(range(start, start + num_target))
            else:
                # Loop padding
                indices = np.linspace(0, total_frames - 1, num_target, dtype=int)
                return indices.tolist()

        elif self.cfg.frame_sampling_strategy == "random":
            # Randomly sample num_target sorted indices
            if total_frames >= num_target:
                indices = sorted(np.random.choice(total_frames, num_target, replace=False))
                return list(indices)
            else:
                # Padding required
                indices = sorted(np.random.choice(total_frames, num_target, replace=True))
                return list(indices)

        else:
            # Fallback to uniform
            indices = np.linspace(0, total_frames - 1, num_target, dtype=int)
            return indices.tolist()

    def _load_faces(self, face_dir_path: Path) -> torch.Tensor:
        """
        Loads and samples face crops, returning a stacked visual tensor.

        Args:
            face_dir_path: Path to the video's face crops directory.

        Returns:
            Tensor of shape (num_frames, 3, image_size, image_size).
        """
        # Find all face crop files
        face_files = sorted(list(face_dir_path.glob("face_*.jpg")))
        total_files = len(face_files)

        sampled_indices = self._sample_frame_indices(total_files)
        face_tensors = []

        for idx in sampled_indices:
            if total_files > 0:
                img_path = face_files[idx]
                try:
                    face_tensor = preprocess_face_crop(img_path, self.cfg.image_size)
                except Exception as e:
                    # In case of loading failure, fallback to zero tensor
                    logger.warning(f"Error loading crop {img_path}: {e}")
                    face_tensor = torch.zeros((3, self.cfg.image_size, self.cfg.image_size), dtype=torch.float32)
            else:
                # Empty folder fallback
                face_tensor = torch.zeros((3, self.cfg.image_size, self.cfg.image_size), dtype=torch.float32)
            
            face_tensors.append(face_tensor)

        # Stack into [num_frames, 3, H, W]
        return torch.stack(face_tensors, dim=0)

    def _load_audio(self, audio_file_path: Path) -> torch.Tensor:
        """
        Loads the audio WAV track, normalizes it, and pads or truncates to max duration.

        Args:
            audio_file_path: Path to the WAV file.

        Returns:
            1D waveform tensor of shape (max_samples,).
        """
        target_samples = int(self.cfg.audio_sample_rate * self.cfg.audio_max_duration_sec)

        try:
            # Load audio using torchaudio
            # waveform shape: [channels, samples]
            waveform, sr = torchaudio.load(str(audio_file_path))
            
            # Wav2Vec2 operates on a mono channel, 1D tensor
            waveform = waveform[0]  # Take first channel if multi-channel

            # Optional resampling (backup check, though extracted at 16kHz)
            if sr != self.cfg.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.cfg.audio_sample_rate)
                waveform = resampler(waveform)

            # Pad or truncate
            if len(waveform) > target_samples:
                # Truncate
                waveform = waveform[:target_samples]
            elif len(waveform) < target_samples:
                # Zero padding
                padding = target_samples - len(waveform)
                waveform = torch.nn.functional.pad(waveform, (0, padding), "constant", 0.0)

            # Standardization
            if self.cfg.audio_normalize:
                mean = waveform.mean()
                std = waveform.std()
                waveform = (waveform - mean) / (std + 1e-7)

        except Exception as e:
            logger.warning(f"Error loading audio file {audio_file_path}: {e}")
            waveform = torch.zeros(target_samples, dtype=torch.float32)

        return waveform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            A tuple containing:
                - faces: Visual frames tensor of shape (num_frames, 3, image_size, image_size).
                - audio: Audio waveform tensor of shape (max_samples,).
                - label: Binary target label tensor (scalar, float32: 0.0 for real, 1.0 for fake).
                - metadata: Dictionary containing metadata keys (video_id, subject).
        """
        sample = self.samples[idx]

        # Construct absolute paths from the relative directory columns
        face_dir_abs = self.base_data_dir / sample["face_dir"]
        audio_path_abs = self.base_data_dir / sample["audio_path"]

        # Load modalities
        faces_tensor = self._load_faces(face_dir_abs)
        audio_tensor = self._load_audio(audio_path_abs)

        # Label representation
        label_val = float(sample["label"])
        label_tensor = torch.tensor(label_val, dtype=torch.float32)

        metadata = {
            "video_id": sample["video_id"],
            "subject": sample["subject"],
            "raw_video_path": sample["raw_video_path"],
        }

        # Apply augmentation if defined and enabled
        if self.transform is not None:
            faces_tensor = self.transform(faces_tensor)

        return faces_tensor, audio_tensor, label_tensor, metadata
