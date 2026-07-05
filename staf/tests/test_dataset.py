"""
Unit tests for STAF PyTorch Dataset: FakeAVCeleb Loader.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torchaudio

from staf.configs.schema import DataConfig, DataPathConfig
from staf.datasets.fakeavceleb import FakeAVCelebDataset, preprocess_face_crop


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_preprocess_face_crop(temp_dir: Path) -> None:
    """Verifies that an image crop is loaded, resized, and normalized to ImageNet format."""
    img_path = temp_dir / "test_face.jpg"
    # Create random image
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    tensor = preprocess_face_crop(img_path, image_size=224)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_dataset_loading_and_sampling(temp_dir: Path) -> None:
    """End-to-end test verifying Dataset sampling, shapes, and normalization logic."""
    # 1. Setup split directories
    processed_dir = temp_dir / "processed"
    faces_root = processed_dir / "faces"
    audio_root = processed_dir / "audio"
    
    faces_root.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    # Create dummy video details
    video_id = "mock_video_01"
    
    # Create dummy audio (1 second of 16kHz mono)
    audio_path = audio_root / f"{video_id}.wav"
    dummy_wav = torch.randn(1, 16000)
    torchaudio.save(str(audio_path), dummy_wav, 16000)

    # Create dummy face crops (5 crops, we will sample 4)
    video_faces_dir = faces_root / video_id
    video_faces_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        img_path = video_faces_dir / f"face_{i:06d}_00.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

    # Create dummy split CSV
    split_csv = temp_dir / "train.csv"
    headers = ["video_id", "raw_video_path", "face_dir", "audio_path", "label", "num_faces", "subject", "split"]
    
    # Paths are stored relative to the parent of processed_dir (which is temp_dir)
    rel_face_dir = "processed/faces/mock_video_01"
    rel_audio_path = "processed/audio/mock_video_01.wav"

    with open(split_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow({
            "video_id": video_id,
            "raw_video_path": "dummy_raw.mp4",
            "face_dir": rel_face_dir,
            "audio_path": rel_audio_path,
            "label": "1.0",  # Fake
            "num_faces": "5",
            "subject": "subj_1",
            "split": "train"
        })

    # Setup configuration
    paths_cfg = DataPathConfig(
        fakeavceleb_root=str(temp_dir),
        processed_dir=str(processed_dir),
        splits_dir=str(temp_dir),
    )
    
    data_cfg = DataConfig(
        paths=paths_cfg,
        num_frames=4,  # sample 4 frames out of 5
        frame_sampling_strategy="uniform",
        image_size=224,
        audio_sample_rate=16000,
        audio_max_duration_sec=2.0,  # 2 seconds = 32000 samples
        audio_normalize=True,
    )

    dataset = FakeAVCelebDataset(split_csv_path=split_csv, data_config=data_cfg)
    assert len(dataset) == 1

    faces, audio, label, metadata = dataset[0]

    # Verify visual tensor shape: [num_frames, 3, image_size, image_size]
    assert faces.shape == (4, 3, 224, 224)
    assert faces.dtype == torch.float32

    # Verify audio shape: [sample_rate * duration]
    assert audio.shape == (32000,)
    assert audio.dtype == torch.float32

    # Verify label and metadata
    assert label.item() == 1.0
    assert metadata["video_id"] == video_id
    assert metadata["subject"] == "subj_1"
