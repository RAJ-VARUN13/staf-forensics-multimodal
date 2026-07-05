"""
Unit tests for STAF Preprocessing Stage 5: Modality Synchronization.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from staf.preprocessing.sync_modalities import (
    extract_subject_and_label,
    build_splits,
    run_modality_sync,
)


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_extract_subject_and_label() -> None:
    """Verifies label and subject ID extraction logic from path structures."""
    # Test FakeAVCeleb Real Video/Audio path
    real_path = Path("FakeAVCeleb/RealVideo-RealAudio/African/men/id00076/00109.mp4")
    subj, label = extract_subject_and_label(real_path)
    assert subj == "id00076"
    assert label == 0

    # Test FakeAVCeleb Fake Video/Audio path
    fake_path = Path("FakeAVCeleb/FakeVideo-FakeAudio/African/men/id00076/00109.mp4")
    subj, label = extract_subject_and_label(fake_path)
    assert subj == "id00076"
    assert label == 1

    # Test path with no explicit id folder but containing subject folder name
    fallback_path = Path("custom_dataset/fake/subject_name/video.mp4")
    subj, label = extract_subject_and_label(fallback_path)
    assert subj == "subject_name"
    assert label == 1


def test_build_splits_subject_independent() -> None:
    """Verifies subject-independent partitioning preserves splits group boundary (no subject overlap)."""
    samples = [
        {"subject": "subj_A", "video_id": "v1"},
        {"subject": "subj_A", "video_id": "v2"},
        {"subject": "subj_B", "video_id": "v3"},
        {"subject": "subj_B", "video_id": "v4"},
        {"subject": "subj_C", "video_id": "v5"},
        {"subject": "subj_D", "video_id": "v6"},
        {"subject": "subj_E", "video_id": "v7"},
    ]

    splits = build_splits(
        samples=samples,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        subject_independent=True,
        seed=42,
    )

    # All samples must be assigned to exactly one split
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(samples)

    # Collect subject sets for each split
    train_subjects = {s["subject"] for s in splits["train"]}
    val_subjects = {s["subject"] for s in splits["val"]}
    test_subjects = {s["subject"] for s in splits["test"]}

    # Ensure intersection of subject sets across splits is empty
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)


def test_run_modality_sync_end_to_end(temp_dir: Path) -> None:
    """End-to-end sync verification on simulated raw database and preprocessed folders."""
    # 1. Setup mock raw videos directory
    raw_dir = temp_dir / "raw"
    raw_dir.mkdir()
    
    # 3 raw videos
    v1_path = raw_dir / "RealVideo-RealAudio" / "men" / "id0001" / "vid_01.mp4"
    v2_path = raw_dir / "FakeVideo-FakeAudio" / "men" / "id0002" / "vid_02.mp4"
    v3_path = raw_dir / "FakeVideo-FakeAudio" / "men" / "id0001" / "vid_03.mp4" # same subject as v1

    for vp in [v1_path, v2_path, v3_path]:
        vp.parent.mkdir(parents=True, exist_ok=True)
        vp.touch()

    # 2. Setup mock preprocessed outputs directory
    processed_dir = temp_dir / "processed"
    faces_root = processed_dir / "faces"
    audio_root = processed_dir / "audio"
    manifests_root = processed_dir / "manifests"

    for r in [faces_root, audio_root, manifests_root]:
        r.mkdir(parents=True, exist_ok=True)

    # Simulate full preprocessing output for v1, v2, v3
    for v_id in ["vid_01", "vid_02", "vid_03"]:
        # Write dummy manifest
        (manifests_root / f"{v_id}_manifest.json").write_text("{}")
        
        # Write dummy WAV
        (audio_root / f"{v_id}.wav").touch()

        # Write dummy face crops
        v_faces_dir = faces_root / v_id
        v_faces_dir.mkdir()
        (v_faces_dir / "face_000001.jpg").touch()

    splits_dir = temp_dir / "splits"

    # Run synchronizer
    run_modality_sync(
        raw_videos_dir=raw_dir,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        subject_independent=True,
        seed=123,
    )

    # Verify master file exists and contains correct columns
    master_csv = processed_dir / "metadata.csv"
    assert master_csv.exists()

    with open(master_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 3
    assert all("video_id" in row for row in rows)
    assert all("label" in row for row in rows)
    assert all("subject" in row for row in rows)

    # Verify separate train/val/test split CSV files exist
    assert (splits_dir / "train.csv").exists()
    assert (splits_dir / "val.csv").exists()
    assert (splits_dir / "test.csv").exists()
