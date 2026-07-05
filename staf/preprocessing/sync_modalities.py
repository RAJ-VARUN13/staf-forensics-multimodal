"""
STAF Preprocessing Stage 5: Modality Synchronization & Metadata Builder.

Scans the raw video files and preprocessed directories (faces and audio) to build
a unified metadata manifest. Resolves real/fake labels, extracts speaker identities,
filters out corrupted or incomplete preprocessing samples, and generates subject-independent
data splits (train/val/test).

Saves outputs to:
    - processed_dir/metadata.csv (Unified master manifest)
    - processed_dir/splits/train.csv
    - processed_dir/splits/val.csv
    - processed_dir/splits/test.csv

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from staf.configs.schema import load_config
from staf.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def extract_subject_and_label(video_path: Path) -> Tuple[str, int]:
    """
    Extracts the subject ID and binary label (0 = Real, 1 = Fake) from a video's path.

    Designed for the FakeAVCeleb structure:
        - RealVideo-RealAudio/ -> label 0
        - FakeVideo-RealAudio/ -> label 1
        - RealVideo-FakeAudio/ -> label 1
        - FakeVideo-FakeAudio/ -> label 1
    Subject ID matches idXXXXX (e.g., id00076).

    Args:
        video_path: Path to the raw video.

    Returns:
        A tuple of (subject_id, binary_label).
    """
    path_str = str(video_path).replace("\\", "/")
    
    # Label extraction
    if "RealVideo-RealAudio" in path_str:
        label = 0
    elif any(x in path_str for x in ["FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"]):
        label = 1
    else:
        # Fallback heuristic: check if word "fake" or "Fake" is present in the path
        if "fake" in path_str.lower():
            label = 1
        else:
            label = 0

    # Subject ID extraction: look for folder matching idXXXXX
    # match e.g. "id02043" or similar
    subject_match = re.search(r"/(id\d+)/", path_str)
    if subject_match:
        subject_id = subject_match.group(1)
    else:
        # Fallback: use grandparent directory name
        if len(video_path.parts) >= 3:
            subject_id = video_path.parent.name
        else:
            subject_id = "unknown"

    return subject_id, label


def build_splits(
    samples: List[Dict[str, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    subject_independent: bool = True,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Partitions the dataset into train, val, and test splits.

    Supports subject-independent splitting to prevent subject leakage:
        - Groups all samples by subject ID.
        - Randomly assigns subjects to train, val, or test splits.
        - All videos from the same subject go into the same split.

    Args:
        samples: A list of dicts representing dataset entries.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        subject_independent: If True, splits by subject instead of individual videos.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary mapping split names to lists of samples.
    """
    # Normalize ratios so they sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    rng = random.Random(seed)
    splits: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}

    if not samples:
        return splits

    if subject_independent:
        # Group by subject
        subject_to_samples: Dict[str, List[Dict[str, str]]] = {}
        for s in samples:
            subj = s["subject"]
            subject_to_samples.setdefault(subj, []).append(s)

        subjects = sorted(list(subject_to_samples.keys()))
        rng.shuffle(subjects)

        num_subjects = len(subjects)
        train_end = int(num_subjects * train_ratio)
        val_end = train_end + int(num_subjects * val_ratio)

        train_subjects = set(subjects[:train_end])
        val_subjects = set(subjects[train_end:val_end])
        test_subjects = set(subjects[val_end:])

        logger.info(f"Subject-independent split: {len(train_subjects)} train, {len(val_subjects)} val, {len(test_subjects)} test subjects")

        for subj, subj_samples in subject_to_samples.items():
            if subj in train_subjects:
                for s in subj_samples:
                    s["split"] = "train"
                splits["train"].extend(subj_samples)
            elif subj in val_subjects:
                for s in subj_samples:
                    s["split"] = "val"
                splits["val"].extend(subj_samples)
            else:
                for s in subj_samples:
                    s["split"] = "test"
                splits["test"].extend(subj_samples)
    else:
        # Simple random split
        shuffled = list(samples)
        rng.shuffle(shuffled)

        num_samples = len(shuffled)
        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)

        for i, s in enumerate(shuffled):
            if i < train_end:
                s["split"] = "train"
                splits["train"].append(s)
            elif i < val_end:
                s["split"] = "val"
                splits["val"].append(s)
            else:
                s["split"] = "test"
                splits["test"].append(s)

    return splits


def run_modality_sync(
    raw_videos_dir: Path,
    processed_dir: Path,
    splits_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    subject_independent: bool = True,
    seed: int = 42,
) -> None:
    """
    Main pipeline for modality synchronization and metadata construction.
    """
    faces_root = processed_dir / "faces"
    audio_root = processed_dir / "audio"
    manifests_root = processed_dir / "manifests"

    logger.info("Starting modality synchronization pipeline...")
    logger.info(f"  Faces root: {faces_root}")
    logger.info(f"  Audio root: {audio_root}")

    # 1. Scan for raw videos to find corresponding targets
    extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_paths: List[Path] = []
    for ext in extensions:
        video_paths.extend(raw_videos_dir.rglob(f"*{ext}"))
        video_paths.extend(raw_videos_dir.rglob(f"*{ext.upper()}"))

    # Deduplicate paths (e.g. on Windows case-insensitive filesystem)
    video_paths = list(dict.fromkeys(video_paths))

    if not video_paths:
        logger.warning(f"No raw video files found in {raw_videos_dir}")
        return

    logger.info(f"Found {len(video_paths)} raw videos. Syncing modalities...")

    valid_samples: List[Dict[str, str]] = []
    skipped_no_faces = 0
    skipped_no_audio = 0
    skipped_no_manifest = 0

    for v_path in video_paths:
        video_name = v_path.stem
        
        # Check face manifest exists
        manifest_path = manifests_root / f"{video_name}_manifest.json"
        if not manifest_path.exists():
            skipped_no_manifest += 1
            continue

        # Check face crops folder exists and is non-empty
        faces_dir = faces_root / video_name
        face_files = list(faces_dir.glob("face_*.jpg")) if faces_dir.exists() else []
        if not face_files:
            skipped_no_faces += 1
            continue

        # Check audio WAV exists
        audio_path = audio_root / f"{video_name}.wav"
        if not audio_path.exists():
            skipped_no_audio += 1
            continue

        # Extract subject and label
        subject, label = extract_subject_and_label(v_path)

        # Get relative paths from workspace/root or save absolute/relative to processed_dir
        # We save paths relative to processed_dir to keep metadata portable
        rel_face_dir = str(faces_dir.relative_to(processed_dir.parent)).replace("\\", "/")
        rel_audio_path = str(audio_path.relative_to(processed_dir.parent)).replace("\\", "/")

        valid_samples.append({
            "video_id": video_name,
            "raw_video_path": str(v_path).replace("\\", "/"),
            "face_dir": rel_face_dir,
            "audio_path": rel_audio_path,
            "label": str(label),
            "num_faces": str(len(face_files)),
            "subject": subject,
            "split": "unassigned"
        })

    logger.info(f"Sync Results:")
    logger.info(f"  Valid synced AV samples:  {len(valid_samples)}")
    logger.info(f"  Skipped (No manifest):    {skipped_no_manifest}")
    logger.info(f"  Skipped (No face crops):   {skipped_no_faces}")
    logger.info(f"  Skipped (No WAV audio):    {skipped_no_audio}")

    if not valid_samples:
        logger.error("No valid synchronized samples found. Training cannot proceed.")
        return

    # 2. Split dataset
    splits = build_splits(
        samples=valid_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        subject_independent=subject_independent,
        seed=seed
    )

    # 3. Save master metadata.csv
    metadata_csv_path = processed_dir / "metadata.csv"
    headers = ["video_id", "raw_video_path", "face_dir", "audio_path", "label", "num_faces", "subject", "split"]
    
    with open(metadata_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(valid_samples)
    
    logger.info(f"Saved master manifest to: {metadata_csv_path}")

    # 4. Save splits csv
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, split_samples in splits.items():
        split_file = splits_dir / f"{name}.csv"
        with open(split_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(split_samples)
        logger.info(f"  Saved split file: {split_file} ({len(split_samples)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STAF Preprocessing: Modality Synchronization")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--raw_dir", type=str, help="Override path to raw videos")
    parser.add_argument("--processed_dir", type=str, help="Override processed data directory")
    parser.add_argument("--splits_dir", type=str, help="Override splits directory")
    parser.add_argument("--seed", type=int, help="Override split seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    raw_dir_str = args.raw_dir or cfg.data.paths.fakeavceleb_root
    if not raw_dir_str:
        logger.error("No input videos path specified. Set paths in config or use --raw_dir.")
        sys.exit(1)

    raw_path = Path(raw_dir_str)
    if (raw_path / "raw").exists():
        raw_path = raw_path / "raw"

    processed_dir_str = args.processed_dir or cfg.data.paths.processed_dir or "data/processed"
    processed_path = Path(processed_dir_str)

    splits_dir_str = args.splits_dir or cfg.data.paths.splits_dir or "data/splits"
    splits_path = Path(splits_dir_str)

    seed = args.seed if args.seed is not None else cfg.data.split_seed

    run_modality_sync(
        raw_videos_dir=raw_path,
        processed_dir=processed_path,
        splits_dir=splits_path,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        subject_independent=cfg.data.subject_independent,
        seed=seed
    )
