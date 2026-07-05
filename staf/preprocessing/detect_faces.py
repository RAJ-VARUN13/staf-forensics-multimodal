"""
STAF Preprocessing Stage 2: Face Detection.

Scans extracted frames folders, runs the configured pluggable face detector backend,
and extracts rich frame-by-frame face bounding boxes and landmarks. Saves results
as a video-level JSON manifest file.

Supports resume/checkpointing (skipping completed videos) and logging of detailed run statistics.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from tqdm import tqdm

from staf.configs.schema import load_config
from staf.preprocessing.detectors import get_face_detector
from staf.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def detect_faces_in_video_frames(
    frames_dir: Path,
    manifest_path: Path,
    detector_name: str,
    detector_kwargs: Dict[str, Any],
    fps: float = 30.0,
    overwrite: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Runs face detection on all frame images in a folder and writes a JSON manifest.

    Args:
        frames_dir: Directory containing frame images (e.g. frame_000000.jpg).
        manifest_path: Destination path for the face detections manifest (.json).
        detector_name: Name of the face detector backend.
        detector_kwargs: Dict of parameters for the detector backend.
        fps: Video frames per second (for timestamp calculation).
        overwrite: Overwrite existing manifest.

    Returns:
        A tuple of (success_status, stats_dict).
    """
    if manifest_path.exists() and not overwrite:
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
            # Verify the manifest is valid
            if "detections" in data and "stats" in data:
                return True, {
                    "status": "SKIPPED",
                    "frames_processed": data["stats"].get("frames_processed", 0),
                    "faces_detected": data["stats"].get("faces_detected", 0)
                }
        except Exception:
            # If JSON is corrupted, proceed to rewrite
            pass

    # Find and sort frame images
    frame_files = sorted(list(frames_dir.glob("frame_*.jpg")))
    num_frames = len(frame_files)
    if num_frames == 0:
        return False, {"status": "FAILED", "msg": f"No frame images found in {frames_dir}"}

    try:
        # Load detector
        detector = get_face_detector(detector_name, **detector_kwargs)
    except Exception as e:
        return False, {"status": "FAILED", "msg": f"Failed to load detector {detector_name}: {str(e)}"}

    detections: List[Dict[str, Any]] = []
    total_faces = 0
    failed_frames = 0

    for idx, frame_path in enumerate(frame_files):
        try:
            # Load frame image
            img = cv2.imread(str(frame_path))
            if img is None:
                failed_frames += 1
                continue
                
            height, width, _ = img.shape
            
            # Detect faces
            faces = detector.detect_faces(img)
            
            # Formulate timestamp
            timestamp = idx / fps

            for face in faces:
                total_faces += 1
                detection_item = {
                    "frame_id": idx,
                    "bbox": face["bbox"],
                    "landmarks": face["landmarks"],
                    "confidence": face["confidence"],
                    "detector": detector_name,
                    "timestamp": float(f"{timestamp:.4f}"),
                    "image_width": width,
                    "image_height": height
                }
                detections.append(detection_item)

        except Exception as e:
            failed_frames += 1
            logger.warning(f"Failed to process frame {frame_path.name}: {e}")

    # Write manifest output
    stats = {
        "status": "SUCCESS",
        "frames_processed": num_frames,
        "faces_detected": total_faces,
        "failed_frames": failed_frames,
        "detector_backend": detector_name
    }
    
    output_data = {
        "video_name": frames_dir.name,
        "stats": stats,
        "detections": detections
    }

    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(output_data, f, indent=2)
        return True, stats
    except Exception as e:
        return False, {"status": "FAILED", "msg": f"Failed to save manifest {manifest_path}: {e}"}


def run_face_detection_pipeline(
    frames_root: Path,
    manifests_root: Path,
    detector_name: str,
    detector_kwargs: Dict[str, Any],
    fps: float = 30.0,
    overwrite: bool = False,
) -> None:
    """
    Runs the face detection pipeline across all frame folders in frames_root.
    """
    logger.info(f"Scanning for frames subfolders in: {frames_root}")
    
    # Get all subdirectories (each corresponds to a video)
    frames_dirs = [d for d in frames_root.iterdir() if d.is_dir()]
    num_videos = len(frames_dirs)
    
    if num_videos == 0:
        logger.warning(f"No frames directories found under: {frames_root}")
        return

    logger.info(f"Found {num_videos} frame directories. Starting face detection...")
    
    total_frames = 0
    total_faces = 0
    failed_frames = 0
    skipped_videos = 0
    success_videos = 0
    failed_videos = 0

    # Sequential loop is recommended for GPU-based detectors to avoid VRAM thrashing
    with tqdm(total=num_videos, desc="Detecting faces") as pbar:
        for f_dir in frames_dirs:
            manifest_path = manifests_root / f"{f_dir.name}_manifest.json"
            success, stats = detect_faces_in_video_frames(
                frames_dir=f_dir,
                manifest_path=manifest_path,
                detector_name=detector_name,
                detector_kwargs=detector_kwargs,
                fps=fps,
                overwrite=overwrite
            )
            
            if success:
                if stats.get("status") == "SKIPPED":
                    skipped_videos += 1
                else:
                    success_videos += 1
                total_frames += stats.get("frames_processed", 0)
                total_faces += stats.get("faces_detected", 0)
                failed_frames += stats.get("failed_frames", 0)
            else:
                failed_videos += 1
                logger.error(f"Failed to process {f_dir.name}: {stats.get('msg')}")
                
            pbar.update(1)

    logger.info("=== Face Detection Stage Summary ===")
    logger.info(f"  Videos Successfully Processed: {success_videos}")
    logger.info(f"  Videos Skipped (Resume):        {skipped_videos}")
    logger.info(f"  Videos Failed:                 {failed_videos}")
    logger.info(f"  Total Frames Processed:        {total_frames}")
    logger.info(f"  Total Faces Detected:          {total_faces}")
    if total_frames > 0:
        logger.info(f"  Average Faces Per Frame:       {total_faces / total_frames:.3f}")
    logger.info(f"  Failed Frame Reads:            {failed_frames}")
    logger.info("=====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STAF Preprocessing: Run Face Detection")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--detector", type=str, help="Override detector name")
    parser.add_argument("--threshold", type=float, help="Override detector threshold")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing manifest files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    # Paths
    processed_dir = Path(cfg.data.paths.processed_dir or "data/processed")
    frames_dir = processed_dir / "frames"
    manifests_dir = processed_dir / "manifests"

    det_name = args.detector or cfg.data.face_detector or "retinaface"
    threshold = args.threshold or 0.9

    logger.info(f"Frames Source:      {frames_dir}")
    logger.info(f"Manifests Output:   {manifests_dir}")
    logger.info(f"Detector Backend:   {det_name}")
    logger.info(f"Detector Threshold: {threshold}")
    logger.info(f"Overwrite:          {args.overwrite}")

    run_face_detection_pipeline(
        frames_root=frames_dir,
        manifests_root=manifests_dir,
        detector_name=det_name,
        detector_kwargs={"threshold": threshold} if det_name == "retinaface" else {},
        overwrite=args.overwrite
    )
