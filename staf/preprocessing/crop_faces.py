"""
STAF Preprocessing Stage 3: Face Cropping with Five-Point Alignment.

Pipeline per frame:
    1. Read bounding box and landmarks from JSON manifest
    2. Align face using five-point eye landmarks (rotate so eyes are horizontal)
    3. Crop face with configurable margin
    4. Resize to target image_size (from config)
    5. Normalize pixel values to [0, 1] float32 (optional — can also be done at Dataset level)
    6. Save aligned + cropped face to processed/faces/

Supports resume/checkpointing and multiprocessing.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from staf.configs.schema import load_config
from staf.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


# =============================================================================
# Five-Point Eye Alignment
# =============================================================================

def align_face_by_eyes(
    image: np.ndarray,
    left_eye: List[int],
    right_eye: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotates an image so that the line between the two eyes is horizontal.

    This ensures consistent face orientation across all samples, which is
    critical for temporal modeling — the model should learn semantic features,
    not orientation artifacts.

    Args:
        image: BGR image (H, W, 3).
        left_eye: [x, y] coordinates of the left eye.
        right_eye: [x, y] coordinates of the right eye.

    Returns:
        A tuple of (rotated_image, rotation_matrix).
    """
    # Compute angle between the two eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))

    # Center of rotation = midpoint between eyes
    eye_center = (
        (left_eye[0] + right_eye[0]) / 2.0,
        (left_eye[1] + right_eye[1]) / 2.0,
    )

    # Compute rotation matrix (no scaling)
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)

    # Rotate the entire image
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated, rotation_matrix


def transform_point(point: List[int], rotation_matrix: np.ndarray) -> List[int]:
    """Applies a 2x3 affine rotation matrix to a 2D point."""
    pt = np.array([point[0], point[1], 1.0])
    transformed = rotation_matrix @ pt
    return [int(transformed[0]), int(transformed[1])]


# =============================================================================
# Crop + Resize
# =============================================================================

def crop_face_with_margin(
    image: np.ndarray,
    bbox: List[int],
    margin: float = 0.3,
    image_size: int = 224,
) -> Optional[np.ndarray]:
    """
    Crops a face from an image using the bounding box with a margin,
    then resizes to target dimensions.

    Args:
        image: BGR image (H, W, 3).
        bbox: [x1, y1, x2, y2] bounding box coordinates.
        margin: Fractional margin to add around the bounding box.
        image_size: Target output size (square: image_size x image_size).

    Returns:
        Cropped and resized face image, or None if the crop is invalid.
    """
    if len(bbox) != 4:
        return None

    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Compute margin in pixels
    box_w = x2 - x1
    box_h = y2 - y1
    margin_w = int(box_w * margin)
    margin_h = int(box_h * margin)

    # Expand with margin, clamped to image bounds
    x1_m = max(0, x1 - margin_w)
    y1_m = max(0, y1 - margin_h)
    x2_m = min(w, x2 + margin_w)
    y2_m = min(h, y2 + margin_h)

    # Validate crop region
    if x2_m <= x1_m or y2_m <= y1_m:
        return None

    crop = image[y1_m:y2_m, x1_m:x2_m]

    # Resize to target
    resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    return resized


# =============================================================================
# Per-Video Processing
# =============================================================================

def crop_faces_for_video(
    frames_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    margin: float = 0.3,
    image_size: int = 224,
    alignment_enabled: bool = True,
    overwrite: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Processes all frames for a single video: align → crop → resize → save.

    Args:
        frames_dir: Directory containing frame_XXXXXX.jpg files.
        manifest_path: Path to the face detection manifest JSON.
        output_dir: Destination for cropped face images.
        margin: Fractional margin around bbox.
        image_size: Target crop dimensions.
        alignment_enabled: Whether to perform five-point eye alignment.
        overwrite: Overwrite existing cropped faces.

    Returns:
        A tuple of (success, stats_dict).
    """
    video_name = frames_dir.name
    faces_dir = output_dir / video_name

    # Resume check
    if faces_dir.exists() and not overwrite:
        existing = list(faces_dir.glob("face_*.jpg"))
        if len(existing) > 0:
            return True, {"status": "SKIPPED", "faces_saved": len(existing)}

    # Load manifest
    if not manifest_path.exists():
        return False, {"status": "FAILED", "msg": f"Manifest not found: {manifest_path}"}

    try:
        with open(manifest_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return False, {"status": "FAILED", "msg": f"Failed to read manifest: {e}"}

    detections = data.get("detections", [])
    if not detections:
        return False, {"status": "FAILED", "msg": "No detections in manifest"}

    faces_dir.mkdir(parents=True, exist_ok=True)

    # Clear if overwriting
    if overwrite:
        for f in faces_dir.glob("face_*.jpg"):
            f.unlink()

    faces_saved = 0
    alignment_failures = 0

    # Group detections by frame_id (handle multi-face)
    dets_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for det in detections:
        fid = det.get("frame_id")
        if fid is not None:
            dets_by_frame.setdefault(fid, []).append(det)

    for frame_id, frame_dets in sorted(dets_by_frame.items()):
        frame_path = frames_dir / f"frame_{frame_id:06d}.jpg"
        if not frame_path.exists():
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        for face_idx, det in enumerate(frame_dets):
            bbox = det.get("bbox", [])
            landmarks = det.get("landmarks", {})

            # Alignment step
            working_img = img
            working_bbox = bbox

            if alignment_enabled:
                left_eye = landmarks.get("left_eye", [])
                right_eye = landmarks.get("right_eye", [])

                if len(left_eye) == 2 and len(right_eye) == 2:
                    try:
                        rotated_img, rot_matrix = align_face_by_eyes(
                            img, left_eye, right_eye
                        )
                        # Transform bbox corners through rotation
                        tl = transform_point([bbox[0], bbox[1]], rot_matrix)
                        br = transform_point([bbox[2], bbox[3]], rot_matrix)
                        working_img = rotated_img
                        working_bbox = [
                            min(tl[0], br[0]),
                            min(tl[1], br[1]),
                            max(tl[0], br[0]),
                            max(tl[1], br[1]),
                        ]
                    except Exception:
                        alignment_failures += 1
                        # Fall back to unaligned crop
                        working_img = img
                        working_bbox = bbox

            # Crop + resize
            face_crop = crop_face_with_margin(
                working_img, working_bbox, margin=margin, image_size=image_size
            )

            if face_crop is None:
                continue

            # Save face
            face_filename = f"face_{frame_id:06d}_{face_idx:02d}.jpg"
            cv2.imwrite(
                str(faces_dir / face_filename), face_crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            faces_saved += 1

    if faces_saved == 0:
        return False, {"status": "FAILED", "msg": "Zero faces cropped"}

    return True, {
        "status": "SUCCESS",
        "faces_saved": faces_saved,
        "alignment_failures": alignment_failures,
    }


# =============================================================================
# Pipeline Runner
# =============================================================================

def run_crop_faces_pipeline(
    frames_root: Path,
    manifests_root: Path,
    output_root: Path,
    margin: float = 0.3,
    image_size: int = 224,
    alignment_enabled: bool = True,
    num_workers: int = 4,
    overwrite: bool = False,
) -> None:
    """
    Runs the crop + alignment pipeline across all videos.
    """
    logger.info(f"Scanning frame directories in: {frames_root}")
    frames_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    num_videos = len(frames_dirs)

    if num_videos == 0:
        logger.warning(f"No frame directories found under: {frames_root}")
        return

    logger.info(f"Found {num_videos} videos. Starting crop pipeline...")
    logger.info(f"  Margin: {margin} | Image Size: {image_size} | Alignment: {alignment_enabled}")

    success_count = 0
    skipped_count = 0
    failed_count = 0
    total_faces = 0
    total_alignment_failures = 0

    # Use ProcessPoolExecutor for CPU-bound alignment/crop operations
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for f_dir in frames_dirs:
            manifest_path = manifests_root / f"{f_dir.name}_manifest.json"
            future = executor.submit(
                crop_faces_for_video,
                frames_dir=f_dir,
                manifest_path=manifest_path,
                output_dir=output_root,
                margin=margin,
                image_size=image_size,
                alignment_enabled=alignment_enabled,
                overwrite=overwrite,
            )
            futures[future] = f_dir.name

        with tqdm(total=num_videos, desc="Cropping faces") as pbar:
            for future in as_completed(futures):
                video_name = futures[future]
                try:
                    success, stats = future.result()
                    if success:
                        if stats.get("status") == "SKIPPED":
                            skipped_count += 1
                        else:
                            success_count += 1
                            total_faces += stats.get("faces_saved", 0)
                            total_alignment_failures += stats.get("alignment_failures", 0)
                    else:
                        failed_count += 1
                        logger.error(f"Failed {video_name}: {stats.get('msg')}")
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Exception for {video_name}: {e}")
                pbar.update(1)

    logger.info("=== Face Cropping Summary ===")
    logger.info(f"  Videos Successfully Cropped: {success_count}")
    logger.info(f"  Videos Skipped (Resume):     {skipped_count}")
    logger.info(f"  Videos Failed:               {failed_count}")
    logger.info(f"  Total Faces Saved:           {total_faces:,}")
    logger.info(f"  Alignment Failures:          {total_alignment_failures}")
    logger.info(f"  Output Directory:            {output_root}")
    logger.info("==============================")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STAF Preprocessing: Crop & Align Faces")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--margin", type=float, help="Override face margin")
    parser.add_argument("--image_size", type=int, help="Override image size")
    parser.add_argument("--no_alignment", action="store_true", help="Disable five-point alignment")
    parser.add_argument("--num_workers", type=int, help="Override number of workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crops")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    processed_dir = Path(cfg.data.paths.processed_dir or "data/processed")
    frames_root = processed_dir / "frames"
    manifests_root = processed_dir / "manifests"
    faces_root = processed_dir / "faces"

    margin = args.margin if args.margin is not None else cfg.data.face_margin
    image_size = args.image_size if args.image_size is not None else cfg.data.image_size
    alignment = not args.no_alignment and cfg.data.face_alignment_enabled
    workers = args.num_workers or cfg.data.num_workers or 4

    logger.info(f"Frames Root:     {frames_root}")
    logger.info(f"Manifests Root:  {manifests_root}")
    logger.info(f"Faces Output:    {faces_root}")
    logger.info(f"Margin:          {margin}")
    logger.info(f"Image Size:      {image_size}")
    logger.info(f"Alignment:       {alignment}")
    logger.info(f"Workers:         {workers}")

    run_crop_faces_pipeline(
        frames_root=frames_root,
        manifests_root=manifests_root,
        output_root=faces_root,
        margin=margin,
        image_size=image_size,
        alignment_enabled=alignment,
        num_workers=workers,
        overwrite=args.overwrite,
    )
