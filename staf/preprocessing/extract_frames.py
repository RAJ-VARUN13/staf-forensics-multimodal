"""
STAF Preprocessing Stage 1: Frame Extraction.

Extracts individual frame images from raw MP4 video files, saving them as sequential
JPEGs. Supports multiprocessing, logging, and smart resume/checkpointing (skipping
videos that have already been extracted).

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
from tqdm import tqdm

from staf.configs.schema import load_config
from staf.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Tuple[bool, str]:
    """
    Extracts all frames from a single video and saves them to a subfolder.

    Args:
        video_path: Path to the raw input video (.mp4).
        output_dir: Root directory where the frames subfolder should be created.
        overwrite: If True, clears existing output frames before extraction.

    Returns:
        A tuple of (success_status, message).
    """
    video_name = video_path.stem
    # The frames subfolder matches the video filename
    frames_dir = output_dir / video_name

    # Checkpoint/Resume verification
    if frames_dir.exists() and not overwrite:
        # Check if the folder contains extracted frames
        existing_frames = list(frames_dir.glob("frame_*.jpg"))
        if len(existing_frames) > 0:
            return True, f"SKIPPED: Frames already exist at {frames_dir}"

    try:
        # Create output directory
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Clear folder if overwriting
        if overwrite:
            for f in frames_dir.glob("frame_*.jpg"):
                f.unlink()

        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, f"FAILED: Could not open video {video_path}"

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame as JPEG
            frame_filename = frames_dir / f"frame_{frame_idx:06d}.jpg"
            # Use high quality JPEG encoding (95)
            cv2.imwrite(str(frame_filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            frame_idx += 1

        cap.release()

        if frame_idx == 0:
            # Clean up empty directory
            try:
                frames_dir.rmdir()
            except OSError:
                pass
            return False, f"FAILED: Zero frames extracted from {video_path}"

        return True, f"SUCCESS: Extracted {frame_idx} frames to {frames_dir}"

    except Exception as e:
        return False, f"ERROR: Exception during extraction of {video_path}: {str(e)}"


def run_frame_extraction_pipeline(
    raw_videos_dir: Path,
    output_frames_dir: Path,
    num_workers: int = 4,
    overwrite: bool = False,
) -> None:
    """
    Scans for raw videos and extracts frames using multiprocessing.

    Args:
        raw_videos_dir: Path to directory containing raw videos.
        output_frames_dir: Destination path for extracted frame subfolders.
        num_workers: Number of parallel worker processes.
        overwrite: Overwrite existing extracted frames.
    """
    logger.info(f"Scanning for videos in: {raw_videos_dir}")
    
    # Supported raw video extensions
    extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_paths: List[Path] = []
    
    # Recursive search for video files
    for ext in extensions:
        video_paths.extend(raw_videos_dir.rglob(f"*{ext}"))
        video_paths.extend(raw_videos_dir.rglob(f"*{ext.upper()}"))
        
    # Deduplicate paths (e.g. on Windows case-insensitive filesystem)
    video_paths = list(dict.fromkeys(video_paths))

    num_videos = len(video_paths)
    if num_videos == 0:
        logger.warning(f"No videos found under: {raw_videos_dir}")
        return

    logger.info(f"Found {num_videos} videos. Launching ProcessPoolExecutor with {num_workers} workers...")
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skipped_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        futures = {
            executor.submit(extract_frames_from_video, path, output_frames_dir, overwrite): path
            for path in video_paths
        }

        # Progress bar
        with tqdm(total=num_videos, desc="Extracting frames") as pbar:
            for future in as_completed(futures):
                video_path = futures[future]
                try:
                    success, msg = future.result()
                    if success:
                        if "SKIPPED" in msg:
                            skipped_count += 1
                            logger.debug(msg)
                        else:
                            success_count += 1
                            logger.info(msg)
                    else:
                        failed_count += 1
                        logger.error(msg)
                except Exception as e:
                    failed_count += 1
                    logger.error(f"EXCEPTION: Future returned exception for {video_path}: {e}")
                
                pbar.update(1)

    logger.info("=== Frame Extraction Summary ===")
    logger.info(f"  Total Processed: {num_videos}")
    logger.info(f"  Successfully Extracted: {success_count}")
    logger.info(f"  Skipped (Already Existed): {skipped_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info("=================================")


if __name__ == "__main__":
    # Command line argument parser
    parser = argparse.ArgumentParser(description="STAF Preprocessing: Extract Video Frames")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--raw_dir", type=str, help="Override path to raw videos")
    parser.add_argument("--output_dir", type=str, help="Override path to output frames")
    parser.add_argument("--num_workers", type=int, help="Override number of workers")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing frame folders")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    # Resolve paths (prioritize args, fall back to config, then defaults)
    raw_dir_str = args.raw_dir or cfg.data.paths.fakeavceleb_root
    if not raw_dir_str:
        logger.error("No input videos path specified. Set paths in config or use --raw_dir.")
        sys.exit(1)

    raw_path = Path(raw_dir_str)
    
    # If the root has a 'raw' subfolder, use it, otherwise use raw_path directly
    if (raw_path / "raw").exists():
        raw_path = raw_path / "raw"

    processed_dir_str = args.output_dir or cfg.data.paths.processed_dir or "data/processed"
    output_path = Path(processed_dir_str) / "frames"

    workers = args.num_workers or cfg.data.num_workers or 4

    logger.info(f"Source Directory: {raw_path}")
    logger.info(f"Output Directory: {output_path}")
    logger.info(f"Workers:          {workers}")
    logger.info(f"Overwrite:        {args.overwrite}")

    run_frame_extraction_pipeline(
        raw_videos_dir=raw_path,
        output_frames_dir=output_path,
        num_workers=workers,
        overwrite=args.overwrite,
    )
