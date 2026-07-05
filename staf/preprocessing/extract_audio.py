"""
STAF Preprocessing Stage 4: Audio Extraction.

Extracts the audio track from raw video files and resamples them to mono, 16kHz PCM WAV
as required by speech models like Wav2Vec 2.0. Supports multiprocessing, logging,
and resume/checkpointing.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from staf.configs.schema import load_config
from staf.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def extract_audio_from_video(
    video_path: Path,
    output_dir: Path,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> Tuple[bool, str]:
    """
    Extracts the audio track from a single video file, converting it to mono 16kHz PCM WAV.

    Args:
        video_path: Path to the raw input video file.
        output_dir: Root directory where the extracted audio file should be saved.
        sample_rate: Target audio sample rate (Hz).
        overwrite: If True, overwrites existing audio file.

    Returns:
        A tuple of (success_status, message).
    """
    video_name = video_path.stem
    audio_path = output_dir / f"{video_name}.wav"

    # Checkpoint/Resume verification
    if audio_path.exists() and not overwrite:
        return True, f"SKIPPED: Audio already exists at {audio_path}"

    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command
        # -y: overwrite output files without asking
        # -i: input file
        # -vn: disable video recording
        # -acodec pcm_s16le: 16-bit PCM codec
        # -ac 1: mono channel
        # -ar: sample rate
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", str(sample_rate),
            str(audio_path)
        ]

        # Run ffmpeg, suppressing output unless there is an error
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            return False, f"FAILED: ffmpeg error for {video_path}: {result.stderr.strip()}"

        # Double check file is non-empty
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            if audio_path.exists():
                audio_path.unlink()
            return False, f"FAILED: Extracted audio file is empty for {video_path}"

        return True, f"SUCCESS: Extracted audio to {audio_path}"

    except FileNotFoundError:
        return False, "FAILED: ffmpeg executable not found. Make sure ffmpeg is installed and added to PATH."
    except Exception as e:
        return False, f"ERROR: Exception during extraction of {video_path}: {str(e)}"


def run_audio_extraction_pipeline(
    raw_videos_dir: Path,
    output_audio_dir: Path,
    sample_rate: int = 16000,
    num_workers: int = 4,
    overwrite: bool = False,
) -> None:
    """
    Scans for raw videos and extracts audio using multiprocessing.

    Args:
        raw_videos_dir: Path to directory containing raw videos.
        output_audio_dir: Destination path for extracted WAV files.
        sample_rate: Target sample rate.
        num_workers: Number of parallel worker processes.
        overwrite: Overwrite existing extracted audio files.
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
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skipped_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        futures = {
            executor.submit(
                extract_audio_from_video,
                path,
                output_audio_dir,
                sample_rate,
                overwrite
            ): path
            for path in video_paths
        }

        # Progress bar
        with tqdm(total=num_videos, desc="Extracting audio") as pbar:
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

    logger.info("=== Audio Extraction Summary ===")
    logger.info(f"  Total Processed: {num_videos}")
    logger.info(f"  Successfully Extracted: {success_count}")
    logger.info(f"  Skipped (Already Existed): {skipped_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info("=================================")


if __name__ == "__main__":
    # Command line argument parser
    parser = argparse.ArgumentParser(description="STAF Preprocessing: Extract Video Audio")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--raw_dir", type=str, help="Override path to raw videos")
    parser.add_argument("--output_dir", type=str, help="Override path to output audio")
    parser.add_argument("--sample_rate", type=int, help="Override audio sample rate")
    parser.add_argument("--num_workers", type=int, help="Override number of workers")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing audio files")
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
    processed_path = Path(processed_dir_str)
    output_audio_path = processed_path / "audio"

    sample_rate = args.sample_rate or cfg.data.audio_sample_rate
    workers = args.num_workers or cfg.data.num_workers or 4

    logger.info(f"Raw Videos Path: {raw_path}")
    logger.info(f"Output Audio Path: {output_audio_path}")
    logger.info(f"Sample Rate: {sample_rate} Hz")
    logger.info(f"Workers: {workers}")

    run_audio_extraction_pipeline(
        raw_videos_dir=raw_path,
        output_audio_dir=output_audio_path,
        sample_rate=sample_rate,
        num_workers=workers,
        overwrite=args.overwrite
    )
