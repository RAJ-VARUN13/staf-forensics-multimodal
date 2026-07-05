"""
Unit tests for STAF Preprocessing Stage 1: Frame Extraction.

Tests frame extraction logic, directory creation, frame verification,
resume behavior, and overwrite options using generated mock videos.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from staf.preprocessing.extract_frames import (
    extract_frames_from_video,
    run_frame_extraction_pipeline,
)


@pytest.fixture
def temp_dir() -> Path:
    """Fixture providing a temporary directory cleaned up after test run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_mock_video(path: Path, num_frames: int = 5, width: int = 100, height: int = 100) -> None:
    """Helper function to create a dummy video file with random colored noise frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    
    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
        
    out.release()


def test_extract_frames_from_video(temp_dir: Path) -> None:
    """Verifies that frame extraction successfully decodes all frames to JPEGs."""
    video_path = temp_dir / "test_video.mp4"
    output_dir = temp_dir / "processed"
    
    # Create mock video with 5 frames
    create_mock_video(video_path, num_frames=5)
    
    # Run frame extraction
    success, msg = extract_frames_from_video(video_path, output_dir)
    
    assert success is True
    assert "SUCCESS" in msg
    
    frames_dir = output_dir / "test_video"
    assert frames_dir.exists()
    
    # Check that exactly 5 frames were extracted
    frame_files = list(frames_dir.glob("frame_*.jpg"))
    assert len(frame_files) == 5
    
    # Ensure they can be successfully loaded as images
    for frame_file in frame_files:
        img = cv2.imread(str(frame_file))
        assert img is not None
        assert img.shape == (100, 100, 3)


def test_extract_frames_resume_logic(temp_dir: Path) -> None:
    """Verifies that the resume check skips videos that are already extracted."""
    video_path = temp_dir / "test_video.mp4"
    output_dir = temp_dir / "processed"
    
    create_mock_video(video_path, num_frames=5)
    
    # First run (extracts)
    success1, msg1 = extract_frames_from_video(video_path, output_dir)
    assert success1 is True
    assert "SUCCESS" in msg1
    
    # Modify one of the files to verify it's not overwritten
    frames_dir = output_dir / "test_video"
    first_frame = frames_dir / "frame_000000.jpg"
    assert first_frame.exists()
    
    with open(first_frame, "w") as f:
        f.write("modified_content")
        
    # Second run (should be skipped)
    success2, msg2 = extract_frames_from_video(video_path, output_dir, overwrite=False)
    assert success2 is True
    assert "SKIPPED" in msg2
    
    # File content should remain "modified_content"
    with open(first_frame, "r") as f:
        content = f.read()
    assert content == "modified_content"
    
    # Third run with overwrite=True (should rewrite)
    success3, msg3 = extract_frames_from_video(video_path, output_dir, overwrite=True)
    assert success3 is True
    assert "SUCCESS" in msg3
    
    # File content should be updated back to a valid image
    img = cv2.imread(str(first_frame))
    assert img is not None


def test_run_frame_extraction_pipeline(temp_dir: Path) -> None:
    """Tests the entire multiprocessing run pipeline with multiple videos."""
    raw_dir = temp_dir / "raw"
    raw_dir.mkdir()
    
    # Create 3 raw videos
    for i in range(3):
        create_mock_video(raw_dir / f"video_{i}.mp4", num_frames=3)
        
    output_dir = temp_dir / "processed"
    
    # Run the pipeline
    run_frame_extraction_pipeline(
        raw_videos_dir=raw_dir,
        output_frames_dir=output_dir,
        num_workers=2,
        overwrite=False
    )
    
    # Check that all 3 folders were created
    for i in range(3):
        frames_dir = output_dir / f"video_{i}"
        assert frames_dir.exists()
        assert len(list(frames_dir.glob("frame_*.jpg"))) == 3
