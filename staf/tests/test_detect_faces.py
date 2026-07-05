"""
Unit tests for STAF Preprocessing Stage 2: Face Detection & Visualization.

Tests the pluggable detector interfaces, manifest schema compliance,
resume verification, and stats logger.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pytest

from staf.preprocessing.detectors.base import BaseFaceDetector
from staf.preprocessing.detect_faces import (
    detect_faces_in_video_frames,
    run_face_detection_pipeline,
)
from staf.preprocessing.visualize_faces import (
    visualize_video_detections,
    render_detections_on_image,
)


class MockFaceDetector(BaseFaceDetector):
    """
    Mock face detector returning fixed bounding boxes and landmarks for tests.
    """

    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        return [
            {
                "bbox": [10, 20, 80, 90],
                "landmarks": {
                    "left_eye": [30, 40],
                    "right_eye": [60, 40],
                    "nose": [45, 55],
                    "mouth_left": [35, 70],
                    "mouth_right": [55, 70]
                },
                "confidence": self.confidence
            }
        ]


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_detect_faces_in_video_frames(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies face detection manifest generation and schema compliance."""
    frames_dir = temp_dir / "test_video"
    frames_dir.mkdir()
    
    # Save 3 dummy images
    for i in range(3):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)
        
    manifest_path = temp_dir / "test_video_manifest.json"
    
    # Mock the get_face_detector factory inside detect_faces module
    monkeypatch.setattr(
        "staf.preprocessing.detect_faces.get_face_detector",
        lambda name, **kwargs: MockFaceDetector()
    )
    
    success, stats = detect_faces_in_video_frames(
        frames_dir=frames_dir,
        manifest_path=manifest_path,
        detector_name="mock_detector",
        detector_kwargs={},
        fps=10.0,
        overwrite=True
    )
    
    assert success is True
    assert stats["frames_processed"] == 3
    assert stats["faces_detected"] == 3
    assert stats["failed_frames"] == 0
    assert stats["status"] == "SUCCESS"
    
    # Read manifest and verify schema
    assert manifest_path.exists()
    with open(manifest_path, "r") as f:
        data = json.load(f)
        
    assert data["video_name"] == "test_video"
    assert "stats" in data
    assert len(data["detections"]) == 3
    
    # Verify rich schema keys on first detection
    det = data["detections"][0]
    assert det["frame_id"] == 0
    assert det["bbox"] == [10, 20, 80, 90]
    assert "landmarks" in det
    assert det["landmarks"]["left_eye"] == [30, 40]
    assert det["confidence"] == 0.95
    assert det["detector"] == "mock_detector"
    assert det["timestamp"] == 0.0  # index 0 / 10fps
    assert det["image_width"] == 100
    assert det["image_height"] == 100


def test_detect_faces_resume_logic(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that completed manifests are skipped unless overwrite=True."""
    frames_dir = temp_dir / "test_video"
    frames_dir.mkdir()
    cv2.imwrite(str(frames_dir / "frame_000000.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))
    
    manifest_path = temp_dir / "test_video_manifest.json"
    
    # Mock factory
    monkeypatch.setattr(
        "staf.preprocessing.detect_faces.get_face_detector",
        lambda name, **kwargs: MockFaceDetector()
    )
    
    # First run (creates manifest)
    success1, stats1 = detect_faces_in_video_frames(
        frames_dir=frames_dir,
        manifest_path=manifest_path,
        detector_name="mock_detector",
        detector_kwargs={},
        overwrite=False
    )
    assert success1 is True
    assert stats1["status"] == "SUCCESS"
    
    # Edit manifest to verify we don't rewrite it
    with open(manifest_path, "w") as f:
        json.dump({"detections": [], "stats": {"frames_processed": 0, "faces_detected": 0}}, f)
        
    # Second run with overwrite=False (should load and skip)
    success2, stats2 = detect_faces_in_video_frames(
        frames_dir=frames_dir,
        manifest_path=manifest_path,
        detector_name="mock_detector",
        detector_kwargs={},
        overwrite=False
    )
    assert success2 is True
    assert stats2["status"] == "SKIPPED"


def test_visualize_face_detections(temp_dir: Path) -> None:
    """Tests overlays and annotated video file compiling."""
    frames_dir = temp_dir / "test_video"
    frames_dir.mkdir()
    cv2.imwrite(str(frames_dir / "frame_000000.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))
    
    manifest_path = temp_dir / "test_video_manifest.json"
    manifest_data = {
        "video_name": "test_video",
        "stats": {},
        "detections": [
            {
                "frame_id": 0,
                "bbox": [10, 20, 80, 90],
                "landmarks": {
                    "left_eye": [30, 40],
                    "right_eye": [60, 40]
                },
                "confidence": 0.99
            }
        ]
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)
        
    output_dir = temp_dir / "vis_output"
    
    # Run visualization
    visualize_video_detections(
        frames_dir=frames_dir,
        manifest_path=manifest_path,
        output_dir=output_dir,
        compile_video=True,
        fps=10.0
    )
    
    # Assert annotated frame and compiled video exist
    assert (output_dir / "frame_000000.jpg").exists()
    assert (output_dir.parent / "test_video_annotated.mp4").exists()
