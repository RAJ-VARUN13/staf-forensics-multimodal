"""
Unit tests for STAF Preprocessing: Manifest Validation and Face Cropping.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from staf.preprocessing.validate_manifest import (
    validate_manifest,
    run_validation_pipeline,
    ValidationReport,
)
from staf.preprocessing.crop_faces import (
    align_face_by_eyes,
    transform_point,
    crop_face_with_margin,
    crop_faces_for_video,
)


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _make_frame(path: Path, w: int = 200, h: int = 200) -> None:
    """Helper: save a blank frame image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_manifest(
    path: Path,
    video_name: str,
    detections: list,
) -> None:
    """Helper: write a manifest JSON."""
    data = {
        "video_name": video_name,
        "stats": {"frames_processed": len(detections), "faces_detected": len(detections)},
        "detections": detections,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidateManifest:

    def test_valid_manifest_passes(self, temp_dir: Path) -> None:
        """A well-formed manifest with matching frames should pass all checks."""
        frames_dir = temp_dir / "video_A"
        frames_dir.mkdir()
        for i in range(3):
            _make_frame(frames_dir / f"frame_{i:06d}.jpg")

        manifest_path = temp_dir / "video_A_manifest.json"
        dets = [
            {
                "frame_id": i,
                "bbox": [10, 20, 80, 90],
                "landmarks": {"left_eye": [30, 40], "right_eye": [60, 40]},
                "confidence": 0.98,
                "detector": "mock",
                "timestamp": i * 0.1,
                "image_width": 200,
                "image_height": 200,
            }
            for i in range(3)
        ]
        _make_manifest(manifest_path, "video_A", dets)

        result = validate_manifest(manifest_path, frames_dir)
        assert result.valid is True
        # Only possible issue is multi_face_frames (INFO-level) or missing frames (WARNING)
        errors = [i for i in result.issues if i.severity == "ERROR"]
        assert len(errors) == 0

    def test_broken_json(self, temp_dir: Path) -> None:
        """Corrupted JSON should be flagged as ERROR."""
        manifest_path = temp_dir / "bad_manifest.json"
        manifest_path.write_text("{broken json!!!")

        result = validate_manifest(manifest_path, temp_dir / "nonexistent")
        assert result.valid is False
        assert any(i.check_name == "broken_json" for i in result.issues)

    def test_negative_bbox(self, temp_dir: Path) -> None:
        """Negative bounding box coordinates should be flagged."""
        frames_dir = temp_dir / "video_B"
        frames_dir.mkdir()
        _make_frame(frames_dir / "frame_000000.jpg")

        manifest_path = temp_dir / "video_B_manifest.json"
        _make_manifest(manifest_path, "video_B", [
            {
                "frame_id": 0, "bbox": [-5, 20, 80, 90],
                "landmarks": {}, "confidence": 0.9,
                "timestamp": 0.0, "image_width": 200, "image_height": 200,
            }
        ])

        result = validate_manifest(manifest_path, frames_dir)
        assert result.valid is False
        assert any(i.check_name == "negative_bbox" for i in result.issues)

    def test_bbox_out_of_bounds(self, temp_dir: Path) -> None:
        """Bounding box exceeding image dimensions should produce a warning."""
        frames_dir = temp_dir / "video_C"
        frames_dir.mkdir()
        _make_frame(frames_dir / "frame_000000.jpg", w=100, h=100)

        manifest_path = temp_dir / "video_C_manifest.json"
        _make_manifest(manifest_path, "video_C", [
            {
                "frame_id": 0, "bbox": [10, 10, 150, 150],
                "landmarks": {}, "confidence": 0.9,
                "timestamp": 0.0, "image_width": 100, "image_height": 100,
            }
        ])

        result = validate_manifest(manifest_path, frames_dir)
        assert any(i.check_name == "bbox_out_of_bounds" for i in result.issues)

    def test_low_confidence_warning(self, temp_dir: Path) -> None:
        """Low confidence detections should trigger a warning."""
        frames_dir = temp_dir / "video_D"
        frames_dir.mkdir()
        _make_frame(frames_dir / "frame_000000.jpg")

        manifest_path = temp_dir / "video_D_manifest.json"
        _make_manifest(manifest_path, "video_D", [
            {
                "frame_id": 0, "bbox": [10, 20, 80, 90],
                "landmarks": {}, "confidence": 0.1,
                "timestamp": 0.0, "image_width": 200, "image_height": 200,
            }
        ])

        result = validate_manifest(manifest_path, frames_dir, confidence_threshold=0.5)
        assert any(i.check_name == "low_confidence" for i in result.issues)

    def test_non_monotonic_timestamp(self, temp_dir: Path) -> None:
        """Non-monotonic timestamps should produce a warning."""
        frames_dir = temp_dir / "video_E"
        frames_dir.mkdir()
        for i in range(3):
            _make_frame(frames_dir / f"frame_{i:06d}.jpg")

        manifest_path = temp_dir / "video_E_manifest.json"
        _make_manifest(manifest_path, "video_E", [
            {"frame_id": 0, "bbox": [10, 20, 80, 90], "landmarks": {},
             "confidence": 0.9, "timestamp": 0.2, "image_width": 200, "image_height": 200},
            {"frame_id": 1, "bbox": [10, 20, 80, 90], "landmarks": {},
             "confidence": 0.9, "timestamp": 0.1, "image_width": 200, "image_height": 200},
            {"frame_id": 2, "bbox": [10, 20, 80, 90], "landmarks": {},
             "confidence": 0.9, "timestamp": 0.3, "image_width": 200, "image_height": 200},
        ])

        result = validate_manifest(manifest_path, frames_dir)
        assert any(i.check_name == "non_monotonic_timestamp" for i in result.issues)

    def test_malformed_bbox(self, temp_dir: Path) -> None:
        """Bounding box where x2<=x1 should be flagged as ERROR."""
        frames_dir = temp_dir / "video_F"
        frames_dir.mkdir()
        _make_frame(frames_dir / "frame_000000.jpg")

        manifest_path = temp_dir / "video_F_manifest.json"
        _make_manifest(manifest_path, "video_F", [
            {
                "frame_id": 0, "bbox": [80, 90, 10, 20],
                "landmarks": {}, "confidence": 0.9,
                "timestamp": 0.0, "image_width": 200, "image_height": 200,
            }
        ])

        result = validate_manifest(manifest_path, frames_dir)
        assert result.valid is False
        assert any(i.check_name == "malformed_bbox" for i in result.issues)


# =============================================================================
# Alignment Tests
# =============================================================================

class TestAlignment:

    def test_horizontal_eyes_no_rotation(self) -> None:
        """If eyes are already horizontal, rotation angle should be ~0."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        left_eye = [60, 100]
        right_eye = [140, 100]

        rotated, rot_mat = align_face_by_eyes(img, left_eye, right_eye)
        assert rotated.shape == img.shape

        # The rotation matrix should be close to identity
        # rot_mat[0][2] and rot_mat[1][2] are translation, check rotation part
        assert abs(rot_mat[0][0] - 1.0) < 0.01  # cos(0) ≈ 1
        assert abs(rot_mat[0][1]) < 0.01         # sin(0) ≈ 0

    def test_tilted_eyes_are_corrected(self) -> None:
        """Tilted eyes should result in a non-trivial rotation."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        left_eye = [60, 120]   # lower
        right_eye = [140, 80]  # higher → ~27 degree tilt

        rotated, rot_mat = align_face_by_eyes(img, left_eye, right_eye)
        assert rotated.shape == img.shape
        # Rotation matrix should differ from identity
        assert abs(rot_mat[0][0] - 1.0) > 0.01

    def test_transform_point(self) -> None:
        """Point transformation through identity-like matrix should be stable."""
        # Identity rotation matrix (no rotation, centered at origin)
        rot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        pt = transform_point([50, 75], rot)
        assert pt == [50, 75]


# =============================================================================
# Crop Tests
# =============================================================================

class TestCropFaces:

    def test_basic_crop_and_resize(self) -> None:
        """Verifies that crop_face_with_margin outputs correct dimensions."""
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        bbox = [30, 40, 130, 160]
        result = crop_face_with_margin(img, bbox, margin=0.2, image_size=112)
        assert result is not None
        assert result.shape == (112, 112, 3)

    def test_crop_with_zero_margin(self) -> None:
        """Zero margin should crop exactly at bbox coordinates."""
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        bbox = [50, 50, 150, 150]
        result = crop_face_with_margin(img, bbox, margin=0.0, image_size=224)
        assert result is not None
        assert result.shape == (224, 224, 3)

    def test_invalid_bbox_returns_none(self) -> None:
        """Invalid bbox (empty or reversed) should return None."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        assert crop_face_with_margin(img, [], margin=0.3, image_size=224) is None
        assert crop_face_with_margin(img, [100, 100, 50, 50], margin=0.0, image_size=224) is None

    def test_crop_faces_for_video_end_to_end(self, temp_dir: Path) -> None:
        """Full end-to-end test of cropping a video's faces from manifest."""
        frames_dir = temp_dir / "video_X"
        frames_dir.mkdir()

        # Create 2 frames with a white rectangle (fake face region)
        for i in range(2):
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(img, (30, 40), (130, 160), (255, 255, 255), -1)
            cv2.imwrite(str(frames_dir / f"frame_{i:06d}.jpg"), img)

        manifest_path = temp_dir / "video_X_manifest.json"
        _make_manifest(manifest_path, "video_X", [
            {
                "frame_id": i,
                "bbox": [30, 40, 130, 160],
                "landmarks": {
                    "left_eye": [60, 80],
                    "right_eye": [100, 80],
                    "nose": [80, 100],
                    "mouth_left": [65, 130],
                    "mouth_right": [95, 130],
                },
                "confidence": 0.95,
                "timestamp": i * 0.033,
                "image_width": 200,
                "image_height": 200,
            }
            for i in range(2)
        ])

        output_dir = temp_dir / "faces"
        success, stats = crop_faces_for_video(
            frames_dir=frames_dir,
            manifest_path=manifest_path,
            output_dir=output_dir,
            margin=0.3,
            image_size=112,
            alignment_enabled=True,
            overwrite=True,
        )

        assert success is True
        assert stats["status"] == "SUCCESS"
        assert stats["faces_saved"] == 2

        # Verify saved face files
        saved_faces = list((output_dir / "video_X").glob("face_*.jpg"))
        assert len(saved_faces) == 2

        # Verify dimensions
        for fp in saved_faces:
            face_img = cv2.imread(str(fp))
            assert face_img.shape == (112, 112, 3)
