"""
STAF Preprocessing Validation: Manifest Integrity Checker.

Runs comprehensive checks on face detection manifests to catch data issues
BEFORE they propagate into cropping, dataset construction, and training.

Checks performed per manifest:
    1. Missing frames (frames in directory but not in manifest, or vice versa)
    2. Broken/corrupted JSON
    3. Negative bounding box coordinates
    4. Bounding boxes outside image boundaries
    5. Landmark coordinates outside image boundaries
    6. Confidence below threshold
    7. Duplicate frame IDs
    8. Timestamp monotonicity (timestamps should be non-decreasing)

Produces a structured preprocessing report summarizing all issues found.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from staf.configs.schema import load_config
from staf.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


# =============================================================================
# Validation Result Types
# =============================================================================

@dataclass
class ValidationIssue:
    """A single validation issue found in a manifest."""

    video_name: str
    check_name: str
    severity: str  # "ERROR", "WARNING"
    message: str
    frame_id: Optional[int] = None


@dataclass
class VideoValidationResult:
    """Validation result for a single video manifest."""

    video_name: str
    manifest_path: str
    valid: bool
    frames_in_dir: int = 0
    frames_in_manifest: int = 0
    total_detections: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Aggregated validation report across all manifests."""

    total_videos: int = 0
    valid_videos: int = 0
    invalid_videos: int = 0
    skipped_videos: int = 0

    total_frames_on_disk: int = 0
    total_frames_in_manifests: int = 0
    total_detections: int = 0

    total_errors: int = 0
    total_warnings: int = 0

    issues_by_check: Dict[str, int] = field(default_factory=dict)
    video_results: List[VideoValidationResult] = field(default_factory=list)


# =============================================================================
# Per-Manifest Validation
# =============================================================================

def validate_manifest(
    manifest_path: Path,
    frames_dir: Path,
    confidence_threshold: float = 0.5,
) -> VideoValidationResult:
    """
    Validates a single face detection manifest against its source frames.

    Args:
        manifest_path: Path to the JSON manifest file.
        frames_dir: Path to the corresponding extracted frames directory.
        confidence_threshold: Minimum acceptable confidence score.

    Returns:
        A VideoValidationResult with all issues found.
    """
    video_name = manifest_path.stem.replace("_manifest", "")
    result = VideoValidationResult(
        video_name=video_name,
        manifest_path=str(manifest_path),
        valid=True,
    )

    # --- Check 1: JSON integrity ---
    try:
        with open(manifest_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result.valid = False
        result.issues.append(ValidationIssue(
            video_name=video_name,
            check_name="broken_json",
            severity="ERROR",
            message=f"Corrupted JSON: {e}",
        ))
        return result
    except Exception as e:
        result.valid = False
        result.issues.append(ValidationIssue(
            video_name=video_name,
            check_name="file_read_error",
            severity="ERROR",
            message=f"Cannot read manifest: {e}",
        ))
        return result

    detections = data.get("detections", [])
    result.total_detections = len(detections)

    # --- Check 2: Missing frames (on disk vs manifest) ---
    if frames_dir.exists():
        frame_files = sorted(list(frames_dir.glob("frame_*.jpg")))
        result.frames_in_dir = len(frame_files)
        disk_frame_ids: Set[int] = set()
        for f in frame_files:
            try:
                # Parse frame_000042.jpg -> 42
                fid = int(f.stem.split("_")[1])
                disk_frame_ids.add(fid)
            except (IndexError, ValueError):
                pass
    else:
        result.frames_in_dir = 0
        disk_frame_ids = set()
        result.issues.append(ValidationIssue(
            video_name=video_name,
            check_name="missing_frames_dir",
            severity="ERROR",
            message=f"Frames directory not found: {frames_dir}",
        ))
        result.valid = False

    manifest_frame_ids: Set[int] = set()
    for det in detections:
        fid = det.get("frame_id")
        if fid is not None:
            manifest_frame_ids.add(fid)

    result.frames_in_manifest = len(manifest_frame_ids)

    # Frames on disk but not in manifest
    missing_from_manifest = disk_frame_ids - manifest_frame_ids
    if missing_from_manifest:
        result.issues.append(ValidationIssue(
            video_name=video_name,
            check_name="frames_missing_from_manifest",
            severity="WARNING",
            message=f"{len(missing_from_manifest)} frames on disk have no detections in manifest "
                    f"(could be no-face frames). First few: {sorted(missing_from_manifest)[:5]}",
        ))

    # Frames in manifest but not on disk
    missing_from_disk = manifest_frame_ids - disk_frame_ids
    if missing_from_disk:
        result.issues.append(ValidationIssue(
            video_name=video_name,
            check_name="frames_missing_from_disk",
            severity="ERROR",
            message=f"{len(missing_from_disk)} frame IDs in manifest not found on disk: "
                    f"{sorted(missing_from_disk)[:5]}",
        ))
        result.valid = False

    # --- Per-detection checks ---
    seen_frame_ids_for_dup: Dict[int, int] = {}
    prev_timestamp: float = -1.0

    for det_idx, det in enumerate(detections):
        frame_id = det.get("frame_id")
        bbox = det.get("bbox", [])
        landmarks = det.get("landmarks", {})
        confidence = det.get("confidence", 0.0)
        timestamp = det.get("timestamp", 0.0)
        img_w = det.get("image_width", 0)
        img_h = det.get("image_height", 0)

        # --- Check 3: Negative bounding box coordinates ---
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            if any(v < 0 for v in [x1, y1, x2, y2]):
                result.issues.append(ValidationIssue(
                    video_name=video_name,
                    check_name="negative_bbox",
                    severity="ERROR",
                    message=f"Negative bbox coordinate: {bbox}",
                    frame_id=frame_id,
                ))
                result.valid = False

            # --- Check 4: Bounding box outside image boundaries ---
            if img_w > 0 and img_h > 0:
                if x1 > img_w or x2 > img_w or y1 > img_h or y2 > img_h:
                    result.issues.append(ValidationIssue(
                        video_name=video_name,
                        check_name="bbox_out_of_bounds",
                        severity="WARNING",
                        message=f"Bbox {bbox} exceeds image dimensions ({img_w}x{img_h})",
                        frame_id=frame_id,
                    ))

            # Check bbox is well-formed (x2 > x1, y2 > y1)
            if x2 <= x1 or y2 <= y1:
                result.issues.append(ValidationIssue(
                    video_name=video_name,
                    check_name="malformed_bbox",
                    severity="ERROR",
                    message=f"Malformed bbox (x2<=x1 or y2<=y1): {bbox}",
                    frame_id=frame_id,
                ))
                result.valid = False

        # --- Check 5: Landmark coordinates outside image ---
        if img_w > 0 and img_h > 0:
            for lm_name, lm_coords in landmarks.items():
                if len(lm_coords) == 2:
                    lx, ly = lm_coords
                    if lx < 0 or ly < 0 or lx > img_w or ly > img_h:
                        result.issues.append(ValidationIssue(
                            video_name=video_name,
                            check_name="landmark_out_of_bounds",
                            severity="WARNING",
                            message=f"Landmark '{lm_name}' at ({lx},{ly}) outside "
                                    f"image ({img_w}x{img_h})",
                            frame_id=frame_id,
                        ))

        # --- Check 6: Low confidence ---
        if confidence < confidence_threshold:
            result.issues.append(ValidationIssue(
                video_name=video_name,
                check_name="low_confidence",
                severity="WARNING",
                message=f"Confidence {confidence:.4f} below threshold {confidence_threshold}",
                frame_id=frame_id,
            ))

        # --- Check 7: Duplicate frame IDs ---
        if frame_id is not None:
            seen_frame_ids_for_dup[frame_id] = seen_frame_ids_for_dup.get(frame_id, 0) + 1

        # --- Check 8: Timestamp monotonicity ---
        if timestamp < prev_timestamp:
            result.issues.append(ValidationIssue(
                video_name=video_name,
                check_name="non_monotonic_timestamp",
                severity="WARNING",
                message=f"Timestamp {timestamp:.4f} at frame {frame_id} is less than "
                        f"previous {prev_timestamp:.4f}",
                frame_id=frame_id,
            ))
        prev_timestamp = timestamp

    # Aggregate duplicate check (frame IDs with >1 detection are EXPECTED for multi-face)
    # but flag it as INFO for awareness
    multi_face_frames = {fid: cnt for fid, cnt in seen_frame_ids_for_dup.items() if cnt > 1}
    if multi_face_frames:
        max_faces = max(multi_face_frames.values())
        result.issues.append(ValidationIssue(
            video_name=video_name,
            check_name="multi_face_frames",
            severity="WARNING",
            message=f"{len(multi_face_frames)} frames have multiple face detections "
                    f"(max {max_faces} faces in a single frame). This is normal for multi-person scenes.",
        ))

    return result


# =============================================================================
# Pipeline: Validate All Manifests
# =============================================================================

def run_validation_pipeline(
    manifests_dir: Path,
    frames_root: Path,
    confidence_threshold: float = 0.5,
) -> ValidationReport:
    """
    Validates all manifests in a directory against their corresponding frame folders.

    Args:
        manifests_dir: Directory containing *_manifest.json files.
        frames_root: Root directory containing video frame subfolders.
        confidence_threshold: Minimum acceptable confidence score.

    Returns:
        A ValidationReport summarizing all findings.
    """
    report = ValidationReport()

    manifest_files = sorted(list(manifests_dir.glob("*_manifest.json")))
    report.total_videos = len(manifest_files)

    if report.total_videos == 0:
        logger.warning(f"No manifest files found in {manifests_dir}")
        return report

    logger.info(f"Validating {report.total_videos} manifests...")

    with tqdm(total=report.total_videos, desc="Validating manifests") as pbar:
        for manifest_path in manifest_files:
            video_name = manifest_path.stem.replace("_manifest", "")
            frames_dir = frames_root / video_name

            result = validate_manifest(
                manifest_path=manifest_path,
                frames_dir=frames_dir,
                confidence_threshold=confidence_threshold,
            )

            report.video_results.append(result)
            report.total_frames_on_disk += result.frames_in_dir
            report.total_frames_in_manifests += result.frames_in_manifest
            report.total_detections += result.total_detections

            if result.valid:
                report.valid_videos += 1
            else:
                report.invalid_videos += 1

            for issue in result.issues:
                if issue.severity == "ERROR":
                    report.total_errors += 1
                else:
                    report.total_warnings += 1
                report.issues_by_check[issue.check_name] = (
                    report.issues_by_check.get(issue.check_name, 0) + 1
                )

            pbar.update(1)

    return report


def print_validation_report(report: ValidationReport) -> None:
    """Prints a formatted validation report to the logger."""
    logger.info("=" * 55)
    logger.info("       STAF PREPROCESSING VALIDATION REPORT")
    logger.info("=" * 55)
    logger.info(f"  Videos Validated:            {report.total_videos}")
    logger.info(f"  Valid Videos:                {report.valid_videos}")
    logger.info(f"  Invalid Videos:              {report.invalid_videos}")
    logger.info(f"  Total Frames on Disk:        {report.total_frames_on_disk:,}")
    logger.info(f"  Total Frames in Manifests:   {report.total_frames_in_manifests:,}")
    logger.info(f"  Total Face Detections:       {report.total_detections:,}")
    if report.total_frames_on_disk > 0:
        avg = report.total_detections / report.total_frames_on_disk
        logger.info(f"  Avg Faces Per Frame:         {avg:.3f}")
    logger.info(f"  Total Errors:                {report.total_errors}")
    logger.info(f"  Total Warnings:              {report.total_warnings}")
    logger.info("-" * 55)

    if report.issues_by_check:
        logger.info("  Issues by Check:")
        for check, count in sorted(report.issues_by_check.items()):
            logger.info(f"    {check:35s} {count}")
    else:
        logger.info("  No issues found. All manifests are clean.")

    logger.info("=" * 55)


def save_validation_report(report: ValidationReport, output_path: Path) -> None:
    """Saves the validation report to a JSON file for programmatic use."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "total_videos": report.total_videos,
        "valid_videos": report.valid_videos,
        "invalid_videos": report.invalid_videos,
        "total_frames_on_disk": report.total_frames_on_disk,
        "total_frames_in_manifests": report.total_frames_in_manifests,
        "total_detections": report.total_detections,
        "total_errors": report.total_errors,
        "total_warnings": report.total_warnings,
        "issues_by_check": report.issues_by_check,
        "invalid_video_details": [
            {
                "video_name": r.video_name,
                "issues": [
                    {"check": i.check_name, "severity": i.severity, "message": i.message, "frame_id": i.frame_id}
                    for i in r.issues if i.severity == "ERROR"
                ]
            }
            for r in report.video_results if not r.valid
        ]
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Validation report saved to: {output_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STAF Preprocessing: Validate Face Detection Manifests")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Minimum acceptable face detection confidence")
    parser.add_argument("--save_report", type=str, default="",
                        help="Path to save JSON validation report")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    processed_dir = Path(cfg.data.paths.processed_dir or "data/processed")
    manifests_dir = processed_dir / "manifests"
    frames_root = processed_dir / "frames"

    report = run_validation_pipeline(
        manifests_dir=manifests_dir,
        frames_root=frames_root,
        confidence_threshold=args.confidence_threshold,
    )

    print_validation_report(report)

    if args.save_report:
        save_validation_report(report, Path(args.save_report))
