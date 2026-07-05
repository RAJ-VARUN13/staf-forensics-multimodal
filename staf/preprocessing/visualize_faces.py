"""
STAF Preprocessing Utility: Face Detection Visualization.

Overlays detected bounding boxes, confidence scores, and facial landmark coordinates
onto the extracted video frames. Optionally compiles the annotated frames into an
MP4 video for human review and debugging.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np

from staf.configs.schema import load_config
from staf.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def render_detections_on_image(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Draws bounding boxes and landmarks for all detections on a single frame.

    Args:
        image: Frame image in BGR format.
        detections: List of detection items for this frame.

    Returns:
        BGR image with rendered annotations.
    """
    annotated = image.copy()
    
    # Draw annotations
    for face in detections:
        bbox = face.get("bbox", [])
        landmarks = face.get("landmarks", {})
        confidence = face.get("confidence", 0.0)
        
        # Draw bounding box
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            # Draw box in Green
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Write confidence score
            label_text = f"{confidence:.3f}"
            cv2.putText(
                annotated,
                label_text,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw landmarks
        # landmarks scheme: {"left_eye": [x, y], "right_eye": [x, y], ...}
        # Draw each landmark in Blue
        colors = {
            "left_eye": (255, 0, 0),
            "right_eye": (255, 0, 0),
            "nose": (0, 0, 255),
            "mouth_left": (0, 255, 255),
            "mouth_right": (0, 255, 255)
        }
        for name, pt in landmarks.items():
            if len(pt) == 2:
                x, y = map(int, pt)
                color = colors.get(name, (255, 0, 255))
                cv2.circle(annotated, (x, y), 3, color, -1)

    return annotated


def visualize_video_detections(
    frames_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    compile_video: bool = True,
    fps: float = 30.0,
) -> None:
    """
    Overlays face detection manifest coordinates onto frame images.
    """
    if not frames_dir.exists():
        logger.error(f"Frames directory does not exist: {frames_dir}")
        return

    if not manifest_path.exists():
        logger.error(f"Face manifest does not exist: {manifest_path}")
        return

    # Load manifest data
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)

    # Group detections by frame_id
    detections_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for det in manifest_data.get("detections", []):
        f_id = det.get("frame_id")
        if f_id is not None:
            if f_id not in detections_by_frame:
                detections_by_frame[f_id] = []
            detections_by_frame[f_id].append(det)

    frame_files = sorted(list(frames_dir.glob("frame_*.jpg")))
    num_frames = len(frame_files)
    if num_frames == 0:
        logger.error(f"No frames found to annotate in {frames_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Visualizing {num_frames} frames to {output_dir}...")

    video_writer: cv2.VideoWriter | None = None
    width, height = 0, 0

    for idx, frame_path in enumerate(frame_files):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        h, w, _ = img.shape
        if idx == 0:
            height, width = h, w
            if compile_video:
                video_out_path = output_dir.parent / f"{frames_dir.name}_annotated.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))
                logger.info(f"Writing annotated video to {video_out_path}")

        frame_dets = detections_by_frame.get(idx, [])
        annotated_img = render_detections_on_image(img, frame_dets)

        # Save annotated image
        out_frame_path = output_dir / frame_path.name
        cv2.imwrite(str(out_frame_path), annotated_img)

        # Write to video
        if video_writer is not None:
            video_writer.write(annotated_img)

    if video_writer is not None:
        video_writer.release()
        
    logger.info("Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STAF Preprocessing: Visualize Face Detections")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--video_name", type=str, required=True, help="Video name (folder stem)")
    parser.add_argument("--no_video", action="store_true", help="Skip compiling annotated frames to MP4 video")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    processed_dir = Path(cfg.data.paths.processed_dir or "data/processed")
    frames_dir = processed_dir / "frames" / args.video_name
    manifest_path = processed_dir / "manifests" / f"{args.video_name}_manifest.json"
    output_dir = processed_dir / "visualizations" / args.video_name

    logger.info(f"Visualizing Video:   {args.video_name}")
    logger.info(f"Frames Source:      {frames_dir}")
    logger.info(f"Manifest Path:      {manifest_path}")
    logger.info(f"Output Directory:   {output_dir}")

    visualize_video_detections(
        frames_dir=frames_dir,
        manifest_path=manifest_path,
        output_dir=output_dir,
        compile_video=not args.no_video
    )
