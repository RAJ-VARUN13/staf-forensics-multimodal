"""
YOLO Face detector stub implementation.

YOLOFace is a high-speed face detection framework using YOLO architectures.
Users can implement their custom model weights using this pluggable class.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from staf.preprocessing.detectors.base import BaseFaceDetector


class YOLOFaceDetector(BaseFaceDetector):
    """
    YOLO Face Detector implementation placeholder.
    """

    def __init__(self, model_path: str = "") -> None:
        """
        Initializes the YOLOFace detector.
        """
        self.model_path = model_path
        raise NotImplementedError(
            "YOLO Face Detector is currently a stub. To use YOLOFace:\n"
            "1. Install ultralytics/yolov8 or custom YOLOFace repositories.\n"
            "2. Implement detect_faces() to run inference using YOLO face weights."
        )

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in BGR image using YOLO Face.
        """
        return []
