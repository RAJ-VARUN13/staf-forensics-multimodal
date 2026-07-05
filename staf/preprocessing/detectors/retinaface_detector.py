"""
RetinaFace implementation for the STAF face detection interface.

Uses the `retina-face` Serengil implementation or similar under the hood.
Validates inputs and converts the output to the standardized STAF schema.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from staf.preprocessing.detectors.base import BaseFaceDetector

try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False


class RetinaFaceDetector(BaseFaceDetector):
    """
    RetinaFace backend implementation.
    """

    def __init__(self, threshold: float = 0.9) -> None:
        """
        Initializes the RetinaFace detector.

        Args:
            threshold: Confidence threshold to filter out low-confidence faces.
        """
        if not RETINAFACE_AVAILABLE:
            raise ImportError(
                "RetinaFace is not installed. Please install it using:\n"
                "pip install retina-face"
            )
        self.threshold = threshold

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in BGR image using RetinaFace.

        Args:
            image: A numpy array representing the image (BGR format).

        Returns:
            List of detected face dictionaries.
        """
        # RetinaFace Serengil expects RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if hasattr(image, "shape") else image
        
        try:
            # detect_faces returns:
            # {
            #   "face_1": {
            #     "score": float,
            #     "facial_area": [x1, y1, x2, y2],
            #     "landmarks": {
            #       "left_eye": [x, y],
            #       "right_eye": [x, y],
            #       "nose": [x, y],
            #       "mouth_left": [x, y],
            #       "mouth_right": [x, y]
            #     }
            #   },
            #   ...
            # }
            raw_detections = RetinaFace.detect_faces(image_rgb)
        except Exception:
            # Handle potential failures gracefully (e.g. empty images)
            return []

        if not isinstance(raw_detections, dict):
            return []

        faces: List[Dict[str, Any]] = []

        for face_key, details in raw_detections.items():
            confidence = details.get("score", 1.0)
            if confidence < self.threshold:
                continue

            facial_area = details.get("facial_area", [])
            landmarks = details.get("landmarks", {})

            # Format to the standard STAF schema
            face_dict = {
                "bbox": [
                    int(facial_area[0]),  # x1
                    int(facial_area[1]),  # y1
                    int(facial_area[2]),  # x2
                    int(facial_area[3])   # y2
                ] if len(facial_area) == 4 else [],
                "landmarks": {
                    "left_eye": [int(v) for v in landmarks.get("left_eye", [])] if "left_eye" in landmarks else [],
                    "right_eye": [int(v) for v in landmarks.get("right_eye", [])] if "right_eye" in landmarks else [],
                    "nose": [int(v) for v in landmarks.get("nose", [])] if "nose" in landmarks else [],
                    "mouth_left": [int(v) for v in landmarks.get("mouth_left", [])] if "mouth_left" in landmarks else [],
                    "mouth_right": [int(v) for v in landmarks.get("mouth_right", [])] if "mouth_right" in landmarks else []
                },
                "confidence": float(confidence)
            }
            faces.append(face_dict)

        return faces


# Ensure OpenCV cv2 is imported inside helper functions
import cv2
