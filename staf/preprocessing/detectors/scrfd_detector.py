"""
SCRFD face detector stub implementation.

SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
is a high-efficiency face detector. Users can implement their custom model weights
using this pluggable class.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from staf.preprocessing.detectors.base import BaseFaceDetector


class SCRFDDetector(BaseFaceDetector):
    """
    SCRFD Face Detector implementation placeholder.
    """

    def __init__(self, model_path: str = "") -> None:
        """
        Initializes the SCRFD detector.
        """
        self.model_path = model_path
        # Note: True SCRFD requires ONNX Runtime and Model Zoo download
        raise NotImplementedError(
            "SCRFD Detector is currently a stub. To use SCRFD:\n"
            "1. Download SCRFD ONNX model weights.\n"
            "2. Implement ONNX runtime forward pass in detect_faces()."
        )

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in BGR image using SCRFD.
        """
        return []
