"""
Base abstract face detector class defining the interface for all pluggable backends.

Every face detector implementation in STAF must inherit from this class
and implement the abstract `detect_faces` method, returning a list of Dicts
matching the standard detection schema.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List

import numpy as np


class BaseFaceDetector(abc.ABC):
    """
    Abstract base class for all face detection backends.
    """

    @abc.abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects all faces in a given frame.

        Args:
            image: A numpy array representing the image (BGR format from cv2).

        Returns:
            A list of dictionaries, one per detected face.
            Each dictionary MUST conform to the following schema:
            {
                "bbox": [x1, y1, x2, y2],  # coordinates in pixels
                "landmarks": {              # facial keypoints
                    "left_eye": [x, y],
                    "right_eye": [x, y],
                    "nose": [x, y],
                    "mouth_left": [x, y],
                    "mouth_right": [x, y]
                },
                "confidence": float        # detector confidence score (0.0 to 1.0)
            }
        """
        pass
