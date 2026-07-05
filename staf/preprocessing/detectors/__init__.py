"""
Face Detector backend registry and factory.

Exports the BaseFaceDetector interface and provides the factory function
`get_face_detector` to instantiate configured backends at runtime.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

from staf.preprocessing.detectors.base import BaseFaceDetector
from staf.preprocessing.detectors.retinaface_detector import RetinaFaceDetector
from staf.preprocessing.detectors.scrfd_detector import SCRFDDetector
from staf.preprocessing.detectors.yoloface_detector import YOLOFaceDetector


def get_face_detector(detector_name: str, **kwargs) -> BaseFaceDetector:
    """
    Factory function to instantiate the configured face detection backend.

    Args:
        detector_name: Configured name of the detector ("retinaface", "scrfd", "yoloface").
        **kwargs: Arguments passed directly to the detector constructor.

    Returns:
        An instance of BaseFaceDetector.
    """
    name_lower = detector_name.lower()
    
    if name_lower == "retinaface":
        return RetinaFaceDetector(**kwargs)
    elif name_lower == "scrfd":
        return SCRFDDetector(**kwargs)
    elif name_lower == "yoloface":
        return YOLOFaceDetector(**kwargs)
    else:
        raise ValueError(
            f"Unsupported face detector backend: {detector_name}. "
            f"Supported options: 'retinaface', 'scrfd', 'yoloface'."
        )
