"""
STAF Evaluation Module: Metric Calculators.

Calculates key classification metrics for deepfake detection, including
Accuracy, Precision, Recall, F1, and threshold-independent metrics (ROC-AUC, PR-AUC).

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc

from staf.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes standard evaluation metrics from predictions and target arrays.

    Args:
        targets: 1D numpy array of binary ground truth labels (0.0 or 1.0).
        probabilities: 1D numpy array of prediction probabilities in [0.0, 1.0].
        threshold: Decision boundary threshold to classify as positive (fake).

    Returns:
        A dictionary containing computed metrics.
    """
    # Convert probabilities to binary predictions
    binary_preds = (probabilities >= threshold).astype(int)

    metrics = {}

    # Ensure inputs are flat 1D arrays
    targets_flat = targets.ravel()
    probs_flat = probabilities.ravel()
    binary_flat = binary_preds.ravel()

    # Core threshold-dependent metrics
    metrics["accuracy"] = float(accuracy_score(targets_flat, binary_flat))
    metrics["precision"] = float(precision_score(targets_flat, binary_flat, zero_division=0))
    metrics["recall"] = float(recall_score(targets_flat, binary_flat, zero_division=0))
    metrics["f1"] = float(f1_score(targets_flat, binary_flat, zero_division=0))

    # Threshold-independent metrics
    # ROC-AUC requires samples from both classes to be present
    if len(np.unique(targets_flat)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(targets_flat, probs_flat))
    else:
        logger.warning("ROC-AUC calculation requires samples from both target classes. Setting to 0.5.")
        metrics["roc_auc"] = 0.5

    # PR-AUC (Area under Precision-Recall Curve)
    try:
        if len(np.unique(targets_flat)) > 1:
            p, r, _ = precision_recall_curve(targets_flat, probs_flat)
            metrics["pr_auc"] = float(auc(r, p))
        else:
            metrics["pr_auc"] = 0.0
    except Exception as e:
        logger.error(f"Error calculating PR-AUC: {e}")
        metrics["pr_auc"] = 0.0

    return metrics
