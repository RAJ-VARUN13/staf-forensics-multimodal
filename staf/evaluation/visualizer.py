"""
STAF Evaluation Module: Visualizer.

Generates and saves performance visualization plots, including
Confusion Matrices, ROC Curves, and Precision-Recall (PR) Curves.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from staf.utils.logging import get_logger

logger = get_logger(__name__)


def plot_confusion_matrix(
    targets: np.ndarray,
    binary_preds: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """
    Generates and saves a confusion matrix heatmap.

    Args:
        targets: Ground truth binary labels.
        binary_preds: Model binary predictions.
        output_path: Output file destination path (saves as image).
        title: Title of the plot.
    """
    cm = confusion_matrix(targets, binary_preds)
    
    plt.figure(figsize=(6, 5))
    # Labels: 0 = Real, 1 = Fake
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        cbar=False
    )
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.debug(f"Confusion matrix plot saved to: {output_path}")


def plot_roc_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
    auc_score: float,
    title: str = "ROC Curve",
) -> None:
    """
    Generates and saves a Receiver Operating Characteristic (ROC) curve plot.

    Args:
        targets: Ground truth binary labels.
        probabilities: Predicted probabilities.
        output_path: File output destination.
        auc_score: Pre-calculated ROC-AUC value.
        title: Plot title.
    """
    if len(np.unique(targets)) <= 1:
        logger.warning("Skipping ROC curve plot since only one target class is present.")
        return

    fpr, tpr, _ = roc_curve(targets, probabilities)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(title, fontsize=13, pad=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.debug(f"ROC curve plot saved to: {output_path}")


def plot_precision_recall_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
    pr_auc_score: float,
    title: str = "Precision-Recall Curve",
) -> None:
    """
    Generates and saves a Precision-Recall (PR) curve plot.

    Args:
        targets: Ground truth binary labels.
        probabilities: Predicted probabilities.
        output_path: File output destination.
        pr_auc_score: Pre-calculated PR-AUC score.
        title: Plot title.
    """
    if len(np.unique(targets)) <= 1:
        logger.warning("Skipping PR curve plot since only one target class is present.")
        return

    precision, recall, _ = precision_recall_curve(targets, probabilities)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="forestgreen", lw=2, label=f"PR Curve (AUC = {pr_auc_score:.4f})")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title(title, fontsize=13, pad=12)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.debug(f"Precision-Recall curve plot saved to: {output_path}")
