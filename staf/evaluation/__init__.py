"""Evaluation metrics, reporting, and benchmarking tools."""

from staf.evaluation.metrics import calculate_metrics
from staf.evaluation.visualizer import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from staf.evaluation.evaluator import Evaluator

__all__ = [
    "calculate_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "Evaluator",
]
