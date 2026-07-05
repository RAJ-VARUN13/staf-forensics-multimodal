"""
STAF Evaluation Module: Evaluator Pipeline.

Runs model evaluation on test sets and cross-dataset benchmarks, computes
overall and per-category metrics, generates performance visualizations,
and exports structured JSON reports.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from staf.configs.schema import STAFConfig
from staf.evaluation.metrics import calculate_metrics
from staf.evaluation.visualizer import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from staf.utils.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Manages evaluation runs on test splits and cross-dataset files.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        cfg: STAFConfig,
        output_dir: Path,
    ) -> None:
        """
        Args:
            model: Active trained PyTorch model module.
            device: Active device (CPU or CUDA).
            cfg: Top-level configuration object.
            output_dir: Destination folder under results/ for metrics and plots.
        """
        self.model = model
        self.device = device
        self.cfg = cfg
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        self.model.eval()

    def get_fine_grained_category(self, path_str: str) -> str:
        """Helper to extract FakeAVCeleb directory categories from video paths."""
        path_str = path_str.replace("\\", "/")
        categories = [
            "RealVideo-RealAudio",
            "FakeVideo-RealAudio",
            "RealVideo-FakeAudio",
            "FakeVideo-FakeAudio"
        ]
        for cat in categories:
            if cat in path_str:
                return cat
        return "Unknown"

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str = "test") -> Dict[str, Any]:
        """
        Runs evaluation on a DataLoader, computes overall/per-class metrics, and saves outputs.

        Args:
            loader: The evaluation DataLoader.
            split_name: Name of split (e.g. "test", "cross_dataset_prof").

        Returns:
            Dictionary containing computed metrics and per-class reports.
        """
        logger.info(f"Running evaluation on split: {split_name} ({len(loader.dataset)} samples)...")

        all_probs: List[float] = []
        all_targets: List[float] = []
        all_categories: List[str] = []
        all_video_ids: List[str] = []

        use_amp = self.cfg.training.use_amp and self.device.type == "cuda"

        pbar = tqdm(loader, desc=f"Evaluating [{split_name}]")
        for faces, audio, labels, metadata in pbar:
            faces = faces.to(self.device, non_blocking=True)
            audio = audio.to(self.device, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = self.model(faces, audio)
            
            # Convert logits to probabilities
            probs = torch.sigmoid(logits).view(-1).cpu().numpy().tolist()
            targets = labels.view(-1).cpu().numpy().tolist()

            all_probs.extend(probs)
            all_targets.extend(targets)
            all_video_ids.extend(metadata["video_id"])
            
            # Categorization helper
            for raw_path in metadata["raw_video_path"]:
                all_categories.append(self.get_fine_grained_category(raw_path))

        probs_arr = np.array(all_probs)
        targets_arr = np.array(all_targets)
        binary_preds = (probs_arr >= self.cfg.evaluation.binary_threshold).astype(int)

        # 1. Compute overall metrics
        overall_metrics = calculate_metrics(
            targets=targets_arr,
            probabilities=probs_arr,
            threshold=self.cfg.evaluation.binary_threshold
        )

        # 2. Compute per-category accuracy (e.g. RealVideo-RealAudio vs Lip-sync Fake)
        unique_categories = sorted(list(set(all_categories)))
        category_metrics: Dict[str, Dict[str, Any]] = {}

        for cat in unique_categories:
            indices = [i for i, c in enumerate(all_categories) if c == cat]
            if not indices:
                continue
            cat_targets = targets_arr[indices]
            cat_preds = binary_preds[indices]
            
            acc = float(np.mean(cat_targets == cat_preds))
            category_metrics[cat] = {
                "count": len(indices),
                "accuracy": acc
            }

        # 3. Compile full report
        report = {
            "experiment_name": self.cfg.experiment_name,
            "split_name": split_name,
            "overall_metrics": overall_metrics,
            "category_metrics": category_metrics,
        }

        # Save JSON metrics report
        report_path = self.output_dir / f"metrics_{split_name}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved metrics report to: {report_path}")

        # 4. Generate visual plots if configured
        if self.cfg.evaluation.save_plots:
            plot_confusion_matrix(
                targets=targets_arr,
                binary_preds=binary_preds,
                output_path=self.output_dir / f"confusion_matrix_{split_name}.png",
                title=f"Confusion Matrix ({split_name.capitalize()})"
            )
            plot_roc_curve(
                targets=targets_arr,
                probabilities=probs_arr,
                output_path=self.output_dir / f"roc_curve_{split_name}.png",
                auc_score=overall_metrics.get("roc_auc", 0.5),
                title=f"ROC Curve ({split_name.capitalize()})"
            )
            plot_precision_recall_curve(
                targets=targets_arr,
                probabilities=probs_arr,
                output_path=self.output_dir / f"pr_curve_{split_name}.png",
                pr_auc_score=overall_metrics.get("pr_auc", 0.0),
                title=f"Precision-Recall Curve ({split_name.capitalize()})"
            )
            logger.info(f"Generated evaluation plots in: {self.output_dir}")

        return report
