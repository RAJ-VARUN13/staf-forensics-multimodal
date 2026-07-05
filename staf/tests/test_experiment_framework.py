"""
Unit tests for STAF Training and Evaluation Framework.

Verifies loss functions, optimizers, CosineAnnealingWithWarmup scheduler,
training/validation loops (Trainer class), and evaluation pipeline (Evaluator class).

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from staf.configs.schema import STAFConfig, TrainingConfig, OptimizerConfig, SchedulerConfig, EvaluationConfig
from staf.training.losses import FocalLoss, get_loss_function
from staf.training.optimizers import CosineAnnealingWithWarmup, get_optimizer, get_scheduler
from staf.training.trainer import Trainer
from staf.evaluation.metrics import calculate_metrics
from staf.evaluation.evaluator import Evaluator


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Helper: Toy Model and Dataloaders for Testing
# =============================================================================

class ToyDetector(nn.Module):
    """Simple linear model to act as a mock model during training tests."""
    def __init__(self) -> None:
        super().__init__()
        # Visual input mock: [B, T, C, H, W] -> mean pool -> linear
        self.fc = nn.Linear(3 * 4 * 4, 1)

    def forward(self, faces: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        # faces: [B, T, C, H, W]
        # audio: [B, A_samples] - ignore in toy model
        b, t, c, h, w = faces.shape
        x = faces.view(b, -1)
        # Project down to logits: [B, 1]
        return self.fc(x[:, :3*4*4])


class ToyDataset(torch.utils.data.Dataset):
    """Generates mock dataset sample items for data loader testing."""
    def __init__(self, size: int = 4) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, str]]:
        # faces: [T=2, C=3, H=10, W=10]
        faces = torch.randn(2, 3, 10, 10)
        # audio: [16000] (1s raw)
        audio = torch.randn(16000)
        # label: 0 or 1
        label = torch.tensor(float(idx % 2))
        metadata = {
            "video_id": f"vid_{idx}",
            "subject": f"subj_{idx}",
            "raw_video_path": f"FakeAVCeleb/FakeVideo-FakeAudio/men/id0001/vid_{idx}.mp4"
        }
        return faces, audio, label, metadata


# =============================================================================
# Loss & Optimization Tests
# =============================================================================

def test_focal_loss_values() -> None:
    """Verifies FocalLoss computes correct reduction shapes and valid loss values."""
    fl = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.tensor([[2.0], [-1.0]])
    targets = torch.tensor([[1.0], [0.0]])
    loss = fl(logits, targets)
    assert loss.dim() == 0  # Should be scalar
    assert loss.item() > 0.0


def test_cosine_annealing_warmup_scheduler() -> None:
    """Verifies CosineAnnealingWithWarmup learning rate progression."""
    model = nn.Linear(10, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler = CosineAnnealingWithWarmup(
        optimizer=opt,
        warmup_epochs=2,
        max_epochs=5,
        min_lr=1e-5
    )
    
    # Init learning rate
    assert scheduler.get_last_lr()[0] == 1e-5

    # Epoch 1 (Warmup midway)
    scheduler.step()
    assert scheduler.get_last_lr()[0] > 1e-5
    
    # Epoch 2 (Max learning rate at end of warmup)
    scheduler.step()
    assert abs(scheduler.get_last_lr()[0] - 1e-3) < 1e-6

    # Epoch 3 (Cosine decay starts)
    scheduler.step()
    assert scheduler.get_last_lr()[0] < 1e-3

    # Epoch 5 (Last epoch -> min learning rate)
    scheduler.step()
    scheduler.step()
    assert abs(scheduler.get_last_lr()[0] - 1e-5) < 1e-6


# =============================================================================
# Metrics Tests
# =============================================================================

def test_calculate_metrics() -> None:
    """Verifies scikit-learn metrics calculation from predictions."""
    targets = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    
    res = calculate_metrics(targets, probs, threshold=0.5)
    assert res["accuracy"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["roc_auc"] == 1.0
    assert res["pr_auc"] == 1.0


# =============================================================================
# Trainer & Evaluator End-to-End Tests
# =============================================================================

def test_trainer_and_evaluator_cycle(temp_dir: Path) -> None:
    """Performs a mini training epoch and validation run using mock data."""
    # 1. Configuration setup
    cfg = STAFConfig()
    cfg.training.max_epochs = 1
    cfg.training.early_stopping_patience = 2
    cfg.training.use_amp = False  # Keep off for CPU test stability
    cfg.training.save_top_k = 1
    cfg.evaluation.save_plots = True
    cfg.evaluation.binary_threshold = 0.5
    
    # Paths config
    cfg.data.paths.processed_dir = str(temp_dir)
    cfg.data.paths.splits_dir = str(temp_dir)

    model = ToyDetector()
    train_loader = DataLoader(ToyDataset(size=4), batch_size=2)
    val_loader = DataLoader(ToyDataset(size=2), batch_size=2)

    # Output directory
    output_dir = temp_dir / "experiment_run"

    # 2. Run Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=torch.device("cpu"),
        output_dir=output_dir,
    )

    fit_results = trainer.fit()
    assert fit_results["best_epoch"] == 1
    assert "val_loss" in fit_results["best_metrics"]

    # Verify checkpoint got written
    ckpt_file = output_dir / "checkpoints" / "best_model.pt"
    assert ckpt_file.exists()

    # Load from saved checkpoint to test load state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    # 3. Run Evaluator
    evaluator = Evaluator(
        model=model,
        device=torch.device("cpu"),
        cfg=cfg,
        output_dir=output_dir / "evaluation_results"
    )

    eval_report = evaluator.evaluate(val_loader, split_name="test")
    assert "overall_metrics" in eval_report
    assert "accuracy" in eval_report["overall_metrics"]
    assert "category_metrics" in eval_report

    # Verify visual image plots got generated
    assert (output_dir / "evaluation_results" / "confusion_matrix_test.png").exists()
    assert (output_dir / "evaluation_results" / "roc_curve_test.png").exists()
    assert (output_dir / "evaluation_results" / "pr_curve_test.png").exists()
    assert (output_dir / "evaluation_results" / "metrics_test.json").exists()
