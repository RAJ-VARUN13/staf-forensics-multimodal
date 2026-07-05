"""
STAF Experimentation: Model Evaluator CLI.

Entry point script to evaluate trained model checkpoints on test sets or
cross-dataset benchmarks. Generates classification metrics and visual plots.

Usage:
    python evaluate.py --checkpoint results/baseline_v1_XXXX/checkpoints/best_model.pt --split test
    python evaluate.py --checkpoint results/baseline_v1_XXXX/checkpoints/best_model.pt --split test --prof_dataset

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from staf.configs.schema import load_config, STAFConfig
from staf.datasets.fakeavceleb import FakeAVCelebDataset
from staf.models.baseline.baseline_detector import BaselineDetector
from staf.evaluation.evaluator import Evaluator
from staf.utils.logging import setup_logging, get_logger
from staf.utils.reproducibility import set_seed, resolve_device

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="STAF Multimodal Deepfake Detector Evaluation CLI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved PyTorch model checkpoint (.pt)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split file name (e.g. 'test', 'val', 'train')"
    )
    parser.add_argument(
        "--prof_dataset",
        action="store_true",
        help="Evaluate on the professor's dataset instead of FakeAVCeleb"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint file not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load Checkpoint State
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    # Map storage to CPU first for safe loading on different hardware
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Reconstruct configuration from the checkpoint to ensure matching architectures
    cfg_dict = checkpoint.get("config")
    if cfg_dict is None:
        logger.error("Checkpoint does not contain config metadata! Cannot reconstruct model.")
        sys.exit(1)

    # Convert raw config dictionary back into STAFConfig object
    from omegaconf import OmegaConf
    schema = OmegaConf.structured(STAFConfig)
    cfg_omega = OmegaConf.merge(schema, OmegaConf.create(cfg_dict))
    cfg: STAFConfig = OmegaConf.to_object(cfg_omega)

    # Setup session logging (save evaluation log in checkpoint's directory)
    eval_dir = checkpoint_path.parent.parent / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.logging, log_dir=str(eval_dir))

    logger.info(f"Loaded configuration for experiment: {cfg.experiment_name}")

    # Set reproducibility seeds
    set_seed(cfg.training.seed, deterministic=cfg.training.deterministic)
    device = resolve_device(cfg.device)
    logger.info(f"Using device: {device}")

    # 2. Resolve Dataloader Paths
    splits_dir = Path(cfg.data.paths.splits_dir or "data/splits")
    
    # Choose split CSV based on flags
    if args.prof_dataset:
        split_csv = Path(cfg.data.paths.professor_metadata_csv or splits_dir / "professor.csv")
        split_name = "professor_dataset"
    else:
        split_csv = splits_dir / f"{args.split}.csv"
        split_name = args.split

    if not split_csv.exists():
        logger.error(f"Manifest split CSV not found: {split_csv}")
        sys.exit(1)

    # 3. Initialize Dataset and DataLoader
    logger.info(f"Initializing {split_name} dataset loader...")
    test_dataset = FakeAVCelebDataset(split_csv_path=split_csv, data_config=cfg.data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=getattr(cfg.data, "persistent_workers", False) if cfg.data.num_workers > 0 else False
    )

    logger.info(f"Loaded {len(test_dataset)} evaluation samples.")

    # 4. Reconstruct Model & Load Weights
    logger.info("Reconstructing model architecture...")
    model = BaselineDetector(cfg.model, loss_function=cfg.training.loss_function)
    
    model_state = checkpoint["model_state_dict"]
    model.load_state_dict(model_state)
    logger.info("Successfully loaded trained weights.")

    # 5. Initialize Evaluator and run evaluation loop
    evaluator = Evaluator(
        model=model,
        device=device,
        cfg=cfg,
        output_dir=eval_dir
    )

    report = evaluator.evaluate(test_loader, split_name=split_name)
    
    # Print metrics summary
    logger.info("=" * 55)
    logger.info(f"   EVALUATION METRICS SUMMARY - {split_name.upper()}")
    logger.info("=" * 55)
    for k, v in report["overall_metrics"].items():
        logger.info(f"  {k:20s}: {v:.4f}")
    logger.info("-" * 55)
    logger.info("  Per-Category Accuracies:")
    for cat, metrics in report["category_metrics"].items():
        logger.info(f"    {cat:30s}: {metrics['accuracy']:.4f} (count: {metrics['count']})")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
