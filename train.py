"""
STAF Experimentation: Model Trainer CLI.

Entry point script to train multimodal deepfake detectors.
Loads configurations, sets up PyTorch datasets and DataLoaders,
initializes the model, and runs the Trainer loop.

Usage:
    python train.py --config configs/baseline.yaml
    python train.py --config experiments/baseline_v1.yaml --overrides training.max_epochs=10

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from staf.configs.schema import load_config, STAFConfig, config_to_dict
from staf.datasets.fakeavceleb import FakeAVCelebDataset
from staf.models.baseline.baseline_detector import BaselineDetector
from staf.training.trainer import Trainer
from staf.utils.logging import setup_logging, get_logger
from staf.utils.reproducibility import set_seed, resolve_device

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="STAF Multimodal Deepfake Detector Training CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="staf/configs/baseline.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        help="Dotted configuration overrides, e.g. training.batch_size=32"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Force disable Weights & Biases logging for this run"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load and resolve configuration
    try:
        cfg = load_config(args.config, overrides=args.overrides)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Force disable W&B if CLI flag is provided
    if args.no_wandb:
        cfg.logging.wandb.enabled = False

    # 2. Setup output folder structures for the experiment run
    results_root = Path(cfg.output_dir or "results")
    results_root.mkdir(parents=True, exist_ok=True)

    # Auto-number experiments: find the highest existing number and increment
    existing = [d.name for d in results_root.iterdir() if d.is_dir()]
    max_num = 0
    for name in existing:
        parts = name.split("_", 1)
        if parts[0].isdigit():
            max_num = max(max_num, int(parts[0]))
    run_number = max_num + 1
    run_dir_name = f"{run_number:03d}_{cfg.experiment_name}"

    output_dir = results_root / run_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure session local logging
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.logging, log_dir=str(log_dir))
    
    logger.info(f"Initialized experiment run: {run_dir_name}")
    logger.info(f"Results output folder: {output_dir}")

    # Save resolved config for absolute reproducibility
    resolved_config_path = output_dir / "config.yaml"
    from staf.configs.schema import save_config
    save_config(cfg, str(resolved_config_path))
    logger.info(f"Saved resolved configuration to: {resolved_config_path}")

    # 3. Setup reproducibility
    set_seed(cfg.training.seed, deterministic=cfg.training.deterministic)
    device = resolve_device(cfg.device)
    logger.info(f"Using device: {device}")

    # 4. Initialize Datasets & DataLoaders
    splits_dir = Path(cfg.data.paths.splits_dir or "data/splits")
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        logger.error(
            f"Split manifests not found! Check train.csv and val.csv are present in: {splits_dir}\n"
            f"Please run sync_modalities.py first to build dataset splits."
        )
        sys.exit(1)

    logger.info("Initializing datasets...")
    train_dataset = FakeAVCelebDataset(split_csv_path=train_csv, data_config=cfg.data)
    val_dataset = FakeAVCelebDataset(split_csv_path=val_csv, data_config=cfg.data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        persistent_workers=getattr(cfg.data, "persistent_workers", False) if cfg.data.num_workers > 0 else False,
        drop_last=len(train_dataset) >= cfg.data.batch_size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        persistent_workers=getattr(cfg.data, "persistent_workers", False) if cfg.data.num_workers > 0 else False
    )

    logger.info(f"Dataset summary:")
    logger.info(f"  Train samples: {len(train_dataset)} ({len(train_loader)} batches)")
    logger.info(f"  Val samples:   {len(val_dataset)} ({len(val_loader)} batches)")

    # 5. Initialize Model
    logger.info("Building model architecture...")
    # Currently only supports baseline model. STAF models will be plugged in Phase 6.
    model = BaselineDetector(cfg.model, loss_function=cfg.training.loss_function)

    # 6. Setup Weights & Biases if enabled
    wandb_run = None
    if cfg.logging.wandb.enabled:
        try:
            import wandb
            logger.info("Initializing Weights & Biases...")
            wandb_run = wandb.init(
                project=cfg.logging.wandb.project,
                entity=cfg.logging.wandb.entity or None,
                name=cfg.logging.wandb.run_name or run_dir_name,
                config=config_to_dict(cfg),
                notes=cfg.logging.wandb.notes,
                tags=cfg.logging.wandb.tags + ["train"]
            )
        except ImportError:
            logger.warning("wandb package not installed. Disabling W&B logging.")
        except Exception as e:
            logger.warning(f"Could not initialize W&B: {e}. Disabling W&B logging.")

    # 7. Initialize Trainer and run fit loop
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
        wandb_run=wandb_run
    )

    try:
        results = trainer.fit()
        logger.info("Training cycle finished successfully.")
        logger.info(f"Best metrics: {results['best_metrics']}")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saved top checkpoints are in output directory.")
    finally:
        if wandb_run:
            wandb.finish()


if __name__ == "__main__":
    main()
