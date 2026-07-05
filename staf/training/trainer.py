"""
STAF Training Module: Unified Model Trainer.

Orchestrates the training and validation loops, manages mixed precision (AMP),
gradient clipping, learning rate scheduling, early stopping, checkpointing,
and experiment logging via console, file, TensorBoard, and Weights & Biases.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from staf.configs.schema import STAFConfig, config_to_dict
from staf.training.losses import get_loss_function
from staf.training.optimizers import get_optimizer, get_scheduler
from staf.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Manages end-to-end model training, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: STAFConfig,
        device: torch.device,
        output_dir: Path,
        wandb_run: Any = None,
    ) -> None:
        """
        Args:
            model: PyTorch model to train.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            cfg: Top-level configuration object.
            device: Active computing device (CPU or CUDA).
            output_dir: Experiment session local output folder under results/.
            wandb_run: Optional active W&B run instance.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.wandb_run = wandb_run

        self.model.to(self.device)

        # Setup Loss function
        # Check if we should compute positive class weight for imbalanced training
        pos_weight = None
        if self.cfg.training.class_weights is not None:
            pos_weight = torch.tensor([self.cfg.training.class_weights[1]], device=self.device)
        self.loss_fn = get_loss_function(self.cfg.training, pos_weight=pos_weight)

        # Setup Optimizer and Scheduler
        self.optimizer = get_optimizer(self.model, self.cfg.training.optimizer)
        self.scheduler, self.is_epoch_scheduler = get_scheduler(
            self.optimizer,
            self.cfg.training.scheduler,
            self.cfg.training.max_epochs
        )

        # Mixed precision gradient scaling
        self.use_amp = self.cfg.training.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Early Stopping State
        self.patience = self.cfg.training.early_stopping_patience
        self.patience_counter = 0
        self.best_metric_val = -float("inf") if self.cfg.training.early_stopping_mode == "max" else float("inf")
        self.monitor_metric = self.cfg.training.early_stopping_metric
        self.monitor_mode = self.cfg.training.early_stopping_mode

        # Checkpoints State
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.saved_checkpoints: List[Tuple[float, Path]] = []  # List of (metric_val, path)

        # TensorBoard Writer
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.output_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"TensorBoard logging enabled: {tb_dir}")
        except ImportError:
            self.tb_writer = None
            logger.warning("TensorBoard not available. Install tensorboard for local experiment tracking.")

    def train_epoch(self, epoch: int) -> float:
        """
        Runs one complete epoch of training.

        Args:
            epoch: Current epoch index.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch:03d} [Train]"
        )

        for batch_idx, (faces, audio, labels, _) in pbar:
            # Transfer data to device
            faces = faces.to(self.device, non_blocking=True)
            audio = audio.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).view(-1, 1)

            self.optimizer.zero_grad()

            # Forward pass with AMP autocast
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(faces, audio)
                loss = self.loss_fn(logits, labels)

            # Backward pass & Optimizer step using scaler
            self.scaler.scale(loss).backward()

            if self.cfg.training.gradient_clip_val > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.training.gradient_clip_val
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(1, len(self.train_loader))
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Runs evaluation on the validation set.

        Args:
            epoch: Current epoch index.

        Returns:
            Dictionary containing average validation metrics.
        """
        self.model.eval()
        total_loss = 0.0

        all_preds: List[float] = []
        all_targets: List[float] = []

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch:03d} [Val]"
        )

        for faces, audio, labels, _ in pbar:
            faces = faces.to(self.device, non_blocking=True)
            audio = audio.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).view(-1, 1)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(faces, audio)
                loss = self.loss_fn(logits, labels)

            total_loss += loss.item()

            # Convert logits to probabilities (sigmoid)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            targets = labels.view(-1).cpu().numpy()

            all_preds.extend(probs.tolist())
            all_targets.extend(targets.tolist())

        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate evaluation metrics
        preds_array = np.array(all_preds)
        targets_array = np.array(all_targets)
        binary_preds = (preds_array >= self.cfg.evaluation.binary_threshold).astype(int)

        metrics = {"val_loss": avg_loss}
        
        # Compute metrics safely (handling zero divisions / single-class edge cases)
        try:
            metrics["val_accuracy"] = float(accuracy_score(targets_array, binary_preds))
            metrics["val_precision"] = float(precision_score(targets_array, binary_preds, zero_division=0))
            metrics["val_recall"] = float(recall_score(targets_array, binary_preds, zero_division=0))
            metrics["val_f1"] = float(f1_score(targets_array, binary_preds, zero_division=0))
            
            # ROC-AUC requires both classes present in targets
            if len(np.unique(targets_array)) > 1:
                metrics["val_auc"] = float(roc_auc_score(targets_array, preds_array))
            else:
                metrics["val_auc"] = 0.5
        except Exception as e:
            logger.warning(f"Error computing validation metrics: {e}")
            metrics.update({"val_accuracy": 0.0, "val_precision": 0.0, "val_recall": 0.0, "val_f1": 0.0, "val_auc": 0.0})

        return metrics

    def save_checkpoint(self, epoch: int, val_metric: float, is_best: bool = False) -> None:
        """
        Saves model weights and state dict. Maintains top-k checkpoints to save disk space.

        Args:
            epoch: Current epoch.
            val_metric: Metric value achieved.
            is_best: True if this is the overall best model checkpoint.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": config_to_dict(self.cfg),
            "val_metric": val_metric
        }

        # Save standard epoch checkpoint
        ckpt_path = self.checkpoints_dir / f"model_epoch_{epoch:03d}.pt"
        torch.save(state, ckpt_path)

        # Track and maintain top-k checkpoints
        self.saved_checkpoints.append((val_metric, ckpt_path))
        # Sort based on monitoring mode
        reverse_sort = (self.monitor_mode == "max")
        self.saved_checkpoints.sort(key=lambda x: x[0], reverse=reverse_sort)

        # If we exceeded save_top_k, delete the worst one
        if len(self.saved_checkpoints) > self.cfg.training.save_top_k:
            _, worst_path = self.saved_checkpoints.pop()
            try:
                if worst_path.exists():
                    worst_path.unlink()
            except Exception as e:
                logger.warning(f"Could not remove old checkpoint {worst_path}: {e}")

        # Save absolute best checkpoint
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info(f"Saved new best model checkpoint to: {best_path}")

    def fit(self) -> Dict[str, Any]:
        """
        Executes the full training and validation pipeline across all epochs.

        Returns:
            Dictionary containing best validation epoch metrics.
        """
        logger.info(f"Starting training run: {self.cfg.experiment_name}")
        logger.info(f"  Device: {self.device} | Epochs: {self.cfg.training.max_epochs}")

        best_epoch = -1
        best_metrics: Dict[str, float] = {}

        for epoch in range(1, self.cfg.training.max_epochs + 1):
            # Run one epoch
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            # Retrieve active learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler if epoch-based
            if self.scheduler:
                if self.is_epoch_scheduler:
                    self.scheduler.step()
                else:
                    # ReduceLROnPlateau steps based on validation loss
                    self.scheduler.step(val_metrics["val_loss"])

            # Log metrics
            log_str = (
                f"Epoch {epoch:03d} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_accuracy']:.4f} | "
                f"Val AUC: {val_metrics['val_auc']:.4f} | Val F1: {val_metrics['val_f1']:.4f}"
            )
            logger.info(log_str)

            # Log to TensorBoard
            if self.tb_writer:
                self.tb_writer.add_scalar("Loss/train", train_loss, epoch)
                self.tb_writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
                self.tb_writer.add_scalar("Metrics/accuracy", val_metrics["val_accuracy"], epoch)
                self.tb_writer.add_scalar("Metrics/precision", val_metrics["val_precision"], epoch)
                self.tb_writer.add_scalar("Metrics/recall", val_metrics["val_recall"], epoch)
                self.tb_writer.add_scalar("Metrics/f1", val_metrics["val_f1"], epoch)
                self.tb_writer.add_scalar("Metrics/auc", val_metrics["val_auc"], epoch)
                self.tb_writer.add_scalar("LearningRate", current_lr, epoch)
                self.tb_writer.flush()

            # Log to W&B
            if self.wandb_run:
                wb_data = {
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    "train_loss": train_loss,
                    **val_metrics
                }
                self.wandb_run.log(wb_data)

            # Determine if current epoch improves the monitored metric
            val_metric_val = val_metrics[self.monitor_metric]
            
            if self.monitor_mode == "max":
                is_improved = (val_metric_val > self.best_metric_val)
            else:
                is_improved = (val_metric_val < self.best_metric_val)

            if is_improved:
                self.best_metric_val = val_metric_val
                self.patience_counter = 0
                best_epoch = epoch
                best_metrics = val_metrics
                self.save_checkpoint(epoch, val_metric_val, is_best=True)
            else:
                self.patience_counter += 1
                # Save normal top-k checkpoint
                self.save_checkpoint(epoch, val_metric_val, is_best=False)

            # Always save last checkpoint (overwrites each epoch)
            self._save_last_checkpoint(epoch, val_metrics)

            # Early Stopping Check
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered! No improvement for {self.patience} epochs.")
                break

        logger.info(f"Training completed. Best epoch: {best_epoch:03d} (val_{self.monitor_metric}={self.best_metric_val:.4f})")

        # Save training summary metrics.json
        self._save_metrics_summary(best_epoch, best_metrics)

        # Close TensorBoard writer
        if self.tb_writer:
            self.tb_writer.close()

        return {"best_epoch": best_epoch, "best_metrics": best_metrics}

    def _save_last_checkpoint(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        """Saves the latest model state as last_model.pt (overwritten each epoch)."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": config_to_dict(self.cfg),
            "val_metrics": val_metrics
        }
        last_path = self.checkpoints_dir / "last_model.pt"
        torch.save(state, last_path)

    def _save_metrics_summary(self, best_epoch: int, best_metrics: Dict[str, float]) -> None:
        """Writes a JSON summary of the training run to the output directory."""
        summary = {
            "experiment_name": self.cfg.experiment_name,
            "best_epoch": best_epoch,
            "total_epochs": self.cfg.training.max_epochs,
            "device": str(self.device),
            "best_metrics": best_metrics,
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "best_metric_value": self.best_metric_val,
        }
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved training metrics summary to: {metrics_path}")
