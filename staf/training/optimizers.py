"""
STAF Training Module: Optimizers & Schedulers.

Configures training optimizers (AdamW, SGD) and learning rate schedulers
(Cosine Annealing, StepLR, ReduceOnPlateau) with support for learning rate warmup.

Author: Varun (IIIT Ranchi)
License: MIT
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from staf.configs.schema import OptimizerConfig, SchedulerConfig
from staf.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Learning Rate Scheduler with Warmup
# =============================================================================

class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine Annealing learning rate scheduler with linear warmup.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculates learning rate for the current epoch."""
        if not self._get_lr_called_within_step:
            logger.warning("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")

        # Warmup phase
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_epochs == 0:
                return [base_lr for base_lr in self.base_lrs]
            alpha = self.last_epoch / self.warmup_epochs
            return [self.min_lr + (base_lr - self.min_lr) * alpha for base_lr in self.base_lrs]

        # Cosine Annealing phase
        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


# =============================================================================
# Optimizer & Scheduler Factories
# =============================================================================

def get_optimizer(model: nn.Module, cfg: OptimizerConfig) -> optim.Optimizer:
    """
    Factory function returning the configured PyTorch optimizer.

    Args:
        model: PyTorch model module containing parameters to optimize.
        cfg: Optimizer configuration section.

    Returns:
        optim.Optimizer instance.
    """
    opt_name = cfg.name.lower()
    logger.info(f"Setting up optimizer: {opt_name} (lr={cfg.learning_rate:.6f}, weight_decay={cfg.weight_decay})")

    # Filter out parameters that do not require gradients (e.g. frozen backbones)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = len(trainable_params)
    logger.info(f"Number of parameters to optimize: {num_trainable}")

    if num_trainable == 0:
        logger.warning("No trainable parameters found in the model! Optimizing all parameters instead.")
        trainable_params = list(model.parameters())

    if opt_name == "adamw":
        return optim.AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(cfg.betas[0], cfg.betas[1])
        )
    elif opt_name == "adam":
        return optim.Adam(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(cfg.betas[0], cfg.betas[1])
        )
    elif opt_name == "sgd":
        return optim.SGD(
            trainable_params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
    else:
        logger.warning(f"Unknown optimizer '{opt_name}'. Falling back to AdamW.")
        return optim.AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )


def get_scheduler(
    optimizer: optim.Optimizer,
    cfg: SchedulerConfig,
    max_epochs: int,
) -> tuple[torch.optim.lr_scheduler._LRScheduler, bool]:
    """
    Factory function returning the configured PyTorch learning rate scheduler.

    Args:
        optimizer: Active optimizer instance.
        cfg: Scheduler configuration section.
        max_epochs: Total number of planned epochs.

    Returns:
        A tuple of (scheduler, is_epoch_based).
        is_epoch_based indicates if the scheduler step should be called after each epoch
        (True) or after each validation round (False, for ReduceLROnPlateau).
    """
    sched_name = cfg.name.lower()
    logger.info(f"Setting up learning rate scheduler: {sched_name}")

    if sched_name == "cosine":
        scheduler = CosineAnnealingWithWarmup(
            optimizer=optimizer,
            warmup_epochs=cfg.warmup_epochs,
            max_epochs=max_epochs,
            min_lr=cfg.min_lr
        )
        return scheduler, True

    elif sched_name == "step":
        # Linear step decay
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma
        )
        return scheduler, True

    elif sched_name == "plateau":
        # Reduce learning rate when validation metric plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr
        )
        return scheduler, False

    else:
        logger.warning(f"Unknown scheduler '{sched_name}'. Falling back to Cosine Annealing.")
        scheduler = CosineAnnealingWithWarmup(
            optimizer=optimizer,
            warmup_epochs=cfg.warmup_epochs,
            max_epochs=max_epochs,
            min_lr=cfg.min_lr
        )
        return scheduler, True
