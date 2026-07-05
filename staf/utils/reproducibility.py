"""
STAF Reproducibility Utilities — Seed management and device resolution.

Every source of randomness in the framework (Python, NumPy, PyTorch, CUDA)
is seeded from a single integer. This module ensures deterministic behavior
across runs when configured.

Usage:
    from staf.utils.reproducibility import set_seed, resolve_device

    set_seed(42, deterministic=True)
    device = resolve_device("auto")
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

from staf.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all randomness sources.

    Args:
        seed: Integer seed value. Applied to Python, NumPy, and PyTorch.
        deterministic: If True, enables PyTorch deterministic algorithms
                       and disables cuDNN benchmark mode. This may reduce
                       performance but guarantees reproducibility.

    Note:
        Some operations (e.g., scatter_add on CUDA) do not have deterministic
        implementations and will raise an error if deterministic=True.
        Set CUBLAS_WORKSPACE_CONFIG=:4096:8 to handle cuBLAS non-determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+ deterministic mode
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # PyTorch < 1.8
            pass
    else:
        torch.backends.cudnn.benchmark = True

    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")


def resolve_device(device_str: str = "auto") -> torch.device:
    """
    Resolve a device string to a torch.device.

    Args:
        device_str: One of "auto", "cpu", "cuda", "cuda:0", "cuda:1", etc.
                    "auto" selects CUDA if available, otherwise CPU.

    Returns:
        Resolved torch.device instance.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            logger.info(f"Device: CUDA ({gpu_name}, {gpu_mem:.1f} GB)")
        else:
            device = torch.device("cpu")
            logger.info("Device: CPU (no CUDA available)")
    else:
        device = torch.device(device_str)
        logger.info(f"Device: {device}")

    return device


def get_gpu_memory_summary() -> Optional[str]:
    """
    Get a summary of GPU memory usage (if CUDA is available).

    Returns:
        Formatted string with allocated/reserved/total memory, or None if no GPU.
    """
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)

    return (
        f"GPU Memory — Allocated: {allocated:.2f} GB | "
        f"Reserved: {reserved:.2f} GB | "
        f"Total: {total:.1f} GB"
    )
