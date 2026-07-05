"""
STAF Logging & Experiment Tracking Module.

Provides a unified logging interface that writes to:
1. Console (with color-coded severity)
2. Log files (with rotation)
3. Weights & Biases (metrics, configs, artifacts)

Design Decision:
    We wrap Python's stdlib logging + W&B instead of using a custom solution
    because: stdlib logging is thread-safe, well-tested, and integrates with
    every library we use. W&B handles experiment tracking, artifact storage,
    and team collaboration. The wrapper ensures consistent formatting and
    makes W&B optional (graceful fallback if not installed/configured).

Usage:
    from staf.utils.logging import setup_logging, get_logger, log_metrics

    # Initialize once at program start
    setup_logging(cfg.logging)

    # Get module-specific logger
    logger = get_logger(__name__)
    logger.info("Training started")

    # Log metrics to W&B
    log_metrics({"train_loss": 0.45, "val_auc": 0.92}, step=10)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Attempt to import W&B; gracefully degrade if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Module-level state
# =============================================================================

_wandb_run: Optional[Any] = None  # Active W&B run, if any
_log_dir: Optional[str] = None
_initialized: bool = False


# =============================================================================
# Console Formatter with Color Support
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Logging formatter that adds ANSI color codes to console output.

    Colors are applied based on log level:
        DEBUG    → Grey
        INFO     → Cyan
        WARNING  → Yellow
        ERROR    → Red
        CRITICAL → Bold Red
    """

    COLORS = {
        logging.DEBUG: "\033[90m",      # Grey
        logging.INFO: "\033[36m",       # Cyan
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[1;31m", # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        # Only colorize the level name
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# =============================================================================
# Core Setup Functions
# =============================================================================

def setup_logging(
    logging_config: Optional[Any] = None,
    log_dir: str = "logs",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> None:
    """
    Initialize the STAF logging system.

    Call this once at the start of your program. Configures root logger,
    file handler, console handler, and optionally initializes W&B.

    Args:
        logging_config: A LoggingConfig dataclass (from staf.configs.schema).
                       If provided, overrides all other arguments.
        log_dir: Directory for log files.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_to_file: Whether to write logs to a file.
        log_to_console: Whether to print logs to console.
    """
    global _log_dir, _initialized

    # Extract settings from config dataclass if provided
    if logging_config is not None:
        if hasattr(logging_config, "log_dir"):
            log_dir = logging_config.log_dir
        if hasattr(logging_config, "level"):
            level = logging_config.level
        if hasattr(logging_config, "log_to_file"):
            log_to_file = logging_config.log_to_file
        if hasattr(logging_config, "log_to_console"):
            log_to_console = logging_config.log_to_console

    _log_dir = log_dir

    # Resolve log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers (prevents duplicate logs on re-init)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # File format (no colors, full timestamp)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console format (with colors, compact)
    console_fmt = ColoredFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Console Handler ---
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_fmt)
        root_logger.addHandler(console_handler)

    # --- File Handler ---
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"staf_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File always captures everything
        file_handler.setFormatter(file_fmt)
        root_logger.addHandler(file_handler)

    _initialized = True

    # Log initialization
    logger = get_logger("staf.logging")
    logger.info(f"Logging initialized (level={level}, dir={log_dir})")

    # Initialize W&B if configured
    if logging_config is not None and hasattr(logging_config, "wandb"):
        wandb_cfg = logging_config.wandb
        if hasattr(wandb_cfg, "enabled") and wandb_cfg.enabled:
            init_wandb(wandb_cfg)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    return logging.getLogger(name)


# =============================================================================
# Weights & Biases Integration
# =============================================================================

def init_wandb(wandb_config: Any, staf_config: Optional[Dict] = None) -> None:
    """
    Initialize a Weights & Biases run.

    Args:
        wandb_config: WandbConfig dataclass with project, entity, etc.
        staf_config: Full STAF config dict to log as the run's config.
    """
    global _wandb_run

    if not WANDB_AVAILABLE:
        logger = get_logger("staf.logging")
        logger.warning(
            "wandb is not installed. Install with: pip install wandb. "
            "Experiment tracking will be disabled."
        )
        return

    logger = get_logger("staf.logging")

    try:
        # Build wandb.init kwargs
        init_kwargs: Dict[str, Any] = {
            "project": getattr(wandb_config, "project", "staf-deepfake-detection"),
            "config": staf_config or {},
            "reinit": True,
        }

        entity = getattr(wandb_config, "entity", "")
        if entity:
            init_kwargs["entity"] = entity

        run_name = getattr(wandb_config, "run_name", "")
        if run_name:
            init_kwargs["name"] = run_name

        tags = getattr(wandb_config, "tags", [])
        if tags:
            init_kwargs["tags"] = list(tags)

        notes = getattr(wandb_config, "notes", "")
        if notes:
            init_kwargs["notes"] = notes

        _wandb_run = wandb.init(**init_kwargs)
        logger.info(f"W&B initialized: project={init_kwargs['project']}, "
                     f"run={_wandb_run.name}")

    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}. Continuing without tracking.")
        _wandb_run = None


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """
    Log metrics to W&B (if active) and to the standard logger.

    Args:
        metrics: Dictionary of metric name → value.
        step: Training step or epoch number.
        prefix: Optional prefix for metric names (e.g., "train/", "val/").
    """
    logger = get_logger("staf.metrics")

    # Prepend prefix
    if prefix:
        prefixed = {f"{prefix}{k}": v for k, v in metrics.items()}
    else:
        prefixed = metrics

    # Log to Python logger
    metrics_str = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in prefixed.items())
    step_str = f"[step={step}] " if step is not None else ""
    logger.info(f"{step_str}{metrics_str}")

    # Log to W&B
    if _wandb_run is not None:
        log_kwargs = {"step": step} if step is not None else {}
        _wandb_run.log(prefixed, **log_kwargs)


def log_artifact(
    artifact_path: str,
    artifact_name: str,
    artifact_type: str = "model",
) -> None:
    """
    Log a file or directory as a W&B artifact.

    Args:
        artifact_path: Local path to the file or directory.
        artifact_name: Name for the artifact in W&B.
        artifact_type: Type of artifact ("model", "dataset", "config").
    """
    if _wandb_run is None:
        return

    logger = get_logger("staf.logging")
    try:
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        if os.path.isdir(artifact_path):
            artifact.add_dir(artifact_path)
        else:
            artifact.add_file(artifact_path)
        _wandb_run.log_artifact(artifact)
        logger.info(f"Logged artifact: {artifact_name} ({artifact_type})")
    except Exception as e:
        logger.warning(f"Failed to log artifact {artifact_name}: {e}")


def finish_wandb() -> None:
    """Finalize the W&B run. Call at the end of training."""
    global _wandb_run
    if _wandb_run is not None:
        _wandb_run.finish()
        _wandb_run = None
        get_logger("staf.logging").info("W&B run finalized.")
