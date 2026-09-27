"""
checkpointing.py

Checkpoint saving/loading for model training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


CHECKPOINT_VERSION = 1


def _unwrap_model(
    model: nn.Module,
) -> nn.Module:
    """
    Return underlying model when wrapped by DataParallel/DDP-style
    wrappers exposing .module.
    """

    if hasattr(model, "module"):
        return model.module

    return model


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    monitor_name: str,
    monitor_value: float,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    early_stopping: Optional[Any] = None,
    config: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """
    Save complete training checkpoint atomically.

    Returns
    -------
    Path
        Final checkpoint path.
    """

    path = Path(
        path
    ).expanduser().resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_to_save = (
        _unwrap_model(
            model
        )
    )

    checkpoint = {
        "checkpoint_version": (
            CHECKPOINT_VERSION
        ),

        "epoch": int(
            epoch
        ),

        "monitor_name": str(
            monitor_name
        ),

        "monitor_value": float(
            monitor_value
        ),

        "model_state_dict": (
            model_to_save.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),

        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),

        "early_stopping_state_dict": (
            early_stopping.state_dict()
            if early_stopping is not None
            else None
        ),

        "config": (
            dict(config)
            if config is not None
            else {}
        ),

        "metadata": (
            dict(metadata)
            if metadata is not None
            else {}
        ),
    }

    # -------------------------------------------------------------
    # Atomic-style save:
    #
    # write temporary checkpoint first
    # then replace target
    # -------------------------------------------------------------

    temporary_path = (
        path.parent
        / f".{path.name}.tmp"
    )

    try:

        torch.save(
            checkpoint,
            temporary_path,
        )

        os.replace(
            temporary_path,
            path,
        )

    finally:

        if temporary_path.exists():
            temporary_path.unlink()

    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    early_stopping: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
    strict_model: bool = True,
) -> dict:
    """
    Restore complete training state.

    This function is intended for checkpoints created by this project.

    Returns
    -------
    dict
        Complete checkpoint dictionary.
    """

    path = Path(
        path
    ).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint must contain a dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model_to_load = (
        _unwrap_model(
            model
        )
    )

    model_to_load.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=strict_model,
    )

    optimizer_state = (
        checkpoint.get(
            "optimizer_state_dict"
        )
    )

    if (
        optimizer is not None
        and optimizer_state
        is not None
    ):
        optimizer.load_state_dict(
            optimizer_state
        )

    scheduler_state = (
        checkpoint.get(
            "scheduler_state_dict"
        )
    )

    if (
        scheduler is not None
        and scheduler_state
        is not None
    ):
        scheduler.load_state_dict(
            scheduler_state
        )

    scaler_state = (
        checkpoint.get(
            "scaler_state_dict"
        )
    )

    if (
        scaler is not None
        and scaler_state
        is not None
    ):
        scaler.load_state_dict(
            scaler_state
        )

    early_stopping_state = (
        checkpoint.get(
            "early_stopping_state_dict"
        )
    )

    if (
        early_stopping is not None
        and early_stopping_state
        is not None
    ):
        early_stopping.load_state_dict(
            early_stopping_state
        )

    return checkpoint


def load_model_weights(
    path: str | Path,
    *,
    model: nn.Module,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    """
    Load model weights only.

    Useful for:
        evaluation
        inference
        ensemble prediction
    """

    path = Path(
        path
    ).expanduser().resolve()

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict"
        in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif isinstance(
        checkpoint,
        dict,
    ):
        state_dict = checkpoint

    else:
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    _unwrap_model(
        model
    ).load_state_dict(
        state_dict,
        strict=strict,
    )

    return checkpoint