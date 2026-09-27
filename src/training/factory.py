"""
factory.py

Construction of the complete shared training stack from TrainingConfig.

Experiment scripts should use this module instead of manually repeating
loss/optimizer/scheduler/early-stopping/Trainer wiring for every model and
scenario.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from src.configs.training_config import TrainingConfig
from src.evaluation.metrics import make_metric_function
from src.training.early_stopping import EarlyStopping
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer


def build_trainer_from_config(
    *,
    model: nn.Module,
    config: TrainingConfig,
    device: torch.device,
    checkpoint_dir: str | Path | None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Trainer:
    """
    Build the project's Trainer and all of its shared dependencies.

    The supplied ``TrainingConfig`` is the single source of truth for:

        loss
        optimizer
        scheduler
        early stopping
        AMP
        gradient clipping
        checkpoint selection metric/mode
        classification threshold used by metric-based selection

    Experiment scripts should normally vary only model/scenario/fold,
    batch size, DataLoader resources, and output paths.
    """

    config.validate()

    loss_name = config.loss.name.lower().strip()
    if loss_name not in {
        "bce_with_logits",
        "bce",
    }:
        raise ValueError(
            f"Unsupported loss configuration: {config.loss.name!r}."
        )

    criterion = build_loss(
        pos_weight=config.loss.pos_weight,
    )

    optimizer = build_optimizer(
        model,
        name=config.optimizer.name,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=(
            config.optimizer.beta1,
            config.optimizer.beta2,
        ),
        eps=config.optimizer.eps,
    )

    scheduler = build_scheduler(
        optimizer,
        name=config.scheduler.name,
        max_epochs=config.training.max_epochs,
        plateau_factor=config.scheduler.factor,
        plateau_patience=config.scheduler.patience,
        plateau_threshold=config.scheduler.threshold,
        min_learning_rate=config.scheduler.min_learning_rate,
    )

    early_stopping = None
    if config.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=config.early_stopping.patience,
            mode=config.early_stopping.mode,
            min_delta=config.early_stopping.min_delta,
            warmup_epochs=config.early_stopping.warmup_epochs,
        )

    checkpoint_metric = (
        config.training.checkpoint_metric
        .lower()
        .strip()
    )

    if checkpoint_metric == "val_loss":
        selection_metric_fn = None
    else:
        selection_metric_fn = make_metric_function(
            checkpoint_metric,
            threshold=(
                config.training.classification_threshold
            ),
        )

    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        early_stopping=early_stopping,
        use_amp=config.training.use_amp,
        gradient_clip_norm=(
            config.training.gradient_clip_norm
        ),
        checkpoint_dir=checkpoint_dir,
        selection_metric_fn=selection_metric_fn,
        selection_metric_name=checkpoint_metric,
        selection_mode=config.training.checkpoint_mode,
        config=asdict(config),
        metadata=metadata,
    )
