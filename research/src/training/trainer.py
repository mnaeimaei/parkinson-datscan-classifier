"""
trainer.py

Generic binary-classification training engine.

Supports:
    - whole-volume input
    - ROI input
    - whole + ROI fusion
    - CUDA AMP
    - gradient clipping
    - validation
    - early stopping
    - learning-rate scheduling
    - best checkpoint
    - last checkpoint
    - resume from checkpoint
    - OOF prediction collection

Models must return:
    logits [B,1]

Models must NOT apply sigmoid internally.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)

from src.training.early_stopping import (
    EarlyStopping,
)

from src.training.losses import (
    prepare_binary_targets,
)

from src.training.optimizer import (
    get_learning_rates,
)

from src.training.scheduler import (
    SchedulerController,
)


# =====================================================================
# EPOCH OUTPUT
# =====================================================================


@dataclass
class EpochOutput:
    """
    Results from one complete train/validation epoch.
    """

    loss: float

    logits: Tensor

    probabilities: Tensor

    targets: Tensor

    uids: list[str]

    n_samples: int


# =====================================================================
# FULL TRAINING RESULT
# =====================================================================


@dataclass
class TrainingResult:
    """
    Final result returned by Trainer.fit().
    """

    best_epoch: int

    best_metric: float

    epochs_completed: int

    stopped_early: bool

    best_checkpoint_path: Optional[
        Path
    ]

    last_checkpoint_path: Optional[
        Path
    ]

    history: list[dict[str, Any]]


# =====================================================================
# TRAINER
# =====================================================================


class Trainer:
    """
    Generic project training engine.

    Parameters
    ----------
    model:
        Classification model.

    optimizer:
        PyTorch optimizer.

    criterion:
        Normally BCEWithLogitsLoss.

    device:
        CUDA or CPU device.

    scheduler:
        Optional SchedulerController.

    early_stopping:
        Optional EarlyStopping.

    use_amp:
        Use CUDA automatic mixed precision.

    gradient_clip_norm:
        Optional maximum gradient norm.

    checkpoint_dir:
        Directory where:

            best_model.pt
            last_model.pt

        will be written.

    selection_metric_fn:
        Optional function:

            metric = f(targets, probabilities)

        Example:
            AUROC

        If None:
            validation loss is used.

    selection_metric_name:
        Name saved in checkpoints.

    selection_mode:
        "min" or "max".
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[
            SchedulerController
        ] = None,
        early_stopping: Optional[
            EarlyStopping
        ] = None,
        use_amp: bool = True,
        gradient_clip_norm: Optional[
            float
        ] = 1.0,
        checkpoint_dir: Optional[
            str | Path
        ] = None,
        selection_metric_fn: Optional[
            Callable[
                [Tensor, Tensor],
                float,
            ]
        ] = None,
        selection_metric_name: str = (
            "val_loss"
        ),
        selection_mode: str = "min",
        config: Optional[
            Mapping[str, Any]
        ] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> None:

        if selection_mode not in {
            "min",
            "max",
        }:
            raise ValueError(
                "selection_mode must be "
                "'min' or 'max'."
            )

        if (
            early_stopping is not None
            and early_stopping.mode != selection_mode
        ):
            raise ValueError(
                "EarlyStopping.mode must match Trainer.selection_mode. "
                f"Received early_stopping.mode={early_stopping.mode!r}, "
                f"selection_mode={selection_mode!r}."
            )

        if (
            gradient_clip_norm
            is not None
            and gradient_clip_norm
            <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must "
                "be > 0 or None."
            )

        self.device = device

        self.model = model.to(
            device
        )

        self.optimizer = optimizer

        self.criterion = (
            criterion.to(
                device
            )
        )

        self.scheduler = scheduler

        self.early_stopping = (
            early_stopping
        )

        self.gradient_clip_norm = (
            gradient_clip_norm
        )

        self.selection_metric_fn = (
            selection_metric_fn
        )

        self.selection_metric_name = (
            selection_metric_name
        )

        self.selection_mode = (
            selection_mode
        )

        self.config = (
            dict(config)
            if config is not None
            else {}
        )

        self.metadata = (
            dict(metadata)
            if metadata is not None
            else {}
        )

        # -------------------------------------------------------------
        # AMP
        #
        # Use FP16 AMP only on CUDA.
        # -------------------------------------------------------------

        self.amp_enabled = bool(
            use_amp
            and device.type == "cuda"
        )

        self.scaler = (
            torch.amp.GradScaler(
                "cuda",
                enabled=self.amp_enabled,
            )
        )

        # -------------------------------------------------------------
        # Checkpoint paths
        # -------------------------------------------------------------

        if checkpoint_dir is not None:

            self.checkpoint_dir = (
                Path(
                    checkpoint_dir
                )
                .expanduser()
                .resolve()
            )

            self.checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.best_checkpoint_path = (
                self.checkpoint_dir
                / "best_model.pt"
            )

            self.last_checkpoint_path = (
                self.checkpoint_dir
                / "last_model.pt"
            )

        else:

            self.checkpoint_dir = None
            self.best_checkpoint_path = None
            self.last_checkpoint_path = None

        self.best_metric: Optional[
            float
        ] = None

        self.best_epoch: Optional[
            int
        ] = None

    # =================================================================
    # AMP CONTEXT
    # =================================================================

    def _autocast_context(
        self,
    ):

        if self.amp_enabled:

            return torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )

        return nullcontext()

    # =================================================================
    # BATCH HANDLING
    # =================================================================

    def _move_tensor(
        self,
        x: Tensor,
    ) -> Tensor:

        return x.to(
            self.device,
            non_blocking=True,
        )

    def _parse_batch(
        self,
        batch: Any,
    ) -> tuple[
        Any,
        Tensor,
        list[str],
    ]:
        """
        Convert DataLoader batch into:

            inputs
            labels
            uids

        Supported dictionary formats
        ----------------------------

        Single input:

            {
                "input": tensor,
                "label": tensor
            }

        Whole:

            {
                "whole": tensor,
                "label": tensor
            }

        ROI:

            {
                "roi": tensor,
                "label": tensor
            }

        Scenario C:

            {
                "whole": tensor,
                "roi": tensor,
                "label": tensor
            }

        Label keys accepted:

            label
            target
            is_pathologic
        """

        if not isinstance(
            batch,
            Mapping,
        ):
            raise TypeError(
                "Trainer currently expects each DataLoader "
                "batch to be a dictionary/mapping."
            )

        # -------------------------------------------------------------
        # Label
        # -------------------------------------------------------------

        target = None

        for key in (
            "label",
            "target",
            "is_pathologic",
        ):

            if key in batch:

                target = batch[key]

                break

        if target is None:

            raise KeyError(
                "Batch does not contain a label. "
                "Expected one of: "
                "'label', 'target', 'is_pathologic'."
            )

        if not isinstance(
            target,
            Tensor,
        ):
            target = torch.as_tensor(
                target
            )

        target = prepare_binary_targets(
            target,
            device=self.device,
        )

        # -------------------------------------------------------------
        # UID
        # -------------------------------------------------------------

        raw_uids = batch.get(
            "uid",
            [],
        )

        if isinstance(
            raw_uids,
            str,
        ):

            uids = [
                raw_uids
            ]

        elif isinstance(
            raw_uids,
            Tensor,
        ):

            uids = [
                str(item)
                for item
                in raw_uids.tolist()
            ]

        elif raw_uids:

            uids = [
                str(item)
                for item
                in raw_uids
            ]

        else:

            uids = []

        # -------------------------------------------------------------
        # Scenario C:
        #
        # whole + ROI
        # -------------------------------------------------------------

        if (
            "whole" in batch
            and "roi" in batch
        ):

            whole = batch[
                "whole"
            ]

            roi = batch[
                "roi"
            ]

            if not isinstance(
                whole,
                Tensor,
            ):
                raise TypeError(
                    "'whole' must be a Tensor."
                )

            if not isinstance(
                roi,
                Tensor,
            ):
                raise TypeError(
                    "'roi' must be a Tensor."
                )

            inputs = (
                self._move_tensor(
                    whole
                ),
                self._move_tensor(
                    roi
                ),
            )

            return (
                inputs,
                target,
                uids,
            )

        # -------------------------------------------------------------
        # Explicit generic input
        # -------------------------------------------------------------

        if "input" in batch:

            x = batch[
                "input"
            ]

        elif "image" in batch:

            x = batch[
                "image"
            ]

        elif "whole" in batch:

            x = batch[
                "whole"
            ]

        elif "roi" in batch:

            x = batch[
                "roi"
            ]

        else:

            raise KeyError(
                "Could not determine model input from batch."
            )

        if not isinstance(
            x,
            Tensor,
        ):
            raise TypeError(
                "Model input must be a Tensor."
            )

        x = self._move_tensor(
            x
        )

        return (
            x,
            target,
            uids,
        )

    # =================================================================
    # MODEL FORWARD
    # =================================================================

    def _forward_model(
        self,
        inputs: Any,
    ) -> Tensor:
        """
        Support both:

            model(x)

        and Scenario C:

            model(whole, roi)
        """

        if isinstance(
            inputs,
            tuple,
        ):

            logits = self.model(
                *inputs
            )

        else:

            logits = self.model(
                inputs
            )

        if not isinstance(
            logits,
            Tensor,
        ):
            raise TypeError(
                "Model must return torch.Tensor logits."
            )

        if logits.ndim == 1:

            logits = logits.unsqueeze(
                1
            )

        if (
            logits.ndim != 2
            or logits.shape[1] != 1
        ):

            raise RuntimeError(
                "Binary classifier must return "
                "[B,1] raw logits. "
                f"Received {tuple(logits.shape)}."
            )

        return logits

    # =================================================================
    # TRAIN ONE EPOCH
    # =================================================================

    def train_one_epoch(
        self,
        loader: DataLoader,
    ) -> EpochOutput:
        """
        Perform one complete training epoch.
        """

        self.model.train()

        total_loss = 0.0
        total_samples = 0

        all_logits = []
        all_targets = []
        all_uids: list[str] = []

        for batch in loader:

            (
                inputs,
                targets,
                uids,
            ) = self._parse_batch(
                batch
            )

            batch_size = (
                targets.shape[0]
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            with self._autocast_context():

                logits = (
                    self._forward_model(
                        inputs
                    )
                )

                if (
                    logits.shape
                    != targets.shape
                ):

                    raise RuntimeError(
                        "Logit/target shape mismatch.\n"
                        f"Logits : {tuple(logits.shape)}\n"
                        f"Targets: {tuple(targets.shape)}"
                    )

                loss = self.criterion(
                    logits,
                    targets,
                )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Non-finite training loss detected: "
                    f"{loss.item()}"
                )

            # ---------------------------------------------------------
            # Backpropagation
            # ---------------------------------------------------------

            self.scaler.scale(
                loss
            ).backward()

            # ---------------------------------------------------------
            # Gradient clipping
            # ---------------------------------------------------------

            if (
                self.gradient_clip_norm
                is not None
            ):

                self.scaler.unscale_(
                    self.optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=(
                        self.gradient_clip_norm
                    ),
                )

            # ---------------------------------------------------------
            # Parameter update
            # ---------------------------------------------------------

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            # ---------------------------------------------------------
            # Statistics
            # ---------------------------------------------------------

            total_loss += (
                float(
                    loss.detach().item()
                )
                * batch_size
            )

            total_samples += (
                batch_size
            )

            all_logits.append(
                logits.detach().float().cpu()
            )

            all_targets.append(
                targets.detach().float().cpu()
            )

            all_uids.extend(
                uids
            )

        if total_samples == 0:
            raise RuntimeError(
                "Training DataLoader produced zero samples."
            )

        logits_tensor = torch.cat(
            all_logits,
            dim=0,
        )

        targets_tensor = torch.cat(
            all_targets,
            dim=0,
        )

        probabilities = torch.sigmoid(
            logits_tensor
        )

        if all_uids and len(all_uids) != total_samples:
            raise RuntimeError(
                "UIDs were present for only part of the training epoch. "
                f"UID count={len(all_uids)}, samples={total_samples}."
            )

        return EpochOutput(
            loss=(
                total_loss
                / total_samples
            ),
            logits=logits_tensor,
            probabilities=probabilities,
            targets=targets_tensor,
            uids=all_uids,
            n_samples=total_samples,
        )

    # =================================================================
    # VALIDATION
    # =================================================================

    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader,
    ) -> EpochOutput:
        """
        Deterministic validation.

        IMPORTANT:
            validation augmentation must already be OFF
            in the validation Dataset/DataLoader.
        """

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_logits = []
        all_targets = []
        all_uids: list[str] = []

        for batch in loader:

            (
                inputs,
                targets,
                uids,
            ) = self._parse_batch(
                batch
            )

            batch_size = (
                targets.shape[0]
            )

            with self._autocast_context():

                logits = (
                    self._forward_model(
                        inputs
                    )
                )

                if (
                    logits.shape
                    != targets.shape
                ):

                    raise RuntimeError(
                        "Logit/target shape mismatch during validation."
                    )

                loss = self.criterion(
                    logits,
                    targets,
                )

            if not torch.isfinite(
                loss
            ):
                raise FloatingPointError(
                    "Non-finite validation loss detected."
                )

            total_loss += (
                float(
                    loss.detach().item()
                )
                * batch_size
            )

            total_samples += (
                batch_size
            )

            all_logits.append(
                logits.detach().float().cpu()
            )

            all_targets.append(
                targets.detach().float().cpu()
            )

            all_uids.extend(
                uids
            )

        if total_samples == 0:
            raise RuntimeError(
                "Validation DataLoader produced zero samples."
            )

        logits_tensor = torch.cat(
            all_logits,
            dim=0,
        )

        targets_tensor = torch.cat(
            all_targets,
            dim=0,
        )

        probabilities = torch.sigmoid(
            logits_tensor
        )

        if all_uids and len(all_uids) != total_samples:
            raise RuntimeError(
                "UIDs were present for only part of the validation epoch. "
                f"UID count={len(all_uids)}, samples={total_samples}."
            )

        return EpochOutput(
            loss=(
                total_loss
                / total_samples
            ),
            logits=logits_tensor,
            probabilities=probabilities,
            targets=targets_tensor,
            uids=all_uids,
            n_samples=total_samples,
        )

    # =================================================================
    # MODEL-SELECTION METRIC
    # =================================================================

    def _calculate_selection_metric(
        self,
        validation: EpochOutput,
    ) -> float:
        """
        Return metric used for:
            best checkpoint
            early stopping

        Default:
            validation loss
        """

        if self.selection_metric_fn is None:

            return float(
                validation.loss
            )

        value = (
            self.selection_metric_fn(
                validation.targets,
                validation.probabilities,
            )
        )

        return float(
            value
        )

    def _is_better(
        self,
        value: float,
    ) -> bool:

        if self.best_metric is None:
            return True

        if self.selection_mode == "min":

            return (
                value
                < self.best_metric
            )

        return (
            value
            > self.best_metric
        )

    # =================================================================
    # CHECKPOINT SAVE
    # =================================================================

    def _save(
        self,
        *,
        path: Path,
        epoch: int,
        metric: float,
    ) -> None:

        checkpoint_metadata = dict(
            self.metadata
        )

        checkpoint_metadata[
            "trainer_state"
        ] = {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "selection_metric_name": self.selection_metric_name,
            "selection_mode": self.selection_mode,
        }

        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            early_stopping=(
                self.early_stopping
            ),
            epoch=epoch,
            monitor_name=(
                self.selection_metric_name
            ),
            monitor_value=metric,
            config=self.config,
            metadata=checkpoint_metadata,
        )

    # =================================================================
    # RESUME
    # =================================================================

    def resume_from_checkpoint(
        self,
        path: str | Path,
    ) -> int:
        """
        Restore complete training state.

        Returns
        -------
        int
            Next epoch number.
        """

        checkpoint = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            early_stopping=(
                self.early_stopping
            ),
            map_location=self.device,
            strict_model=True,
        )

        epoch = int(
            checkpoint[
                "epoch"
            ]
        )

        monitor_value = float(
            checkpoint[
                "monitor_value"
            ]
        )

        # Restore historical best state. A last_model.pt checkpoint stores
        # the CURRENT epoch's monitor value, which is not necessarily the
        # best value seen before the interruption. New checkpoints persist
        # explicit trainer_state; older checkpoints can still recover the
        # best state from EarlyStopping when available.
        trainer_state = (
            checkpoint.get(
                "metadata",
                {},
            ).get(
                "trainer_state",
                {},
            )
        )

        restored_best_metric = trainer_state.get(
            "best_metric"
        )
        restored_best_epoch = trainer_state.get(
            "best_epoch"
        )

        if (
            restored_best_metric is not None
            and restored_best_epoch is not None
        ):
            self.best_metric = float(
                restored_best_metric
            )
            self.best_epoch = int(
                restored_best_epoch
            )

        elif (
            self.early_stopping is not None
            and self.early_stopping.best_value is not None
            and self.early_stopping.best_epoch is not None
        ):
            self.best_metric = float(
                self.early_stopping.best_value
            )
            self.best_epoch = int(
                self.early_stopping.best_epoch
            )

        else:
            # Backward-compatible fallback for older checkpoints that do
            # not contain historical trainer state.
            self.best_metric = monitor_value
            self.best_epoch = epoch

        return epoch + 1

    # =================================================================
    # FULL TRAINING
    # =================================================================

    def fit(
        self,
        *,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        max_epochs: int,
        start_epoch: int = 1,
        verbose: bool = True,
    ) -> TrainingResult:
        """
        Full model training.

        This is the first project stage that performs:
            optimizer updates
            backward propagation
            weight updates
        """

        if max_epochs < 1:
            raise ValueError(
                "max_epochs must be >= 1."
            )

        if start_epoch < 1:
            raise ValueError(
                "start_epoch must be >= 1."
            )

        if start_epoch > max_epochs:
            raise ValueError(
                "start_epoch cannot be greater than max_epochs. "
                f"Received start_epoch={start_epoch}, "
                f"max_epochs={max_epochs}."
            )

        history: list[
            dict[str, Any]
        ] = []

        stopped_early = False

        epochs_completed = 0

        for epoch in range(
            start_epoch,
            max_epochs + 1,
        ):

            # ---------------------------------------------------------
            # TRAIN
            # ---------------------------------------------------------

            train_output = (
                self.train_one_epoch(
                    train_loader
                )
            )

            # ---------------------------------------------------------
            # VALIDATE
            # ---------------------------------------------------------

            validation_output = (
                self.validate(
                    validation_loader
                )
            )

            # ---------------------------------------------------------
            # Metric used for checkpoint selection.
            # ---------------------------------------------------------

            selection_metric = (
                self._calculate_selection_metric(
                    validation_output
                )
            )

            # ---------------------------------------------------------
            # Scheduler
            #
            # ReduceLROnPlateau monitors validation loss.
            #
            # Cosine ignores metric.
            # ---------------------------------------------------------

            if self.scheduler is not None:

                self.scheduler.step(
                    metric=(
                        validation_output.loss
                    )
                )

            # ---------------------------------------------------------
            # Early stopping
            # ---------------------------------------------------------

            if (
                self.early_stopping
                is not None
            ):

                decision = (
                    self.early_stopping.step(
                        value=(
                            selection_metric
                        ),
                        epoch=epoch,
                    )
                )

                improved = (
                    decision.improved
                )

                should_stop = (
                    decision.should_stop
                )

            else:

                improved = self._is_better(
                    selection_metric
                )

                should_stop = False

            # ---------------------------------------------------------
            # Best model
            # ---------------------------------------------------------

            if improved:

                self.best_metric = (
                    selection_metric
                )

                self.best_epoch = (
                    epoch
                )

                if (
                    self.best_checkpoint_path
                    is not None
                ):

                    self._save(
                        path=(
                            self.best_checkpoint_path
                        ),
                        epoch=epoch,
                        metric=(
                            selection_metric
                        ),
                    )

            # ---------------------------------------------------------
            # Current learning rates
            # ---------------------------------------------------------

            learning_rates = (
                get_learning_rates(
                    self.optimizer
                )
            )

            # ---------------------------------------------------------
            # History row
            # ---------------------------------------------------------

            row = {
                "epoch": epoch,

                "train_loss": (
                    train_output.loss
                ),

                "val_loss": (
                    validation_output.loss
                ),

                "selection_metric_name": (
                    self.selection_metric_name
                ),

                "selection_metric": (
                    selection_metric
                ),

                "learning_rates": (
                    learning_rates
                ),

                "best_epoch": (
                    self.best_epoch
                ),

                "best_metric": (
                    self.best_metric
                ),
            }

            history.append(
                row
            )

            epochs_completed = (
                epoch
            )

            # ---------------------------------------------------------
            # Always save last checkpoint.
            # ---------------------------------------------------------

            if (
                self.last_checkpoint_path
                is not None
            ):

                self._save(
                    path=(
                        self.last_checkpoint_path
                    ),
                    epoch=epoch,
                    metric=(
                        selection_metric
                    ),
                )

            # ---------------------------------------------------------
            # Console output
            # ---------------------------------------------------------

            if verbose:

                lr_string = ",".join(
                    f"{lr:.3e}"
                    for lr
                    in learning_rates
                )

                print(
                    f"Epoch {epoch:03d}/{max_epochs:03d} | "
                    f"train_loss={train_output.loss:.6f} | "
                    f"val_loss={validation_output.loss:.6f} | "
                    f"{self.selection_metric_name}="
                    f"{selection_metric:.6f} | "
                    f"lr={lr_string}"
                )

            # ---------------------------------------------------------
            # Stop only AFTER saving current state.
            # ---------------------------------------------------------

            if should_stop:

                stopped_early = True

                if verbose:

                    print(
                        "Early stopping triggered at "
                        f"epoch {epoch}."
                    )

                break

        if self.best_epoch is None:
            raise RuntimeError(
                "Training completed without identifying "
                "a best epoch."
            )

        if self.best_metric is None:
            raise RuntimeError(
                "Training completed without a best metric."
            )

        return TrainingResult(
            best_epoch=self.best_epoch,
            best_metric=self.best_metric,
            epochs_completed=(
                epochs_completed
            ),
            stopped_early=(
                stopped_early
            ),
            best_checkpoint_path=(
                self.best_checkpoint_path
            ),
            last_checkpoint_path=(
                self.last_checkpoint_path
            ),
            history=history,
        )