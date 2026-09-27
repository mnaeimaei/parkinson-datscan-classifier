"""
early_stopping.py

Early-stopping logic independent of the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional


EarlyStoppingMode = Literal[
    "min",
    "max",
]


@dataclass(frozen=True)
class EarlyStoppingDecision:
    """
    Result of one early-stopping update.
    """

    improved: bool

    should_stop: bool

    best_value: float

    best_epoch: int

    bad_epochs: int


class EarlyStopping:
    """
    Generic early stopping.

    Examples
    --------
    Validation loss:

        mode="min"

    AUROC:

        mode="max"
    """

    def __init__(
        self,
        *,
        patience: int = 15,
        mode: EarlyStoppingMode = "min",
        min_delta: float = 0.0,
        warmup_epochs: int = 0,
    ) -> None:

        if patience < 1:
            raise ValueError(
                "patience must be >= 1."
            )

        if mode not in {
            "min",
            "max",
        }:
            raise ValueError(
                "mode must be 'min' or 'max'."
            )

        if min_delta < 0:
            raise ValueError(
                "min_delta must be >= 0."
            )

        if warmup_epochs < 0:
            raise ValueError(
                "warmup_epochs must be >= 0."
            )

        self.patience = patience
        self.mode = mode
        self.min_delta = float(
            min_delta
        )
        self.warmup_epochs = (
            warmup_epochs
        )

        self.best_value: Optional[
            float
        ] = None

        self.best_epoch: Optional[
            int
        ] = None

        self.bad_epochs = 0

    def _is_improvement(
        self,
        value: float,
    ) -> bool:

        if self.best_value is None:
            return True

        if self.mode == "min":

            return (
                value
                < self.best_value
                - self.min_delta
            )

        return (
            value
            > self.best_value
            + self.min_delta
        )

    def step(
        self,
        *,
        value: float,
        epoch: int,
    ) -> EarlyStoppingDecision:
        """
        Update early-stopping state.
        """

        if epoch < 1:
            raise ValueError(
                "epoch must start at 1."
            )

        if not math.isfinite(value):

            self.bad_epochs += 1

            should_stop = (
                epoch > self.warmup_epochs
                and self.bad_epochs
                >= self.patience
            )

            return EarlyStoppingDecision(
                improved=False,
                should_stop=should_stop,
                best_value=(
                    self.best_value
                    if self.best_value
                    is not None
                    else float("nan")
                ),
                best_epoch=(
                    self.best_epoch
                    if self.best_epoch
                    is not None
                    else -1
                ),
                bad_epochs=self.bad_epochs,
            )

        improved = (
            self._is_improvement(
                value
            )
        )

        if improved:

            self.best_value = float(
                value
            )

            self.best_epoch = int(
                epoch
            )

            self.bad_epochs = 0

        elif epoch > self.warmup_epochs:

            self.bad_epochs += 1

        should_stop = (
            epoch > self.warmup_epochs
            and self.bad_epochs
            >= self.patience
        )

        assert self.best_value is not None
        assert self.best_epoch is not None

        return EarlyStoppingDecision(
            improved=improved,
            should_stop=should_stop,
            best_value=self.best_value,
            best_epoch=self.best_epoch,
            bad_epochs=self.bad_epochs,
        )

    def state_dict(
        self,
    ) -> dict:
        """
        Return serializable early-stopping state.
        """

        return {
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "warmup_epochs": (
                self.warmup_epochs
            ),
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(
        self,
        state_dict: dict,
    ) -> None:
        """
        Restore early-stopping state.
        """

        self.best_value = (
            state_dict.get(
                "best_value"
            )
        )

        self.best_epoch = (
            state_dict.get(
                "best_epoch"
            )
        )

        self.bad_epochs = int(
            state_dict.get(
                "bad_epochs",
                0,
            )
        )

    def reset(
        self,
    ) -> None:
        """
        Clear history.
        """

        self.best_value = None
        self.best_epoch = None
        self.bad_epochs = 0