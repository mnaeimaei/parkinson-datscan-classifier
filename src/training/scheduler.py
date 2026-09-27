"""
scheduler.py

Learning-rate scheduler construction and unified stepping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
)


SchedulerStepMode = Literal[
    "none",
    "epoch",
    "metric",
]


@dataclass
class SchedulerController:
    """
    Uniform interface around different scheduler types.

    step_mode:
        none
            no scheduler

        epoch
            scheduler.step()

        metric
            scheduler.step(validation_metric)
    """

    scheduler: Optional[Any]

    step_mode: SchedulerStepMode

    name: str

    def step(
        self,
        *,
        metric: Optional[float] = None,
    ) -> None:

        if self.scheduler is None:
            return

        if self.step_mode == "epoch":

            self.scheduler.step()

            return

        if self.step_mode == "metric":

            if metric is None:
                raise ValueError(
                    f"Scheduler {self.name} requires "
                    "a validation metric."
                )

            self.scheduler.step(
                metric
            )

            return

        if self.step_mode == "none":
            return

        raise RuntimeError(
            f"Unknown step mode: {self.step_mode}"
        )

    def state_dict(
        self,
    ) -> dict:
        """
        Return scheduler state for checkpointing.
        """

        if self.scheduler is None:
            return {
                "name": self.name,
                "step_mode": self.step_mode,
                "scheduler_state": None,
            }

        return {
            "name": self.name,
            "step_mode": self.step_mode,
            "scheduler_state": (
                self.scheduler.state_dict()
            ),
        }

    def load_state_dict(
        self,
        state_dict: dict,
    ) -> None:
        """
        Restore scheduler state.
        """

        scheduler_state = (
            state_dict.get(
                "scheduler_state"
            )
        )

        if (
            self.scheduler is not None
            and scheduler_state is not None
        ):
            self.scheduler.load_state_dict(
                scheduler_state
            )


def build_scheduler(
    optimizer: Optimizer,
    *,
    name: str = "reduce_on_plateau",
    max_epochs: int = 100,
    plateau_factor: float = 0.5,
    plateau_patience: int = 5,
    plateau_threshold: float = 1e-4,
    min_learning_rate: float = 1e-7,
) -> SchedulerController:
    """
    Construct learning-rate scheduler.

    Supported
    ---------
    none

    reduce_on_plateau
        monitors validation loss

    cosine
        cosine decay over max_epochs
    """

    name = name.lower().strip()

    if name in {
        "none",
        "off",
    }:

        return SchedulerController(
            scheduler=None,
            step_mode="none",
            name="none",
        )

    if name in {
        "reduce_on_plateau",
        "plateau",
    }:

        if not 0 < plateau_factor < 1:
            raise ValueError(
                "plateau_factor must satisfy 0 < factor < 1."
            )

        if plateau_patience < 0:
            raise ValueError(
                "plateau_patience must be >= 0."
            )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            threshold_mode="rel",
            cooldown=0,
            min_lr=min_learning_rate,
        )

        return SchedulerController(
            scheduler=scheduler,
            step_mode="metric",
            name="reduce_on_plateau",
        )

    if name in {
        "cosine",
        "cosine_annealing",
    }:

        if max_epochs <= 0:
            raise ValueError(
                "max_epochs must be > 0."
            )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=min_learning_rate,
        )

        return SchedulerController(
            scheduler=scheduler,
            step_mode="epoch",
            name="cosine",
        )

    raise ValueError(
        f"Unknown scheduler: {name!r}."
    )