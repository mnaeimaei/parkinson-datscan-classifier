"""
training_config.py

Central configuration for all model-training experiments.

Purpose
-------
Freeze shared training behavior across:

    10 models
    x
    6 scenarios
    x
    5 CV folds

The experiment scripts should change ONLY the dimensions that define
the experiment:

    model
    scenario
    fold
    batch size if required by memory

Shared optimizer/loss/scheduler/early-stopping behavior should come
from this file.

Important distinction
---------------------
Fold checkpoint selection:
    validation loss

Final experiment ranking:
    global OOF AUROC
    unless the competition specifies another official metric.

This prevents fold checkpoint selection and experiment comparison from
being accidentally conflated.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import (
    asdict,
    dataclass,
    field,
)
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.augmentation.augmentations import AugmentationConfig


# =====================================================================
# SCENARIO DEFINITIONS
# =====================================================================

InputScenario = Literal[
    "whole",
    "roi",
    "whole_roi",
]


@dataclass(frozen=True)
class ScenarioDefinition:

    scenario_id: int

    name: str

    input_type: InputScenario

    augmentation: bool


SCENARIOS = {
    1: ScenarioDefinition(
        scenario_id=1,
        name="scenario01_whole_noaug",
        input_type="whole",
        augmentation=False,
    ),

    2: ScenarioDefinition(
        scenario_id=2,
        name="scenario02_roi_noaug",
        input_type="roi",
        augmentation=False,
    ),

    3: ScenarioDefinition(
        scenario_id=3,
        name="scenario03_whole_roi_noaug",
        input_type="whole_roi",
        augmentation=False,
    ),

    4: ScenarioDefinition(
        scenario_id=4,
        name="scenario04_whole_aug",
        input_type="whole",
        augmentation=True,
    ),

    5: ScenarioDefinition(
        scenario_id=5,
        name="scenario05_roi_aug",
        input_type="roi",
        augmentation=True,
    ),

    6: ScenarioDefinition(
        scenario_id=6,
        name="scenario06_whole_roi_aug",
        input_type="whole_roi",
        augmentation=True,
    ),
}


# =====================================================================
# LOSS
# =====================================================================


@dataclass(frozen=True)
class LossConfig:

    name: str = "bce_with_logits"

    # Modest imbalance:
    #
    # default baseline does not reweight.
    pos_weight: float | None = None


# =====================================================================
# OPTIMIZER
# =====================================================================


@dataclass(frozen=True)
class OptimizerConfig:

    name: str = "adamw"

    learning_rate: float = 1e-4

    weight_decay: float = 1e-4

    beta1: float = 0.9

    beta2: float = 0.999

    eps: float = 1e-8


# =====================================================================
# SCHEDULER
# =====================================================================


@dataclass(frozen=True)
class SchedulerConfig:

    name: str = "reduce_on_plateau"

    factor: float = 0.5

    patience: int = 5

    threshold: float = 1e-4

    min_learning_rate: float = 1e-7


# =====================================================================
# EARLY STOPPING
# =====================================================================


@dataclass(frozen=True)
class EarlyStoppingConfig:

    enabled: bool = True

    # Fold checkpoint selection uses validation loss.
    monitor: str = "val_loss"

    mode: Literal[
        "min",
        "max",
    ] = "min"

    patience: int = 15

    min_delta: float = 0.0

    # Let the network train for a few epochs before bad epochs
    # contribute toward stopping.
    warmup_epochs: int = 5


# =====================================================================
# AUGMENTATION
# =====================================================================


@dataclass(frozen=True)
class AugmentationTrainingConfig(AugmentationConfig):
    """
    Shared training augmentation configuration.

    This intentionally inherits the exact augmentation implementation
    config so the project has one source of truth for augmentation
    hyperparameters.  ScenarioDefinition.augmentation decides whether
    the augmenter is actually enabled for a specific experiment.
    """

    pass


# =====================================================================
# DATA LOADER
# =====================================================================


@dataclass(frozen=True)
class DataLoaderConfig:

    num_workers: int = 4

    pin_memory: bool = True

    persistent_workers: bool = True

    prefetch_factor: int = 2

    # Validation and inference are never shuffled.
    train_shuffle: bool = True

    validation_shuffle: bool = False

    drop_last_train: bool = False

    drop_last_validation: bool = False


# =====================================================================
# REPRODUCIBILITY
# =====================================================================


@dataclass(frozen=True)
class ReproducibilityConfig:

    seed: int = 2026

    # True gives stronger reproducibility but may reduce speed and may
    # fail on operations without deterministic implementations.
    deterministic_algorithms: bool = False

    deterministic_warn_only: bool = True

    cudnn_benchmark: bool = False


# =====================================================================
# CORE TRAINING
# =====================================================================


@dataclass(frozen=True)
class CoreTrainingConfig:

    max_epochs: int = 100

    use_amp: bool = True

    gradient_clip_norm: float | None = 1.0

    # -------------------------------------------------------------
    # Fold-level checkpoint policy
    # -------------------------------------------------------------

    checkpoint_metric: str = "val_loss"

    checkpoint_mode: Literal[
        "min",
        "max",
    ] = "min"

    # -------------------------------------------------------------
    # Experiment-ranking policy
    #
    # This is used AFTER complete 5-fold OOF evaluation.
    # -------------------------------------------------------------

    primary_cv_metric: str = "auroc"

    primary_cv_mode: Literal[
        "min",
        "max",
    ] = "max"

    # Frozen hard-decision baseline.
    classification_threshold: float = 0.5


# =====================================================================
# FULL SHARED CONFIG
# =====================================================================


@dataclass(frozen=True)
class TrainingConfig:

    loss: LossConfig = field(
        default_factory=LossConfig
    )

    optimizer: OptimizerConfig = field(
        default_factory=OptimizerConfig
    )

    scheduler: SchedulerConfig = field(
        default_factory=SchedulerConfig
    )

    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig
    )

    augmentation: AugmentationTrainingConfig = field(
        default_factory=AugmentationTrainingConfig
    )

    dataloader: DataLoaderConfig = field(
        default_factory=DataLoaderConfig
    )

    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )

    training: CoreTrainingConfig = field(
        default_factory=CoreTrainingConfig
    )

    def validate(
        self,
    ) -> None:

        if (
            self.training.max_epochs
            < 1
        ):

            raise ValueError(
                "max_epochs must be >= 1."
            )

        if (
            self.optimizer.learning_rate
            <= 0
        ):

            raise ValueError(
                "learning_rate must be > 0."
            )

        if (
            self.optimizer.weight_decay
            < 0
        ):

            raise ValueError(
                "weight_decay must be >= 0."
            )

        if (
            self.early_stopping.patience
            < 1
        ):

            raise ValueError(
                "early-stopping patience must be >= 1."
            )

        # The augmentation config used by training is the same dataclass
        # consumed by augmentation/augmentations.py.
        self.augmentation.validate()

        if self.loss.name.lower().strip() not in {
            "bce_with_logits",
            "bce",
        }:
            raise ValueError(
                "This binary-classification pipeline currently supports "
                "BCEWithLogitsLoss only."
            )

        if (
            self.loss.pos_weight is not None
            and self.loss.pos_weight <= 0
        ):
            raise ValueError(
                "loss.pos_weight must be > 0 or None."
            )

        if self.optimizer.name.lower().strip() not in {
            "adamw", "adam", "sgd"
        }:
            raise ValueError(
                f"Unsupported optimizer: {self.optimizer.name!r}."
            )

        if not (0.0 <= self.optimizer.beta1 < 1.0):
            raise ValueError("optimizer.beta1 must be in [0,1).")

        if not (0.0 <= self.optimizer.beta2 < 1.0):
            raise ValueError("optimizer.beta2 must be in [0,1).")

        if self.optimizer.eps <= 0:
            raise ValueError("optimizer.eps must be > 0.")

        if self.scheduler.name.lower().strip() not in {
            "none", "off", "reduce_on_plateau", "plateau",
            "cosine", "cosine_annealing",
        }:
            raise ValueError(
                f"Unsupported scheduler: {self.scheduler.name!r}."
            )

        if not 0.0 < self.scheduler.factor < 1.0:
            raise ValueError("scheduler.factor must be in (0,1).")

        if self.scheduler.patience < 0:
            raise ValueError("scheduler.patience must be >= 0.")

        if self.scheduler.threshold < 0:
            raise ValueError("scheduler.threshold must be >= 0.")

        if self.scheduler.min_learning_rate < 0:
            raise ValueError("scheduler.min_learning_rate must be >= 0.")

        if self.dataloader.num_workers < 0:
            raise ValueError("dataloader.num_workers must be >= 0.")

        if self.dataloader.prefetch_factor < 1:
            raise ValueError("dataloader.prefetch_factor must be >= 1.")

        if self.early_stopping.enabled:
            if (
                self.early_stopping.monitor
                != self.training.checkpoint_metric
            ):
                raise ValueError(
                    "early_stopping.monitor and "
                    "training.checkpoint_metric must match."
                )

            if (
                self.early_stopping.mode
                != self.training.checkpoint_mode
            ):
                raise ValueError(
                    "early_stopping.mode and "
                    "training.checkpoint_mode must match."
                )

        threshold = (
            self.training
            .classification_threshold
        )

        if not 0 <= threshold <= 1:

            raise ValueError(
                "classification_threshold must be in [0,1]."
            )


# =====================================================================
# PER-EXPERIMENT CONFIG
# =====================================================================


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration specific to one model/scenario/fold training run.
    """

    model_name: str

    scenario_id: int

    fold: int

    batch_size: int

    output_dir: str

    shared: TrainingConfig = field(
        default_factory=TrainingConfig
    )

    @property
    def scenario(
        self,
    ) -> ScenarioDefinition:

        if (
            self.scenario_id
            not in SCENARIOS
        ):

            raise ValueError(
                f"Unknown scenario ID: "
                f"{self.scenario_id}"
            )

        return SCENARIOS[
            self.scenario_id
        ]

    @property
    def augmentation_enabled(
        self,
    ) -> bool:

        return (
            self.scenario
            .augmentation
        )

    @property
    def input_type(
        self,
    ) -> InputScenario:

        return (
            self.scenario
            .input_type
        )

    def validate(
        self,
    ) -> None:

        self.shared.validate()

        if not self.model_name:

            raise ValueError(
                "model_name cannot be empty."
            )

        if self.fold not in {
            0,
            1,
            2,
            3,
            4,
        }:

            raise ValueError(
                "Fold must be one of 0,1,2,3,4."
            )

        if self.batch_size < 1:

            raise ValueError(
                "batch_size must be >= 1."
            )

        # Trigger scenario validation.
        _ = self.scenario


# =====================================================================
# BATCH-SIZE STARTING POINTS
# =====================================================================

# IMPORTANT:
#
# These are deliberately conservative STARTING values.
#
# Batch size must be finalized using a GPU-memory dry run on the actual
# training infrastructure.
#
# They are not scientific hyperparameters unless deliberately varied.
#
# Scenario keys:
#     whole
#     roi
#     whole_roi

INITIAL_BATCH_SIZE_SUGGESTIONS = {

    "model01_simple3d_scratch": {
        "whole": 2,
        "roi": 8,
        "whole_roi": 1,
    },

    "model02_resnet18_3d_scratch": {
        "whole": 1,
        "roi": 4,
        "whole_roi": 1,
    },

    "model03_resnet18_3d_medicalnet_pretrained": {
        "whole": 1,
        "roi": 4,
        "whole_roi": 1,
    },

    "model04_resnet18_2p5d_attention_imagenet_pretrained": {
        "whole": 2,
        "roi": 4,
        "whole_roi": 1,
    },

    "model05_r3d18_scratch": {
        "whole": 1,
        "roi": 2,
        "whole_roi": 1,
    },

    "model06_r3d18_kinetics400_pretrained": {
        "whole": 1,
        "roi": 2,
        "whole_roi": 1,
    },

    "model07_resnet18_2p5d_attention_scratch": {
        "whole": 2,
        "roi": 4,
        "whole_roi": 1,
    },

    "model08_densenet121_3d_scratch": {
        "whole": 1,
        "roi": 2,
        "whole_roi": 1,
    },

    "model09_swin3d_t_scratch": {
        "whole": 1,
        "roi": 1,
        "whole_roi": 1,
    },

    "model10_swin3d_t_kinetics400_pretrained": {
        "whole": 1,
        "roi": 1,
        "whole_roi": 1,
    },
}


def suggested_batch_size(
    model_name: str,
    input_type: InputScenario,
) -> int:
    """
    Return conservative starting batch size.

    Must still be tested against actual GPU VRAM.
    """

    if (
        model_name
        not in INITIAL_BATCH_SIZE_SUGGESTIONS
    ):

        return 1

    return int(
        INITIAL_BATCH_SIZE_SUGGESTIONS[
            model_name
        ].get(
            input_type,
            1,
        )
    )


# =====================================================================
# RANDOM SEED
# =====================================================================


def seed_everything(
    config: ReproducibilityConfig,
) -> None:
    """
    Seed Python, NumPy and PyTorch.

    Note:
        PyTorch does not guarantee perfect reproducibility across every
        release/platform/device combination.

    This function controls the major randomness sources within one
    experiment environment.
    """

    seed = int(
        config.seed
    )

    os.environ[
        "PYTHONHASHSEED"
    ] = str(
        seed
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = (
        config.cudnn_benchmark
    )

    if hasattr(
        torch.backends.cudnn,
        "deterministic",
    ):

        torch.backends.cudnn.deterministic = (
            config.deterministic_algorithms
        )

    torch.use_deterministic_algorithms(
        config.deterministic_algorithms,
        warn_only=(
            config.deterministic_warn_only
        ),
    )


# =====================================================================
# DATALOADER WORKER SEED
# =====================================================================


def seed_worker(
    worker_id: int,
) -> None:
    """
    DataLoader worker initialization.

    Uses PyTorch's worker seed to seed NumPy and Python RNGs.
    """

    del worker_id

    worker_seed = (
        torch.initial_seed()
        % 2**32
    )

    np.random.seed(
        worker_seed
    )

    random.seed(
        worker_seed
    )


def make_dataloader_generator(
    seed: int,
) -> torch.Generator:
    """
    Generator useful for deterministic DataLoader shuffling.
    """

    generator = torch.Generator()

    generator.manual_seed(
        int(
            seed
        )
    )

    return generator


# =====================================================================
# SERIALIZATION
# =====================================================================


def save_experiment_config(
    config: ExperimentConfig,
    path: str | Path,
) -> Path:
    """
    Save exact configuration used by one fold.
    """

    config.validate()

    path = (
        Path(
            path
        )
        .expanduser()
        .resolve()
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = asdict(
        config
    )

    payload[
        "resolved_scenario"
    ] = asdict(
        config.scenario
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )

    return path


# =====================================================================
# DEFAULT SHARED CONFIG
# =====================================================================


DEFAULT_TRAINING_CONFIG = (
    TrainingConfig()
)