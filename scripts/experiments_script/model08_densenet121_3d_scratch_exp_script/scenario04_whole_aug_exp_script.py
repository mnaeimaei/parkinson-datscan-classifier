#!/usr/bin/env python3
"""
E8 — Scenario 04
Whole / Augmentation.

Model:
    src.models.model08_densenet121_3d_scratch

Architecture:
    MONAI 3D DenseNet-121 trained completely from scratch.

Input:
    [B, 1, D, H, W]

Frozen E8 configuration:
    spatial_dims   = 3
    in_channels    = 1
    out_channels   = 1
    init_features  = 64
    growth_rate    = 32
    bn_size        = 4
    block_config   = (6,12,24,16)
    dropout_prob   = 0.0

Stem:
    Conv3d(
        1 -> 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    -> BatchNorm3d
    -> ReLU
    -> MaxPool3d(kernel=3, stride=2, padding=1)

Dense blocks:
    [6,12,24,16]

Feature progression:
    64
    -> dense1 256 -> transition1 128
    -> dense2 512 -> transition2 256
    -> dense3 1024 -> transition3 512
    -> dense4 1024

Dense layer:
    BN -> ReLU -> Conv3d(1x1x1)
    -> BN -> ReLU -> Conv3d(3x3x3)

Transition:
    BN -> ReLU -> Conv3d(1x1x1)
    -> AvgPool3d(kernel=2, stride=2)

Final representation:
    ReLU
    -> AdaptiveAvgPool3d(1)
    -> Flatten
    -> 1024-D subject feature

Binary head:
    Linear(1024 -> 1)

Initialization:
    scratch only
    no pretrained weights


Single-branch architecture:
    MONAI DenseNet-121 convolutional features
    -> final ReLU
    -> AdaptiveAvgPool3d(1)
    -> Flatten
    -> 1024-D subject representation
    -> Linear(1024 -> 1)


Augmentation:
    ON for TRAINING ONLY.
    Uses src.augmentation.augmentations with spatial_mode="3d".

Validation/OOF:
    augmentation OFF.


No sigmoid inside the model.
No additional normalization.
No additional clipping.
No registration during training.
No resampling during training.
Frozen Step-10 manifest and Step-11 folds are used exactly.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model08_densenet121_3d_scratch import build_model

from scripts.experiments_script.common_cv_experiment import (
    ExperimentSpec,
    run_experiment,
)




SPEC = ExperimentSpec(
    experiment_name="model08_densenet121_3d_scratch__scenario04_whole_aug",
    model_name="model08_densenet121_3d_scratch",
    scenario_id=4,
    output_subdir=(
        "model08_densenet121_3d_scratch_exp_data/"
        "scenario04_whole_aug_exp_data"
    ),
    model_builder=build_model,
    augmentation_spatial_mode="3d",
    metadata={
        "experiment_id": "E8",
        "pretraining": "none",
        "model_family": "MONAI 3D DenseNet-121",
        "architecture": "monai_densenet121_3d",
        "spatial_dims": 3,
        "input_channels": 1,
        "init_features": 64,
        "growth_rate": 32,
        "bn_size": 4,
        "block_config": [6, 12, 24, 16],
        "dense_dropout_probability": 0.0,
        "feature_dim": 1024,
        "scratch_initialization": True,
        "project_frozen_normalization_only": True,
        "fusion": "none",
        "feature_dim": 1024,
        "augmentation_source": "src.augmentation.augmentations",
        "augmentation_scope": "training_only",
        "augmentation_spatial_mode": "3d",
    },
)


if __name__ == "__main__":
    raise SystemExit(
        run_experiment(
            SPEC,
            default_project_root=PROJECT_ROOT,
        )
    )
