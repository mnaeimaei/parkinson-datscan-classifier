#!/usr/bin/env python3
"""
E5 — Scenario 04
Whole / Augmentation.

Model:
    src.models.model05_r3d18_scratch

Architecture:
    Torchvision R3D-18 trained completely from scratch.

Input:
    [B, 1, D, H, W]

Interpretation:
    D is treated as Torchvision's temporal dimension.

Native single-channel stem:
    Conv3d(
        1 -> 64,
        kernel_size=(3,7,7),
        stride=(1,2,2),
        padding=(1,3,3),
        bias=False,
    )
    -> BatchNorm3d(64)
    -> ReLU

Residual stages:
    BasicBlock [2,2,2,2]
    channels 64 -> 128 -> 256 -> 512
    full 3x3x3 convolutions

Initialization:
    scratch / random only
    Conv3d stem initialized with Kaiming
    no Kinetics pretrained weights

Parameter count:
    33,147,969


Single-branch architecture:
    native grayscale R3D-18
    -> AdaptiveAvgPool3d
    -> 512-D subject feature
    -> Linear(512 -> 1)


Augmentation:
    ON for TRAINING ONLY.
    Uses src.augmentation.augmentations with spatial_mode="3d".

Validation/OOF:
    augmentation OFF.


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

from src.models.model05_r3d18_scratch import build_model

from scripts.experiments_script.common_cv_experiment import (
    ExperimentSpec,
    run_experiment,
)




SPEC = ExperimentSpec(
    experiment_name="model05_r3d18_scratch__scenario04_whole_aug",
    model_name="model05_r3d18_scratch",
    scenario_id=4,
    output_subdir=(
        "model05_r3d18_scratch_exp_data/"
        "scenario04_whole_aug_exp_data"
    ),
    model_builder=build_model,
    augmentation_spatial_mode="3d",
    metadata={
        "experiment_id": "E5",
        "pretraining": "none",
        "model_family": "Torchvision R3D-18 3D CNN",
        "architecture": "torchvision_r3d_18",
        "input_channels": 1,
        "stem_kernel": [3, 7, 7],
        "stem_stride": [1, 2, 2],
        "stem_padding": [1, 3, 3],
        "block_structure": [2, 2, 2, 2],
        "stage_channels": [64, 128, 256, 512],
        "feature_dim": 512,
        "parameter_count": 33147969,
        "depth_role": "temporal_dimension",
        "grayscale_stem": "native_scratch_single_channel",
        "kinetics_normalization": False,
        "project_frozen_normalization_only": True,
        "fusion": "none",
        "feature_dim": 512,
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
