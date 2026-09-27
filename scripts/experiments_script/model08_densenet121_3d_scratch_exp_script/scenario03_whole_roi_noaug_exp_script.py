#!/usr/bin/env python3
"""
E8 — Scenario 03
Whole + ROI / No Augmentation.

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


Scenario-C architecture:
    whole -> E8.forward_features() -> 1024
    ROI   -> E8.forward_features() -> 1024
    concat -> 2048 -> Linear(2048 -> 1)

No image-level concatenation.
No branch weight sharing.


Augmentation:
    OFF for training, validation, and OOF prediction.


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

from src.models.dual_input_fusion import build_dual_input_fusion
from src.models.model08_densenet121_3d_scratch import (
    build_model as build_single_e8,
)

from scripts.experiments_script.common_cv_experiment import (
    ExperimentSpec,
    run_experiment,
)


def build_model():
    """
    Build a fresh dual-input E8 model for one CV fold.

    Every call creates:
        - one new scratch whole-volume DenseNet-121 branch
        - one new scratch ROI DenseNet-121 branch
        - one new late-fusion classifier

    The two branches are independent and do not share parameters.
    """
    return build_dual_input_fusion(
        whole_branch=build_single_e8(),
        roi_branch=build_single_e8(),
        dropout=0.0,
    )


SPEC = ExperimentSpec(
    experiment_name="model08_densenet121_3d_scratch__scenario03_whole_roi_noaug",
    model_name="model08_densenet121_3d_scratch",
    scenario_id=3,
    output_subdir=(
        "model08_densenet121_3d_scratch_exp_data/"
        "scenario03_whole_roi_noaug_exp_data"
    ),
    model_builder=build_model,
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
        "fusion": "late_feature_concatenation",
        "whole_feature_dim": 1024,
        "roi_feature_dim": 1024,
        "fusion_feature_dim": 2048,
        "fusion_dropout": 0.0,
        "weight_sharing": False,
        "augmentation_scope": "off",
    },
)


if __name__ == "__main__":
    raise SystemExit(
        run_experiment(
            SPEC,
            default_project_root=PROJECT_ROOT,
        )
    )
