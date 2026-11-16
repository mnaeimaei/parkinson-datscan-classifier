#!/usr/bin/env python3
"""
E6 — Scenario 01
Whole / No Augmentation.

Model:
    src.models.model06_r3d18_kinetics400_pretrained

Architecture:
    exactly identical to E5 / model05_r3d18_scratch.py

Input:
    [B, 1, D, H, W]

Depth:
    D is treated as Torchvision R3D-18's temporal dimension.

Stem:
    Conv3d(
        1 -> 64,
        kernel_size=(3,7,7),
        stride=(1,2,2),
        padding=(1,3,3),
        bias=False,
    )

Residual stages:
    BasicBlock [2,2,2,2]
    channels 64 -> 128 -> 256 -> 512
    full 3x3x3 convolutions

Pretraining:
    R3D_18_Weights.KINETICS400_V1

Original pretrained RGB stem:
    [64,3,3,7,7]

Frozen grayscale conversion:
    W_gray = W_R + W_G + W_B
    resulting stem: [64,1,3,7,7]

Pretrained Kinetics classifier:
    discarded

Task classifier:
    newly initialized Linear(512 -> 1)

IMPORTANT preprocessing rule:
    DO NOT use Kinetics RGB normalization/transforms.
    Use only the project's frozen DaT-SPECT preprocessing/normalization.


Single-branch architecture:
    Kinetics-pretrained grayscale R3D-18
    -> AdaptiveAvgPool3d
    -> 512-D subject feature
    -> new Linear(512 -> 1) binary head


Augmentation:
    OFF for training, validation, and OOF prediction.


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

from scripts.experiments_script.model06_r3d18_kinetics400_pretrained_exp_script.kinetics_pretrained_factory import (
    build_single_e6,
)

from scripts.experiments_script.common_cv_experiment import (
    ExperimentSpec,
    run_experiment,
)


def build_model():
    """Build one fresh Kinetics-pretrained E6 model for this CV fold."""
    return build_single_e6()


SPEC = ExperimentSpec(
    experiment_name="model06_r3d18_kinetics400_pretrained__scenario01_whole_noaug",
    model_name="model06_r3d18_kinetics400_pretrained",
    scenario_id=1,
    output_subdir=(
        "model06_r3d18_kinetics400_pretrained_exp_data/"
        "scenario01_whole_noaug_exp_data"
    ),
    model_builder=build_model,
    metadata={
        "experiment_id": "E6",
        "pretraining": "Kinetics-400",
        "torchvision_weights": "R3D_18_Weights.KINETICS400_V1",
        "model_family": "Torchvision R3D-18 3D CNN",
        "architecture": "torchvision_r3d_18",
        "e5_architecture_identical": True,
        "input_channels": 1,
        "stem_kernel": [3, 7, 7],
        "stem_stride": [1, 2, 2],
        "stem_padding": [1, 3, 3],
        "block_structure": [2, 2, 2, 2],
        "stage_channels": [64, 128, 256, 512],
        "feature_dim": 512,
        "parameter_count": 33147969,
        "depth_role": "temporal_dimension",
        "grayscale_stem": "RGB_SUM",
        "rgb_to_grayscale_formula": "W_gray=W_R+W_G+W_B",
        "original_kinetics_head_discarded": True,
        "new_binary_head": True,
        "kinetics_rgb_normalization": False,
        "torchvision_kinetics_transforms": False,
        "project_frozen_normalization_only": True,
        "pretrained_transfer_qc": "one_time_exact_tensor_check_per_process",
        "fusion": "none",
        "feature_dim": 512,
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
