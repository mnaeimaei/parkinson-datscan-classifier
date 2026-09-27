#!/usr/bin/env python3
"""
E10 — Scenario 03
Whole + ROI / No Augmentation.

Model:
    src.models.model10_swin3d_t_kinetics400_pretrained

Architecture:
    exactly identical to E9 / model09_swin3d_t_scratch.py

Input:
    [B,1,D,H,W]

Patch embedding:
    Conv3d(
        1 -> 96,
        kernel_size=(2,4,4),
        stride=(2,4,4),
        padding=0,
    )
    -> LayerNorm(96)

Swin stages:
    depths           = [2,2,6,2]
    attention heads  = [3,6,12,24]
    window size      = (8,7,7)
    channels         = 96 -> 192 -> 384 -> 768
    stochastic depth = 0.1

Pretraining:
    Swin3D_T_Weights.KINETICS400_V1

Original pretrained RGB patch embedding:
    [96,3,2,4,4]

Frozen grayscale conversion:
    W_gray = W_R + W_G + W_B

Final grayscale patch embedding:
    [96,1,2,4,4]

The pretrained patch-embedding bias is preserved exactly.

Original Kinetics classifier:
    Linear(768 -> 400)

Task classifier:
    newly initialized Linear(768 -> 1)

IMPORTANT preprocessing rule:
    DO NOT use Torchvision Kinetics transforms.
    DO NOT use Kinetics RGB mean/std normalization.
    DO NOT convert the DaT input to RGB.
    DO NOT apply Kinetics resize/crop preprocessing.

    Use only the project's frozen DaT-SPECT preprocessing and normalization.

Frozen input-shape compatibility:
    Whole [D,H,W]=[160,192,192]
      -> [80,48,48]
      -> [80,24,24]
      -> [80,12,12]
      -> [80,6,6]

    ROI [D,H,W]=[36,44,44]
      -> [18,11,11]
      -> [18,6,6]
      -> [18,3,3]
      -> [18,2,2]


Scenario-C:
    whole -> E10.forward_features() -> 768
    ROI   -> E10.forward_features() -> 768
    concat -> 1536 -> Linear(1536 -> 1)

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
from scripts.experiments_script.model10_swin3d_t_kinetics400_pretrained_exp_script.swin3d_kinetics_pretrained_factory import (
    build_single_e10,
)

from scripts.experiments_script.common_cv_experiment import (
    ExperimentSpec,
    run_experiment,
)


def build_model():
    """
    Build a fresh dual-input E10 model for one CV fold.

    Both branches:
        - are independent model objects,
        - start from the same official Kinetics-400 Swin3D-Tiny weights,
        - use the same RGB-sum grayscale patch-embedding adaptation,
        - do not share trainable parameters.
    """
    return build_dual_input_fusion(
        whole_branch=build_single_e10(),
        roi_branch=build_single_e10(),
        dropout=0.0,
    )


SPEC = ExperimentSpec(
    experiment_name="model10_swin3d_t_kinetics400_pretrained__scenario03_whole_roi_noaug",
    model_name="model10_swin3d_t_kinetics400_pretrained",
    scenario_id=3,
    output_subdir=(
        "model10_swin3d_t_kinetics400_pretrained_exp_data/"
        "scenario03_whole_roi_noaug_exp_data"
    ),
    model_builder=build_model,
    metadata={
        "experiment_id": "E10",
        "pretraining": "Kinetics-400",
        "torchvision_weights": "Swin3D_T_Weights.KINETICS400_V1",
        "model_family": "Torchvision Swin3D-Tiny",
        "architecture": "torchvision_swin3d_t",
        "e9_architecture_identical": True,
        "input_channels": 1,
        "patch_size": [2, 4, 4],
        "embedding_dim": 96,
        "stage_depths": [2, 2, 6, 2],
        "attention_heads": [3, 6, 12, 24],
        "window_size": [8, 7, 7],
        "stage_channels": [96, 192, 384, 768],
        "stochastic_depth_probability": 0.1,
        "feature_dim": 768,
        "parameter_count": 27845095,
        "grayscale_patch_method": "RGB_SUM",
        "rgb_to_grayscale_formula": "W_gray=W_R+W_G+W_B",
        "pretrained_patch_bias_preserved": True,
        "original_kinetics_head_discarded": True,
        "new_binary_head": True,
        "kinetics_rgb_normalization": False,
        "torchvision_kinetics_transforms": False,
        "project_frozen_normalization_only": True,
        "pretrained_transfer_qc": "one_time_exact_tensor_check_per_process",
        "whole_swin_stage_grids": [
            [80,48,48], [80,24,24], [80,12,12], [80,6,6]
        ],
        "roi_swin_stage_grids": [
            [18,11,11], [18,6,6], [18,3,3], [18,2,2]
        ],
        "fusion": "late_feature_concatenation",
        "whole_feature_dim": 768,
        "roi_feature_dim": 768,
        "fusion_feature_dim": 1536,
        "fusion_dropout": 0.0,
        "weight_sharing": False,
        "independent_kinetics_pretrained_branches": True,
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
