#!/usr/bin/env python3
"""
E9 — Scenario 01
Whole / No Augmentation.

Model:
    src.models.model09_swin3d_t_scratch

Architecture:
    Torchvision Swin3D-Tiny trained completely from scratch.

Input:
    [B,1,D,H,W]

Depth:
    D is the temporal/depth dimension used by Torchvision Video Swin.

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

Attention:
    shifted-window 3-D self-attention.

Patch merging:
    between stages 1-2, 2-3 and 3-4;
    Torchvision Video Swin preserves temporal/depth token count during
    PatchMerging while reducing H/W.

Final representation:
    LayerNorm(768)
    -> global average pooling
    -> 768-D subject feature

Binary head:
    Linear(768 -> 1)

Initialization:
    scratch/random only
    no Kinetics-400 weights

Parameter count:
    27,845,095

Frozen input-shape compatibility:
    Whole [D,H,W] = [160,192,192]
      stage1 = [80,48,48]
      stage2 = [80,24,24]
      stage3 = [80,12,12]
      stage4 = [80,6,6]

    ROI [D,H,W] = [36,44,44]
      stage1 = [18,11,11]
      stage2 = [18,6,6]
      stage3 = [18,3,3]
      stage4 = [18,2,2]

Both are compatible with the frozen Swin3D hierarchy.


Single branch:
    single-channel Swin3D-Tiny
    -> final LayerNorm
    -> global average pooling
    -> 768-D subject feature
    -> Linear(768 -> 1)


Augmentation:
    OFF for training, validation, and OOF prediction.


No sigmoid inside the model.
No additional normalization.
No Kinetics normalization.
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

from scripts.experiments_script.model09_swin3d_t_scratch_exp_script.swin3d_scratch_factory import (
    build_single_e9,
)

from scripts.experiments_script.common_cv_experiment import (
    ExperimentSpec,
    run_experiment,
)


def build_model():
    """Build one fresh scratch E9 Swin3D-Tiny model for this CV fold."""
    return build_single_e9()


SPEC = ExperimentSpec(
    experiment_name="model09_swin3d_t_scratch__scenario01_whole_noaug",
    model_name="model09_swin3d_t_scratch",
    scenario_id=1,
    output_subdir=(
        "model09_swin3d_t_scratch_exp_data/"
        "scenario01_whole_noaug_exp_data"
    ),
    model_builder=build_model,
    metadata={
        "experiment_id": "E9",
        "pretraining": "none",
        "model_family": "Torchvision Swin3D-Tiny",
        "architecture": "torchvision_swin3d_t",
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
        "scratch_initialization": True,
        "kinetics_pretraining": False,
        "kinetics_rgb_normalization": False,
        "project_frozen_normalization_only": True,
        "whole_swin_stage_grids": [
            [80,48,48], [80,24,24], [80,12,12], [80,6,6]
        ],
        "roi_swin_stage_grids": [
            [18,11,11], [18,6,6], [18,3,3], [18,2,2]
        ],
        "fusion": "none",
        "feature_dim": 768,
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
