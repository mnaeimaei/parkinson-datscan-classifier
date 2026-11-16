#!/usr/bin/env python3
"""
E7 — Scenario 01
Whole / No Augmentation.

Model:
    src.models.model07_resnet18_2p5d_attention_scratch

Architecture:
    exactly the same 2.5D ResNet-18 + learned slice-attention architecture
    used by E4.

Initialization:
    SCRATCH ONLY.
    No ImageNet weights.

Shared 2D backbone:
    ResNet-18 BasicBlock [2,2,2,2]
    channels 64 -> 128 -> 256 -> 512

Native grayscale stem:
    Conv2d(
        1 -> 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

Per-slice feature:
    512-D

Learned attention:
    Linear(512 -> 1) -> softmax across selected axial slices

Subject representation:
    weighted sum -> 512-D

Classification:
    Dropout(0.3) -> Linear(512 -> 1)

Parameter count:
    11,171,266

FROZEN E4/E7 SLICE POLICY
--------------------------
Whole-volume:
    select exactly 40 consecutive axial slices centered on the frozen
    subject-specific striatal ROI axial extent.

    The location is derived from:
        frozen ROI NIfTI affine
        -> world coordinates
        -> inverse frozen whole-volume affine
        -> whole-volume Z / model depth D

    No raw-central or full-depth-uniform fallback is allowed.

ROI:
    use all 36 frozen axial slices.

This same shared selector MUST be used by E4 and E7.


Single branch:
    selected axial slices
      -> SAME shared single-channel 2D ResNet-18 for every slice
      -> one 512-D embedding per slice
      -> Linear(512->1) slice score
      -> softmax across slices
      -> weighted 512-D subject representation
      -> Dropout(0.3)
      -> Linear(512->1)


Augmentation:
    OFF for training, validation, and OOF prediction.


No extra normalization.
No ImageNet normalization.
No extra clipping.
No registration during training.
No resampling during training.
"""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep the selector tied to the exact official frozen supervised manifest.
os.environ.setdefault(
    "DATSCAN_SUPERVISED_MANIFEST",
    str(
        PROJECT_ROOT
        / "data"
        / "preprocessing_supervised_data"
        / "step10_supervised_dataset_manifest_data"
        / "supervised_dataset"
        / "supervised_dataset_manifest.csv"
    ),
)

from scripts.experiments_script.model07_resnet18_2p5d_attention_scratch_exp_script.scratch_2p5d_factory import (
    build_single_e7,
)

from scripts.experiments_script.common_cv_experiment import (
    ExperimentSpec,
    run_experiment,
)
from scripts.experiments_script.two_point_five_d_input import (
    select_2p5d_slices,
    slice_selection_metadata,
)


def build_model():
    """Build one fresh scratch E7 model for this CV fold."""
    return build_single_e7()


SPEC = ExperimentSpec(
    experiment_name="model07_resnet18_2p5d_attention_scratch__scenario01_whole_noaug",
    model_name="model07_resnet18_2p5d_attention_scratch",
    scenario_id=1,
    output_subdir=(
        "model07_resnet18_2p5d_attention_scratch_exp_data/"
        "scenario01_whole_noaug_exp_data"
    ),
    model_builder=build_model,
    sample_transform=select_2p5d_slices,
    metadata={
        "experiment_id": "E7",
        "pretraining": "none",
        "model_family": "2.5D ResNet-18 + learned slice attention",
        "architecture": "torchvision_resnet18_2p5d_attention",
        "e4_architecture_identical": True,
        "input_channels": 1,
        "block_structure": [2, 2, 2, 2],
        "stage_channels": [64, 128, 256, 512],
        "slice_feature_dim": 512,
        "attention": "Linear(512->1)+softmax_across_slices",
        "model_dropout": 0.3,
        "parameter_count": 11171266,
        "scratch_initialization": True,
        "imagenet_pretraining": False,
        "imagenet_rgb_mean_std_normalization": False,
        "project_frozen_normalization_only": True,
        "slice_selection": slice_selection_metadata(),
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
