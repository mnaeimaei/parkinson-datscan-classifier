#!/usr/bin/env python3
"""
E4 — Scenario 06
Whole + ROI / Augmentation.

Model:
    src.models.model04_resnet18_2p5d_attention_imagenet_pretrained

Initialization:
    ResNet18_Weights.IMAGENET1K_V1

RGB -> grayscale:
    pretrained conv1 RGB kernels are SUMMED:
        W_gray = W_R + W_G + W_B

IMPORTANT:
    No ImageNet RGB mean/std normalization.
    No torchvision ImageNet preprocessing transform.
    Frozen project-normalized DaT tensors are used.

Frozen E4/E7 slice selection:
    whole: 40 uniformly spaced axial slices from D=160
    ROI:   all 36 axial slices

Scenario-C:
    whole selected slices -> E4.forward_features() -> 512
    ROI selected slices   -> E4.forward_features() -> 512
    concat -> 1024 -> Linear(1024 -> 1)

No image-level concatenation and no branch weight sharing.


Augmentation:
    ON for TRAINING ONLY.
    spatial_mode="inplane_2d" is mandatory for E4/E7.
    Validation and OOF prediction are unaugmented.

No extra normalization, clipping, registration or resampling.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dual_input_fusion import build_dual_input_fusion
from scripts.experiments_script.model04_resnet18_2p5d_attention_imagenet_pretrained_exp_script.imagenet_pretrained_factory import (
    build_single_e4,
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
    """
    Build a fresh dual-input E4 model for one CV fold.

    Both branches use the same E4 architecture and ImageNet initialization,
    but they are independent model objects with no parameter sharing.
    """
    return build_dual_input_fusion(
        whole_branch=build_single_e4(),
        roi_branch=build_single_e4(),
        dropout=0.0,
    )


SPEC = ExperimentSpec(
    experiment_name="model04_resnet18_2p5d_attention_imagenet_pretrained__scenario06_whole_roi_aug",
    model_name="model04_resnet18_2p5d_attention_imagenet_pretrained",
    scenario_id=6,
    output_subdir=(
        "model04_resnet18_2p5d_attention_imagenet_pretrained_exp_data/"
        "scenario06_whole_roi_aug_exp_data"
    ),
    model_builder=build_model,
    augmentation_spatial_mode="inplane_2d",
    sample_transform=select_2p5d_slices,
    metadata={
        "experiment_id": "E4",
        "pretraining": "ImageNet-1K",
        "torchvision_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "model_family": "2.5D ResNet-18 + learned slice attention",
        "grayscale_conv1_method": "RGB_SUM",
        "imagenet_rgb_mean_std_normalization": False,
        "torchvision_preprocessing_transforms": False,
        "project_frozen_normalization_only": True,
        "slice_selection": slice_selection_metadata(),
        "attention": "Linear(512->1)+softmax_across_slices",
        "model_dropout": 0.3,
        "fusion": "late_feature_concatenation",
        "whole_feature_dim": 512,
        "roi_feature_dim": 512,
        "fusion_feature_dim": 1024,
        "fusion_dropout": 0.0,
        "weight_sharing": False,
        "independent_imagenet_pretrained_branches": True,
        "augmentation_source": "src.augmentation.augmentations",
        "augmentation_scope": "training_only",
        "augmentation_spatial_mode": "inplane_2d",
        "paired_augmentation": True,
        "paired_translation_policy": "same_voxel_displacement",
    },
)


if __name__ == "__main__":
    raise SystemExit(
        run_experiment(
            SPEC,
            default_project_root=PROJECT_ROOT,
        )
    )
