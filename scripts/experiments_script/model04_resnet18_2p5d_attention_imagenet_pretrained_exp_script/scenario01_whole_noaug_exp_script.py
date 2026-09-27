#!/usr/bin/env python3
"""
E4 — Scenario 01
Whole / No Augmentation.

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

Single branch:
    selected axial slices
      -> shared 2D ResNet-18 slice encoder
      -> one 512-D embedding per slice
      -> learned Linear(512->1) slice scores
      -> softmax across slices
      -> weighted 512-D subject feature
      -> Dropout(0.3)
      -> Linear(512->1)


Augmentation:
    OFF for training, validation and OOF prediction.

No extra normalization, clipping, registration or resampling.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    """Build one fresh E4 model for this CV fold."""
    return build_single_e4()


SPEC = ExperimentSpec(
    experiment_name="model04_resnet18_2p5d_attention_imagenet_pretrained__scenario01_whole_noaug",
    model_name="model04_resnet18_2p5d_attention_imagenet_pretrained",
    scenario_id=1,
    output_subdir=(
        "model04_resnet18_2p5d_attention_imagenet_pretrained_exp_data/"
        "scenario01_whole_noaug_exp_data"
    ),
    model_builder=build_model,
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
