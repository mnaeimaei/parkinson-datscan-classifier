#!/usr/bin/env python3
"""
E3 — Scenario 06
Whole + ROI / Augmentation.

Model:
    src.models.model03_resnet18_3d_medicalnet_pretrained

MedicalNet initialization:
    resnet_18_23dataset.pth

The authoritative src loader:
    - strips supported checkpoint wrapper prefixes,
    - discards pretrained task/head tensors,
    - checks names and exact shapes,
    - requires essentially complete intended encoder transfer,
    - loads ONLY the encoder,
    - keeps a new binary classification head.


Scenario-C architecture:
    whole -> E3.forward_features() -> 512
    ROI   -> E3.forward_features() -> 512
    concat -> 1024 -> Linear(1024 -> 1)

The two branches are independent objects with no weight sharing.
Both are initialized from the same frozen MedicalNet encoder checkpoint.


Augmentation:
    ON for TRAINING ONLY through src.augmentation.
    Validation and OOF prediction remain unaugmented.


For Scenario 06, whole/ROI augmentation uses the shared augment_pair()
policy so paired spatial transformations remain synchronized.

No additional normalization, clipping, registration, or resampling.
Frozen Step-10 manifest and Step-11 5-fold assignments are used exactly.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dual_input_fusion import build_dual_input_fusion
from scripts.experiments_script.model03_resnet18_3d_medicalnet_pretrained_exp_script.medicalnet_pretrained_factory import (
    build_single_e3,
    resolve_medicalnet_checkpoint,
    sha256_file,
)

from scripts.experiments_script.common_cv_experiment import (
    ExperimentSpec,
    run_experiment,
)


CHECKPOINT_PATH = resolve_medicalnet_checkpoint(PROJECT_ROOT)
CHECKPOINT_SHA256 = sha256_file(CHECKPOINT_PATH)


def build_model():
    """
    Build a fresh dual-input E3 model for one CV fold.

    Each call creates two independent E3 branches. Both encoders are
    initialized from the same frozen MedicalNet checkpoint; branch objects
    do not share parameters.

    The branch binary classifiers are not used by Scenario-C fusion because
    fusion consumes forward_features(), exactly as in E2 Scenario-C.
    """
    whole_branch = build_single_e3(CHECKPOINT_PATH)
    roi_branch = build_single_e3(CHECKPOINT_PATH)

    return build_dual_input_fusion(
        whole_branch=whole_branch,
        roi_branch=roi_branch,
        dropout=0.0,
    )


SPEC = ExperimentSpec(
    experiment_name="model03_resnet18_3d_medicalnet_pretrained__scenario06_whole_roi_aug",
    model_name="model03_resnet18_3d_medicalnet_pretrained",
    scenario_id=6,
    output_subdir=(
        "model03_resnet18_3d_medicalnet_pretrained_exp_data/"
        "scenario06_whole_roi_aug_exp_data"
    ),
    model_builder=build_model,
    augmentation_spatial_mode="3d",
    metadata={
        "experiment_id": "E3",
        "pretraining": "MedicalNet/Med3D 23-dataset",
        "pretrained_checkpoint_name": "resnet_18_23dataset.pth",
        "pretrained_checkpoint_path": str(CHECKPOINT_PATH),
        "pretrained_checkpoint_sha256": CHECKPOINT_SHA256,
        "pretrained_encoder_tensor_coverage_required": 0.999999,
        "pretrained_encoder_parameter_coverage_required": 0.999999,
        "new_binary_head": True,
        "model_family": "MedicalNet-compatible MONAI 3D ResNet-18",
        "e2_architecture_identical": True,
        "fusion": "late_feature_concatenation",
        "whole_feature_dim": 512,
        "roi_feature_dim": 512,
        "fusion_feature_dim": 1024,
        "fusion_dropout": 0.0,
        "weight_sharing": False,
        "independent_pretrained_branches": True,
        "augmentation_source": "src.augmentation.augmentations",
        "augmentation_scope": "training_only",
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
