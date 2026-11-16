#!/usr/bin/env python3
"""
E3 — Scenario 02
ROI / No Augmentation.

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


Architecture:
    Exactly identical to E2:
    BasicBlock [2,2,2,2]
    channels 64 -> 128 -> 256 -> 512
    MedicalNet-compatible shortcut_type / bias_downsample conventions
    GAP -> Linear(512 -> 1)

Only the encoder initialization differs from E2.


Augmentation:
    OFF for training, validation, and OOF prediction.


No additional normalization, clipping, registration, or resampling.
Frozen Step-10 manifest and Step-11 5-fold assignments are used exactly.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    """Build one fresh MedicalNet-pretrained E3 model for this CV fold."""
    return build_single_e3(CHECKPOINT_PATH)


SPEC = ExperimentSpec(
    experiment_name="model03_resnet18_3d_medicalnet_pretrained__scenario02_roi_noaug",
    model_name="model03_resnet18_3d_medicalnet_pretrained",
    scenario_id=2,
    output_subdir=(
        "model03_resnet18_3d_medicalnet_pretrained_exp_data/"
        "scenario02_roi_noaug_exp_data"
    ),
    model_builder=build_model,
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
