#!/usr/bin/env python3
"""
E2 — Scenario 05
Striatal ROI / training augmentation.

Uses exactly the same E2 scratch architecture, frozen ROI input, and
shared training policy as Scenario 02. The intended experimental
difference is augmentation.

Augmentation:
    ON for TRAINING ONLY.
    Uses src.augmentation.augmentations with the central configuration in
    src.configs.training_config.

Validation/OOF:
    augmentation OFF.

No additional normalization, clipping, registration, or resampling.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model02_resnet18_3d_scratch import build_model
from scripts.experiments_script.common_cv_experiment import (
    ExperimentSpec,
    run_experiment,
)


SPEC = ExperimentSpec(
    experiment_name="model02_resnet18_3d_scratch__scenario05_roi_aug",
    model_name="model02_resnet18_3d_scratch",
    scenario_id=5,
    output_subdir=(
        "model02_resnet18_3d_scratch_exp_data/"
        "scenario05_roi_aug_exp_data"
    ),
    model_builder=build_model,
    augmentation_spatial_mode="3d",
    metadata={
        "experiment_id": "E2",
        "pretraining": "none",
        "model_family": "MedicalNet-compatible MONAI 3D ResNet-18",
        "fusion": "none",
        "augmentation_source": "src.augmentation.augmentations",
        "augmentation_scope": "training_only",
    },
)


if __name__ == "__main__":
    raise SystemExit(
        run_experiment(
            SPEC,
            default_project_root=PROJECT_ROOT,
        )
    )
