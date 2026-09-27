#!/usr/bin/env python3
"""
E2 — Scenario 02
Striatal ROI / no augmentation.

Uses exactly the same E2 scratch architecture and shared training policy
as Scenario 01; only the frozen input scenario changes from whole to ROI.
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
    experiment_name="model02_resnet18_3d_scratch__scenario02_roi_noaug",
    model_name="model02_resnet18_3d_scratch",
    scenario_id=2,
    output_subdir=(
        "model02_resnet18_3d_scratch_exp_data/"
        "scenario02_roi_noaug_exp_data"
    ),
    model_builder=build_model,
    metadata={
        "experiment_id": "E2",
        "pretraining": "none",
        "model_family": "MedicalNet-compatible MONAI 3D ResNet-18",
        "fusion": "none",
    },
)


if __name__ == "__main__":
    raise SystemExit(
        run_experiment(
            SPEC,
            default_project_root=PROJECT_ROOT,
        )
    )
