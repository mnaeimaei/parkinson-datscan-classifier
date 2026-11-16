#!/usr/bin/env python3
"""
E1 — Scenario 01
Whole / No Augmentation.

Model:
    src.models.model01_simple3d_scratch

Architecture:
    Simple3D compact residual 3D CNN trained from scratch.
    Single-channel input.
    Stem: Conv3d 1->32 + BN + ReLU + MaxPool.
    Residual stages: 32 -> 32 -> 64 -> 128 -> 256.
    Adaptive global average pooling.
    Dropout(p=0.3).
    Linear(256 -> 1).
    Random/Kaiming initialization only.

Augmentation:
    OFF for training, validation, and OOF prediction.

No additional normalization, clipping, registration, or resampling.
Frozen Step-10 manifest and Step-11 folds are used exactly.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model01_simple3d_scratch import build_model
from scripts.experiments_script.common_cv_experiment import (
    ExperimentSpec,
    run_experiment,
)


SPEC = ExperimentSpec(
    experiment_name="model01_simple3d_scratch__scenario01_whole_noaug",
    model_name="model01_simple3d_scratch",
    scenario_id=1,
    output_subdir=(
        "model01_simple3d_scratch_exp_data/"
        "scenario01_whole_noaug_exp_data"
    ),
    model_builder=build_model,
    metadata={
        "experiment_id": "E1",
        "pretraining": "none",
        "model_family": "Simple3D compact 3D residual CNN",
        "single_branch_feature_dim": 256,
        "model_dropout": 0.3,
        "fusion": "none",
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
