#!/usr/bin/env python3
"""
E1 — Scenario 03
Whole + ROI / No Augmentation.

The two frozen inputs remain separate.

Each stream receives an independent fresh Model-1 Simple3D branch:

    whole -> Simple3D.forward_features() -> 256
    ROI   -> Simple3D.forward_features() -> 256

Shared Scenario-C fusion:

    concat(256, 256) -> Linear(512 -> 1)

There is:
    - no image-level concatenation
    - no branch weight sharing
    - no pretrained initialization

Augmentation:
    OFF for training, validation, and OOF prediction.

No additional normalization, clipping, registration, or resampling.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dual_input_fusion import build_dual_input_fusion
from src.models.model01_simple3d_scratch import (
    build_model as build_single_e1,
)
from scripts.experiments_script.common_cv_experiment import (
    ExperimentSpec,
    run_experiment,
)


def build_model():
    """
    Build a fresh dual-input E1 model for one CV fold.

    Every call creates:
        - one new scratch whole-volume Simple3D branch
        - one new scratch ROI Simple3D branch
        - one new 512 -> 1 fusion classifier
    """
    return build_dual_input_fusion(
        whole_branch=build_single_e1(),
        roi_branch=build_single_e1(),
        dropout=0.0,
    )


SPEC = ExperimentSpec(
    experiment_name="model01_simple3d_scratch__scenario03_whole_roi_noaug",
    model_name="model01_simple3d_scratch",
    scenario_id=3,
    output_subdir=(
        "model01_simple3d_scratch_exp_data/"
        "scenario03_whole_roi_noaug_exp_data"
    ),
    model_builder=build_model,
    metadata={
        "experiment_id": "E1",
        "pretraining": "none",
        "model_family": "Simple3D compact 3D residual CNN",
        "whole_feature_dim": 256,
        "roi_feature_dim": 256,
        "fusion_feature_dim": 512,
        "branch_model_dropout": 0.3,
        "fusion_dropout": 0.0,
        "fusion": "late_feature_concatenation",
        "weight_sharing": False,
        "augmentation_scope": "off",
        "paired_augmentation": False,
    },
)


if __name__ == "__main__":
    raise SystemExit(
        run_experiment(
            SPEC,
            default_project_root=PROJECT_ROOT,
        )
    )
