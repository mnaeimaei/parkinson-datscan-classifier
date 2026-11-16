#!/usr/bin/env python3
"""
E2 — Scenario 03
Whole volume + striatal ROI / no augmentation.

The two inputs remain separate.

Each stream receives an independent fresh E2 ResNet-18 branch:
    whole -> E2.forward_features() -> 512
    ROI   -> E2.forward_features() -> 512

src.models.dual_input_fusion then performs:
    concat(512, 512) -> Linear(1024 -> 1)

No image-level concatenation is performed.
No weight sharing is used between the two branches.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dual_input_fusion import build_dual_input_fusion
from src.models.model02_resnet18_3d_scratch import (
    build_model as build_single_e2,
)
from scripts.experiments_script.common_cv_experiment import (
    ExperimentSpec,
    run_experiment,
)


def build_model():
    """
    Build a fresh Scenario-C E2 model.

    Every call creates:
        - one new scratch whole-volume branch
        - one new scratch ROI branch
        - one new fusion classifier

    Therefore every CV fold starts from fresh random weights.
    """
    return build_dual_input_fusion(
        whole_branch=build_single_e2(),
        roi_branch=build_single_e2(),
        dropout=0.0,
    )


SPEC = ExperimentSpec(
    experiment_name="model02_resnet18_3d_scratch__scenario03_whole_roi_noaug",
    model_name="model02_resnet18_3d_scratch",
    scenario_id=3,
    output_subdir=(
        "model02_resnet18_3d_scratch_exp_data/"
        "scenario03_whole_roi_noaug_exp_data"
    ),
    model_builder=build_model,
    metadata={
        "experiment_id": "E2",
        "pretraining": "none",
        "model_family": "MedicalNet-compatible MONAI 3D ResNet-18",
        "fusion": "late_feature_concatenation",
        "whole_feature_dim": 512,
        "roi_feature_dim": 512,
        "fusion_feature_dim": 1024,
        "fusion_dropout": 0.0,
        "weight_sharing": False,
    },
)


if __name__ == "__main__":
    raise SystemExit(
        run_experiment(
            SPEC,
            default_project_root=PROJECT_ROOT,
        )
    )
