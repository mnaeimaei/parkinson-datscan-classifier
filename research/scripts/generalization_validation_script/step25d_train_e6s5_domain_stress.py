#!/usr/bin/env python3
"""
STEP 25D — E6-S5 ACQUISITION-DOMAIN STRESS TRAINING

Purpose
-------
Retrain ONLY the current best individual architecture/configuration:

    E6-S5
    model06_r3d18_kinetics400_pretrained
    ROI input
    training-only 3D augmentation
    Kinetics-400 initialization

using the Step-25C recommended 3-fold spacing-family-isolated stress split.

CONTROLLED EXPERIMENT
---------------------
The intended experimental change is ONLY the validation split:

    original Step-11 random/stratified 5-fold CV
        ->
    Step-25C candidate 3-fold acquisition-domain stress split

The following remain authoritative and unchanged:
- src/
- E6 architecture and Kinetics transfer
- frozen Step-10 ROI paths
- NIfTI -> [C,D,H,W] tensor conversion
- central augmentation parameters
- loss / optimizer / scheduler
- early stopping
- checkpoint selection
- evaluation implementation
- AMP policy unless explicitly overridden

This is a GENERALIZATION STRESS TEST, not a replacement for Step-11 and
not a new final competition model-selection procedure.

Why a dedicated runner?
-----------------------
The original scripts/experiments_script/common_cv_experiment.py intentionally
validates exactly five folds [0,1,2,3,4]. Step 25C recommends three candidate
stress folds [0,1,2]. This runner reuses the original data/training/evaluation
components but implements only the 3-fold orchestration locally, leaving the
authoritative src/ and original experiment runner untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    auc,
)

# Project root is inserted in main() before project imports are required.


EXPECTED_SUBJECTS = 1362
EXPECTED_STRESS_FOLDS = 3
MODEL_NAME = "model06_r3d18_kinetics400_pretrained"
SCENARIO_ID = 5
INPUT_TYPE = "roi"
GROUP_COLUMN = "derived_spacing_signature"
CANDIDATE_FOLD_COLUMN = "candidate_fold"
EXPECTED_SCHEME = "spacing_sgkf3"

ORIGINAL_P1_RAW_OOF_LOG_LOSS = 0.297122117
COMPETITION_FULL_TEST_LOG_LOSS = 0.2997


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 25D: train E6-S5 on the Step-25C 3-fold "
            "acquisition-domain stress split."
        )
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--assignments", type=Path, default=None)
    parser.add_argument("--step25c-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument(
        "--fold",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Train one candidate stress fold. Normally supplied by Slurm array.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not train. Aggregate fold_0..fold_2 after all three jobs finish.",
    )

    parser.add_argument("--expected-subjects", type=int, default=EXPECTED_SUBJECTS)
    parser.add_argument("--expected-folds", type=int, default=EXPECTED_STRESS_FOLDS)
    parser.add_argument("--group-column", default=GROUP_COLUMN)
    parser.add_argument("--candidate-fold-column", default=CANDIDATE_FOLD_COLUMN)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Optional override. Leave unset for the same central "
            "suggested_batch_size() used by the original experiment runner."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="DataLoader workers. Default 6 for ROI GPU jobs.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help=(
            "Smoke-test override only. Leave unset for official Step-25D "
            "training so DEFAULT_TRAINING_CONFIG remains authoritative."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Optional AMP override. The Slurm training wrapper explicitly "
            "uses --amp to reproduce the original GPU training policy."
        ),
    )
    parser.add_argument(
        "--overwrite-fold",
        action="store_true",
        help="Delete and rerun the selected fold output directory.",
    )

    return parser.parse_args()


def resolve(project_root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Not JSON serializable: {type(value).__name__}")

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=default)


def load_and_validate_inputs(
    *,
    manifest_path: Path,
    assignments_path: Path,
    summary_path: Path,
    expected_subjects: int,
    expected_folds: int,
    group_column: str,
    candidate_fold_column: str,
    resolve_manifest_columns,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Step-10 manifest not found: {manifest_path}")
    if not assignments_path.is_file():
        raise FileNotFoundError(f"Step-25C assignments not found: {assignments_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Step-25C summary not found: {summary_path}")

    manifest = pd.read_csv(manifest_path)
    assignments = pd.read_csv(assignments_path)

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    columns = resolve_manifest_columns(manifest, INPUT_TYPE)
    uid_col = columns["uid"]
    label_col = columns["label"]

    manifest = manifest.copy()
    manifest[uid_col] = manifest[uid_col].astype(str)
    manifest[label_col] = pd.to_numeric(
        manifest[label_col], errors="raise"
    ).astype(np.int64)

    if len(manifest) != expected_subjects:
        raise RuntimeError(
            f"Expected {expected_subjects} manifest rows, found {len(manifest)}."
        )
    if manifest[uid_col].duplicated().any():
        raise RuntimeError("Duplicate UID in Step-10 manifest.")
    if not np.isin(manifest[label_col].to_numpy(), [0, 1]).all():
        raise RuntimeError("Manifest labels must be binary 0/1.")

    required = {
        "uid",
        "is_pathologic",
        candidate_fold_column,
        group_column,
        "scheme",
    }
    missing = required - set(assignments.columns)
    if missing:
        raise RuntimeError(
            f"Step-25C assignment file missing columns: {sorted(missing)}"
        )

    assignments = assignments.copy()
    assignments["uid"] = assignments["uid"].astype(str)
    assignments["is_pathologic"] = pd.to_numeric(
        assignments["is_pathologic"], errors="raise"
    ).astype(np.int64)
    assignments[candidate_fold_column] = pd.to_numeric(
        assignments[candidate_fold_column], errors="raise"
    ).astype(np.int64)
    assignments[group_column] = assignments[group_column].astype(str)

    if len(assignments) != expected_subjects:
        raise RuntimeError(
            f"Expected {expected_subjects} assignment rows, found {len(assignments)}."
        )
    if assignments["uid"].duplicated().any():
        raise RuntimeError("Duplicate UID in Step-25C assignment file.")

    expected_fold_ids = set(range(expected_folds))
    observed_fold_ids = set(assignments[candidate_fold_column].unique())
    if observed_fold_ids != expected_fold_ids:
        raise RuntimeError(
            f"Expected stress folds {sorted(expected_fold_ids)}, "
            f"found {sorted(observed_fold_ids)}."
        )

    observed_schemes = set(assignments["scheme"].astype(str).unique())
    if observed_schemes != {EXPECTED_SCHEME}:
        raise RuntimeError(
            f"Expected scheme {EXPECTED_SCHEME!r}, found {sorted(observed_schemes)}."
        )

    manifest_uids = set(manifest[uid_col])
    assignment_uids = set(assignments["uid"])
    if manifest_uids != assignment_uids:
        raise RuntimeError(
            "Step-10 / Step-25C UID sets differ: "
            f"missing={len(manifest_uids-assignment_uids)}, "
            f"unknown={len(assignment_uids-manifest_uids)}."
        )

    label_lookup = manifest.set_index(uid_col)[label_col]
    assigned_labels = assignments.set_index("uid")["is_pathologic"]
    mismatch = label_lookup.loc[assigned_labels.index].to_numpy() != assigned_labels.to_numpy()
    if mismatch.any():
        raise RuntimeError(
            f"Label mismatch for {int(mismatch.sum())} subjects between "
            "Step-10 and Step-25C."
        )

    # Strict spacing-family isolation: every domain must occur in exactly one
    # candidate validation fold.
    folds_per_group = assignments.groupby(group_column)[candidate_fold_column].nunique()
    violations = folds_per_group[folds_per_group != 1]
    if len(violations):
        raise RuntimeError(
            f"{len(violations)} acquisition groups cross candidate folds."
        )

    # Validate Step-25C provenance hash.
    expected_sha = str(summary.get("recommended_assignment_sha256", "")).strip()
    actual_sha = sha256_file(assignments_path)
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(
            "Step-25C recommended assignment SHA256 mismatch.\n"
            f"Expected: {expected_sha}\n"
            f"Actual  : {actual_sha}"
        )

    if summary.get("recommended_scheme") != EXPECTED_SCHEME:
        raise RuntimeError(
            "Step-25C summary does not recommend spacing_sgkf3."
        )

    # Keep the original manifest row order and attach ONLY candidate-fold
    # metadata. The old Step-11 `fold` column from the Step-25C file is never
    # used to define training/validation membership.
    # Label consistency was already validated above. Do not merge the
    # assignment label a second time, otherwise a manifest whose canonical
    # label column is also `is_pathologic` would create `_x` / `_y` columns.
    attach = assignments[
        ["uid", candidate_fold_column, group_column, "scheme"]
    ].copy()

    merged = manifest.merge(
        attach,
        left_on=uid_col,
        right_on="uid",
        how="left",
        validate="one_to_one",
    )

    if merged[candidate_fold_column].isna().any():
        raise RuntimeError("Missing candidate-fold assignment after merge.")

    return manifest, merged, columns, summary


def shared_config_for_fold(
    *,
    fold_id: int,
    num_workers: int | None,
    max_epochs: int | None,
    amp: bool | None,
    DEFAULT_TRAINING_CONFIG,
):
    shared = DEFAULT_TRAINING_CONFIG

    # Preserve the exact original fold-seed convention: base seed + fold ID.
    fold_seed = int(shared.reproducibility.seed) + int(fold_id)
    reproducibility = replace(shared.reproducibility, seed=fold_seed)

    dataloader = shared.dataloader
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError("--num-workers must be >= 0.")
        dataloader = replace(dataloader, num_workers=int(num_workers))

    training = shared.training
    if max_epochs is not None:
        if max_epochs < 1:
            raise ValueError("--max-epochs must be >= 1.")
        training = replace(training, max_epochs=int(max_epochs))

    if amp is not None:
        training = replace(training, use_amp=bool(amp))

    shared = replace(
        shared,
        reproducibility=reproducibility,
        dataloader=dataloader,
        training=training,
    )
    shared.validate()
    return shared


def _find_prediction_column(df: pd.DataFrame, candidates: tuple[str, ...], name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not identify {name} in predictions.csv. "
        f"Tried {candidates}; available={list(df.columns)}"
    )


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / total) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def calculate_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0 - eps)

    precision, recall, _ = precision_recall_curve(y, p)
    auprc = float(auc(recall, precision))

    return {
        "log_loss": float(log_loss(y, p_safe, labels=[0, 1])),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": auprc,
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10bin": expected_calibration_error(y, p, n_bins=10),
    }


def independently_aggregate_predictions(
    *,
    output_dir: Path,
    assignments: pd.DataFrame,
    expected_subjects: int,
    expected_folds: int,
    candidate_fold_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    frames = []

    for fold_id in range(expected_folds):
        path = output_dir / f"fold_{fold_id}" / "predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing fold prediction file: {path}")

        df = pd.read_csv(path)
        uid_col = _find_prediction_column(
            df, ("uid", "UID", "subject_uid", "subject_id"), "UID"
        )
        label_col = _find_prediction_column(
            df, ("label", "is_pathologic", "target"), "label"
        )
        prob_col = _find_prediction_column(
            df,
            ("probability", "probabilities", "prediction_probability", "y_prob"),
            "probability",
        )

        part = pd.DataFrame(
            {
                "uid": df[uid_col].astype(str),
                "is_pathologic": pd.to_numeric(df[label_col], errors="raise").astype(np.int64),
                "probability": pd.to_numeric(df[prob_col], errors="raise").astype(np.float64),
                "candidate_fold": int(fold_id),
            }
        )

        if "logit" in df.columns:
            part["logit"] = pd.to_numeric(df["logit"], errors="coerce")
        frames.append(part)

    oof = pd.concat(frames, ignore_index=True)

    if len(oof) != expected_subjects:
        raise RuntimeError(
            f"Independent aggregation expected {expected_subjects} rows, found {len(oof)}."
        )
    if oof["uid"].duplicated().any():
        raise RuntimeError("Duplicate UID across stress-fold OOF predictions.")
    if not np.isfinite(oof["probability"].to_numpy()).all():
        raise RuntimeError("NaN/Inf in OOF probabilities.")
    if ((oof["probability"] < 0.0) | (oof["probability"] > 1.0)).any():
        raise RuntimeError("OOF probabilities outside [0,1].")

    canonical = assignments[
        ["uid", "is_pathologic", candidate_fold_column]
    ].rename(columns={candidate_fold_column: "expected_candidate_fold"})

    checked = oof.merge(canonical, on="uid", how="left", validate="one_to_one")

    if (
        checked["candidate_fold"].to_numpy()
        != checked["expected_candidate_fold"].to_numpy()
    ).any():
        raise RuntimeError(
            "A prediction was generated by a fold different from its "
            "Step-25C candidate validation fold."
        )

    if (
        checked["is_pathologic_x"].to_numpy()
        != checked["is_pathologic_y"].to_numpy()
    ).any():
        raise RuntimeError("Prediction label mismatch against Step-25C.")

    checked = checked.rename(columns={"is_pathologic_x": "is_pathologic"})
    checked = checked.drop(columns=["is_pathologic_y"])

    fold_rows = []
    for fold_id in range(expected_folds):
        g = checked.loc[checked["candidate_fold"] == fold_id]
        metrics = calculate_metrics(
            g["is_pathologic"].to_numpy(),
            g["probability"].to_numpy(),
        )
        fold_rows.append(
            {
                "candidate_fold": fold_id,
                "subjects": len(g),
                "negatives": int((g["is_pathologic"] == 0).sum()),
                "positives": int((g["is_pathologic"] == 1).sum()),
                "positive_rate": float(g["is_pathologic"].mean()),
                **metrics,
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)
    global_metrics = calculate_metrics(
        checked["is_pathologic"].to_numpy(),
        checked["probability"].to_numpy(),
    )

    return checked, fold_metrics, global_metrics


def aggregate_only(
    *,
    output_dir: Path,
    assignments: pd.DataFrame,
    expected_subjects: int,
    expected_folds: int,
    candidate_fold_column: str,
    threshold: float,
    aggregate_cv_results,
) -> None:
    prediction_paths = [
        output_dir / f"fold_{fold_id}" / "predictions.csv"
        for fold_id in range(expected_folds)
    ]
    missing = [p for p in prediction_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot aggregate Step 25D; missing:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    # Produce the project's standard CV artifacts, but explicitly with 3 folds.
    project_aggregate = aggregate_cv_results(
        prediction_paths=prediction_paths,
        output_dir=output_dir,
        threshold=threshold,
        expected_folds=expected_folds,
        expected_subjects=expected_subjects,
    )

    oof, fold_metrics, global_metrics = independently_aggregate_predictions(
        output_dir=output_dir,
        assignments=assignments,
        expected_subjects=expected_subjects,
        expected_folds=expected_folds,
        candidate_fold_column=candidate_fold_column,
    )

    oof.to_csv(
        output_dir / "step25d_domain_stress_oof_predictions.csv",
        index=False,
        float_format="%.9f",
    )
    fold_metrics.to_csv(
        output_dir / "step25d_domain_stress_fold_metrics.csv",
        index=False,
        float_format="%.9f",
    )

    delta_vs_original_p1 = (
        global_metrics["log_loss"] - ORIGINAL_P1_RAW_OOF_LOG_LOSS
    )
    delta_vs_competition = (
        global_metrics["log_loss"] - COMPETITION_FULL_TEST_LOG_LOSS
    )

    # Heuristic interpretation only. The split also changes training-set size.
    if global_metrics["log_loss"] <= 0.32:
        stress_level = "LOW_TO_MODEST"
        interpretation = (
            "E6-S5 remains comparatively robust under this acquisition-domain "
            "stress split. Acquisition shift alone is unlikely to explain the "
            "entire leaderboard gap."
        )
    elif global_metrics["log_loss"] <= 0.38:
        stress_level = "MEANINGFUL"
        interpretation = (
            "E6-S5 degrades meaningfully under acquisition-domain holdout. "
            "Domain robustness should become a major improvement target."
        )
    else:
        stress_level = "SEVERE"
        interpretation = (
            "E6-S5 shows severe degradation under acquisition-domain holdout. "
            "Acquisition-domain generalization is a major weakness."
        )

    summary = {
        "status": "PASS",
        "step": "25D",
        "experiment": "E6-S5 acquisition-domain stress training",
        "model": MODEL_NAME,
        "scenario_id": SCENARIO_ID,
        "input_type": INPUT_TYPE,
        "augmentation": "training_only_3d_same_central_config_as_original_E6_S5",
        "stress_folds": expected_folds,
        "subjects": expected_subjects,
        "global_metrics": global_metrics,
        "original_P1_raw_OOF_log_loss_reference": ORIGINAL_P1_RAW_OOF_LOG_LOSS,
        "competition_full_test_log_loss_reference": COMPETITION_FULL_TEST_LOG_LOSS,
        "stress_minus_original_P1_log_loss": delta_vs_original_p1,
        "stress_minus_competition_log_loss": delta_vs_competition,
        "stress_level": stress_level,
        "interpretation": interpretation,
        "critical_caveat": (
            "The stress folds also change the number of training subjects. "
            "Therefore the entire difference versus the original 5-fold P1 "
            "cannot be attributed purely to acquisition-domain shift."
        ),
        "project_aggregate_summary": project_aggregate.get("summary", {}),
    }
    save_json(summary, output_dir / "step25d_summary.json")

    lines = [
        "# Step 25D — E6-S5 Acquisition-Domain Stress Training",
        "",
        "## Status",
        "",
        "**PASS — all three stress folds trained and aggregated.**",
        "",
        "## Controlled model",
        "",
        "- Model: **E6 — R3D-18 Kinetics-400 pretrained**",
        "- Scenario: **S5 — ROI + training augmentation**",
        "- Validation split: **Step-25C spacing_sgkf3**",
        "- Neural-network configurations tested: **1**",
        "- Fold training runs: **3**",
        "",
        "## Global stress OOF performance",
        "",
        f"- Log Loss: **{global_metrics['log_loss']:.6f}**",
        f"- AUROC: **{global_metrics['auroc']:.6f}**",
        f"- AUPRC: **{global_metrics['auprc']:.6f}**",
        f"- Average Precision: **{global_metrics['average_precision']:.6f}**",
        f"- Brier: **{global_metrics['brier_score']:.6f}**",
        f"- ECE (10-bin diagnostic): **{global_metrics['ece_10bin']:.6f}**",
        "",
        "## Fold-level stress performance",
        "",
        fold_metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## References",
        "",
        f"- Original P1 5-fold random-CV raw OOF LL: **{ORIGINAL_P1_RAW_OOF_LOG_LOSS:.6f}**",
        f"- Current competition full-test LL: **{COMPETITION_FULL_TEST_LOG_LOSS:.4f}**",
        f"- Stress LL − original P1 OOF LL: **{delta_vs_original_p1:+.6f}**",
        f"- Stress LL − competition LL: **{delta_vs_competition:+.6f}**",
        "",
        "## Interpretation",
        "",
        f"Stress level: **{stress_level}**",
        "",
        interpretation,
        "",
        "## Critical caveat",
        "",
        (
            "This experiment simultaneously holds out acquisition families "
            "and changes training-set size (834/863/1027 training subjects "
            "instead of roughly 1089 in ordinary 5-fold CV). Do not attribute "
            "100% of any degradation to domain shift."
        ),
        "",
        "## Decision rule",
        "",
        "- LL <= ~0.32: domain shift probably not the dominant problem.",
        "- LL ~0.33–0.38: domain robustness is a meaningful problem.",
        "- LL > ~0.38, especially with one catastrophic fold: strong evidence of acquisition sensitivity.",
        "",
        "Do not modify the frozen ENS328 submission predictor solely from this stress test.",
    ]
    (output_dir / "step25d_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("STEP 25D — FINAL STRESS-TEST SUMMARY")
    print("=" * 100)
    print(f"Status                         : PASS")
    print(f"Global stress OOF Log Loss     : {global_metrics['log_loss']:.6f}")
    print(f"Global stress OOF AUROC        : {global_metrics['auroc']:.6f}")
    print(f"Global stress OOF AUPRC        : {global_metrics['auprc']:.6f}")
    print(f"Original P1 OOF Log Loss       : {ORIGINAL_P1_RAW_OOF_LOG_LOSS:.6f}")
    print(f"Current competition Log Loss   : {COMPETITION_FULL_TEST_LOG_LOSS:.4f}")
    print(f"Stress - original P1 LL        : {delta_vs_original_p1:+.6f}")
    print(f"Stress level                   : {stress_level}")
    print("=" * 100)
    print()
    print(f"Saved: {output_dir / 'step25d_report.md'}")


def main() -> int:
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else script_path.parents[2]
    )

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Reuse the already-validated original experiment-layer components.
    from scripts.experiments_script.common_cv_experiment import (
        FrozenDATScanDataset,
        make_loader,
        resolve_device,
        resolve_manifest_columns,
    )
    from scripts.experiments_script.model06_r3d18_kinetics400_pretrained_exp_script.kinetics_pretrained_factory import (
        build_single_e6,
    )
    from src.augmentation.augmentations import build_augmentation
    from src.configs.training_config import (
        DEFAULT_TRAINING_CONFIG,
        ExperimentConfig,
        SCENARIOS,
        save_experiment_config,
        seed_everything,
        suggested_batch_size,
    )
    from src.evaluation.aggregate_cv_results import aggregate_cv_results
    from src.evaluation.evaluate import evaluate_predictions, save_evaluation_result
    from src.training.checkpointing import load_checkpoint
    from src.training.factory import build_trainer_from_config

    manifest_path = resolve(
        project_root,
        args.manifest
        or Path(
            "data/preprocessing_supervised_data/"
            "step10_supervised_dataset_manifest_data/"
            "supervised_dataset/supervised_dataset_manifest.csv"
        ),
    )
    assignments_path = resolve(
        project_root,
        args.assignments
        or Path(
            "data/generalization_validation_data/"
            "step25c_domain_stress_split_design/"
            "recommended_step25d_fold_assignments.csv"
        ),
    )
    summary_path = resolve(
        project_root,
        args.step25c_summary
        or Path(
            "data/generalization_validation_data/"
            "step25c_domain_stress_split_design/"
            "step25c_summary.json"
        ),
    )
    output_dir = resolve(
        project_root,
        args.output_dir
        or Path(
            "data/generalization_validation_data/"
            "step25d_e6s5_domain_stress_training"
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, merged, columns, step25c_summary = load_and_validate_inputs(
        manifest_path=manifest_path,
        assignments_path=assignments_path,
        summary_path=summary_path,
        expected_subjects=args.expected_subjects,
        expected_folds=args.expected_folds,
        group_column=args.group_column,
        candidate_fold_column=args.candidate_fold_column,
        resolve_manifest_columns=resolve_manifest_columns,
    )

    scenario = SCENARIOS[SCENARIO_ID]
    if scenario.input_type != INPUT_TYPE or not scenario.augmentation:
        raise RuntimeError(
            "Central Scenario 5 contract changed unexpectedly. "
            f"input_type={scenario.input_type}, augmentation={scenario.augmentation}"
        )

    if args.aggregate_only:
        aggregate_only(
            output_dir=output_dir,
            assignments=merged[
                ["uid", "is_pathologic", args.candidate_fold_column]
            ].copy(),
            expected_subjects=args.expected_subjects,
            expected_folds=args.expected_folds,
            candidate_fold_column=args.candidate_fold_column,
            threshold=DEFAULT_TRAINING_CONFIG.training.classification_threshold,
            aggregate_cv_results=aggregate_cv_results,
        )
        return 0

    if args.fold is None:
        raise ValueError(
            "Training mode requires --fold 0, --fold 1, or --fold 2. "
            "Use the supplied Slurm array wrapper to run all three in parallel."
        )

    fold_id = int(args.fold)
    device = resolve_device(args.device)

    shared = shared_config_for_fold(
        fold_id=fold_id,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        amp=args.amp,
        DEFAULT_TRAINING_CONFIG=DEFAULT_TRAINING_CONFIG,
    )
    seed_everything(shared.reproducibility)

    uid_col = columns["uid"]
    label_col = columns["label"]

    val_uids = set(
        merged.loc[
            merged[args.candidate_fold_column] == fold_id, "uid"
        ].astype(str)
    )

    train_rows = manifest.loc[
        ~manifest[uid_col].astype(str).isin(val_uids)
    ].reset_index(drop=True)
    val_rows = manifest.loc[
        manifest[uid_col].astype(str).isin(val_uids)
    ].reset_index(drop=True)

    if len(train_rows) + len(val_rows) != len(manifest):
        raise RuntimeError("Stress fold does not partition the dataset.")
    if set(train_rows[uid_col].astype(str)) & set(val_rows[uid_col].astype(str)):
        raise RuntimeError("Train/validation UID leakage.")

    expected_val_uids = set(
        merged.loc[
            merged[args.candidate_fold_column] == fold_id, "uid"
        ].astype(str)
    )
    if set(val_rows[uid_col].astype(str)) != expected_val_uids:
        raise RuntimeError("Candidate validation membership mismatch.")

    # Stronger domain isolation check for THIS fold:
    assignment_by_uid = merged.set_index("uid")
    train_groups = set(
        assignment_by_uid.loc[
            train_rows[uid_col].astype(str), args.group_column
        ].astype(str)
    )
    val_groups = set(
        assignment_by_uid.loc[
            val_rows[uid_col].astype(str), args.group_column
        ].astype(str)
    )
    overlap_groups = train_groups & val_groups
    if overlap_groups:
        raise RuntimeError(
            f"Acquisition-domain leakage in stress fold {fold_id}: "
            f"{len(overlap_groups)} shared spacing families."
        )

    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else suggested_batch_size(MODEL_NAME, INPUT_TYPE)
    )
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    fold_dir = output_dir / f"fold_{fold_id}"
    if fold_dir.exists():
        if args.overwrite_fold:
            shutil.rmtree(fold_dir)
        elif any(fold_dir.iterdir()):
            raise FileExistsError(
                f"Fold output already exists and is non-empty: {fold_dir}\n"
                "Use --overwrite-fold only for an intentional rerun."
            )
    fold_dir.mkdir(parents=True, exist_ok=True)

    fold_config = ExperimentConfig(
        model_name=MODEL_NAME,
        scenario_id=SCENARIO_ID,
        fold=fold_id,
        batch_size=batch_size,
        output_dir=str(fold_dir),
        shared=shared,
    )
    fold_config.validate()
    save_experiment_config(fold_config, fold_dir / "experiment_config.json")

    train_augmenter = build_augmentation(
        enabled=True,
        spatial_mode="3d",
        config=shared.augmentation,
    )

    train_dataset = FrozenDATScanDataset(
        rows=train_rows,
        columns=columns,
        project_root=project_root,
        input_type=INPUT_TYPE,
        augmenter=train_augmenter,
        sample_transform=None,
    )
    val_dataset = FrozenDATScanDataset(
        rows=val_rows,
        columns=columns,
        project_root=project_root,
        input_type=INPUT_TYPE,
        augmenter=None,
        sample_transform=None,
    )

    fold_seed = int(shared.reproducibility.seed)
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        training=True,
        shared_config=shared,
        seed=fold_seed,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        training=False,
        shared_config=shared,
        seed=fold_seed,
        device=device,
    )

    # Fresh E6 Kinetics initialization. The existing factory performs the
    # original architecture / pretrained-transfer QC.
    model = build_single_e6()

    metadata = {
        "step": "25D",
        "purpose": "acquisition_domain_stress_validation",
        "experiment_name": "step25d_E6S5_spacing_sgkf3",
        "experiment_id": "E6",
        "model_name": MODEL_NAME,
        "scenario_id": SCENARIO_ID,
        "scenario_name": scenario.name,
        "input_type": INPUT_TYPE,
        "pretraining": "Kinetics-400",
        "torchvision_weights": "R3D_18_Weights.KINETICS400_V1",
        "augmentation": True,
        "augmentation_scope": "training_only",
        "augmentation_spatial_mode": "3d",
        "augmentation_config": asdict(shared.augmentation),
        "candidate_stress_scheme": EXPECTED_SCHEME,
        "candidate_fold": fold_id,
        "domain_group_column": args.group_column,
        "train_domain_groups": len(train_groups),
        "validation_domain_groups": len(val_groups),
        "domain_overlap_groups": 0,
        "train_subjects": len(train_rows),
        "validation_subjects": len(val_rows),
        "fold_seed": fold_seed,
        "step25c_assignment_sha256": sha256_file(assignments_path),
        "controlled_change": "CV_SPLIT_ONLY",
        "src_modified": False,
    }

    trainer = build_trainer_from_config(
        model=model,
        config=shared,
        device=device,
        checkpoint_dir=fold_dir,
        metadata=metadata,
    )

    print("=" * 100)
    print("STEP 25D — E6-S5 ACQUISITION-DOMAIN STRESS TRAINING")
    print("=" * 100)
    print(f"Fold                    : {fold_id}/{args.expected_folds - 1}")
    print(f"Model                   : E6 / R3D-18 Kinetics-400")
    print(f"Scenario                : S5 / ROI + training augmentation")
    print(f"Controlled change       : CV SPLIT ONLY")
    print(f"Train subjects          : {len(train_rows)}")
    print(f"Validation subjects     : {len(val_rows)}")
    print(f"Train spacing families  : {len(train_groups)}")
    print(f"Val spacing families    : {len(val_groups)}")
    print(f"Shared spacing families : {len(overlap_groups)}")
    print(f"Device                  : {device}")
    print(f"Batch size              : {batch_size}")
    print(f"Workers                 : {shared.dataloader.num_workers}")
    print(f"AMP                     : {trainer.amp_enabled}")
    print(f"Seed                    : {fold_seed}")
    print(
        f"Checkpoint selection    : {shared.training.checkpoint_metric} "
        f"({shared.training.checkpoint_mode})"
    )
    print(f"Output                  : {fold_dir}")
    print("=" * 100)

    start = time.monotonic()

    training_result = trainer.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        max_epochs=shared.training.max_epochs,
        verbose=True,
    )

    if training_result.best_checkpoint_path is None:
        raise RuntimeError("Trainer did not create a best checkpoint.")

    load_checkpoint(
        training_result.best_checkpoint_path,
        model=trainer.model,
        map_location=device,
        strict_model=True,
    )

    best_validation = trainer.validate(val_loader)
    if not best_validation.uids:
        raise RuntimeError("Validation UIDs are required for stress OOF evaluation.")

    evaluation = evaluate_predictions(
        uids=best_validation.uids,
        labels=best_validation.targets.numpy(),
        probabilities=best_validation.probabilities.numpy(),
        logits=best_validation.logits.numpy(),
        threshold=shared.training.classification_threshold,
        fold=fold_id,
    )

    save_evaluation_result(evaluation, output_dir=fold_dir)
    pd.DataFrame(training_result.history).to_csv(
        fold_dir / "history.csv", index=False
    )

    training_summary = {
        **metadata,
        "best_epoch": training_result.best_epoch,
        "best_selection_metric": training_result.best_metric,
        "selection_metric_name": shared.training.checkpoint_metric,
        "selection_mode": shared.training.checkpoint_mode,
        "best_checkpoint": str(training_result.best_checkpoint_path),
        "last_checkpoint": (
            str(training_result.last_checkpoint_path)
            if training_result.last_checkpoint_path is not None
            else None
        ),
        "epochs_completed": training_result.epochs_completed,
        "stopped_early": training_result.stopped_early,
        "best_checkpoint_validation_loss": best_validation.loss,
        "best_checkpoint_metrics": evaluation.metrics,
        "runtime_seconds": time.monotonic() - start,
    }
    save_json(training_summary, fold_dir / "training_summary.json")

    print()
    print("=" * 100)
    print(f"STEP 25D FOLD {fold_id} COMPLETE")
    print("=" * 100)
    print(f"Best epoch             : {training_result.best_epoch}")
    print(f"Validation loss        : {best_validation.loss:.6f}")
    print(f"AUROC                  : {evaluation.metrics['auroc']:.6f}")
    print(f"Balanced accuracy      : {evaluation.metrics['balanced_accuracy']:.6f}")
    print(f"Predictions            : {fold_dir / 'predictions.csv'}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
