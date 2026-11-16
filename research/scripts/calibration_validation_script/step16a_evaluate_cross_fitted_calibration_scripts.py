#!/usr/bin/env python3
"""
Evaluate cross-fitted post-hoc calibration for all 60 trained experiments.

No neural network is trained or modified here.

For every experiment, evaluate:
    - none
    - temperature
    - platt
    - isotonic

Calibration uses the existing five OOF folds. For held-out fold f, the
calibrator is fitted on OOF predictions from the other four folds and applied
only to fold f. Concatenating the five held-out results gives one unbiased
cross-fitted calibrated probability for each of the 1,362 subjects.

Competition selection criterion:
    PRIMARY: lower cross-fitted OOF Log Loss

The actual calibration implementation is imported from:
    src.calibration.probability_calibration.ProbabilityCalibrator

Metric definitions are imported from:
    src.evaluation.metrics.compute_binary_metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.probability_calibration import ProbabilityCalibrator
from src.evaluation.metrics import compute_binary_metrics


METHODS = ("none", "temperature", "platt", "isotonic")
FITTED_METHODS = ("temperature", "platt", "isotonic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fitted calibration evaluation for all 60 OOF experiment "
            "predictions using the authoritative src calibration module."
        )
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--candidate-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-candidates", type=int, default=60)
    parser.add_argument("--expected-subjects", type=int, default=1362)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=150)
    return parser.parse_args()


def _validate_oof(
    path: Path,
    *,
    expected_subjects: int,
    expected_folds: int,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"uid", "fold", "is_pathologic", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    if len(df) != expected_subjects:
        raise ValueError(
            f"{path}: expected {expected_subjects} OOF rows, found {len(df)}"
        )

    df = df.copy()
    df["uid"] = df["uid"].astype(str)
    if df["uid"].duplicated().any():
        raise ValueError(f"{path}: duplicate OOF UIDs detected")

    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(np.int64)
    df["is_pathologic"] = pd.to_numeric(
        df["is_pathologic"], errors="raise"
    ).astype(np.int64)
    df["probability"] = pd.to_numeric(
        df["probability"], errors="raise"
    ).astype(np.float64)

    labels = df["is_pathologic"].to_numpy()
    probabilities = df["probability"].to_numpy()
    folds = df["fold"].to_numpy()

    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path}: labels must contain only 0/1")

    if not np.isfinite(probabilities).all():
        raise ValueError(f"{path}: probabilities contain NaN/Inf")

    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(f"{path}: probabilities must be in [0,1]")

    expected_fold_ids = set(range(expected_folds))
    actual_fold_ids = set(int(v) for v in np.unique(folds))
    if actual_fold_ids != expected_fold_ids:
        raise ValueError(
            f"{path}: expected fold IDs {sorted(expected_fold_ids)}, "
            f"got {sorted(actual_fold_ids)}"
        )

    for fold_id in sorted(expected_fold_ids):
        fold_labels = labels[folds == fold_id]
        if fold_labels.size == 0:
            raise ValueError(f"{path}: fold {fold_id} is empty")
        if len(np.unique(fold_labels)) != 2:
            raise ValueError(f"{path}: fold {fold_id} does not contain both classes")

    if len(np.unique(labels)) != 2:
        raise ValueError(f"{path}: complete OOF set must contain both classes")

    return df


def _candidate_dir_name(row: pd.Series) -> str:
    return (
        f"{str(row['Calibration Candidate ID'])}_"
        f"{str(row['Model'])}_"
        f"{str(row['Scenario ID'])}"
    )


def _cross_fit_method(
    df: pd.DataFrame,
    *,
    method: str,
    expected_folds: int,
    epsilon: float,
    max_iter: int,
) -> tuple[np.ndarray, list[dict]]:
    y = df["is_pathologic"].to_numpy(dtype=np.int64)
    raw_p = df["probability"].to_numpy(dtype=np.float64)
    folds = df["fold"].to_numpy(dtype=np.int64)

    if method == "none":
        return raw_p.copy(), []

    calibrated = np.full(len(df), np.nan, dtype=np.float64)
    fold_states: list[dict] = []

    for heldout_fold in range(expected_folds):
        fit_mask = folds != heldout_fold
        apply_mask = folds == heldout_fold

        calibrator = ProbabilityCalibrator(
            method=method,
            output_epsilon=epsilon,
        )
        calibrator.fit(
            labels=y[fit_mask],
            probabilities=raw_p[fit_mask],
            max_iter=max_iter,
        )

        calibrated[apply_mask] = calibrator.predict_proba(raw_p[apply_mask])

        fold_states.append(
            {
                "heldout_fold": int(heldout_fold),
                "fit_subjects": int(fit_mask.sum()),
                "apply_subjects": int(apply_mask.sum()),
                "calibrator": calibrator.to_dict(),
            }
        )

    if not np.isfinite(calibrated).all():
        raise RuntimeError(
            f"Cross-fitted calibration produced non-finite predictions: {method}"
        )

    if np.any(calibrated <= 0.0) or np.any(calibrated >= 1.0):
        raise RuntimeError(
            f"Calibrated probabilities for {method} escaped the open interval (0,1)."
        )

    return calibrated, fold_states


def _result_row(
    *,
    row: pd.Series,
    method: str,
    metrics: dict,
    oof_path: Path,
    prediction_path: Path,
) -> dict:
    return {
        "Calibration Candidate ID": str(row["Calibration Candidate ID"]),
        "Model": str(row["Model"]),
        "Model Used": row.get("Model Used", ""),
        "Type": row.get("Type", ""),
        "Scratch / pretrained": row.get("Scratch / pretrained", ""),
        "File Name": row.get("File Name", ""),
        "Scenario ID": str(row["Scenario ID"]),
        "Scenario": row.get("Scenario", ""),
        "Calibration Method": method,
        "Cross-Fitted OOF Log Loss": float(metrics["log_loss"]),
        "Cross-Fitted OOF AUROC": float(metrics["auroc"]),
        "Cross-Fitted OOF AUPRC": float(metrics["auprc"]),
        "Cross-Fitted OOF Average Precision": float(metrics["average_precision"]),
        "Cross-Fitted OOF Brier Score": float(metrics["brier_score"]),
        "Cross-Fitted OOF ECE": float(metrics["ece"]),
        "Cross-Fitted OOF Balanced Accuracy @ 0.5": float(
            metrics["balanced_accuracy"]
        ),
        "OOF Subjects": int(metrics["n_samples"]),
        "OOF Prediction File": str(oof_path),
        "Cross-Fitted Prediction File": str(prediction_path.resolve()),
    }


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = (
        args.project_root or script_path.parents[2]
    ).expanduser().resolve()

    candidate_manifest = (
        args.candidate_manifest
        or project_root
        / "data"
        / "experiment_selection_data"
        / "calibration_candidate_manifest_data"
        / "all_60_calibration_candidates.csv"
    ).expanduser().resolve()

    output_dir = (
        args.output_dir
        or project_root
        / "data"
        / "calibration_validation_data"
        / "cross_fitted_calibration_data"
    ).expanduser().resolve()

    print("\n" + "=" * 88)
    print("ALL-60 CROSS-FITTED CALIBRATION EVALUATION")
    print("=" * 88)
    print(f"Project root             : {project_root}")
    print(f"Candidate manifest       : {candidate_manifest}")
    print(f"Output                   : {output_dir}")
    print(f"Expected candidates      : {args.expected_candidates}")
    print(f"Expected OOF subjects    : {args.expected_subjects}")
    print(f"Cross-fit folds          : {args.expected_folds}")
    print("Methods                  : none, temperature, platt, isotonic")
    print("Authoritative calibrator : src/calibration/probability_calibration.py")
    print("Primary metric           : Cross-Fitted OOF Log Loss (LOWER is better)")
    print(f"Fitted configurations    : {args.expected_candidates * 3}")
    print(f"Total evaluated rows     : {args.expected_candidates * 4}")
    print()

    if not candidate_manifest.is_file():
        print(f"ERROR: candidate manifest not found: {candidate_manifest}")
        sys.exit(1)

    manifest = pd.read_csv(candidate_manifest)
    if len(manifest) != args.expected_candidates:
        raise RuntimeError(
            f"Expected {args.expected_candidates} candidates, found {len(manifest)}"
        )

    required_manifest = {
        "Calibration Candidate ID",
        "Model",
        "Scenario ID",
        "OOF Prediction File",
        "Advance to Cross-Fitted Calibration",
    }
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise RuntimeError(
            f"Candidate manifest missing columns: {sorted(missing)}"
        )

    if manifest["Calibration Candidate ID"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate Calibration Candidate ID values")

    advance = (
        manifest["Advance to Cross-Fitted Calibration"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    if not (advance == "YES").all():
        raise RuntimeError(
            "Every candidate must be marked YES for cross-fitted calibration."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for index, row in manifest.iterrows():
        cid = str(row["Calibration Candidate ID"])
        model = str(row["Model"])
        scenario = str(row["Scenario ID"])
        oof_path = Path(str(row["OOF Prediction File"])).expanduser().resolve()

        print(f"[{index + 1:02d}/{len(manifest)}] {cid} {model}-{scenario}")

        if not oof_path.is_file():
            raise FileNotFoundError(f"OOF prediction file not found: {oof_path}")

        df = _validate_oof(
            oof_path,
            expected_subjects=args.expected_subjects,
            expected_folds=args.expected_folds,
        )

        candidate_dir = output_dir / _candidate_dir_name(row)
        candidate_dir.mkdir(parents=True, exist_ok=True)

        candidate_rows: list[dict] = []
        none_log_loss: float | None = None

        for method in METHODS:
            calibrated_p, fold_states = _cross_fit_method(
                df,
                method=method,
                expected_folds=args.expected_folds,
                epsilon=args.epsilon,
                max_iter=args.max_iter,
            )

            labels = df["is_pathologic"].to_numpy(dtype=np.int64)
            metrics = compute_binary_metrics(
                labels,
                calibrated_p,
                threshold=0.5,
                ece_bins=args.ece_bins,
            )

            method_dir = candidate_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)

            prediction_path = method_dir / "cross_fitted_predictions.csv"
            prediction_df = df[["uid", "fold", "is_pathologic", "probability"]].copy()
            prediction_df = prediction_df.rename(
                columns={"probability": "raw_probability"}
            )
            prediction_df["calibrated_probability"] = calibrated_p
            prediction_df.to_csv(
                prediction_path,
                index=False,
                float_format="%.12f",
            )

            with (method_dir / "metrics.json").open("w", encoding="utf-8") as file:
                json.dump(metrics, file, indent=2, sort_keys=True)

            with (method_dir / "cross_fit_calibrators.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(
                    {
                        "method": method,
                        "calibration_input": "probability",
                        "cross_fit_policy": (
                            "For heldout fold f, fit calibration on OOF folds != f "
                            "and apply only to OOF fold f."
                        ),
                        "fold_calibrators": fold_states,
                    },
                    file,
                    indent=2,
                    sort_keys=True,
                )

            result = _result_row(
                row=row,
                method=method,
                metrics=metrics,
                oof_path=oof_path,
                prediction_path=prediction_path,
            )

            if method == "none":
                none_log_loss = float(metrics["log_loss"])

            candidate_rows.append(result)
            all_rows.append(result)

            print(
                f"    {method:11s} "
                f"LogLoss={metrics['log_loss']:.6f}  "
                f"AUROC={metrics['auroc']:.6f}  "
                f"Brier={metrics['brier_score']:.6f}  "
                f"ECE={metrics['ece']:.6f}"
            )

        if none_log_loss is None:
            raise RuntimeError(f"Missing uncalibrated baseline for {cid}")

        # Add improvement relative to the raw probability baseline.
        for result in candidate_rows:
            delta = float(result["Cross-Fitted OOF Log Loss"] - none_log_loss)
            result["Delta Log Loss vs None"] = delta
            result["Improves Log Loss vs None"] = bool(delta < 0.0)

        candidate_df = pd.DataFrame(candidate_rows).sort_values(
            [
                "Cross-Fitted OOF Log Loss",
                "Cross-Fitted OOF Brier Score",
                "Cross-Fitted OOF AUROC",
            ],
            ascending=[True, True, False],
            kind="stable",
        ).reset_index(drop=True)
        candidate_df.insert(
            0,
            "Within-Candidate Calibration Rank",
            np.arange(1, len(candidate_df) + 1),
        )
        candidate_df.to_csv(
            candidate_dir / "candidate_calibration_summary.csv",
            index=False,
            float_format="%.9f",
        )

    results = pd.DataFrame(all_rows)
    expected_rows = args.expected_candidates * len(METHODS)
    if len(results) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} calibration rows, found {len(results)}"
        )

    # Recalculate delta columns globally because all_rows dictionaries were
    # appended before candidate_rows received their delta values.
    none_by_candidate = (
        results.loc[results["Calibration Method"] == "none"]
        .set_index("Calibration Candidate ID")["Cross-Fitted OOF Log Loss"]
        .to_dict()
    )
    results["Delta Log Loss vs None"] = results.apply(
        lambda r: float(
            r["Cross-Fitted OOF Log Loss"]
            - none_by_candidate[str(r["Calibration Candidate ID"])]
        ),
        axis=1,
    )
    results["Improves Log Loss vs None"] = results["Delta Log Loss vs None"] < 0.0

    results = results.sort_values(
        [
            "Cross-Fitted OOF Log Loss",
            "Cross-Fitted OOF Brier Score",
            "Cross-Fitted OOF AUROC",
        ],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    results.insert(
        0,
        "Overall Cross-Fitted Log Loss Rank",
        np.arange(1, len(results) + 1),
    )

    all_results_file = output_dir / "all_240_cross_fitted_calibration_results.csv"
    results.to_csv(all_results_file, index=False, float_format="%.9f")

    # One best calibration choice per original experiment.
    best = (
        results.sort_values(
            [
                "Cross-Fitted OOF Log Loss",
                "Cross-Fitted OOF Brier Score",
                "Cross-Fitted OOF AUROC",
            ],
            ascending=[True, True, False],
            kind="stable",
        )
        .groupby("Calibration Candidate ID", as_index=False, sort=False)
        .first()
        .sort_values(
            [
                "Cross-Fitted OOF Log Loss",
                "Cross-Fitted OOF Brier Score",
                "Cross-Fitted OOF AUROC",
            ],
            ascending=[True, True, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    best.insert(
        0,
        "Best-Per-Experiment Log Loss Rank",
        np.arange(1, len(best) + 1),
    )

    best_file = output_dir / "best_calibration_method_per_experiment.csv"
    best.to_csv(best_file, index=False, float_format="%.9f")

    top10_file = output_dir / "top_10_post_calibration_preview.csv"
    best.head(10).to_csv(top10_file, index=False, float_format="%.9f")

    summary = {
        "status": "PASS",
        "candidates": int(args.expected_candidates),
        "subjects_per_candidate": int(args.expected_subjects),
        "cross_fit_folds": int(args.expected_folds),
        "methods_evaluated": list(METHODS),
        "fitted_calibration_methods": list(FITTED_METHODS),
        "fitted_calibration_configurations": int(
            args.expected_candidates * len(FITTED_METHODS)
        ),
        "evaluated_configurations_including_none": int(expected_rows),
        "authoritative_calibration_module": (
            "src.calibration.probability_calibration.ProbabilityCalibrator"
        ),
        "calibration_input": "uncalibrated probability",
        "primary_competition_metric": (
            "cross-fitted OOF log loss (lower is better)"
        ),
        "best_configuration": results.iloc[0].to_dict(),
        "all_results_file": str(all_results_file.resolve()),
        "best_per_experiment_file": str(best_file.resolve()),
        "top_10_preview_file": str(top10_file.resolve()),
    }

    with (output_dir / "cross_fitted_calibration_run_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, default=str)

    print("\n" + "=" * 88)
    print("POST-CALIBRATION COMPETITION RANKING — TOP 10 EXPERIMENTS")
    print("=" * 88)

    columns = [
        "Best-Per-Experiment Log Loss Rank",
        "Calibration Candidate ID",
        "Model",
        "Scenario ID",
        "Calibration Method",
        "Cross-Fitted OOF Log Loss",
        "Delta Log Loss vs None",
        "Cross-Fitted OOF AUROC",
        "Cross-Fitted OOF Brier Score",
        "Cross-Fitted OOF ECE",
    ]
    print(best[columns].head(10).to_string(index=False))

    print(f"\nSaved: {all_results_file}")
    print(f"Saved: {best_file}")
    print(f"Saved: {top10_file}")
    print("\nSTATUS: PASS")
    print("No neural network was retrained or modified.\n")


if __name__ == "__main__":
    main()
