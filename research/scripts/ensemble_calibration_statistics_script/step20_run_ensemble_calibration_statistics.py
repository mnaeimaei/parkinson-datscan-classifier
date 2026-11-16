#!/usr/bin/env python3
"""
Phase 11 — Cross-Fitted Ensemble Calibration + Paired Log-Loss Statistics

Purpose
-------
Starting from Phase 10, automatically retain one representative ensemble for
each ensemble size 2..8 (the lowest-Log-Loss ensemble for that size), then:

1. Reconstruct each ensemble's OOF probabilities from the authoritative P1–P8
   cross-fitted prediction files.
2. Verify the reconstructed raw ensemble Log Loss matches Phase 10.
3. Evaluate cross-fitted calibration:
       none
       temperature
       platt
       isotonic
   using src.calibration.probability_calibration.ProbabilityCalibrator.
4. Select the best calibration method for each representative ensemble by
   cross-fitted OOF Log Loss.
5. Rank the seven representatives after calibration.
6. Compare each selected ensemble configuration against P1 using paired
   subject-level ΔLogLoss and a class-stratified bootstrap.
7. Report nominal 95% and Bonferroni familywise confidence intervals.

No neural network is retrained.
No competition/test labels are used.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


EPS = 1e-6
METHODS = ("none", "temperature", "platt", "isotonic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit calibration for representative Phase-10 ensembles "
            "and compare them with P1 using paired Log-Loss bootstrap."
        )
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--shortlist", type=Path, default=None)
    parser.add_argument("--phase10-best-per-size", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference-id", type=str, default="P1")
    parser.add_argument("--expected-candidates", type=int, default=8)
    parser.add_argument("--expected-subjects", type=int, default=1362)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--min-ensemble-size", type=int, default=2)
    parser.add_argument("--max-ensemble-size", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--max-calibration-iter", type=int, default=150)
    parser.add_argument(
        "--global-match-tolerance",
        type=float,
        default=2e-6,
    )
    return parser.parse_args()


def binary_subject_log_loss(y_true: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)

    if len(y) != len(p):
        raise ValueError("Label/probability length mismatch.")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Labels must contain only 0/1.")
    if not np.isfinite(p).all():
        raise ValueError("Probabilities contain NaN/Inf.")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("Probabilities must be in [0,1].")

    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def binary_log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(binary_subject_log_loss(y_true, p).mean())


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def expected_calibration_error(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])

        n = int(mask.sum())
        if n == 0:
            continue

        confidence = float(p[mask].mean())
        observed = float(y[mask].mean())
        ece += (n / n_total) * abs(observed - confidence)

    return float(ece)


def compute_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    ece_bins: int,
) -> dict:
    precision, recall, _ = precision_recall_curve(y, p)

    return {
        "Cross-Fitted OOF Log Loss": binary_log_loss(y, p),
        "Cross-Fitted OOF AUROC": float(roc_auc_score(y, p)),
        "Cross-Fitted OOF AUPRC": float(auc(recall, precision)),
        "Cross-Fitted OOF Average Precision": float(
            average_precision_score(y, p)
        ),
        "Cross-Fitted OOF Brier Score": float(
            brier_score_loss(y, p)
        ),
        "Cross-Fitted OOF ECE": expected_calibration_error(
            y, p, n_bins=ece_bins
        ),
    }


def validate_prediction_file(
    path: Path,
    *,
    probability_column: str,
    expected_subjects: int,
    expected_folds: int,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)
    required = {
        "uid",
        "fold",
        "is_pathologic",
        probability_column,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    if len(df) != expected_subjects:
        raise ValueError(
            f"{path}: expected {expected_subjects} rows, found {len(df)}"
        )

    df = df.copy()
    df["uid"] = df["uid"].astype(str)

    if df["uid"].duplicated().any():
        raise ValueError(f"{path}: duplicate UIDs detected.")

    df["fold"] = pd.to_numeric(
        df["fold"], errors="raise"
    ).astype(np.int64)
    df["is_pathologic"] = pd.to_numeric(
        df["is_pathologic"], errors="raise"
    ).astype(np.int64)
    df[probability_column] = pd.to_numeric(
        df[probability_column], errors="raise"
    ).astype(np.float64)

    expected = set(range(expected_folds))
    observed = set(int(v) for v in np.unique(df["fold"]))
    if observed != expected:
        raise ValueError(
            f"{path}: expected folds {sorted(expected)}, "
            f"found {sorted(observed)}"
        )

    if not np.isin(df["is_pathologic"].to_numpy(), [0, 1]).all():
        raise ValueError(f"{path}: labels are not binary.")

    p = df[probability_column].to_numpy(dtype=np.float64)
    if not np.isfinite(p).all():
        raise ValueError(f"{path}: probability contains NaN/Inf.")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(f"{path}: probability outside [0,1].")

    return df.sort_values("uid", kind="stable").reset_index(drop=True)


def shortlist_sort_key(value: str) -> int:
    value = str(value).strip()
    if value.startswith("P") and value[1:].isdigit():
        return int(value[1:])
    return 10_000


def parse_members(value: str) -> list[str]:
    members = [v.strip() for v in str(value).split("+") if v.strip()]
    if len(members) < 2:
        raise ValueError(f"Invalid ensemble member list: {value!r}")
    if len(set(members)) != len(members):
        raise ValueError(f"Duplicate ensemble member in: {value!r}")
    return members


def make_ensemble_probability(
    probability_matrix: np.ndarray,
    member_indices: list[int],
    aggregation_rule: str,
) -> np.ndarray:
    selected = probability_matrix[:, member_indices]

    if aggregation_rule == "probability_mean":
        return np.mean(selected, axis=1)

    if aggregation_rule == "logit_mean":
        return sigmoid(np.mean(logit(selected), axis=1))

    raise ValueError(
        f"Unknown Phase-10 aggregation rule: {aggregation_rule}"
    )


def cross_fit_calibration(
    ProbabilityCalibrator,
    *,
    y: np.ndarray,
    raw_probability: np.ndarray,
    fold: np.ndarray,
    method: str,
    n_folds: int,
    max_iter: int,
) -> tuple[np.ndarray, list[dict]]:
    calibrated = np.empty_like(raw_probability, dtype=np.float64)
    fold_parameters = []

    for heldout_fold in range(n_folds):
        test_mask = fold == heldout_fold
        train_mask = ~test_mask

        y_train = y[train_mask]
        p_train = raw_probability[train_mask]
        p_test = raw_probability[test_mask]

        calibrator = ProbabilityCalibrator(
            method=method,
            output_epsilon=EPS,
        )
        calibrator.fit(
            labels=y_train,
            probabilities=p_train,
            max_iter=max_iter,
        )
        calibrated[test_mask] = calibrator.predict_proba(p_test)

        state = calibrator.to_dict()
        state["heldout_fold"] = int(heldout_fold)
        state["calibration_train_subjects"] = int(train_mask.sum())
        state["calibration_heldout_subjects"] = int(test_mask.sum())
        fold_parameters.append(state)

    if not np.isfinite(calibrated).all():
        raise RuntimeError(
            f"{method}: cross-fitted calibrated probabilities contain NaN/Inf."
        )

    return calibrated, fold_parameters


def bootstrap_mean_deltas_stratified(
    delta_matrix: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    chunk_size: int = 250,
) -> np.ndarray:
    delta_matrix = np.asarray(delta_matrix, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    neg_idx = np.flatnonzero(y == 0)
    pos_idx = np.flatnonzero(y == 1)

    if len(neg_idx) == 0 or len(pos_idx) == 0:
        raise ValueError("Both classes are required.")

    n_total = len(y)
    rng = np.random.default_rng(seed)
    out = np.empty(
        (n_bootstrap, delta_matrix.shape[1]),
        dtype=np.float64,
    )

    cursor = 0
    while cursor < n_bootstrap:
        batch = min(chunk_size, n_bootstrap - cursor)

        neg_sample = rng.choice(
            neg_idx,
            size=(batch, len(neg_idx)),
            replace=True,
        )
        pos_sample = rng.choice(
            pos_idx,
            size=(batch, len(pos_idx)),
            replace=True,
        )

        neg_mean = delta_matrix[neg_sample].mean(axis=1)
        pos_mean = delta_matrix[pos_sample].mean(axis=1)

        out[cursor:cursor + batch] = (
            len(neg_idx) * neg_mean + len(pos_idx) * pos_mean
        ) / n_total
        cursor += batch

    return out


def ci_conclusion(low: float, high: float) -> str:
    # Delta = ensemble - P1.
    if high < 0.0:
        return "ENSEMBLE BETTER"
    if low > 0.0:
        return "P1 BETTER"
    return "INCONCLUSIVE"


def main() -> None:
    args = parse_args()

    if args.bootstrap_replicates < 1000:
        raise ValueError("Use at least 1,000 bootstrap replicates.")
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha must be between 0 and 1.")

    script_path = Path(__file__).resolve()
    project_root = (
        args.project_root or script_path.parents[2]
    ).expanduser().resolve()

    shortlist_path = (
        args.shortlist
        or project_root
        / "data"
        / "calibration_validation_data"
        / "competition_shortlist_data"
        / "competition_shortlist_8.csv"
    ).expanduser().resolve()

    phase10_best_path = (
        args.phase10_best_per_size
        or project_root
        / "data"
        / "competition_ensemble_analysis_data"
        / "equal_weight_ensemble_analysis"
        / "best_ensemble_per_size.csv"
    ).expanduser().resolve()

    output_dir = (
        args.output_dir
        or project_root
        / "data"
        / "ensemble_calibration_statistics_data"
        / "representative_ensembles"
    ).expanduser().resolve()

    # Ensure project imports resolve even if PYTHONPATH was not inherited.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.calibration.probability_calibration import ProbabilityCalibrator

    print("\n" + "=" * 108)
    print("PHASE 11 — CROSS-FITTED ENSEMBLE CALIBRATION + PAIRED LOG-LOSS STATISTICS")
    print("=" * 108)
    print(f"Project root               : {project_root}")
    print(f"P1–P8 shortlist            : {shortlist_path}")
    print(f"Phase-10 best per size     : {phase10_best_path}")
    print(f"Output                     : {output_dir}")
    print(f"Representative sizes       : {args.min_ensemble_size}..{args.max_ensemble_size}")
    print(f"Calibration methods        : {', '.join(METHODS)}")
    print(f"Cross-fitting folds        : {args.expected_folds}")
    print(f"Reference                  : {args.reference_id}")
    print(f"Bootstrap replicates       : {args.bootstrap_replicates:,}")
    print(f"Bootstrap seed             : {args.seed}")
    print("Primary metric             : Cross-Fitted OOF Log Loss")
    print("Authoritative calibrator   : src/calibration/probability_calibration.py")
    print("GPU                        : NOT USED")
    print()

    if not shortlist_path.is_file():
        raise FileNotFoundError(f"Shortlist not found: {shortlist_path}")
    if not phase10_best_path.is_file():
        raise FileNotFoundError(
            f"Phase-10 best-per-size file not found: {phase10_best_path}"
        )

    shortlist = pd.read_csv(shortlist_path)
    required_shortlist = {
        "Shortlist ID",
        "Cross-Fitted OOF Log Loss",
        "Cross-Fitted Prediction File",
    }
    missing = required_shortlist - set(shortlist.columns)
    if missing:
        raise RuntimeError(
            f"P1–P8 shortlist missing columns: {sorted(missing)}"
        )
    if len(shortlist) != args.expected_candidates:
        raise RuntimeError(
            f"Expected {args.expected_candidates} P1–P8 rows, "
            f"found {len(shortlist)}."
        )

    shortlist = shortlist.copy()
    shortlist["_sort_key"] = shortlist["Shortlist ID"].map(
        shortlist_sort_key
    )
    shortlist = shortlist.sort_values("_sort_key").drop(
        columns="_sort_key"
    ).reset_index(drop=True)

    candidate_ids = shortlist["Shortlist ID"].astype(str).tolist()
    if args.reference_id not in candidate_ids:
        raise RuntimeError(
            f"Reference {args.reference_id} not present in shortlist."
        )

    # ------------------------------------------------------------------
    # Load authoritative P1–P8 predictions and align all subjects.
    # ------------------------------------------------------------------
    canonical_uid = None
    canonical_y = None
    canonical_fold = None
    probability_columns = []

    print("Loading authoritative P1–P8 probabilities:")

    for _, row in shortlist.iterrows():
        sid = str(row["Shortlist ID"])
        path = Path(
            str(row["Cross-Fitted Prediction File"])
        ).expanduser().resolve()

        df = validate_prediction_file(
            path,
            probability_column="calibrated_probability",
            expected_subjects=args.expected_subjects,
            expected_folds=args.expected_folds,
        )

        uid = df["uid"].to_numpy(dtype=str)
        y = df["is_pathologic"].to_numpy(dtype=np.int64)
        fold = df["fold"].to_numpy(dtype=np.int64)
        p = df["calibrated_probability"].to_numpy(dtype=np.float64)

        if canonical_uid is None:
            canonical_uid = uid
            canonical_y = y
            canonical_fold = fold
        else:
            if not np.array_equal(uid, canonical_uid):
                raise RuntimeError(f"{sid}: UID alignment mismatch.")
            if not np.array_equal(y, canonical_y):
                raise RuntimeError(f"{sid}: label mismatch.")
            if not np.array_equal(fold, canonical_fold):
                raise RuntimeError(f"{sid}: fold assignment mismatch.")

        recomputed_ll = binary_log_loss(y, p)
        stored_ll = float(row["Cross-Fitted OOF Log Loss"])
        if abs(recomputed_ll - stored_ll) > args.global_match_tolerance:
            raise RuntimeError(
                f"{sid}: recomputed LL {recomputed_ll:.9f} does not "
                f"match shortlist {stored_ll:.9f}."
            )

        probability_columns.append(p)
        print(f"  {sid}: LL={recomputed_ll:.6f}")

    probability_matrix = np.column_stack(probability_columns)
    assert canonical_uid is not None
    assert canonical_y is not None
    assert canonical_fold is not None

    p1_index = candidate_ids.index(args.reference_id)
    p1_probability = probability_matrix[:, p1_index]
    p1_subject_loss = binary_subject_log_loss(
        canonical_y, p1_probability
    )
    p1_ll = float(p1_subject_loss.mean())

    # ------------------------------------------------------------------
    # Select one best Phase-10 ensemble for each size 2..8.
    # ------------------------------------------------------------------
    best_per_size = pd.read_csv(phase10_best_path)

    required_best = {
        "Ensemble ID",
        "Members",
        "Number of Members",
        "Aggregation Rule",
        "Global OOF Log Loss",
    }
    missing = required_best - set(best_per_size.columns)
    if missing:
        raise RuntimeError(
            f"Phase-10 best-per-size file missing columns: {sorted(missing)}"
        )

    best_per_size["Number of Members"] = pd.to_numeric(
        best_per_size["Number of Members"], errors="raise"
    ).astype(int)

    wanted_sizes = list(
        range(args.min_ensemble_size, args.max_ensemble_size + 1)
    )
    representatives = best_per_size.loc[
        best_per_size["Number of Members"].isin(wanted_sizes)
    ].copy()

    if sorted(representatives["Number of Members"].tolist()) != wanted_sizes:
        raise RuntimeError(
            "Expected exactly one Phase-10 best ensemble for every size "
            f"{wanted_sizes}; observed "
            f"{sorted(representatives['Number of Members'].tolist())}."
        )

    representatives = representatives.sort_values(
        "Number of Members"
    ).reset_index(drop=True)

    representatives.insert(
        0,
        "Representative ID",
        [f"R{i}" for i in range(1, len(representatives) + 1)],
    )

    # ------------------------------------------------------------------
    # Reconstruct and verify representative ensemble probabilities.
    # ------------------------------------------------------------------
    raw_ensemble_probability: dict[str, np.ndarray] = {}
    representative_manifest_rows = []

    print("\nRepresentative Phase-10 ensembles:")

    for _, row in representatives.iterrows():
        rid = str(row["Representative ID"])
        eid = str(row["Ensemble ID"])
        members = parse_members(row["Members"])
        rule = str(row["Aggregation Rule"])
        size = int(row["Number of Members"])

        if len(members) != size:
            raise RuntimeError(
                f"{eid}: member count {len(members)} != declared size {size}."
            )

        unknown = sorted(set(members) - set(candidate_ids))
        if unknown:
            raise RuntimeError(
                f"{eid}: unknown P1–P8 members: {unknown}"
            )

        indices = [candidate_ids.index(m) for m in members]
        p_raw = make_ensemble_probability(
            probability_matrix,
            indices,
            rule,
        )

        reconstructed_ll = binary_log_loss(canonical_y, p_raw)
        phase10_ll = float(row["Global OOF Log Loss"])

        if abs(reconstructed_ll - phase10_ll) > args.global_match_tolerance:
            raise RuntimeError(
                f"{eid}: reconstructed LL={reconstructed_ll:.9f} "
                f"does not match Phase-10 LL={phase10_ll:.9f}."
            )

        raw_ensemble_probability[rid] = p_raw

        representative_manifest_rows.append(
            {
                "Representative ID": rid,
                "Ensemble ID": eid,
                "Members": "+".join(members),
                "Number of Members": size,
                "Aggregation Rule": rule,
                "Phase-10 Raw OOF Log Loss": reconstructed_ll,
                "Phase-10 Overall Ensemble Rank":
                    row.get("Overall Ensemble Log Loss Rank", np.nan),
            }
        )

        print(
            f"  {rid}: {eid} size={size} {rule} "
            f"members={'+'.join(members)}  LL={reconstructed_ll:.6f}"
        )

    representative_manifest = pd.DataFrame(
        representative_manifest_rows
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_root = output_dir / "cross_fitted_ensemble_calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)

    manifest_file = (
        output_dir / "representative_ensemble_manifest.csv"
    )
    representative_manifest.to_csv(
        manifest_file,
        index=False,
        float_format="%.9f",
    )

    # ------------------------------------------------------------------
    # Cross-fitted calibration for all representative ensembles.
    # ------------------------------------------------------------------
    all_rows = []
    prediction_lookup: dict[tuple[str, str], np.ndarray] = {}

    print("\nCross-fitted ensemble calibration:")

    for _, rep in representative_manifest.iterrows():
        rid = str(rep["Representative ID"])
        eid = str(rep["Ensemble ID"])
        p_raw = raw_ensemble_probability[rid]
        raw_ll = float(rep["Phase-10 Raw OOF Log Loss"])

        print(f"\n{rid} {eid} ({rep['Members']})")

        for method in METHODS:
            calibrated, fold_parameters = cross_fit_calibration(
                ProbabilityCalibrator,
                y=canonical_y,
                raw_probability=p_raw,
                fold=canonical_fold,
                method=method,
                n_folds=args.expected_folds,
                max_iter=args.max_calibration_iter,
            )

            metrics = compute_metrics(
                canonical_y,
                calibrated,
                ece_bins=args.ece_bins,
            )
            delta_vs_raw = (
                metrics["Cross-Fitted OOF Log Loss"] - raw_ll
            )
            delta_vs_p1 = (
                metrics["Cross-Fitted OOF Log Loss"] - p1_ll
            )

            method_dir = calibration_root / f"{rid}_{eid}" / method
            method_dir.mkdir(parents=True, exist_ok=True)

            pred_file = method_dir / "cross_fitted_predictions.csv"
            param_file = method_dir / "fold_parameters.json"
            metric_file = method_dir / "metrics.json"

            pd.DataFrame(
                {
                    "uid": canonical_uid,
                    "fold": canonical_fold,
                    "is_pathologic": canonical_y,
                    "raw_ensemble_probability": p_raw,
                    "calibrated_probability": calibrated,
                }
            ).to_csv(
                pred_file,
                index=False,
                float_format="%.9f",
            )

            with param_file.open("w", encoding="utf-8") as f:
                json.dump(fold_parameters, f, indent=2)

            metric_payload = {
                "Representative ID": rid,
                "Ensemble ID": eid,
                "Members": rep["Members"],
                "Number of Members": int(rep["Number of Members"]),
                "Aggregation Rule": rep["Aggregation Rule"],
                "Calibration Method": method,
                **metrics,
                "Phase-10 Raw OOF Log Loss": raw_ll,
                "Delta Log Loss vs Raw Ensemble": delta_vs_raw,
                "P1 OOF Log Loss": p1_ll,
                "Delta Log Loss vs P1": delta_vs_p1,
            }
            with metric_file.open("w", encoding="utf-8") as f:
                json.dump(metric_payload, f, indent=2)

            all_rows.append(
                {
                    **metric_payload,
                    "Cross-Fitted Prediction File": str(pred_file),
                    "Fold Parameters File": str(param_file),
                }
            )
            prediction_lookup[(rid, method)] = calibrated

            print(
                f"    {method:<11s} "
                f"LL={metrics['Cross-Fitted OOF Log Loss']:.6f}  "
                f"Δraw={delta_vs_raw:+.6f}  "
                f"ΔP1={delta_vs_p1:+.6f}"
            )

    all_results = pd.DataFrame(all_rows)
    all_results = all_results.sort_values(
        [
            "Cross-Fitted OOF Log Loss",
            "Cross-Fitted OOF AUROC",
            "Cross-Fitted OOF Brier Score",
            "Cross-Fitted OOF ECE",
        ],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    all_results.insert(
        0,
        "Overall Calibrated Ensemble Rank",
        np.arange(1, len(all_results) + 1),
    )

    # Best calibration method separately for each representative.
    best_per_ensemble = (
        all_results.sort_values(
            [
                "Representative ID",
                "Cross-Fitted OOF Log Loss",
                "Cross-Fitted OOF AUROC",
            ],
            ascending=[True, True, False],
            kind="stable",
        )
        .groupby("Representative ID", sort=False, as_index=False)
        .first()
    )

    best_per_ensemble = best_per_ensemble.sort_values(
        [
            "Cross-Fitted OOF Log Loss",
            "Cross-Fitted OOF AUROC",
        ],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)

    best_per_ensemble.insert(
        0,
        "Post-Calibration Representative Rank",
        np.arange(1, len(best_per_ensemble) + 1),
    )

    best_overall = best_per_ensemble.iloc[[0]].copy()

    # ------------------------------------------------------------------
    # Fold stability for each best calibrated representative.
    # ------------------------------------------------------------------
    fold_rows = []

    for _, row in best_per_ensemble.iterrows():
        rid = str(row["Representative ID"])
        method = str(row["Calibration Method"])
        p = prediction_lookup[(rid, method)]

        for fold_id in range(args.expected_folds):
            mask = canonical_fold == fold_id
            fold_rows.append(
                {
                    "Representative ID": rid,
                    "Ensemble ID": row["Ensemble ID"],
                    "Members": row["Members"],
                    "Number of Members": int(row["Number of Members"]),
                    "Calibration Method": method,
                    "Fold": int(fold_id),
                    "Fold Subjects": int(mask.sum()),
                    "Fold Log Loss": binary_log_loss(
                        canonical_y[mask], p[mask]
                    ),
                }
            )

    fold_results = pd.DataFrame(fold_rows)

    stability_rows = []
    for rid, group in fold_results.groupby(
        "Representative ID", sort=False
    ):
        losses = group.sort_values("Fold")[
            "Fold Log Loss"
        ].to_numpy(dtype=np.float64)

        first = group.iloc[0]
        stability_rows.append(
            {
                "Representative ID": rid,
                "Ensemble ID": first["Ensemble ID"],
                "Members": first["Members"],
                "Number of Members": int(first["Number of Members"]),
                "Calibration Method": first["Calibration Method"],
                "Mean Fold Log Loss": float(losses.mean()),
                "Fold Log Loss SD": float(np.std(losses, ddof=1)),
                "Minimum Fold Log Loss": float(losses.min()),
                "Maximum Fold Log Loss": float(losses.max()),
                "Fold Log Loss Range": float(losses.max() - losses.min()),
                "Best Fold": int(np.argmin(losses)),
                "Worst Fold": int(np.argmax(losses)),
            }
        )

    stability = pd.DataFrame(stability_rows)

    # ------------------------------------------------------------------
    # Paired subject-level comparison of each selected representative vs P1.
    # ------------------------------------------------------------------
    comparator_ids = best_per_ensemble[
        "Representative ID"
    ].astype(str).tolist()

    delta_columns = []
    subject_table = pd.DataFrame(
        {
            "uid": canonical_uid,
            "fold": canonical_fold,
            "is_pathologic": canonical_y,
            "P1 Probability": p1_probability,
            "P1 Subject Log Loss": p1_subject_loss,
        }
    )

    for rid in comparator_ids:
        row = best_per_ensemble.loc[
            best_per_ensemble["Representative ID"] == rid
        ].iloc[0]
        method = str(row["Calibration Method"])
        p = prediction_lookup[(rid, method)]
        loss = binary_subject_log_loss(canonical_y, p)
        delta = loss - p1_subject_loss

        subject_table[f"{rid} Probability"] = p
        subject_table[f"{rid} Subject Log Loss"] = loss

        delta_col = f"Delta Log Loss {rid} minus P1"
        subject_table[delta_col] = delta
        delta_columns.append(delta_col)

    delta_matrix = subject_table[
        delta_columns
    ].to_numpy(dtype=np.float64)

    print("\nRunning paired class-stratified bootstrap...")

    bootstrap = bootstrap_mean_deltas_stratified(
        delta_matrix,
        canonical_y,
        n_bootstrap=args.bootstrap_replicates,
        seed=args.seed,
    )

    n_comparisons = len(comparator_ids)
    nominal_low_q = args.alpha / 2.0
    nominal_high_q = 1.0 - args.alpha / 2.0

    familywise_alpha_each = args.alpha / n_comparisons
    family_low_q = familywise_alpha_each / 2.0
    family_high_q = 1.0 - familywise_alpha_each / 2.0
    family_confidence = 1.0 - familywise_alpha_each

    paired_rows = []

    for j, rid in enumerate(comparator_ids):
        row = best_per_ensemble.loc[
            best_per_ensemble["Representative ID"] == rid
        ].iloc[0]

        delta = delta_matrix[:, j]
        boot = bootstrap[:, j]

        observed = float(delta.mean())
        expected = (
            float(row["Cross-Fitted OOF Log Loss"]) - p1_ll
        )

        if abs(observed - expected) > 1e-12:
            raise RuntimeError(
                f"{rid}: mean subject ΔLogLoss does not match "
                "the recomputed global difference."
            )

        nominal_low, nominal_high = np.quantile(
            boot,
            [nominal_low_q, nominal_high_q],
        )
        family_low, family_high = np.quantile(
            boot,
            [family_low_q, family_high_q],
        )

        paired_rows.append(
            {
                "Reference ID": args.reference_id,
                "Representative ID": rid,
                "Ensemble ID": row["Ensemble ID"],
                "Members": row["Members"],
                "Number of Members": int(row["Number of Members"]),
                "Aggregation Rule": row["Aggregation Rule"],
                "Calibration Method": row["Calibration Method"],
                "P1 Cross-Fitted OOF Log Loss": p1_ll,
                "Ensemble Cross-Fitted OOF Log Loss":
                    float(row["Cross-Fitted OOF Log Loss"]),
                "Observed Mean Delta Log Loss": observed,
                "Nominal CI Confidence": 1.0 - args.alpha,
                "Nominal CI Lower": float(nominal_low),
                "Nominal CI Upper": float(nominal_high),
                "Nominal CI Conclusion": ci_conclusion(
                    float(nominal_low), float(nominal_high)
                ),
                "Familywise CI Confidence Per Comparison":
                    family_confidence,
                "Familywise CI Lower": float(family_low),
                "Familywise CI Upper": float(family_high),
                "Familywise CI Conclusion": ci_conclusion(
                    float(family_low), float(family_high)
                ),
                "Subjects Ensemble Better Than P1":
                    int(np.sum(delta < 0.0)),
                "Subjects P1 Better Than Ensemble":
                    int(np.sum(delta > 0.0)),
                "Subjects Tied": int(np.sum(delta == 0.0)),
                "Ensemble Better Subject Fraction":
                    float(np.mean(delta < 0.0)),
                "P1 Better Subject Fraction":
                    float(np.mean(delta > 0.0)),
            }
        )

    paired = pd.DataFrame(paired_rows)
    paired = paired.sort_values(
        "Observed Mean Delta Log Loss",
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Save outputs.
    # ------------------------------------------------------------------
    all_file = (
        output_dir
        / "all_28_cross_fitted_ensemble_calibration_results.csv"
    )
    best_file = (
        output_dir
        / "best_calibration_method_per_representative_ensemble.csv"
    )
    winner_file = (
        output_dir / "best_ensemble_after_cross_fitted_calibration.csv"
    )
    fold_file = (
        output_dir / "best_representative_ensemble_fold_log_loss.csv"
    )
    stability_file = (
        output_dir / "best_representative_ensemble_stability_summary.csv"
    )
    subject_file = (
        output_dir
        / "representative_ensemble_subject_level_log_loss_differences.csv"
    )
    paired_file = (
        output_dir
        / "representative_ensemble_vs_p1_paired_bootstrap_results.csv"
    )
    bootstrap_file = (
        output_dir
        / "representative_ensemble_vs_p1_bootstrap_distributions.npz"
    )
    summary_file = (
        output_dir / "ensemble_calibration_statistics_run_summary.json"
    )
    report_file = (
        output_dir / "ensemble_calibration_statistics_report.md"
    )

    all_results.to_csv(
        all_file,
        index=False,
        float_format="%.9f",
    )
    best_per_ensemble.to_csv(
        best_file,
        index=False,
        float_format="%.9f",
    )
    best_overall.to_csv(
        winner_file,
        index=False,
        float_format="%.9f",
    )
    fold_results.to_csv(
        fold_file,
        index=False,
        float_format="%.9f",
    )
    stability.to_csv(
        stability_file,
        index=False,
        float_format="%.9f",
    )
    subject_table.to_csv(
        subject_file,
        index=False,
        float_format="%.9f",
    )
    paired.to_csv(
        paired_file,
        index=False,
        float_format="%.9f",
    )

    np.savez_compressed(
        bootstrap_file,
        representative_ids=np.asarray(
            comparator_ids, dtype=str
        ),
        bootstrap_mean_delta_log_loss=bootstrap,
    )

    winner = best_overall.iloc[0]

    summary_payload = {
        "status": "PASS",
        "representative_rule": (
            "Best Phase-10 ensemble for each size "
            f"{args.min_ensemble_size}..{args.max_ensemble_size}"
        ),
        "representative_count": int(len(representative_manifest)),
        "calibration_methods": list(METHODS),
        "total_calibration_configurations": int(len(all_results)),
        "cross_fit_folds": int(args.expected_folds),
        "subjects": int(args.expected_subjects),
        "reference": {
            "id": args.reference_id,
            "cross_fitted_oof_log_loss": p1_ll,
        },
        "bootstrap": {
            "method": "paired class-stratified subject bootstrap",
            "replicates": int(args.bootstrap_replicates),
            "seed": int(args.seed),
            "nominal_confidence": float(1.0 - args.alpha),
            "familywise_method":
                "Bonferroni simultaneous percentile intervals",
            "familywise_comparisons": int(n_comparisons),
            "familywise_per_comparison_confidence":
                float(family_confidence),
        },
        "best_post_calibration_ensemble": {
            "representative_id": str(
                winner["Representative ID"]
            ),
            "ensemble_id": str(winner["Ensemble ID"]),
            "members": str(winner["Members"]),
            "number_of_members": int(
                winner["Number of Members"]
            ),
            "aggregation_rule": str(
                winner["Aggregation Rule"]
            ),
            "calibration_method": str(
                winner["Calibration Method"]
            ),
            "cross_fitted_oof_log_loss": float(
                winner["Cross-Fitted OOF Log Loss"]
            ),
            "delta_log_loss_vs_p1": float(
                winner["Delta Log Loss vs P1"]
            ),
        },
        "outputs": {
            "representative_manifest": str(manifest_file),
            "all_calibration_results": str(all_file),
            "best_method_per_ensemble": str(best_file),
            "best_overall": str(winner_file),
            "fold_log_loss": str(fold_file),
            "stability_summary": str(stability_file),
            "subject_level_differences": str(subject_file),
            "paired_bootstrap_results": str(paired_file),
            "bootstrap_distributions": str(bootstrap_file),
            "report": str(report_file),
        },
    }

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    report_cols = [
        "Post-Calibration Representative Rank",
        "Representative ID",
        "Ensemble ID",
        "Members",
        "Number of Members",
        "Calibration Method",
        "Cross-Fitted OOF Log Loss",
        "Delta Log Loss vs Raw Ensemble",
        "Delta Log Loss vs P1",
        "Cross-Fitted OOF AUROC",
        "Cross-Fitted OOF Brier Score",
        "Cross-Fitted OOF ECE",
    ]

    md = [
        "# Phase 11 — Ensemble Calibration + Paired Log-Loss Statistics",
        "",
        f"P1 reference Log Loss: **{p1_ll:.6f}**.",
        "",
        (
            "Representative selection: best Phase-10 ensemble for each "
            f"size {args.min_ensemble_size}–{args.max_ensemble_size}."
        ),
        "",
        f"Calibration configurations evaluated: **{len(all_results)}**.",
        "",
        "## Best calibration per representative ensemble",
        "",
        "| " + " | ".join(report_cols) + " |",
        "| " + " | ".join(["---"] * len(report_cols)) + " |",
    ]

    for _, row in best_per_ensemble[report_cols].iterrows():
        values = []
        for col in report_cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        md.append("| " + " | ".join(values) + " |")

    md.extend(
        [
            "",
            "## Best observed post-calibration ensemble",
            "",
            f"- Representative: `{winner['Representative ID']}`",
            f"- Ensemble: `{winner['Ensemble ID']}`",
            f"- Members: `{winner['Members']}`",
            f"- Aggregation: `{winner['Aggregation Rule']}`",
            f"- Calibration: `{winner['Calibration Method']}`",
            (
                "- Cross-fitted OOF Log Loss: "
                f"`{winner['Cross-Fitted OOF Log Loss']:.6f}`"
            ),
            (
                "- ΔLogLoss vs P1: "
                f"`{winner['Delta Log Loss vs P1']:+.6f}`"
            ),
            "",
            "## Statistical interpretation",
            "",
            (
                "Paired bootstrap uses subject-level ΔLogLoss = "
                "ensemble − P1. Negative values favor the ensemble."
            ),
            "",
            (
                "Because ensemble subsets were selected using the same OOF "
                "dataset in Phase 10, these confidence intervals are "
                "supporting evidence rather than a fully independent "
                "post-selection validation."
            ),
            "",
        ]
    )

    report_file.write_text("\n".join(md), encoding="utf-8")

    # ------------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 108)
    print("BEST CALIBRATION METHOD PER REPRESENTATIVE ENSEMBLE")
    print("=" * 108)

    display_cols = [
        "Post-Calibration Representative Rank",
        "Representative ID",
        "Ensemble ID",
        "Members",
        "Number of Members",
        "Aggregation Rule",
        "Calibration Method",
        "Cross-Fitted OOF Log Loss",
        "Delta Log Loss vs Raw Ensemble",
        "Delta Log Loss vs P1",
        "Cross-Fitted OOF AUROC",
        "Cross-Fitted OOF Brier Score",
        "Cross-Fitted OOF ECE",
    ]
    print(
        best_per_ensemble[display_cols].to_string(index=False)
    )

    print("\n" + "=" * 108)
    print("PAIRED ΔLOGLOSS VS P1")
    print("=" * 108)

    paired_display = [
        "Representative ID",
        "Ensemble ID",
        "Members",
        "Calibration Method",
        "Ensemble Cross-Fitted OOF Log Loss",
        "Observed Mean Delta Log Loss",
        "Nominal CI Lower",
        "Nominal CI Upper",
        "Nominal CI Conclusion",
        "Familywise CI Lower",
        "Familywise CI Upper",
        "Familywise CI Conclusion",
    ]
    print(paired[paired_display].to_string(index=False))

    print()
    print(f"P1 Log Loss                  : {p1_ll:.6f}")
    print(
        f"Best post-calibration        : "
        f"{winner['Representative ID']} / {winner['Ensemble ID']}"
    )
    print(
        f"Members                      : {winner['Members']}"
    )
    print(
        f"Aggregation                  : {winner['Aggregation Rule']}"
    )
    print(
        f"Calibration                  : {winner['Calibration Method']}"
    )
    print(
        f"Best Cross-Fitted OOF LL     : "
        f"{winner['Cross-Fitted OOF Log Loss']:.6f}"
    )
    print(
        f"Delta Log Loss vs P1         : "
        f"{winner['Delta Log Loss vs P1']:+.6f}"
    )
    print()
    print(f"Saved: {manifest_file}")
    print(f"Saved: {all_file}")
    print(f"Saved: {best_file}")
    print(f"Saved: {winner_file}")
    print(f"Saved: {fold_file}")
    print(f"Saved: {stability_file}")
    print(f"Saved: {subject_file}")
    print(f"Saved: {paired_file}")
    print(f"Saved: {bootstrap_file}")
    print(f"Saved: {summary_file}")
    print(f"Saved: {report_file}")
    print()
    print("STATUS: PASS")
    print(
        "Next step: freeze the final predictor/calibration choice and "
        "build the deployment/inference configuration."
    )


if __name__ == "__main__":
    main()
