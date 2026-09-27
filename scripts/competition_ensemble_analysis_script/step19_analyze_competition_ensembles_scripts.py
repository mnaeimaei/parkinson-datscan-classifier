#!/usr/bin/env python3
"""
Phase 10 — Competition-Oriented Equal-Weight Ensemble Analysis

Input
-----
Post-calibration P1–P8 shortlist:
    competition_shortlist_8.csv

For every subset of size 2..8, evaluate:
    1) equal-weight probability averaging
    2) equal-weight logit averaging

With 8 candidates:
    subsets = 2^8 - 8 - 1 = 247
    aggregation rules = 2
    total ensembles = 494

Primary criterion:
    Global OOF Log Loss (lower is better)

Supporting metrics:
    AUROC, AUPRC, Average Precision, Brier Score, ECE

No neural network is retrained.
No calibrator is fitted or refitted in this phase.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all equal-weight P1–P8 ensemble subsets."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--shortlist", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-candidates", type=int, default=8)
    parser.add_argument("--expected-subjects", type=int, default=1362)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--reference-id", type=str, default="P1")
    parser.add_argument("--top-n-save-predictions", type=int, default=20)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument(
        "--global-match-tolerance",
        type=float,
        default=2e-6,
    )
    return parser.parse_args()


def binary_log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def expected_calibration_error(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=np.float64)
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

        confidence = float(p[mask].mean())
        accuracy = float(y[mask].mean())
        ece += (n / total) * abs(accuracy - confidence)

    return float(ece)


def compute_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    ece_bins: int,
) -> dict:
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)

    precision, recall, _ = precision_recall_curve(y, p)

    return {
        "Global OOF Log Loss": binary_log_loss(y, p),
        "Global OOF AUROC": float(roc_auc_score(y, p)),
        "Global OOF AUPRC": float(auc(recall, precision)),
        "Global OOF Average Precision": float(
            average_precision_score(y, p)
        ),
        "Global OOF Brier Score": float(brier_score_loss(y, p)),
        "Global OOF ECE": expected_calibration_error(
            y, p, n_bins=ece_bins
        ),
    }


def validate_prediction_file(
    path: Path,
    *,
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
        "calibrated_probability",
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
        raise ValueError(f"{path}: duplicate UIDs detected")

    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(np.int64)
    df["is_pathologic"] = pd.to_numeric(
        df["is_pathologic"], errors="raise"
    ).astype(np.int64)
    df["calibrated_probability"] = pd.to_numeric(
        df["calibrated_probability"], errors="raise"
    ).astype(np.float64)

    expected_folds_set = set(range(expected_folds))
    observed_folds = set(int(x) for x in np.unique(df["fold"]))
    if observed_folds != expected_folds_set:
        raise ValueError(
            f"{path}: expected folds {sorted(expected_folds_set)}, "
            f"found {sorted(observed_folds)}"
        )

    if not np.isin(df["is_pathologic"].to_numpy(), [0, 1]).all():
        raise ValueError(f"{path}: labels are not binary")

    p = df["calibrated_probability"].to_numpy()
    if not np.isfinite(p).all():
        raise ValueError(f"{path}: probability contains NaN/Inf")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(f"{path}: probability outside [0,1]")

    return df.sort_values("uid", kind="stable").reset_index(drop=True)


def shortlist_sort_key(value: str) -> int:
    value = str(value).strip()
    if value.startswith("P") and value[1:].isdigit():
        return int(value[1:])
    return 10000


def make_ensemble_probability(
    probability_matrix: np.ndarray,
    member_indices: tuple[int, ...],
    rule: str,
) -> np.ndarray:
    selected = probability_matrix[:, list(member_indices)]

    if rule == "probability_mean":
        return np.mean(selected, axis=1)

    if rule == "logit_mean":
        mean_logit = np.mean(logit(selected), axis=1)
        return sigmoid(mean_logit)

    raise ValueError(f"Unknown aggregation rule: {rule}")


def main() -> None:
    args = parse_args()

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

    output_dir = (
        args.output_dir
        or project_root
        / "data"
        / "competition_ensemble_analysis_data"
        / "equal_weight_ensemble_analysis"
    ).expanduser().resolve()

    print("\n" + "=" * 104)
    print("PHASE 10 — COMPETITION-ORIENTED EQUAL-WEIGHT ENSEMBLE ANALYSIS")
    print("=" * 104)
    print(f"Project root          : {project_root}")
    print(f"Shortlist             : {shortlist_path}")
    print(f"Output                : {output_dir}")
    print(f"Expected candidates   : {args.expected_candidates}")
    print(f"Expected OOF subjects : {args.expected_subjects}")
    print(f"Expected folds        : {args.expected_folds}")
    print(f"Reference             : {args.reference_id}")
    print("Aggregation rules     : probability_mean, logit_mean")
    print("Subset sizes          : 2 through 8")
    print("Primary metric        : Global OOF Log Loss")
    print("Lower is better")
    print()

    if not shortlist_path.is_file():
        raise FileNotFoundError(f"Shortlist not found: {shortlist_path}")

    shortlist = pd.read_csv(shortlist_path)

    required_shortlist = {
        "Shortlist ID",
        "Calibration Candidate ID",
        "Model",
        "Scenario ID",
        "Calibration Method",
        "Cross-Fitted OOF Log Loss",
        "Cross-Fitted Prediction File",
    }
    missing = required_shortlist - set(shortlist.columns)
    if missing:
        raise RuntimeError(
            f"Shortlist missing required columns: {sorted(missing)}"
        )

    if len(shortlist) != args.expected_candidates:
        raise RuntimeError(
            f"Expected {args.expected_candidates} shortlist rows, "
            f"found {len(shortlist)}"
        )

    shortlist = shortlist.copy()
    shortlist["_sort_key"] = shortlist["Shortlist ID"].map(shortlist_sort_key)
    shortlist = shortlist.sort_values(
        "_sort_key", kind="stable"
    ).drop(columns="_sort_key").reset_index(drop=True)

    ids = shortlist["Shortlist ID"].astype(str).tolist()

    if args.reference_id not in ids:
        raise RuntimeError(
            f"Reference ID {args.reference_id} not present in shortlist."
        )

    # ------------------------------------------------------------------
    # Load and align P1–P8.
    # ------------------------------------------------------------------
    canonical_uid = None
    canonical_y = None
    canonical_fold = None
    probability_columns = []

    print("Loading P1–P8 predictions:")

    for i, row in shortlist.iterrows():
        sid = str(row["Shortlist ID"])
        path = Path(
            str(row["Cross-Fitted Prediction File"])
        ).expanduser().resolve()

        df = validate_prediction_file(
            path,
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
        diff = abs(recomputed_ll - stored_ll)

        if diff > args.global_match_tolerance:
            raise RuntimeError(
                f"{sid}: recomputed LL={recomputed_ll:.9f}, "
                f"stored LL={stored_ll:.9f}, diff={diff:.3e}"
            )

        probability_columns.append(p)

        print(
            f"  {sid}: {row['Model']}-{row['Scenario ID']} "
            f"{row['Calibration Method']}  LL={recomputed_ll:.6f}"
        )

    probability_matrix = np.column_stack(probability_columns)
    assert canonical_y is not None
    assert canonical_uid is not None
    assert canonical_fold is not None

    reference_idx = ids.index(args.reference_id)
    reference_probability = probability_matrix[:, reference_idx]
    reference_metrics = compute_metrics(
        canonical_y,
        reference_probability,
        ece_bins=args.ece_bins,
    )
    reference_ll = reference_metrics["Global OOF Log Loss"]

    # ------------------------------------------------------------------
    # Enumerate all subsets and both equal-weight rules.
    # ------------------------------------------------------------------
    subsets = []
    for size in range(2, len(ids) + 1):
        subsets.extend(itertools.combinations(range(len(ids)), size))

    rules = ["probability_mean", "logit_mean"]

    expected_subsets = (2 ** len(ids)) - len(ids) - 1
    expected_ensembles = expected_subsets * len(rules)

    if len(subsets) != expected_subsets:
        raise RuntimeError("Unexpected subset count.")

    print()
    print(f"Candidate subsets      : {len(subsets)}")
    print(f"Aggregation rules      : {len(rules)}")
    print(f"Total ensembles        : {expected_ensembles}")
    print()

    rows = []
    prediction_cache = {}

    ensemble_counter = 0
    for subset in subsets:
        member_ids = [ids[i] for i in subset]
        member_text = "+".join(member_ids)
        size = len(subset)

        for rule in rules:
            ensemble_counter += 1
            ensemble_id = f"ENS{ensemble_counter:03d}"

            p_ens = make_ensemble_probability(
                probability_matrix,
                subset,
                rule,
            )

            metrics = compute_metrics(
                canonical_y,
                p_ens,
                ece_bins=args.ece_bins,
            )

            delta_ll = metrics["Global OOF Log Loss"] - reference_ll
            relative_improvement = (
                (reference_ll - metrics["Global OOF Log Loss"])
                / reference_ll
                * 100.0
            )

            rows.append(
                {
                    "Ensemble ID": ensemble_id,
                    "Members": member_text,
                    "Number of Members": size,
                    "Contains P1": args.reference_id in member_ids,
                    "Aggregation Rule": rule,
                    **metrics,
                    "P1 Global OOF Log Loss": reference_ll,
                    "Delta Log Loss vs P1": delta_ll,
                    "Improves Log Loss vs P1": bool(delta_ll < 0.0),
                    "Relative Log Loss Improvement vs P1 (%)":
                        relative_improvement,
                }
            )
            prediction_cache[ensemble_id] = p_ens

    results = pd.DataFrame(rows)

    # Competition ranking.
    results = results.sort_values(
        by=[
            "Global OOF Log Loss",
            "Global OOF AUROC",
            "Global OOF Brier Score",
            "Global OOF ECE",
        ],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    results.insert(
        0,
        "Overall Ensemble Log Loss Rank",
        np.arange(1, len(results) + 1, dtype=int),
    )

    # Best within each ensemble size and aggregation rule.
    best_per_size_rule = (
        results.sort_values(
            [
                "Number of Members",
                "Aggregation Rule",
                "Global OOF Log Loss",
            ],
            ascending=[True, True, True],
            kind="stable",
        )
        .groupby(
            ["Number of Members", "Aggregation Rule"],
            sort=True,
            as_index=False,
        )
        .first()
    )

    # Best per size regardless of rule.
    best_per_size = (
        results.sort_values(
            ["Number of Members", "Global OOF Log Loss"],
            ascending=[True, True],
            kind="stable",
        )
        .groupby("Number of Members", sort=True, as_index=False)
        .first()
    )

    top_n = min(args.top_n_save_predictions, len(results))
    top = results.head(top_n).copy()
    best = results.iloc[[0]].copy()

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_root = output_dir / "top_ensemble_predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    all_file = output_dir / "all_494_equal_weight_ensemble_results.csv"
    top_file = output_dir / f"top_{top_n}_ensemble_preview.csv"
    best_file = output_dir / "best_overall_ensemble.csv"
    size_rule_file = output_dir / "best_ensemble_per_size_and_rule.csv"
    size_file = output_dir / "best_ensemble_per_size.csv"
    manifest_file = output_dir / f"top_{top_n}_ensemble_prediction_manifest.csv"
    run_file = output_dir / "competition_ensemble_analysis_run_summary.json"
    report_file = output_dir / "competition_ensemble_analysis_report.md"

    results.to_csv(all_file, index=False, float_format="%.9f")
    top.to_csv(top_file, index=False, float_format="%.9f")
    best.to_csv(best_file, index=False, float_format="%.9f")
    best_per_size_rule.to_csv(
        size_rule_file, index=False, float_format="%.9f"
    )
    best_per_size.to_csv(size_file, index=False, float_format="%.9f")

    manifest_rows = []

    # Top predictions need retrieval by original Ensemble ID.
    for _, row in top.iterrows():
        eid = str(row["Ensemble ID"])
        p = prediction_cache[eid]

        ensemble_dir = pred_root / eid
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        pred_file = ensemble_dir / "cross_fitted_ensemble_predictions.csv"

        pred_df = pd.DataFrame(
            {
                "uid": canonical_uid,
                "fold": canonical_fold,
                "is_pathologic": canonical_y,
                "ensemble_probability": p,
            }
        )
        pred_df.to_csv(pred_file, index=False, float_format="%.9f")

        manifest_rows.append(
            {
                "Overall Ensemble Log Loss Rank":
                    int(row["Overall Ensemble Log Loss Rank"]),
                "Ensemble ID": eid,
                "Members": row["Members"],
                "Number of Members": int(row["Number of Members"]),
                "Aggregation Rule": row["Aggregation Rule"],
                "Global OOF Log Loss": float(row["Global OOF Log Loss"]),
                "Delta Log Loss vs P1": float(row["Delta Log Loss vs P1"]),
                "Improves Log Loss vs P1":
                    bool(row["Improves Log Loss vs P1"]),
                "Cross-Fitted Ensemble Prediction File": str(pred_file),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_file, index=False, float_format="%.9f")

    n_better = int(results["Improves Log Loss vs P1"].sum())
    best_row = results.iloc[0]

    run_summary = {
        "status": "PASS",
        "shortlist_candidates": len(ids),
        "candidate_ids": ids,
        "subjects": int(args.expected_subjects),
        "folds": int(args.expected_folds),
        "subsets_evaluated": len(subsets),
        "aggregation_rules": rules,
        "total_ensembles_evaluated": len(results),
        "primary_metric": "Global OOF Log Loss",
        "reference": {
            "id": args.reference_id,
            "global_oof_log_loss": reference_ll,
        },
        "ensembles_beating_p1": n_better,
        "best_ensemble": {
            "ensemble_id": str(best_row["Ensemble ID"]),
            "members": str(best_row["Members"]),
            "number_of_members": int(best_row["Number of Members"]),
            "aggregation_rule": str(best_row["Aggregation Rule"]),
            "global_oof_log_loss": float(
                best_row["Global OOF Log Loss"]
            ),
            "delta_log_loss_vs_p1": float(
                best_row["Delta Log Loss vs P1"]
            ),
            "relative_improvement_percent": float(
                best_row[
                    "Relative Log Loss Improvement vs P1 (%)"
                ]
            ),
        },
        "outputs": {
            "all_results": str(all_file),
            "top_preview": str(top_file),
            "best_overall": str(best_file),
            "best_per_size_rule": str(size_rule_file),
            "best_per_size": str(size_file),
            "top_prediction_manifest": str(manifest_file),
            "markdown_report": str(report_file),
        },
    }

    with run_file.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    report_cols = [
        "Overall Ensemble Log Loss Rank",
        "Ensemble ID",
        "Members",
        "Number of Members",
        "Aggregation Rule",
        "Global OOF Log Loss",
        "Delta Log Loss vs P1",
        "Global OOF AUROC",
        "Global OOF Brier Score",
        "Global OOF ECE",
    ]

    md = [
        "# Phase 10 — Competition-Oriented Ensemble Analysis",
        "",
        f"P1 reference Log Loss: **{reference_ll:.6f}**.",
        "",
        f"Equal-weight ensembles evaluated: **{len(results)}**.",
        "",
        f"Ensembles with Log Loss below P1: **{n_better}**.",
        "",
        "## Top ensembles",
        "",
        "| " + " | ".join(report_cols) + " |",
        "| " + " | ".join(["---"] * len(report_cols)) + " |",
    ]

    for _, r in top.head(20)[report_cols].iterrows():
        vals = []
        for c in report_cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.6f}")
            else:
                vals.append(str(v))
        md.append("| " + " | ".join(vals) + " |")

    md.extend(
        [
            "",
            "## Best overall",
            "",
            f"- Ensemble ID: `{best_row['Ensemble ID']}`",
            f"- Members: `{best_row['Members']}`",
            f"- Rule: `{best_row['Aggregation Rule']}`",
            f"- OOF Log Loss: `{best_row['Global OOF Log Loss']:.6f}`",
            f"- ΔLogLoss vs P1: `{best_row['Delta Log Loss vs P1']:+.6f}`",
            (
                "- Relative Log-Loss improvement vs P1: "
                f"`{best_row['Relative Log Loss Improvement vs P1 (%)']:.3f}%`"
            ),
            "",
            "This phase is an OOF ensemble-selection analysis. "
            "The best ensemble should next undergo cross-fitted ensemble "
            "calibration and paired statistical comparison against P1.",
            "",
        ]
    )
    report_file.write_text("\n".join(md), encoding="utf-8")

    print("=" * 104)
    print("TOP 20 EQUAL-WEIGHT ENSEMBLES")
    print("=" * 104)

    display_cols = [
        "Overall Ensemble Log Loss Rank",
        "Ensemble ID",
        "Members",
        "Number of Members",
        "Aggregation Rule",
        "Global OOF Log Loss",
        "Delta Log Loss vs P1",
        "Relative Log Loss Improvement vs P1 (%)",
        "Global OOF AUROC",
        "Global OOF Brier Score",
        "Global OOF ECE",
    ]

    print(top.head(20)[display_cols].to_string(index=False))
    print()
    print(f"P1 Log Loss                 : {reference_ll:.6f}")
    print(
        f"Ensembles beating P1        : {n_better}/{len(results)}"
    )
    print(
        f"Best ensemble               : {best_row['Ensemble ID']} "
        f"({best_row['Members']}, {best_row['Aggregation Rule']})"
    )
    print(
        f"Best ensemble Log Loss      : "
        f"{best_row['Global OOF Log Loss']:.6f}"
    )
    print(
        f"Delta Log Loss vs P1        : "
        f"{best_row['Delta Log Loss vs P1']:+.6f}"
    )
    print(
        f"Relative improvement vs P1  : "
        f"{best_row['Relative Log Loss Improvement vs P1 (%)']:.3f}%"
    )
    print()
    print(f"Saved: {all_file}")
    print(f"Saved: {top_file}")
    print(f"Saved: {best_file}")
    print(f"Saved: {size_rule_file}")
    print(f"Saved: {size_file}")
    print(f"Saved: {manifest_file}")
    print(f"Saved: {run_file}")
    print(f"Saved: {report_file}")
    print()
    print("STATUS: PASS")
    print(
        "Next step: cross-fitted calibration of the strongest ensemble "
        "candidates, followed by paired ΔLogLoss comparison with P1."
    )


if __name__ == "__main__":
    main()
