#!/usr/bin/env python3
"""
Phase 8 — Log-Loss Stability Analysis for the post-calibration P1–P8 shortlist.

Inputs
------
competition_shortlist_8.csv

For each shortlisted candidate, this script reads the authoritative
Cross-Fitted Prediction File produced during Phase 7 and computes:
    - Log Loss independently for each of the 5 OOF folds
    - unweighted mean fold Log Loss
    - sample SD of fold Log Loss
    - minimum / maximum fold Log Loss
    - fold-to-fold range
    - coefficient of variation (SD / mean)
    - best and worst fold
    - relative instability flag

Important
---------
The primary competition estimate remains the GLOBAL cross-fitted OOF Log Loss
over all 1,362 subjects. The mean of five fold Log Loss values is a stability
diagnostic and can differ slightly from the global Log Loss if fold sizes are
not exactly equal.

The script does NOT re-rank the competition shortlist by mean fold Log Loss and
does NOT eliminate candidates. It identifies relatively unstable candidates
for review before paired statistical comparison.

Relative instability is flagged using the standard Tukey upper-outlier rule
across the P1–P8 shortlist:
    threshold = Q3 + 1.5 * IQR

A candidate is marked REVIEW if either its fold Log-Loss SD or fold range is
above the corresponding shortlist-wide threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EPSILON = 1e-15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze five-fold Log-Loss stability for P1–P8."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--shortlist", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-candidates", type=int, default=8)
    parser.add_argument("--expected-subjects", type=int, default=1362)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument(
        "--global-match-tolerance",
        type=float,
        default=2e-6,
        help="Allowed difference between recomputed and stored global Log Loss.",
    )
    return parser.parse_args()


def binary_log_loss(y_true: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)

    if len(y) != len(p):
        raise ValueError("Label/probability length mismatch.")
    if len(y) == 0:
        raise ValueError("Cannot calculate Log Loss for an empty array.")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Labels must contain only 0/1.")
    if not np.isfinite(p).all():
        raise ValueError("Probabilities contain NaN/Inf.")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("Probabilities must be in [0,1].")

    p = np.clip(p, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


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
        "raw_probability",
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

    if not np.isin(df["is_pathologic"].to_numpy(), [0, 1]).all():
        raise ValueError(f"{path}: labels must contain only 0/1")

    p = df["calibrated_probability"].to_numpy()
    if not np.isfinite(p).all():
        raise ValueError(f"{path}: calibrated probabilities contain NaN/Inf")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(f"{path}: calibrated probabilities must be in [0,1]")

    expected_fold_ids = set(range(expected_folds))
    actual_fold_ids = set(int(v) for v in np.unique(df["fold"]))
    if actual_fold_ids != expected_fold_ids:
        raise ValueError(
            f"{path}: expected fold IDs {sorted(expected_fold_ids)}, "
            f"found {sorted(actual_fold_ids)}"
        )

    for fold_id in range(expected_folds):
        fold_df = df.loc[df["fold"] == fold_id]
        if fold_df.empty:
            raise ValueError(f"{path}: fold {fold_id} is empty")
        if fold_df["is_pathologic"].nunique() != 2:
            raise ValueError(f"{path}: fold {fold_id} does not contain both classes")

    return df


def tukey_upper_fence(values: pd.Series) -> tuple[float, float, float, float]:
    values = pd.to_numeric(values, errors="raise").astype(float)
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    return q1, q3, iqr, upper


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
        / "post_calibration_stability_data"
        / "log_loss_stability"
    ).expanduser().resolve()

    print("\n" + "=" * 96)
    print("PHASE 8 — P1–P8 LOG-LOSS STABILITY ANALYSIS")
    print("=" * 96)
    print(f"Project root          : {project_root}")
    print(f"Shortlist             : {shortlist_path}")
    print(f"Output                : {output_dir}")
    print(f"Expected candidates   : {args.expected_candidates}")
    print(f"Expected OOF subjects : {args.expected_subjects}")
    print(f"Expected folds        : {args.expected_folds}")
    print("Primary diagnostic    : fold-to-fold Log-Loss stability")
    print("Probability column    : calibrated_probability")
    print()

    if not shortlist_path.is_file():
        print(f"ERROR: shortlist not found: {shortlist_path}")
        sys.exit(1)

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

    if shortlist["Shortlist ID"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate Shortlist ID values detected.")

    if shortlist["Calibration Candidate ID"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate Calibration Candidate ID values detected.")

    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict] = []
    summary_rows: list[dict] = []

    # This also checks that P1..P8 are in expected numeric order if possible.
    def shortlist_sort_key(value: str) -> int:
        value = str(value).strip()
        if value.startswith("P") and value[1:].isdigit():
            return int(value[1:])
        return 10_000

    shortlist = shortlist.copy()
    shortlist["_sort_key"] = shortlist["Shortlist ID"].map(shortlist_sort_key)
    shortlist = shortlist.sort_values(
        ["_sort_key", "Cross-Fitted OOF Log Loss"],
        ascending=[True, True],
        kind="stable",
    ).drop(columns="_sort_key").reset_index(drop=True)

    for i, row in shortlist.iterrows():
        sid = str(row["Shortlist ID"])
        cid = str(row["Calibration Candidate ID"])
        model = str(row["Model"])
        scenario = str(row["Scenario ID"])
        method = str(row["Calibration Method"])
        stored_global = float(row["Cross-Fitted OOF Log Loss"])
        prediction_path = Path(
            str(row["Cross-Fitted Prediction File"])
        ).expanduser().resolve()

        print(
            f"[{i + 1:02d}/{len(shortlist)}] "
            f"{sid} {cid} {model}-{scenario} calibration={method}"
        )

        df = validate_prediction_file(
            prediction_path,
            expected_subjects=args.expected_subjects,
            expected_folds=args.expected_folds,
        )

        y_all = df["is_pathologic"].to_numpy(dtype=np.int64)
        p_all = df["calibrated_probability"].to_numpy(dtype=np.float64)
        recomputed_global = binary_log_loss(y_all, p_all)
        global_abs_diff = abs(recomputed_global - stored_global)

        if global_abs_diff > args.global_match_tolerance:
            raise RuntimeError(
                f"{sid}: recomputed global Log Loss ({recomputed_global:.9f}) "
                f"does not match shortlist value ({stored_global:.9f}); "
                f"absolute difference={global_abs_diff:.9g}. "
                "This usually means the wrong prediction file/column is being analyzed."
            )

        candidate_fold_losses: list[float] = []
        candidate_fold_sizes: list[int] = []

        for fold_id in range(args.expected_folds):
            fold_df = df.loc[df["fold"] == fold_id].copy()
            y = fold_df["is_pathologic"].to_numpy(dtype=np.int64)
            p = fold_df["calibrated_probability"].to_numpy(dtype=np.float64)

            ll = binary_log_loss(y, p)
            candidate_fold_losses.append(ll)
            candidate_fold_sizes.append(len(fold_df))

            positives = int(y.sum())
            negatives = int(len(y) - positives)

            fold_rows.append(
                {
                    "Shortlist ID": sid,
                    "Calibration Candidate ID": cid,
                    "Model": model,
                    "Scenario ID": scenario,
                    "Calibration Method": method,
                    "Fold": int(fold_id),
                    "Fold Subjects": int(len(fold_df)),
                    "Fold Positives": positives,
                    "Fold Negatives": negatives,
                    "Fold Log Loss": ll,
                    "Cross-Fitted Prediction File": str(prediction_path),
                }
            )

            print(
                f"    fold {fold_id}: n={len(fold_df):3d} "
                f"LogLoss={ll:.6f}"
            )

        losses = np.asarray(candidate_fold_losses, dtype=np.float64)
        sizes = np.asarray(candidate_fold_sizes, dtype=np.int64)

        mean_ll = float(np.mean(losses))
        sd_ll = float(np.std(losses, ddof=1))
        min_ll = float(np.min(losses))
        max_ll = float(np.max(losses))
        range_ll = float(max_ll - min_ll)
        cv_ll = float(sd_ll / mean_ll) if mean_ll > 0 else np.nan
        weighted_fold_ll = float(np.average(losses, weights=sizes))
        best_fold = int(np.argmin(losses))
        worst_fold = int(np.argmax(losses))

        # The weighted average of per-fold losses must equal the global loss
        # because each subject contributes exactly once.
        weighted_abs_diff = abs(weighted_fold_ll - recomputed_global)
        if weighted_abs_diff > 1e-12:
            raise RuntimeError(
                f"{sid}: weighted fold Log Loss does not reproduce global Log Loss."
            )

        summary_rows.append(
            {
                "Shortlist ID": sid,
                "Calibration Candidate ID": cid,
                "Model": model,
                "Scenario ID": scenario,
                "Calibration Method": method,
                "Global Cross-Fitted OOF Log Loss": recomputed_global,
                "Stored Cross-Fitted OOF Log Loss": stored_global,
                "Global Log Loss Match Abs Diff": global_abs_diff,
                "Mean Fold Log Loss": mean_ll,
                "Fold Log Loss SD": sd_ll,
                "Minimum Fold Log Loss": min_ll,
                "Maximum Fold Log Loss": max_ll,
                "Fold Log Loss Range": range_ll,
                "Fold Log Loss CV": cv_ll,
                "Best Fold": best_fold,
                "Worst Fold": worst_fold,
                "Worst Fold Excess vs Global": float(max_ll - recomputed_global),
                "OOF Subjects": int(len(df)),
                "Cross-Fitted Prediction File": str(prediction_path),
            }
        )

    fold_results = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)

    # Relative instability thresholds among the P1–P8 shortlist.
    sd_q1, sd_q3, sd_iqr, sd_upper = tukey_upper_fence(
        summary["Fold Log Loss SD"]
    )
    range_q1, range_q3, range_iqr, range_upper = tukey_upper_fence(
        summary["Fold Log Loss Range"]
    )

    summary["SD Relative Outlier"] = (
        summary["Fold Log Loss SD"] > sd_upper
    )
    summary["Range Relative Outlier"] = (
        summary["Fold Log Loss Range"] > range_upper
    )
    summary["Stability Flag"] = np.where(
        summary["SD Relative Outlier"] | summary["Range Relative Outlier"],
        "REVIEW",
        "OK",
    )
    summary["Stability Reason"] = summary.apply(
        lambda r: (
            "SD and range are relative outliers"
            if bool(r["SD Relative Outlier"]) and bool(r["Range Relative Outlier"])
            else "Fold Log-Loss SD is a relative outlier"
            if bool(r["SD Relative Outlier"])
            else "Fold Log-Loss range is a relative outlier"
            if bool(r["Range Relative Outlier"])
            else "No relative fold-stability outlier detected"
        ),
        axis=1,
    )

    summary["Stability Rank by SD"] = (
        summary["Fold Log Loss SD"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    # Keep shortlist order as the main presentation order.
    summary["_sort_key"] = summary["Shortlist ID"].map(shortlist_sort_key)
    summary = summary.sort_values("_sort_key").drop(columns="_sort_key").reset_index(
        drop=True
    )

    fold_results["_sort_key"] = fold_results["Shortlist ID"].map(shortlist_sort_key)
    fold_results = fold_results.sort_values(
        ["_sort_key", "Fold"]
    ).drop(columns="_sort_key").reset_index(drop=True)

    fold_file = output_dir / "p1_p8_fold_log_loss.csv"
    summary_file = output_dir / "p1_p8_log_loss_stability_summary.csv"
    json_file = output_dir / "p1_p8_log_loss_stability_run_summary.json"
    report_file = output_dir / "p1_p8_log_loss_stability_report.md"

    fold_results.to_csv(fold_file, index=False, float_format="%.9f")
    summary.to_csv(summary_file, index=False, float_format="%.9f")

    run_summary = {
        "status": "PASS",
        "shortlist_candidates": int(len(summary)),
        "subjects_per_candidate": int(args.expected_subjects),
        "folds": int(args.expected_folds),
        "primary_competition_metric": "Global Cross-Fitted OOF Log Loss",
        "stability_diagnostics": [
            "Mean Fold Log Loss",
            "Fold Log Loss SD",
            "Minimum Fold Log Loss",
            "Maximum Fold Log Loss",
            "Fold Log Loss Range",
            "Fold Log Loss CV",
            "Best Fold",
            "Worst Fold",
        ],
        "instability_rule": (
            "REVIEW if fold Log-Loss SD or fold Log-Loss range exceeds "
            "the P1-P8 Tukey upper fence Q3 + 1.5*IQR."
        ),
        "sd_tukey": {
            "q1": sd_q1,
            "q3": sd_q3,
            "iqr": sd_iqr,
            "upper_fence": sd_upper,
        },
        "range_tukey": {
            "q1": range_q1,
            "q3": range_q3,
            "iqr": range_iqr,
            "upper_fence": range_upper,
        },
        "review_candidates": summary.loc[
            summary["Stability Flag"] == "REVIEW", "Shortlist ID"
        ].tolist(),
        "outputs": {
            "fold_log_loss": str(fold_file),
            "stability_summary": str(summary_file),
            "run_summary": str(json_file),
            "markdown_report": str(report_file),
        },
    }
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    # Lightweight markdown report.
    report_columns = [
        "Shortlist ID",
        "Model",
        "Scenario ID",
        "Calibration Method",
        "Global Cross-Fitted OOF Log Loss",
        "Mean Fold Log Loss",
        "Fold Log Loss SD",
        "Minimum Fold Log Loss",
        "Maximum Fold Log Loss",
        "Fold Log Loss Range",
        "Worst Fold",
        "Stability Flag",
    ]
    report_view = summary[report_columns].copy()

    md_lines = [
        "# Phase 8 — P1–P8 Log-Loss Stability Report",
        "",
        "Primary competition score remains the global cross-fitted OOF Log Loss.",
        "Fold-level values are stability diagnostics.",
        "",
        "## Stability thresholds",
        "",
        f"- SD upper fence: `{sd_upper:.6f}`",
        f"- Range upper fence: `{range_upper:.6f}`",
        "- `REVIEW` means relative instability within P1–P8, not automatic exclusion.",
        "",
        "## Summary",
        "",
    ]

    # Avoid requiring tabulate.
    header = "| " + " | ".join(report_columns) + " |"
    sep = "| " + " | ".join(["---"] * len(report_columns)) + " |"
    md_lines.extend([header, sep])

    for _, r in report_view.iterrows():
        cells = []
        for col in report_columns:
            val = r[col]
            if isinstance(val, (float, np.floating)):
                cells.append(f"{float(val):.6f}")
            else:
                cells.append(str(val))
        md_lines.append("| " + " | ".join(cells) + " |")

    review = run_summary["review_candidates"]
    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Relative instability review candidates: "
                + (", ".join(review) if review else "None")
                + "."
            ),
            "",
            "Do not eliminate a candidate solely from this flag. "
            "Use Phase 9 paired subject-level Log-Loss statistics next.",
            "",
        ]
    )
    report_file.write_text("\n".join(md_lines), encoding="utf-8")

    display_cols = [
        "Shortlist ID",
        "Calibration Candidate ID",
        "Model",
        "Scenario ID",
        "Calibration Method",
        "Global Cross-Fitted OOF Log Loss",
        "Mean Fold Log Loss",
        "Fold Log Loss SD",
        "Minimum Fold Log Loss",
        "Maximum Fold Log Loss",
        "Fold Log Loss Range",
        "Worst Fold",
        "Stability Flag",
    ]

    print("\n" + "=" * 96)
    print("P1–P8 LOG-LOSS STABILITY SUMMARY")
    print("=" * 96)
    print(summary[display_cols].to_string(index=False))
    print()
    print(f"SD Tukey upper fence    : {sd_upper:.6f}")
    print(f"Range Tukey upper fence : {range_upper:.6f}")
    print(
        "Candidates for review   : "
        + (
            ", ".join(run_summary["review_candidates"])
            if run_summary["review_candidates"]
            else "None"
        )
    )
    print()
    print(f"Saved: {fold_file}")
    print(f"Saved: {summary_file}")
    print(f"Saved: {json_file}")
    print(f"Saved: {report_file}")
    print()
    print("STATUS: PASS")
    print(
        "Next step: paired subject-level ΔLogLoss comparison using P1 as reference."
    )


if __name__ == "__main__":
    main()
