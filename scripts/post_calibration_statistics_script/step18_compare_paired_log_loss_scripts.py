#!/usr/bin/env python3
"""
Phase 9 — Paired subject-level Log-Loss statistical comparison.

Reference
---------
P1 is the fixed reference candidate.

For each candidate P2–P8 and each of the 1,362 aligned OOF subjects:

    subject_delta = subject_log_loss(candidate) - subject_log_loss(P1)

Interpretation
--------------
    delta < 0  -> candidate better
    delta > 0  -> P1 better

The mean subject-level delta is mathematically identical to:

    global_log_loss(candidate) - global_log_loss(P1)

Inference
---------
Paired stratified bootstrap over subjects:
- resample negative subjects with replacement
- resample positive subjects with replacement
- preserve original class counts
- retain pairing because the same sampled subject indices are used for P1
  and every comparator
- 10,000 bootstrap replicates by default
- fixed seed 2026 by default

Outputs include:
- nominal 95% percentile CI
- familywise Bonferroni simultaneous CI across the seven P1-vs-P2...P8
  comparisons

No neural network is retrained and no calibrator is refitted.
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
        description="Paired subject-level Log-Loss comparison of P2–P8 versus P1."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--shortlist", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference-id", type=str, default="P1")
    parser.add_argument("--expected-candidates", type=int, default=8)
    parser.add_argument("--expected-subjects", type=int, default=1362)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--global-match-tolerance",
        type=float,
        default=2e-6,
        help="Tolerance when checking recomputed global Log Loss against shortlist.",
    )
    return parser.parse_args()


def binary_subject_log_loss(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)

    if len(y) != len(p):
        raise ValueError("Label/probability length mismatch.")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Labels must contain only 0/1.")
    if not np.isfinite(p).all():
        raise ValueError("Probabilities contain NaN/Inf.")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("Probabilities must be in [0,1].")

    p = np.clip(p, EPSILON, 1.0 - EPSILON)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


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
        raise ValueError(f"{path}: duplicate UID values detected")

    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(np.int64)
    df["is_pathologic"] = pd.to_numeric(
        df["is_pathologic"], errors="raise"
    ).astype(np.int64)
    df["calibrated_probability"] = pd.to_numeric(
        df["calibrated_probability"], errors="raise"
    ).astype(np.float64)

    expected_fold_ids = set(range(expected_folds))
    actual_fold_ids = set(int(v) for v in np.unique(df["fold"]))
    if actual_fold_ids != expected_fold_ids:
        raise ValueError(
            f"{path}: expected fold IDs {sorted(expected_fold_ids)}, "
            f"found {sorted(actual_fold_ids)}"
        )

    if not np.isin(df["is_pathologic"].to_numpy(), [0, 1]).all():
        raise ValueError(f"{path}: labels must contain only 0/1")

    p = df["calibrated_probability"].to_numpy()
    if not np.isfinite(p).all():
        raise ValueError(f"{path}: probabilities contain NaN/Inf")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(f"{path}: probabilities are outside [0,1]")

    return df.sort_values("uid", kind="stable").reset_index(drop=True)


def shortlist_sort_key(value: str) -> int:
    value = str(value).strip()
    if value.startswith("P") and value[1:].isdigit():
        return int(value[1:])
    return 10_000


def ci_conclusion(low: float, high: float) -> str:
    if high < 0.0:
        return "CANDIDATE BETTER"
    if low > 0.0:
        return "P1 BETTER"
    return "INCONCLUSIVE"


def bootstrap_mean_deltas_stratified(
    delta_matrix: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    chunk_size: int = 250,
) -> np.ndarray:
    """
    delta_matrix shape: [n_subjects, n_comparators]
    Returns shape: [n_bootstrap, n_comparators]

    Class-stratified resampling preserves the observed class counts.
    """
    delta_matrix = np.asarray(delta_matrix, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    neg_idx = np.flatnonzero(y == 0)
    pos_idx = np.flatnonzero(y == 1)

    if len(neg_idx) == 0 or len(pos_idx) == 0:
        raise ValueError("Both classes are required for stratified bootstrap.")

    n_total = len(y)
    rng = np.random.default_rng(seed)
    out = np.empty((n_bootstrap, delta_matrix.shape[1]), dtype=np.float64)

    cursor = 0
    while cursor < n_bootstrap:
        b = min(chunk_size, n_bootstrap - cursor)

        neg_sample = rng.choice(neg_idx, size=(b, len(neg_idx)), replace=True)
        pos_sample = rng.choice(pos_idx, size=(b, len(pos_idx)), replace=True)

        neg_mean = delta_matrix[neg_sample].mean(axis=1)
        pos_mean = delta_matrix[pos_sample].mean(axis=1)

        out[cursor:cursor + b] = (
            (len(neg_idx) * neg_mean + len(pos_idx) * pos_mean) / n_total
        )
        cursor += b

    return out


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

    output_dir = (
        args.output_dir
        or project_root
        / "data"
        / "post_calibration_statistics_data"
        / "paired_log_loss"
    ).expanduser().resolve()

    print("\n" + "=" * 100)
    print("PHASE 9 — PAIRED SUBJECT-LEVEL LOG-LOSS STATISTICAL COMPARISON")
    print("=" * 100)
    print(f"Project root          : {project_root}")
    print(f"Shortlist             : {shortlist_path}")
    print(f"Output                : {output_dir}")
    print(f"Reference             : {args.reference_id}")
    print(f"Expected candidates   : {args.expected_candidates}")
    print(f"Expected OOF subjects : {args.expected_subjects}")
    print(f"Expected folds        : {args.expected_folds}")
    print(f"Bootstrap replicates  : {args.bootstrap_replicates:,}")
    print(f"Seed                  : {args.seed}")
    print(f"Nominal alpha         : {args.alpha}")
    print("Delta definition      : LL(candidate) - LL(P1)")
    print("Negative delta        : candidate better")
    print("Positive delta        : P1 better")
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
    shortlist = shortlist.sort_values("_sort_key", kind="stable").drop(
        columns="_sort_key"
    ).reset_index(drop=True)

    if args.reference_id not in set(shortlist["Shortlist ID"].astype(str)):
        raise RuntimeError(
            f"Reference {args.reference_id} is not present in shortlist."
        )

    # ------------------------------------------------------------------
    # Load and independently validate all prediction files.
    # ------------------------------------------------------------------
    loaded: dict[str, pd.DataFrame] = {}
    stored_global: dict[str, float] = {}
    recomputed_global: dict[str, float] = {}
    metadata: dict[str, dict] = {}

    canonical_uid = None
    canonical_y = None
    canonical_fold = None

    for i, row in shortlist.iterrows():
        sid = str(row["Shortlist ID"])
        path = Path(str(row["Cross-Fitted Prediction File"])).expanduser().resolve()

        print(
            f"[{i + 1:02d}/{len(shortlist)}] Loading {sid}: "
            f"{row['Model']}-{row['Scenario ID']} "
            f"calibration={row['Calibration Method']}"
        )

        df = validate_prediction_file(
            path,
            expected_subjects=args.expected_subjects,
            expected_folds=args.expected_folds,
        )

        uid = df["uid"].to_numpy(dtype=str)
        y = df["is_pathologic"].to_numpy(dtype=np.int64)
        fold = df["fold"].to_numpy(dtype=np.int64)

        if canonical_uid is None:
            canonical_uid = uid
            canonical_y = y
            canonical_fold = fold
        else:
            if not np.array_equal(uid, canonical_uid):
                raise RuntimeError(
                    f"{sid}: UID alignment differs from {args.reference_id}/canonical data."
                )
            if not np.array_equal(y, canonical_y):
                raise RuntimeError(f"{sid}: labels differ from canonical data.")
            if not np.array_equal(fold, canonical_fold):
                raise RuntimeError(f"{sid}: fold assignments differ from canonical data.")

        p = df["calibrated_probability"].to_numpy(dtype=np.float64)
        subject_loss = binary_subject_log_loss(y, p)
        recomputed_ll = float(subject_loss.mean())
        stored = float(row["Cross-Fitted OOF Log Loss"])
        diff = abs(recomputed_ll - stored)

        if diff > args.global_match_tolerance:
            raise RuntimeError(
                f"{sid}: recomputed Log Loss {recomputed_ll:.9f} "
                f"does not match shortlist {stored:.9f}; diff={diff:.9g}"
            )

        df = df.copy()
        df["subject_log_loss"] = subject_loss
        loaded[sid] = df
        stored_global[sid] = stored
        recomputed_global[sid] = recomputed_ll
        metadata[sid] = {
            "Calibration Candidate ID": str(row["Calibration Candidate ID"]),
            "Model": str(row["Model"]),
            "Scenario ID": str(row["Scenario ID"]),
            "Calibration Method": str(row["Calibration Method"]),
            "Cross-Fitted Prediction File": str(path),
        }

    assert canonical_uid is not None
    assert canonical_y is not None
    assert canonical_fold is not None

    reference_df = loaded[args.reference_id]
    reference_loss = reference_df["subject_log_loss"].to_numpy(dtype=np.float64)
    reference_global = float(reference_loss.mean())

    comparator_ids = [
        sid
        for sid in shortlist["Shortlist ID"].astype(str).tolist()
        if sid != args.reference_id
    ]
    n_comparisons = len(comparator_ids)

    if n_comparisons != args.expected_candidates - 1:
        raise RuntimeError("Unexpected number of comparator candidates.")

    # ------------------------------------------------------------------
    # Build subject-level paired-difference table.
    # ------------------------------------------------------------------
    subject_table = pd.DataFrame(
        {
            "uid": canonical_uid,
            "fold": canonical_fold,
            "is_pathologic": canonical_y,
            f"{args.reference_id} Probability":
                reference_df["calibrated_probability"].to_numpy(dtype=np.float64),
            f"{args.reference_id} Subject Log Loss": reference_loss,
        }
    )

    delta_columns = []
    for sid in comparator_ids:
        candidate_df = loaded[sid]
        candidate_loss = candidate_df["subject_log_loss"].to_numpy(dtype=np.float64)
        delta = candidate_loss - reference_loss

        subject_table[f"{sid} Probability"] = candidate_df[
            "calibrated_probability"
        ].to_numpy(dtype=np.float64)
        subject_table[f"{sid} Subject Log Loss"] = candidate_loss

        delta_col = f"Delta Log Loss {sid} minus {args.reference_id}"
        subject_table[delta_col] = delta
        delta_columns.append(delta_col)

    delta_matrix = subject_table[delta_columns].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # Paired stratified bootstrap.
    # ------------------------------------------------------------------
    print("\nRunning paired stratified bootstrap...")
    bootstrap = bootstrap_mean_deltas_stratified(
        delta_matrix,
        canonical_y,
        n_bootstrap=args.bootstrap_replicates,
        seed=args.seed,
    )

    # Nominal 95% interval.
    nominal_low_q = args.alpha / 2.0
    nominal_high_q = 1.0 - args.alpha / 2.0

    # Bonferroni familywise simultaneous interval across 7 comparisons.
    familywise_alpha_each = args.alpha / n_comparisons
    family_low_q = familywise_alpha_each / 2.0
    family_high_q = 1.0 - familywise_alpha_each / 2.0
    family_confidence = 1.0 - familywise_alpha_each

    rows = []

    for j, sid in enumerate(comparator_ids):
        delta = delta_matrix[:, j]
        boot = bootstrap[:, j]

        observed_delta = float(delta.mean())

        # Mathematical identity check must use independently recomputed
        # probabilities, not the rounded/stored shortlist metric.
        expected_delta = float(recomputed_global[sid] - reference_global)
        identity_abs_diff = abs(observed_delta - expected_delta)

        if identity_abs_diff > 1e-12:
            raise RuntimeError(
                f"{sid}: subject-level mean delta does not equal the "
                f"recomputed global Log-Loss difference; "
                f"abs diff={identity_abs_diff:.3e}."
            )

        nominal_low, nominal_high = np.quantile(
            boot, [nominal_low_q, nominal_high_q]
        )
        family_low, family_high = np.quantile(
            boot, [family_low_q, family_high_q]
        )

        candidate_better_count = int(np.sum(delta < 0.0))
        p1_better_count = int(np.sum(delta > 0.0))
        ties = int(np.sum(delta == 0.0))

        row = {
            "Reference ID": args.reference_id,
            "Candidate ID": sid,
            "Calibration Candidate ID": metadata[sid]["Calibration Candidate ID"],
            "Model": metadata[sid]["Model"],
            "Scenario ID": metadata[sid]["Scenario ID"],
            "Calibration Method": metadata[sid]["Calibration Method"],
            "P1 Global Cross-Fitted OOF Log Loss": reference_global,
            "Candidate Global Cross-Fitted OOF Log Loss": recomputed_global[sid],
            "Stored Candidate Cross-Fitted OOF Log Loss": stored_global[sid],
            "Stored-vs-Recomputed Log Loss Abs Diff": abs(
                stored_global[sid] - recomputed_global[sid]
            ),
            "Observed Mean Delta Log Loss": observed_delta,
            "Median Subject Delta Log Loss": float(np.median(delta)),
            "Delta Log Loss SD Across Subjects": float(np.std(delta, ddof=1)),
            "Nominal CI Confidence": 1.0 - args.alpha,
            "Nominal CI Lower": float(nominal_low),
            "Nominal CI Upper": float(nominal_high),
            "Nominal CI Conclusion": ci_conclusion(
                float(nominal_low), float(nominal_high)
            ),
            "Familywise CI Confidence Per Comparison": family_confidence,
            "Familywise CI Lower": float(family_low),
            "Familywise CI Upper": float(family_high),
            "Familywise CI Conclusion": ci_conclusion(
                float(family_low), float(family_high)
            ),
            "Subjects Candidate Better Than P1": candidate_better_count,
            "Subjects P1 Better Than Candidate": p1_better_count,
            "Subjects Tied": ties,
            "Candidate Better Subject Fraction": candidate_better_count / len(delta),
            "P1 Better Subject Fraction": p1_better_count / len(delta),
            "Bootstrap Replicates": args.bootstrap_replicates,
            "Bootstrap Seed": args.seed,
            "Cross-Fitted Prediction File": metadata[sid][
                "Cross-Fitted Prediction File"
            ],
        }
        rows.append(row)

        print(
            f"    {sid} vs {args.reference_id}: "
            f"ΔLL={observed_delta:+.6f}  "
            f"95% CI=[{nominal_low:+.6f}, {nominal_high:+.6f}]  "
            f"familywise=[{family_low:+.6f}, {family_high:+.6f}]  "
            f"{row['Familywise CI Conclusion']}"
        )

    results = pd.DataFrame(rows)
    results = results.sort_values(
        "Observed Mean Delta Log Loss",
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Per-fold paired deltas as a descriptive diagnostic.
    # ------------------------------------------------------------------
    fold_rows = []
    for sid in comparator_ids:
        delta_col = f"Delta Log Loss {sid} minus {args.reference_id}"
        for fold_id in range(args.expected_folds):
            mask = subject_table["fold"].to_numpy() == fold_id
            fold_delta = subject_table.loc[mask, delta_col].to_numpy(dtype=np.float64)
            fold_rows.append(
                {
                    "Reference ID": args.reference_id,
                    "Candidate ID": sid,
                    "Fold": fold_id,
                    "Fold Subjects": int(mask.sum()),
                    "Mean Fold Delta Log Loss": float(fold_delta.mean()),
                    "Median Fold Delta Log Loss": float(np.median(fold_delta)),
                    "Candidate Better Subjects": int(np.sum(fold_delta < 0.0)),
                    "P1 Better Subjects": int(np.sum(fold_delta > 0.0)),
                    "Tied Subjects": int(np.sum(fold_delta == 0.0)),
                }
            )
    fold_results = pd.DataFrame(fold_rows)

    # ------------------------------------------------------------------
    # Save outputs.
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_file = output_dir / "p1_p8_subject_level_log_loss_differences.csv"
    result_file = output_dir / "p1_vs_p2_p8_paired_bootstrap_results.csv"
    fold_file = output_dir / "p1_vs_p2_p8_fold_delta_log_loss.csv"
    bootstrap_file = output_dir / "p1_vs_p2_p8_bootstrap_distributions.npz"
    summary_file = output_dir / "p1_vs_p2_p8_paired_statistics_run_summary.json"
    report_file = output_dir / "p1_vs_p2_p8_paired_statistics_report.md"

    subject_table.to_csv(subject_file, index=False, float_format="%.9f")
    results.to_csv(result_file, index=False, float_format="%.9f")
    fold_results.to_csv(fold_file, index=False, float_format="%.9f")

    np.savez_compressed(
        bootstrap_file,
        candidate_ids=np.asarray(comparator_ids, dtype=str),
        bootstrap_mean_delta_log_loss=bootstrap,
    )

    familywise_supported_better = results.loc[
        results["Familywise CI Conclusion"] == "CANDIDATE BETTER",
        "Candidate ID",
    ].tolist()
    familywise_supported_p1_better = results.loc[
        results["Familywise CI Conclusion"] == "P1 BETTER",
        "Candidate ID",
    ].tolist()
    familywise_inconclusive = results.loc[
        results["Familywise CI Conclusion"] == "INCONCLUSIVE",
        "Candidate ID",
    ].tolist()

    summary = {
        "status": "PASS",
        "reference_id": args.reference_id,
        "reference_global_cross_fitted_oof_log_loss": reference_global,
        "subjects": int(args.expected_subjects),
        "comparisons": int(n_comparisons),
        "bootstrap": {
            "method": "paired class-stratified subject bootstrap",
            "replicates": int(args.bootstrap_replicates),
            "seed": int(args.seed),
            "nominal_confidence": float(1.0 - args.alpha),
            "familywise_method": "Bonferroni simultaneous percentile intervals",
            "familywise_alpha": float(args.alpha),
            "per_comparison_alpha": float(familywise_alpha_each),
            "per_comparison_confidence": float(family_confidence),
        },
        "delta_definition": "candidate subject Log Loss - P1 subject Log Loss",
        "interpretation": {
            "negative": "candidate better",
            "positive": "P1 better",
        },
        "familywise_supported_candidate_better_than_p1": familywise_supported_better,
        "familywise_supported_p1_better_than_candidate": familywise_supported_p1_better,
        "familywise_inconclusive": familywise_inconclusive,
        "outputs": {
            "subject_level_differences": str(subject_file),
            "paired_bootstrap_results": str(result_file),
            "fold_delta_log_loss": str(fold_file),
            "bootstrap_distributions": str(bootstrap_file),
            "run_summary": str(summary_file),
            "markdown_report": str(report_file),
        },
    }

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_columns = [
        "Candidate ID",
        "Model",
        "Scenario ID",
        "Calibration Method",
        "Candidate Global Cross-Fitted OOF Log Loss",
        "Observed Mean Delta Log Loss",
        "Nominal CI Lower",
        "Nominal CI Upper",
        "Nominal CI Conclusion",
        "Familywise CI Lower",
        "Familywise CI Upper",
        "Familywise CI Conclusion",
        "Candidate Better Subject Fraction",
    ]

    md = [
        "# Phase 9 — Paired Subject-Level Log-Loss Statistical Comparison",
        "",
        f"Reference: **{args.reference_id}**",
        "",
        "`ΔLogLoss = candidate − P1`; negative values favor the candidate, positive values favor P1.",
        "",
        f"Bootstrap: {args.bootstrap_replicates:,} paired class-stratified replicates, seed {args.seed}.",
        "",
        f"Nominal CI: {(1.0-args.alpha)*100:.1f}%.",
        "",
        (
            "Familywise CI: Bonferroni simultaneous intervals across "
            f"{n_comparisons} comparisons "
            f"({family_confidence*100:.4f}% per-comparison interval)."
        ),
        "",
        "## Results",
        "",
        "| " + " | ".join(report_columns) + " |",
        "| " + " | ".join(["---"] * len(report_columns)) + " |",
    ]

    for _, r in results[report_columns].iterrows():
        cells = []
        for col in report_columns:
            val = r[col]
            if isinstance(val, (float, np.floating)):
                cells.append(f"{float(val):.6f}")
            else:
                cells.append(str(val))
        md.append("| " + " | ".join(cells) + " |")

    md.extend(
        [
            "",
            "## Familywise interpretation",
            "",
            (
                "- Candidate(s) statistically supported as better than P1: "
                + (", ".join(familywise_supported_better) if familywise_supported_better else "None")
            ),
            (
                "- P1 statistically supported as better than candidate(s): "
                + (", ".join(familywise_supported_p1_better) if familywise_supported_p1_better else "None")
            ),
            (
                "- Inconclusive after familywise correction: "
                + (", ".join(familywise_inconclusive) if familywise_inconclusive else "None")
            ),
            "",
            "This phase does not automatically eliminate candidates from ensemble testing. "
            "A candidate can be individually worse than P1 yet still add complementary information in an ensemble.",
            "",
        ]
    )
    report_file.write_text("\n".join(md), encoding="utf-8")

    display_cols = [
        "Candidate ID",
        "Model",
        "Scenario ID",
        "Calibration Method",
        "Candidate Global Cross-Fitted OOF Log Loss",
        "Observed Mean Delta Log Loss",
        "Nominal CI Lower",
        "Nominal CI Upper",
        "Nominal CI Conclusion",
        "Familywise CI Lower",
        "Familywise CI Upper",
        "Familywise CI Conclusion",
    ]

    print("\n" + "=" * 100)
    print("PAIRED LOG-LOSS COMPARISON SUMMARY")
    print("=" * 100)
    print(results[display_cols].to_string(index=False))
    print()
    print(
        "Familywise supported candidate better than P1 : "
        + (
            ", ".join(familywise_supported_better)
            if familywise_supported_better
            else "None"
        )
    )
    print(
        "Familywise supported P1 better than candidate : "
        + (
            ", ".join(familywise_supported_p1_better)
            if familywise_supported_p1_better
            else "None"
        )
    )
    print(
        "Familywise inconclusive                     : "
        + (
            ", ".join(familywise_inconclusive)
            if familywise_inconclusive
            else "None"
        )
    )
    print()
    print(f"Saved: {subject_file}")
    print(f"Saved: {result_file}")
    print(f"Saved: {fold_file}")
    print(f"Saved: {bootstrap_file}")
    print(f"Saved: {summary_file}")
    print(f"Saved: {report_file}")
    print()
    print("STATUS: PASS")
    print("Next step: competition-oriented ensemble analysis using P1–P8.")


if __name__ == "__main__":
    main()
