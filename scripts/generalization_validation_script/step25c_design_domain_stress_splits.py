#!/usr/bin/env python3
"""
STEP 25C — ACQUISITION-DOMAIN STRESS SPLIT DESIGN

Goal
----
Design candidate acquisition-domain-aware validation splits for a fast
generalization stress test, using the Step-25B subject/domain table.

This step DOES NOT train any model and DOES NOT replace/freeze Step-11 folds.

Candidate designs
-----------------
A) 5-fold StratifiedGroupKFold using derived_spacing_signature
B) 3-fold StratifiedGroupKFold using derived_spacing_signature
C) Explicit major-spacing-family holdouts (top K spacing families by size)

Why spacing signature?
----------------------
Step 25B showed:
- strong mixing of spacing families across the original Step-11 folds,
- measurable acquisition-dependent performance variation,
- the largest spacing family contains 528/1362 subjects.

Because 528 subjects alone exceed the ideal size of one 5-fold split
(~272), a strict 5-fold group-aware split is intrinsically imbalanced.
A 3-fold design is therefore evaluated explicitly.

IMPORTANT
---------
The P1/ENS328 probabilities in the Step-25B table came from the ORIGINAL
Step-11 OOF process. Re-grouping those probabilities by candidate folds
does NOT create new domain-held-out OOF performance. This script therefore
uses them only for optional descriptive context and NEVER reports them as
candidate-CV performance.

Step 25D should retrain ONLY the selected E6-S5 champion architecture on
the recommended candidate folds to measure true domain-held-out performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


UID_COL = "uid"
LABEL_COL = "is_pathologic"
OLD_FOLD_COL = "fold"
DEFAULT_GROUP_COL = "derived_spacing_signature"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 25C: design acquisition-domain stress-validation splits."
    )
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--step25b-subject-table", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)

    p.add_argument("--group-column", default=DEFAULT_GROUP_COL)
    p.add_argument("--expected-subjects", type=int, default=1362)
    p.add_argument("--old-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--major-holdouts", type=int, default=5)

    # Recommendation thresholds.
    p.add_argument("--max-size-cv-for-preferred", type=float, default=0.25)
    p.add_argument("--max-positive-rate-range-for-preferred", type=float, default=0.10)

    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def conv(v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if np.isnan(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v

    path.write_text(json.dumps(conv(payload), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def validate_subject_table(
    path: Path,
    expected_subjects: int,
    old_folds: int,
    group_col: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Step-25B subject table not found: {path}")

    df = pd.read_csv(path)

    required = {UID_COL, LABEL_COL, OLD_FOLD_COL, group_col}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available: {list(df.columns)}"
        )

    if len(df) != expected_subjects:
        raise RuntimeError(
            f"Expected {expected_subjects} subjects, found {len(df)}."
        )

    if df[UID_COL].duplicated().any():
        raise RuntimeError("Duplicate UIDs detected.")

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="raise").astype(int)
    df[OLD_FOLD_COL] = pd.to_numeric(df[OLD_FOLD_COL], errors="raise").astype(int)

    if not np.isin(df[LABEL_COL].to_numpy(), [0, 1]).all():
        raise RuntimeError("Labels must be binary 0/1.")

    observed_old = set(df[OLD_FOLD_COL].unique())
    if observed_old != set(range(old_folds)):
        raise RuntimeError(
            f"Expected old fold IDs 0..{old_folds-1}, found {sorted(observed_old)}."
        )

    if df[group_col].isna().any():
        n = int(df[group_col].isna().sum())
        raise RuntimeError(
            f"Grouping column {group_col!r} contains {n} missing values."
        )

    df[group_col] = df[group_col].astype(str).str.strip()
    if (df[group_col] == "").any():
        raise RuntimeError(f"Grouping column {group_col!r} contains empty values.")

    return df.copy()


def build_sgkf_assignment(
    df: pd.DataFrame,
    group_col: str,
    n_splits: int,
    seed: int,
    scheme_name: str,
) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    assignment = np.full(len(df), -1, dtype=np.int64)

    for fold_id, (_, valid_idx) in enumerate(
        splitter.split(
            X=np.zeros((len(df), 1), dtype=np.float32),
            y=df[LABEL_COL].to_numpy(),
            groups=df[group_col].to_numpy(),
        )
    ):
        if np.any(assignment[valid_idx] != -1):
            raise RuntimeError(f"{scheme_name}: duplicate validation assignment.")
        assignment[valid_idx] = fold_id

    if np.any(assignment < 0):
        raise RuntimeError(f"{scheme_name}: unassigned subjects remain.")

    out = df[[UID_COL, LABEL_COL, OLD_FOLD_COL, group_col]].copy()
    out["candidate_fold"] = assignment
    out["scheme"] = scheme_name
    return out


def check_group_isolation(
    assignment: pd.DataFrame,
    group_col: str,
) -> tuple[int, int]:
    folds_per_group = assignment.groupby(group_col)["candidate_fold"].nunique()
    violating_groups = int((folds_per_group > 1).sum())

    if violating_groups:
        bad = set(folds_per_group[folds_per_group > 1].index.astype(str))
        violating_subjects = int(assignment[group_col].isin(bad).sum())
    else:
        violating_subjects = 0

    return violating_groups, violating_subjects


def summarize_scheme(
    assignment: pd.DataFrame,
    group_col: str,
    scheme_name: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    total = len(assignment)
    global_positive_rate = float(assignment[LABEL_COL].mean())

    rows = []
    domain_rows = []

    for fold_id, g in assignment.groupby("candidate_fold", sort=True):
        fold_groups = sorted(g[group_col].astype(str).unique().tolist())
        n = len(g)
        positives = int(g[LABEL_COL].sum())
        negatives = n - positives

        rows.append(
            {
                "scheme": scheme_name,
                "candidate_fold": int(fold_id),
                "subjects": int(n),
                "fraction_of_dataset": float(n / total),
                "negatives": int(negatives),
                "positives": int(positives),
                "positive_rate": float(positives / n),
                "positive_rate_minus_global": float((positives / n) - global_positive_rate),
                "domain_groups_in_validation": int(len(fold_groups)),
                "training_subjects": int(total - n),
            }
        )

        group_counts = g[group_col].value_counts()
        for group_name, count in group_counts.items():
            gg = g.loc[g[group_col] == group_name]
            domain_rows.append(
                {
                    "scheme": scheme_name,
                    "candidate_fold": int(fold_id),
                    "group_value": str(group_name),
                    "subjects": int(count),
                    "positive_rate": float(gg[LABEL_COL].mean()),
                }
            )

    fold_summary = pd.DataFrame(rows).sort_values("candidate_fold").reset_index(drop=True)
    domain_map = pd.DataFrame(domain_rows).sort_values(
        ["candidate_fold", "subjects", "group_value"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    fold_sizes = fold_summary["subjects"].to_numpy(dtype=float)
    positive_rates = fold_summary["positive_rate"].to_numpy(dtype=float)

    violating_groups, violating_subjects = check_group_isolation(
        assignment, group_col
    )

    size_cv = float(np.std(fold_sizes, ddof=0) / np.mean(fold_sizes))
    rate_range = float(positive_rates.max() - positive_rates.min())
    max_rate_deviation = float(np.max(np.abs(positive_rates - global_positive_rate)))

    summary = {
        "scheme": scheme_name,
        "folds": int(fold_summary["candidate_fold"].nunique()),
        "subjects": int(total),
        "group_column": group_col,
        "groups": int(assignment[group_col].nunique()),
        "global_positive_rate": global_positive_rate,
        "minimum_fold_size": int(fold_sizes.min()),
        "maximum_fold_size": int(fold_sizes.max()),
        "fold_size_cv": size_cv,
        "minimum_positive_rate": float(positive_rates.min()),
        "maximum_positive_rate": float(positive_rates.max()),
        "positive_rate_range": rate_range,
        "maximum_positive_rate_deviation_from_global": max_rate_deviation,
        "minimum_training_subjects": int(
            fold_summary["training_subjects"].min()
        ),
        "group_overlap_violating_groups": violating_groups,
        "group_overlap_violating_subjects": violating_subjects,
        "all_groups_isolated": bool(violating_groups == 0),
        "all_folds_have_both_classes": bool(
            ((fold_summary["positives"] > 0) & (fold_summary["negatives"] > 0)).all()
        ),
    }
    return summary, fold_summary, domain_map


def make_major_holdout_design(
    df: pd.DataFrame,
    group_col: str,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_counts = df[group_col].value_counts()
    major_groups = group_counts.head(top_k).index.tolist()

    long_rows = []
    summary_rows = []

    global_rate = float(df[LABEL_COL].mean())

    for rank, group_name in enumerate(major_groups, start=1):
        test_mask = df[group_col] == group_name
        test = df.loc[test_mask]
        train = df.loc[~test_mask]

        holdout_id = f"H{rank}"

        summary_rows.append(
            {
                "holdout_id": holdout_id,
                "held_out_group": str(group_name),
                "validation_subjects": int(len(test)),
                "validation_fraction": float(len(test) / len(df)),
                "validation_negatives": int((test[LABEL_COL] == 0).sum()),
                "validation_positives": int((test[LABEL_COL] == 1).sum()),
                "validation_positive_rate": float(test[LABEL_COL].mean()),
                "positive_rate_minus_global": float(test[LABEL_COL].mean() - global_rate),
                "training_subjects": int(len(train)),
                "training_positive_rate": float(train[LABEL_COL].mean()),
            }
        )

        for _, row in df[[UID_COL, LABEL_COL, OLD_FOLD_COL, group_col]].iterrows():
            long_rows.append(
                {
                    "holdout_id": holdout_id,
                    UID_COL: row[UID_COL],
                    LABEL_COL: int(row[LABEL_COL]),
                    OLD_FOLD_COL: int(row[OLD_FOLD_COL]),
                    group_col: str(row[group_col]),
                    "role": (
                        "validation"
                        if row[group_col] == group_name
                        else "training"
                    ),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


def pick_recommended_scheme(
    scheme_summaries: pd.DataFrame,
    max_size_cv: float,
    max_rate_range: float,
) -> tuple[str, str]:
    eligible = scheme_summaries.loc[
        scheme_summaries["all_groups_isolated"]
        & scheme_summaries["all_folds_have_both_classes"]
    ].copy()

    if eligible.empty:
        raise RuntimeError("No valid group-isolated candidate split scheme.")

    preferred = eligible.loc[
        (eligible["fold_size_cv"] <= max_size_cv)
        & (eligible["positive_rate_range"] <= max_rate_range)
    ].copy()

    # Lower is better. Size imbalance matters strongly because the purpose is
    # a practical, comparable stress-validation run. Class-rate imbalance is
    # also penalized. More folds get only a small reward, not enough to justify
    # a badly imbalanced 5-fold design.
    pool = preferred if not preferred.empty else eligible
    pool["selection_score"] = (
        pool["fold_size_cv"]
        + 2.0 * pool["positive_rate_range"]
        - 0.02 * pool["folds"]
    )

    best = pool.sort_values(
        ["selection_score", "fold_size_cv", "positive_rate_range"],
        ascending=[True, True, True],
        kind="stable",
    ).iloc[0]

    name = str(best["scheme"])

    if name == "spacing_sgkf3":
        reason = (
            "The 3-fold spacing-group split preserves complete spacing-family "
            "isolation while substantially reducing the fold-size and class-rate "
            "imbalance caused by the 528-subject dominant spacing family. "
            "It also requires only 3 E6-S5 training runs in Step 25D."
        )
    else:
        reason = (
            "Selected by the predefined balance score among valid group-isolated "
            "candidate designs."
        )

    return name, reason


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()

    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else script_path.parents[2]
    )

    subject_path = resolve(
        project_root,
        args.step25b_subject_table
        or Path(
            "data/generalization_validation_data/"
            "step25b_acquisition_domain_analysis/"
            "subject_acquisition_domain_table.csv"
        ),
    )

    output_dir = resolve(
        project_root,
        args.output_dir
        or Path(
            "data/generalization_validation_data/"
            "step25c_domain_stress_split_design"
        ),
    )

    print("=" * 100)
    print("STEP 25C — ACQUISITION-DOMAIN STRESS SPLIT DESIGN")
    print("=" * 100)
    print(f"Project root        : {project_root}")
    print(f"Step-25B subject    : {subject_path}")
    print(f"Grouping variable   : {args.group_column}")
    print(f"Output              : {output_dir}")
    print(f"Seed                : {args.seed}")
    print("Neural-net training : NO")
    print("Step-11 modification: NO")
    print("New folds frozen    : NO")
    print()

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"Output directory is non-empty: {output_dir}\n"
            "Use --overwrite to rebuild."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = validate_subject_table(
        subject_path,
        expected_subjects=args.expected_subjects,
        old_folds=args.old_folds,
        group_col=args.group_column,
    )

    print("Dataset:")
    print(f"  Subjects              : {len(df)}")
    print(f"  Normal                 : {(df[LABEL_COL] == 0).sum()}")
    print(f"  Pathologic             : {(df[LABEL_COL] == 1).sum()}")
    print(f"  Positive rate          : {df[LABEL_COL].mean():.4f}")
    print(f"  Unique spacing groups  : {df[args.group_column].nunique()}")
    print(f"  Largest spacing group  : {df[args.group_column].value_counts().iloc[0]}")
    print()

    # Candidate A: 5-fold spacing-group split.
    scheme5 = build_sgkf_assignment(
        df,
        args.group_column,
        n_splits=5,
        seed=args.seed,
        scheme_name="spacing_sgkf5",
    )
    s5, f5, d5 = summarize_scheme(
        scheme5, args.group_column, "spacing_sgkf5"
    )

    # Candidate B: 3-fold spacing-group split.
    scheme3 = build_sgkf_assignment(
        df,
        args.group_column,
        n_splits=3,
        seed=args.seed,
        scheme_name="spacing_sgkf3",
    )
    s3, f3, d3 = summarize_scheme(
        scheme3, args.group_column, "spacing_sgkf3"
    )

    scheme_summaries = pd.DataFrame([s5, s3])

    recommended_scheme, recommendation_reason = pick_recommended_scheme(
        scheme_summaries,
        max_size_cv=args.max_size_cv_for_preferred,
        max_rate_range=args.max_positive_rate_range_for_preferred,
    )

    recommended_assignment = (
        scheme3.copy()
        if recommended_scheme == "spacing_sgkf3"
        else scheme5.copy()
    )
    recommended_fold_summary = (
        f3.copy()
        if recommended_scheme == "spacing_sgkf3"
        else f5.copy()
    )
    recommended_domain_map = (
        d3.copy()
        if recommended_scheme == "spacing_sgkf3"
        else d5.copy()
    )

    # Explicit top-K major spacing holdouts for later targeted stress checks.
    major_summary, major_manifest = make_major_holdout_design(
        df,
        group_col=args.group_column,
        top_k=args.major_holdouts,
    )

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    scheme_summary_path = output_dir / "candidate_split_scheme_comparison.csv"
    s5_path = output_dir / "candidate_spacing_sgkf5_assignments.csv"
    s3_path = output_dir / "candidate_spacing_sgkf3_assignments.csv"
    f5_path = output_dir / "candidate_spacing_sgkf5_fold_summary.csv"
    f3_path = output_dir / "candidate_spacing_sgkf3_fold_summary.csv"
    d5_path = output_dir / "candidate_spacing_sgkf5_domain_map.csv"
    d3_path = output_dir / "candidate_spacing_sgkf3_domain_map.csv"

    rec_path = output_dir / "recommended_step25d_fold_assignments.csv"
    rec_fold_path = output_dir / "recommended_step25d_training_plan.csv"
    rec_domain_path = output_dir / "recommended_step25d_domain_map.csv"

    major_summary_path = output_dir / "major_spacing_family_holdout_summary.csv"
    major_manifest_path = output_dir / "major_spacing_family_holdout_manifest.csv"

    summary_json_path = output_dir / "step25c_summary.json"
    report_path = output_dir / "step25c_report.md"

    scheme_summaries.to_csv(
        scheme_summary_path, index=False, float_format="%.9f"
    )
    scheme5.to_csv(s5_path, index=False)
    scheme3.to_csv(s3_path, index=False)
    f5.to_csv(f5_path, index=False, float_format="%.9f")
    f3.to_csv(f3_path, index=False, float_format="%.9f")
    d5.to_csv(d5_path, index=False, float_format="%.9f")
    d3.to_csv(d3_path, index=False, float_format="%.9f")

    recommended_assignment.to_csv(rec_path, index=False)

    training_plan = recommended_fold_summary.copy()
    training_plan = training_plan.rename(
        columns={"candidate_fold": "validation_fold"}
    )
    training_plan["recommended_model"] = "E6-S5"
    training_plan["purpose"] = "domain_stress_validation"
    training_plan["train_on_other_candidate_folds"] = True
    training_plan.to_csv(
        rec_fold_path, index=False, float_format="%.9f"
    )

    recommended_domain_map.to_csv(
        rec_domain_path, index=False, float_format="%.9f"
    )

    major_summary.to_csv(
        major_summary_path, index=False, float_format="%.9f"
    )
    major_manifest.to_csv(major_manifest_path, index=False)

    # Hash the recommended assignment only for reproducibility.
    # This does NOT mean the split is frozen as an authoritative project CV.
    rec_sha = sha256_file(rec_path)

    selected_summary = (
        s3 if recommended_scheme == "spacing_sgkf3" else s5
    )

    payload = {
        "status": "PASS",
        "step": "25C",
        "analysis_type": "acquisition_domain_stress_split_design",
        "subjects": len(df),
        "group_column": args.group_column,
        "seed": args.seed,
        "candidate_schemes": [s5, s3],
        "recommended_scheme": recommended_scheme,
        "recommendation_reason": recommendation_reason,
        "recommended_scheme_summary": selected_summary,
        "recommended_assignment_file": str(rec_path),
        "recommended_assignment_sha256": rec_sha,
        "major_holdout_count": args.major_holdouts,
        "important_limits": [
            "Step 25C does not train any model.",
            "Step 25C does not replace or modify frozen Step-11 folds.",
            "The recommended folds are candidate stress-validation folds, not yet the project's final CV.",
            "Old P1/ENS328 OOF predictions cannot be treated as domain-held-out predictions under these new folds.",
            "True domain-held-out performance requires retraining E6-S5 in Step 25D.",
        ],
        "next_step": (
            f"Train only E6-S5 across the {int(selected_summary['folds'])} "
            "recommended candidate folds, producing one genuinely held-out "
            "prediction for every subject."
        ),
    }
    write_json(summary_json_path, payload)

    # Human-readable report.
    lines = [
        "# Step 25C — Acquisition-Domain Stress Split Design",
        "",
        "## Status",
        "",
        "**PASS — candidate split design completed. No model was trained.**",
        "",
        "## Purpose",
        "",
        (
            "Design a small number of acquisition-domain-aware validation "
            "splits for the next E6-S5 stress test without modifying the "
            "original Step-11 folds."
        ),
        "",
        "## Candidate schemes",
        "",
        scheme_summaries[
            [
                "scheme",
                "folds",
                "minimum_fold_size",
                "maximum_fold_size",
                "fold_size_cv",
                "minimum_positive_rate",
                "maximum_positive_rate",
                "positive_rate_range",
                "minimum_training_subjects",
                "all_groups_isolated",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Recommended Step-25D design",
        "",
        f"**{recommended_scheme}**",
        "",
        recommendation_reason,
        "",
        "Recommended fold composition:",
        "",
        recommended_fold_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Major spacing-family holdouts",
        "",
        (
            "These are saved as optional targeted diagnostics. They are not "
            "the primary recommended Step-25D CV because running all of them "
            "would require additional training runs."
        ),
        "",
        major_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Critical methodological warning",
        "",
        (
            "The P1/ENS328 predictions stored in Step 25B were generated using "
            "the original Step-11 folds. Re-grouping them into the new candidate "
            "folds would **not** produce valid domain-held-out OOF performance."
        ),
        "",
        (
            "Step 25D must actually retrain **only E6-S5** on the recommended "
            f"{int(selected_summary['folds'])}-fold candidate design."
        ),
        "",
        "## Reproducibility",
        "",
        f"- Recommended assignment SHA256: `{rec_sha}`",
        "- This hash is for reproducibility only; Step 25C does not freeze the split.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Console.
    # ------------------------------------------------------------------
    print("=" * 100)
    print("STEP 25C SUMMARY")
    print("=" * 100)
    print("Status                         : PASS")
    print(f"Grouping variable              : {args.group_column}")
    print(f"Candidate 5-fold size CV       : {s5['fold_size_cv']:.4f}")
    print(f"Candidate 5-fold class range   : {s5['positive_rate_range']:.4f}")
    print(f"Candidate 3-fold size CV       : {s3['fold_size_cv']:.4f}")
    print(f"Candidate 3-fold class range   : {s3['positive_rate_range']:.4f}")
    print(f"Recommended Step-25D scheme    : {recommended_scheme}")
    print(f"Recommended training runs      : {selected_summary['folds']}")
    print(f"Group isolation                : {selected_summary['all_groups_isolated']}")
    print(f"Assignment SHA256              : {rec_sha}")
    print(f"Saved                          : {output_dir}")
    print("=" * 100)
    print()
    print("NEXT:")
    print(
        f"  Step 25D should train ONLY E6-S5 across "
        f"{int(selected_summary['folds'])} candidate folds."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
