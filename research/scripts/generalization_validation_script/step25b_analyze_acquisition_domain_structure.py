#!/usr/bin/env python3
"""
STEP 25B — ACQUISITION / DOMAIN GENERALIZATION ANALYSIS

Purpose
-------
Diagnose whether the current subject-level StratifiedKFold validation mixes
acquisition/site/scanner/geometry domains across train and validation folds.

This step is ANALYSIS ONLY:
- no neural-network training
- no checkpoint modification
- no calibration fitting
- no ensemble search
- no new CV folds are frozen here

Primary inputs
--------------
1) Step-10 supervised manifest
2) Step-11 frozen fold assignments
3) Step-5B acquisition_cohort_per_scan.csv (when available)
4) Step-4 resampling_summary.json (used as a robust geometry fallback)
5) P1 cross-fitted OOF predictions
6) Step-20 winner cross-fitted OOF predictions (ENS328)

Main questions
--------------
A) Which acquisition/site/scanner/geometry group variables exist?
B) Are those groups spread across all existing CV folds?
C) Is pathology prevalence associated with acquisition/domain groups?
D) Are P1 / ENS328 OOF errors concentrated in particular domains?
E) Which grouping variable is the best candidate for Step 25C
   domain-aware fold construction?

Important
---------
A geometry/acquisition proxy is NOT automatically equivalent to hospital/site.
The report explicitly distinguishes true site metadata from proxy domains.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2_contingency
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Step 25B requires scipy (scipy.stats.chi2_contingency)."
    ) from exc

try:
    from sklearn.metrics import roc_auc_score
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Step 25B requires scikit-learn (sklearn.metrics.roc_auc_score)."
    ) from exc


# =============================================================================
# Constants
# =============================================================================

UID_CANDIDATES = [
    "uid",
    "UID",
    "subject_id",
    "subject",
    "file_name",
    "filename",
    "file",
]

LABEL_CANDIDATES = [
    "is_pathologic",
    "label",
    "target",
]

FOLD_CANDIDATES = [
    "fold",
    "Fold",
    "cv_fold",
]

TRUE_SITE_PATTERNS = (
    "site",
    "center",
    "centre",
    "hospital",
    "institution",
)

SCANNER_PATTERNS = (
    "scanner",
    "manufacturer",
    "model",
    "station",
    "protocol",
    "series",
    "camera",
)

PROXY_PRIORITY_NAMES = (
    "acquisition_cohort",
    "geometry_cohort",
    "geometry_signature",
    "spacing_family",
    "shape_family",
)

DIAGNOSTIC_ONLY_PATTERNS = (
    "scale_decade",
    "intensity",
    "reference_qc",
)


# =============================================================================
# Utility helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 25B: analyze acquisition/site/domain structure, "
            "current-fold domain overlap, label association, and "
            "domain-specific P1/ENS328 OOF performance."
        )
    )

    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--folds", type=Path, default=None)
    parser.add_argument("--acquisition-csv", type=Path, default=None)
    parser.add_argument("--resampling-summary", type=Path, default=None)
    parser.add_argument("--shortlist", type=Path, default=None)
    parser.add_argument("--step20-winner", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--expected-subjects", type=int, default=1362)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--min-domain-size", type=int, default=10)
    parser.add_argument("--top-domains", type=int, default=40)

    parser.add_argument(
        "--spacing-round-decimals",
        type=int,
        default=3,
        help="Decimals used for derived geometry spacing signature.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow reusing a non-empty output directory.",
    )

    return parser.parse_args()


def strip_nifti_suffix(value: Any) -> str:
    text = str(value).strip()
    name = Path(text).name
    low = name.lower()
    if low.endswith(".nii.gz"):
        return name[:-7]
    if low.endswith(".nii"):
        return name[:-4]
    return name


def resolve_input_path(project_root: Path, value: Path) -> Path:
    path = value.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def resolve_stored_path(project_root: Path, value: Any) -> Path:
    """
    Resolve a path stored in a CSV/JSON.

    Older artifacts may contain absolute paths from another machine.
    When that old absolute path does not exist, remap its /data/... suffix
    into the current project root.
    """
    raw = str(value).strip()
    path = Path(raw).expanduser()

    if path.is_file():
        return path.resolve()

    if not path.is_absolute():
        candidate = (project_root / path).resolve()
        if candidate.is_file():
            return candidate

    normalized = raw.replace("\\", "/")
    marker = "/data/"
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
        candidate = (project_root / "data" / rel).resolve()
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not resolve stored artifact path:\n"
        f"  stored : {raw}\n"
        f"  project: {project_root}"
    )


def first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise RuntimeError(
            f"None of the expected columns exist: {list(candidates)}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def prepare_uid_column(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    uid_col = first_existing_column(df, UID_CANDIDATES)
    df["uid"] = df[uid_col].map(strip_nifti_suffix)

    if df["uid"].isna().any() or (df["uid"].astype(str).str.len() == 0).any():
        raise RuntimeError(f"{source_name}: empty UID values detected.")

    if df["uid"].duplicated().any():
        duplicated = df.loc[df["uid"].duplicated(keep=False), "uid"].tolist()
        raise RuntimeError(
            f"{source_name}: duplicate UIDs detected, e.g. {duplicated[:10]}"
        )
    return df


def clean_group_values(series: pd.Series) -> pd.Series:
    out = series.copy()

    def normalize(v: Any) -> str | float:
        if pd.isna(v):
            return np.nan
        text = str(v).strip()
        if text == "" or text.lower() in {"nan", "none", "null", "unknown"}:
            return np.nan
        return text

    return out.map(normalize)


def binary_log_loss_per_subject(
    y: np.ndarray,
    p: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def global_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(binary_log_loss_per_subject(y, p)))


def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def cramers_v_from_table(table: pd.DataFrame) -> tuple[float, float]:
    """
    Bias-corrected Cramer's V + chi-square p-value.
    Returns (V, p).
    """
    obs = table.to_numpy(dtype=np.float64)
    if obs.size == 0 or obs.shape[0] < 2 or obs.shape[1] < 2:
        return float("nan"), float("nan")
    if np.any(obs.sum(axis=0) == 0) or np.any(obs.sum(axis=1) == 0):
        return float("nan"), float("nan")

    chi2, p_value, _, _ = chi2_contingency(obs, correction=False)
    n = float(obs.sum())
    if n <= 1:
        return float("nan"), float(p_value)

    r, k = obs.shape
    phi2 = chi2 / n
    phi2_corr = max(
        0.0,
        phi2 - ((k - 1) * (r - 1)) / (n - 1),
    )
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1.0, r_corr - 1.0)

    if denom <= 0:
        return 0.0, float(p_value)

    return float(math.sqrt(phi2_corr / denom)), float(p_value)


def normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    positive = counts[counts > 0]
    if total <= 0 or len(positive) <= 1:
        return 0.0

    probs = positive / total
    entropy = -float(np.sum(probs * np.log(probs)))
    max_entropy = math.log(len(counts))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, indent=2, sort_keys=True)


# =============================================================================
# Input loading
# =============================================================================

def load_manifest(path: Path, expected_subjects: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Step-10 manifest not found: {path}")

    df = pd.read_csv(path)
    df = prepare_uid_column(df, "Step-10 manifest")

    label_col = first_existing_column(df, LABEL_CANDIDATES)
    df["is_pathologic"] = pd.to_numeric(
        df[label_col], errors="raise"
    ).astype(np.int64)

    if not np.isin(df["is_pathologic"].to_numpy(), [0, 1]).all():
        raise RuntimeError("Step-10 labels must contain only 0/1.")

    if len(df) != expected_subjects:
        raise RuntimeError(
            f"Expected {expected_subjects} subjects in Step-10 manifest, "
            f"found {len(df)}."
        )

    return df[["uid", "is_pathologic"]].copy()


def load_folds(
    path: Path,
    expected_subjects: int,
    expected_folds: int,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Step-11 fold file not found: {path}")

    df = pd.read_csv(path)
    df = prepare_uid_column(df, "Step-11 folds")

    fold_col = first_existing_column(df, FOLD_CANDIDATES)
    df["fold"] = pd.to_numeric(df[fold_col], errors="raise").astype(np.int64)

    if len(df) != expected_subjects:
        raise RuntimeError(
            f"Expected {expected_subjects} rows in Step-11 folds, "
            f"found {len(df)}."
        )

    expected = set(range(expected_folds))
    observed = set(int(v) for v in df["fold"].unique())
    if observed != expected:
        raise RuntimeError(
            f"Expected fold IDs {sorted(expected)}, found {sorted(observed)}."
        )

    return df[["uid", "fold"]].copy()


def load_acquisition_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["uid"])

    df = pd.read_csv(path)
    df = prepare_uid_column(df, "Step-5B acquisition cohort CSV")

    # Prefix only columns that would collide with canonical fields.
    rename = {}
    for col in df.columns:
        if col == "uid":
            continue
        if col in {"is_pathologic", "fold"}:
            rename[col] = f"step5b_{col}"
    if rename:
        df = df.rename(columns=rename)

    return df


def _find_record_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        preferred = (
            "records",
            "results",
            "subjects",
            "files",
            "resampling_results",
            "images",
        )
        for key in preferred:
            value = payload.get(key)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value

        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value

    raise RuntimeError(
        "Could not identify the per-scan record list in resampling_summary.json."
    )


def load_resampling_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["uid"])

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = _find_record_list(payload)
    rows: list[dict[str, Any]] = []

    for rec in records:
        uid_value = None
        for key in UID_CANDIDATES:
            if key in rec:
                uid_value = rec[key]
                break
        if uid_value is None:
            continue

        row: dict[str, Any] = {"uid": strip_nifti_suffix(uid_value)}

        for field in (
            "original_shape",
            "resampled_shape",
            "original_spacing",
            "resampled_spacing",
        ):
            value = rec.get(field)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                prefix = f"rs_{field}"
                row[f"{prefix}_x"] = value[0]
                row[f"{prefix}_y"] = value[1]
                row[f"{prefix}_z"] = value[2]

        orientation = rec.get("original_orientation")
        if isinstance(orientation, (list, tuple)):
            row["rs_original_orientation"] = "".join(map(str, orientation))

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No per-scan rows could be extracted from resampling_summary.json."
        )

    if df["uid"].duplicated().any():
        raise RuntimeError("Duplicate UIDs in resampling summary.")

    return df


# =============================================================================
# Derived acquisition features
# =============================================================================

def derive_geometry_features(df: pd.DataFrame, spacing_decimals: int) -> pd.DataFrame:
    df = df.copy()

    # Prefer Step-4 resampling-summary original geometry because it is
    # consistently available for the full dataset.
    shape_cols = [
        "rs_original_shape_x",
        "rs_original_shape_y",
        "rs_original_shape_z",
    ]
    spacing_cols = [
        "rs_original_spacing_x",
        "rs_original_spacing_y",
        "rs_original_spacing_z",
    ]

    if all(col in df.columns for col in shape_cols):
        for col in shape_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        def shape_signature(row: pd.Series) -> str | float:
            vals = [row[c] for c in shape_cols]
            if not all(np.isfinite(vals)):
                return np.nan
            return "x".join(str(int(round(v))) for v in vals)

        df["derived_shape_signature"] = df.apply(shape_signature, axis=1)

    if all(col in df.columns for col in spacing_cols):
        for col in spacing_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        def spacing_signature(row: pd.Series) -> str | float:
            vals = [row[c] for c in spacing_cols]
            if not all(np.isfinite(vals)):
                return np.nan
            return "x".join(
                f"{float(v):.{spacing_decimals}f}" for v in vals
            )

        df["derived_spacing_signature"] = df.apply(
            spacing_signature, axis=1
        )

        df["derived_voxel_volume_mm3"] = (
            df[spacing_cols[0]]
            * df[spacing_cols[1]]
            * df[spacing_cols[2]]
        )

    if (
        all(col in df.columns for col in shape_cols)
        and all(col in df.columns for col in spacing_cols)
    ):
        for axis, shape_col, spacing_col in zip(
            ("x", "y", "z"), shape_cols, spacing_cols
        ):
            df[f"derived_fov_{axis}_mm"] = (
                df[shape_col] * df[spacing_col]
            )

        def geometry_signature(row: pd.Series) -> str | float:
            shape = row.get("derived_shape_signature")
            spacing = row.get("derived_spacing_signature")
            if pd.isna(shape) or pd.isna(spacing):
                return np.nan
            return f"shape={shape}|spacing={spacing}"

        df["derived_geometry_signature"] = df.apply(
            geometry_signature, axis=1
        )

    return df


def classify_group_column(column: str) -> tuple[str, int]:
    low = column.lower()

    if any(token in low for token in TRUE_SITE_PATTERNS):
        return "actual_site_or_center", 1

    if any(token in low for token in SCANNER_PATTERNS):
        return "scanner_or_protocol", 2

    if low in {"acquisition_cohort", "geometry_cohort"}:
        return "existing_acquisition_proxy", 3

    if low in {"derived_geometry_signature", "geometry_signature"}:
        return "derived_geometry_proxy", 4

    if low in {"spacing_family", "shape_family",
               "derived_spacing_signature", "derived_shape_signature"}:
        return "coarse_geometry_proxy", 5

    if any(token in low for token in DIAGNOSTIC_ONLY_PATTERNS):
        return "diagnostic_only", 9

    return "other_categorical", 8


def candidate_group_columns(df: pd.DataFrame) -> list[str]:
    explicit = {
        "acquisition_cohort",
        "geometry_cohort",
        "spacing_family",
        "shape_family",
        "scale_decade",
        "derived_geometry_signature",
        "derived_spacing_signature",
        "derived_shape_signature",
    }

    candidates: list[str] = []

    for col in df.columns:
        if col in {"uid", "is_pathologic", "fold"}:
            continue

        low = col.lower()

        relevant_name = (
            col in explicit
            or any(token in low for token in TRUE_SITE_PATTERNS)
            or any(token in low for token in SCANNER_PATTERNS)
            or "cohort" in low
            or "family" in low
        )
        if not relevant_name:
            continue

        series = clean_group_values(df[col])
        non_missing = series.notna().sum()
        n_unique = series.nunique(dropna=True)

        # Avoid unusable constants and effectively subject-ID-like columns.
        if non_missing == 0 or n_unique < 2:
            continue
        if n_unique > max(250, int(0.5 * len(df))):
            continue

        candidates.append(col)

    return sorted(
        set(candidates),
        key=lambda c: (classify_group_column(c)[1], c.lower()),
    )


# =============================================================================
# Group/domain diagnostics
# =============================================================================

def analyze_group_variable(
    df: pd.DataFrame,
    column: str,
    expected_folds: int,
    min_domain_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    work = df[["uid", "is_pathologic", "fold", column]].copy()
    work[column] = clean_group_values(work[column])
    work = work.loc[work[column].notna()].copy()

    coverage = len(work) / len(df) if len(df) else 0.0
    source_type, priority = classify_group_column(column)

    if work.empty:
        raise RuntimeError(f"No usable rows for grouping variable {column!r}.")

    group_rows: list[dict[str, Any]] = []

    for group_name, g in work.groupby(column, sort=False):
        fold_counts = (
            g["fold"]
            .value_counts()
            .reindex(range(expected_folds), fill_value=0)
            .sort_index()
        )
        label_counts = (
            g["is_pathologic"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        n = len(g)
        positives = int(label_counts.loc[1])
        negatives = int(label_counts.loc[0])
        folds_present = int((fold_counts > 0).sum())
        max_fold_share = float(fold_counts.max() / n)

        row = {
            "group_variable": column,
            "group_value": str(group_name),
            "subjects": int(n),
            "negatives": negatives,
            "positives": positives,
            "positive_rate": float(positives / n),
            "folds_present": folds_present,
            "present_in_all_folds": bool(folds_present == expected_folds),
            "max_fold_share": max_fold_share,
            "normalized_fold_entropy": normalized_entropy(
                fold_counts.to_numpy()
            ),
        }

        for fold_id in range(expected_folds):
            row[f"fold_{fold_id}_subjects"] = int(fold_counts.loc[fold_id])

        group_rows.append(row)

    details = pd.DataFrame(group_rows).sort_values(
        ["subjects", "group_value"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    group_sizes = details["subjects"].to_numpy(dtype=np.int64)
    weighted_entropy = float(
        np.average(
            details["normalized_fold_entropy"],
            weights=details["subjects"],
        )
    )
    weighted_max_fold_share = float(
        np.average(
            details["max_fold_share"],
            weights=details["subjects"],
        )
    )

    subjects_in_all_fold_groups = int(
        details.loc[details["present_in_all_folds"], "subjects"].sum()
    )

    fold_table = pd.crosstab(work[column], work["fold"])
    fold_v, fold_p = cramers_v_from_table(fold_table)

    label_table = pd.crosstab(work[column], work["is_pathologic"])
    label_v, label_p = cramers_v_from_table(label_table)

    groups_with_positive = int(
        work.loc[work["is_pathologic"] == 1, column].nunique()
    )
    groups_with_negative = int(
        work.loc[work["is_pathologic"] == 0, column].nunique()
    )

    n_groups = int(work[column].nunique())
    largest_group_fraction = float(group_sizes.max() / len(work))
    both_class_groups = int(
        ((details["positives"] > 0) & (details["negatives"] > 0)).sum()
    )
    groups_at_least_min_size = int(
        (details["subjects"] >= min_domain_size).sum()
    )

    basic_group_cv_feasible = bool(
        n_groups >= expected_folds
        and groups_with_positive >= expected_folds
        and groups_with_negative >= expected_folds
    )

    # Large individual groups can make balanced group CV difficult even
    # when technically feasible.
    balance_risk = (
        "HIGH"
        if largest_group_fraction > 0.35
        else "MODERATE"
        if largest_group_fraction > 0.20
        else "LOW"
    )

    # This is a domain-overlap diagnostic, not a formal proof of leakage.
    all_fold_subject_fraction = (
        subjects_in_all_fold_groups / len(work)
        if len(work)
        else float("nan")
    )

    if (
        all_fold_subject_fraction >= 0.60
        and weighted_entropy >= 0.75
    ):
        overlap_level = "HIGH"
    elif (
        all_fold_subject_fraction >= 0.25
        or weighted_entropy >= 0.55
    ):
        overlap_level = "MODERATE"
    else:
        overlap_level = "LOW"

    positive_rate_range = float(
        details["positive_rate"].max()
        - details["positive_rate"].min()
    )

    result = {
        "group_variable": column,
        "source_type": source_type,
        "selection_priority": priority,
        "coverage_subjects": int(len(work)),
        "coverage_fraction": float(coverage),
        "number_of_groups": n_groups,
        "minimum_group_size": int(group_sizes.min()),
        "median_group_size": float(np.median(group_sizes)),
        "maximum_group_size": int(group_sizes.max()),
        "largest_group_fraction": largest_group_fraction,
        "groups_at_least_min_domain_size": groups_at_least_min_size,
        "groups_with_both_classes": both_class_groups,
        "groups_with_positive_subjects": groups_with_positive,
        "groups_with_negative_subjects": groups_with_negative,
        "positive_rate_min": float(details["positive_rate"].min()),
        "positive_rate_max": float(details["positive_rate"].max()),
        "positive_rate_range": positive_rate_range,
        "label_association_cramers_v": label_v,
        "label_association_chi2_p": label_p,
        "fold_association_cramers_v": fold_v,
        "fold_association_chi2_p": fold_p,
        "weighted_mean_normalized_fold_entropy": weighted_entropy,
        "weighted_mean_max_fold_share": weighted_max_fold_share,
        "subjects_in_groups_present_all_folds": subjects_in_all_fold_groups,
        "subject_fraction_in_groups_present_all_folds":
            float(all_fold_subject_fraction),
        "current_cv_domain_overlap_level": overlap_level,
        "basic_5fold_group_cv_feasible": basic_group_cv_feasible,
        "group_balance_risk": balance_risk,
    }

    return result, details


# =============================================================================
# OOF performance loading
# =============================================================================

def validate_prediction_table(
    df: pd.DataFrame,
    probability_col: str,
    expected_subjects: int,
    expected_folds: int,
    source_name: str,
) -> pd.DataFrame:
    required = {"uid", "fold", "is_pathologic", probability_col}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{source_name}: missing columns {sorted(missing)}."
        )

    out = df.copy()
    out["uid"] = out["uid"].astype(str)
    out["fold"] = pd.to_numeric(out["fold"], errors="raise").astype(np.int64)
    out["is_pathologic"] = pd.to_numeric(
        out["is_pathologic"], errors="raise"
    ).astype(np.int64)
    out[probability_col] = pd.to_numeric(
        out[probability_col], errors="raise"
    ).astype(np.float64)

    if len(out) != expected_subjects:
        raise RuntimeError(
            f"{source_name}: expected {expected_subjects} rows, found {len(out)}."
        )
    if out["uid"].duplicated().any():
        raise RuntimeError(f"{source_name}: duplicate UIDs.")
    if not np.isin(out["is_pathologic"], [0, 1]).all():
        raise RuntimeError(f"{source_name}: invalid labels.")
    expected = set(range(expected_folds))
    if set(out["fold"].unique()) != expected:
        raise RuntimeError(
            f"{source_name}: fold IDs do not equal {sorted(expected)}."
        )

    p = out[probability_col].to_numpy()
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise RuntimeError(f"{source_name}: invalid probabilities.")

    return out


def load_p1_predictions(
    project_root: Path,
    shortlist_path: Path,
    expected_subjects: int,
    expected_folds: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    shortlist = pd.read_csv(shortlist_path)
    if "Shortlist ID" not in shortlist.columns:
        raise RuntimeError("Shortlist CSV has no 'Shortlist ID' column.")

    p1 = shortlist.loc[shortlist["Shortlist ID"].astype(str) == "P1"]
    if len(p1) != 1:
        raise RuntimeError(f"Expected one P1 row, found {len(p1)}.")

    row = p1.iloc[0]
    pred_col_path = "Cross-Fitted Prediction File"
    if pred_col_path not in shortlist.columns:
        raise RuntimeError(
            f"Shortlist CSV missing {pred_col_path!r}."
        )

    path = resolve_stored_path(project_root, row[pred_col_path])
    df = pd.read_csv(path)
    df = validate_prediction_table(
        df,
        probability_col="calibrated_probability",
        expected_subjects=expected_subjects,
        expected_folds=expected_folds,
        source_name="P1 prediction file",
    )

    meta = {
        "prediction_file": str(path),
        "stored_log_loss": (
            float(row["Cross-Fitted OOF Log Loss"])
            if "Cross-Fitted OOF Log Loss" in row.index
            else None
        ),
    }
    return df, meta


def load_step20_winner_predictions(
    project_root: Path,
    winner_path: Path,
    expected_subjects: int,
    expected_folds: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    winner = pd.read_csv(winner_path)
    if len(winner) != 1:
        raise RuntimeError(
            f"Expected exactly one Step-20 winner row, found {len(winner)}."
        )
    row = winner.iloc[0]

    pred_col = "Cross-Fitted Prediction File"
    if pred_col not in winner.columns:
        raise RuntimeError(
            f"Step-20 winner file missing {pred_col!r}."
        )

    path = resolve_stored_path(project_root, row[pred_col])
    df = pd.read_csv(path)
    df = validate_prediction_table(
        df,
        probability_col="calibrated_probability",
        expected_subjects=expected_subjects,
        expected_folds=expected_folds,
        source_name="Step-20 winner prediction file",
    )

    meta = {
        "prediction_file": str(path),
        "ensemble_id": str(row.get("Ensemble ID", "")),
        "members": str(row.get("Members", "")),
        "calibration_method": str(row.get("Calibration Method", "")),
        "stored_log_loss": (
            float(row["Cross-Fitted OOF Log Loss"])
            if "Cross-Fitted OOF Log Loss" in row.index
            else None
        ),
    }
    return df, meta


def merge_predictions(
    base: pd.DataFrame,
    p1: pd.DataFrame,
    ens: pd.DataFrame,
) -> pd.DataFrame:
    p1_small = p1[
        ["uid", "fold", "is_pathologic", "calibrated_probability"]
    ].rename(
        columns={"calibrated_probability": "p1_probability"}
    )

    ens_small = ens[
        ["uid", "fold", "is_pathologic", "calibrated_probability"]
    ].rename(
        columns={"calibrated_probability": "ens328_probability"}
    )

    merged = base.merge(
        p1_small,
        on=["uid", "fold", "is_pathologic"],
        how="left",
        validate="one_to_one",
    ).merge(
        ens_small,
        on=["uid", "fold", "is_pathologic"],
        how="left",
        validate="one_to_one",
    )

    if merged["p1_probability"].isna().any():
        raise RuntimeError("Missing P1 probabilities after UID/fold merge.")
    if merged["ens328_probability"].isna().any():
        raise RuntimeError("Missing ENS328 probabilities after UID/fold merge.")

    y = merged["is_pathologic"].to_numpy(dtype=np.int64)
    merged["p1_subject_log_loss"] = binary_log_loss_per_subject(
        y, merged["p1_probability"].to_numpy()
    )
    merged["ens328_subject_log_loss"] = binary_log_loss_per_subject(
        y, merged["ens328_probability"].to_numpy()
    )
    merged["ens328_minus_p1_subject_log_loss"] = (
        merged["ens328_subject_log_loss"]
        - merged["p1_subject_log_loss"]
    )

    return merged


def domain_performance_table(
    df: pd.DataFrame,
    group_columns: list[str],
    expected_folds: int,
    min_domain_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    global_p1 = global_log_loss(
        df["is_pathologic"].to_numpy(),
        df["p1_probability"].to_numpy(),
    )
    global_ens = global_log_loss(
        df["is_pathologic"].to_numpy(),
        df["ens328_probability"].to_numpy(),
    )

    for column in group_columns:
        work = df[
            [
                "uid",
                "is_pathologic",
                "fold",
                "p1_probability",
                "ens328_probability",
                column,
            ]
        ].copy()
        work[column] = clean_group_values(work[column])
        work = work.loc[work[column].notna()].copy()

        for group_name, g in work.groupby(column, sort=False):
            y = g["is_pathologic"].to_numpy(dtype=np.int64)
            p1 = g["p1_probability"].to_numpy(dtype=np.float64)
            ens = g["ens328_probability"].to_numpy(dtype=np.float64)

            p1_ll = global_log_loss(y, p1)
            ens_ll = global_log_loss(y, ens)

            rows.append(
                {
                    "group_variable": column,
                    "group_value": str(group_name),
                    "subjects": int(len(g)),
                    "negatives": int((y == 0).sum()),
                    "positives": int((y == 1).sum()),
                    "positive_rate": float(y.mean()),
                    "folds_present": int(g["fold"].nunique()),
                    "P1_log_loss": p1_ll,
                    "ENS328_log_loss": ens_ll,
                    "ENS328_minus_P1_log_loss": float(ens_ll - p1_ll),
                    "P1_excess_vs_global": float(p1_ll - global_p1),
                    "ENS328_excess_vs_global": float(ens_ll - global_ens),
                    "P1_AUROC": safe_auroc(y, p1),
                    "ENS328_AUROC": safe_auroc(y, ens),
                    "eligible_min_domain_size": bool(
                        len(g) >= min_domain_size
                    ),
                    "contains_both_classes": bool(len(np.unique(y)) == 2),
                    "present_in_all_folds": bool(
                        g["fold"].nunique() == expected_folds
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["group_variable", "subjects", "group_value"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


# =============================================================================
# Recommendation logic
# =============================================================================

def choose_primary_group(
    diagnostics: pd.DataFrame,
) -> tuple[str | None, str]:
    if diagnostics.empty:
        return None, "NO_GROUP_VARIABLES_FOUND"

    usable = diagnostics.loc[
        (diagnostics["coverage_fraction"] >= 0.95)
        & diagnostics["basic_5fold_group_cv_feasible"]
    ].copy()

    if usable.empty:
        return None, "NO_GROUP_VARIABLE_BASICALLY_FEASIBLE_FOR_5FOLD_CV"

    # Do not pick intensity-only diagnostic variables for CV grouping.
    usable = usable.loc[
        usable["source_type"] != "diagnostic_only"
    ].copy()

    if usable.empty:
        return None, "ONLY_DIAGNOSTIC_ONLY_GROUP_VARIABLES_AVAILABLE"

    usable = usable.copy()
    risk_rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    usable["_balance_risk_rank"] = (
        usable["group_balance_risk"]
        .map(risk_rank)
        .fillna(99)
        .astype(int)
    )

    usable = usable.sort_values(
        [
            "selection_priority",
            "_balance_risk_rank",
            "number_of_groups",
            "coverage_fraction",
        ],
        ascending=[True, True, False, False],
        kind="stable",
    )

    row = usable.iloc[0]
    column = str(row["group_variable"])
    source_type = str(row["source_type"])

    if source_type == "actual_site_or_center":
        reason = "PREFER_TRUE_SITE_CENTER_METADATA"
    elif source_type == "scanner_or_protocol":
        reason = "PREFER_SCANNER_PROTOCOL_METADATA_NO_TRUE_SITE_AVAILABLE"
    elif source_type == "existing_acquisition_proxy":
        reason = "USE_EXISTING_ACQUISITION_GEOMETRY_PROXY_NO_TRUE_SITE_AVAILABLE"
    else:
        reason = "USE_DERIVED_GEOMETRY_PROXY_WITH_CAUTION"

    return column, reason


def label_association_level(v: float) -> str:
    if not np.isfinite(v):
        return "UNKNOWN"
    if v >= 0.30:
        return "STRONG"
    if v >= 0.15:
        return "MODERATE"
    if v >= 0.05:
        return "WEAK"
    return "VERY_LOW"


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = (
        args.project_root
        if args.project_root is not None
        else script_path.parents[2]
    ).expanduser().resolve()

    manifest_path = resolve_input_path(
        project_root,
        args.manifest
        or Path(
            "data/preprocessing_supervised_data/"
            "step10_supervised_dataset_manifest_data/"
            "supervised_dataset/"
            "supervised_dataset_manifest.csv"
        ),
    )

    folds_path = resolve_input_path(
        project_root,
        args.folds
        or Path(
            "data/preprocessing_supervised_data/"
            "step11_create_freeze_cv_splits_data/"
            "fold_assignments.csv"
        ),
    )

    acquisition_path = resolve_input_path(
        project_root,
        args.acquisition_csv
        or Path(
            "data/preprocessing_image_data/"
            "step5b_acquisition_geometry_analysis_data/"
            "acquisition_cohort_per_scan.csv"
        ),
    )

    resampling_path = resolve_input_path(
        project_root,
        args.resampling_summary
        or Path(
            "data/preprocessing_image_data/"
            "step4_voxel_resampler_data/"
            "resampling_summary.json"
        ),
    )

    shortlist_path = resolve_input_path(
        project_root,
        args.shortlist
        or Path(
            "data/calibration_validation_data/"
            "competition_shortlist_data/"
            "competition_shortlist_8.csv"
        ),
    )

    step20_winner_path = resolve_input_path(
        project_root,
        args.step20_winner
        or Path(
            "data/ensemble_calibration_statistics_data/"
            "representative_ensembles/"
            "best_ensemble_after_cross_fitted_calibration.csv"
        ),
    )

    output_dir = resolve_input_path(
        project_root,
        args.output_dir
        or Path(
            "data/generalization_validation_data/"
            "step25b_acquisition_domain_analysis"
        ),
    )

    print("=" * 100)
    print("STEP 25B — ACQUISITION / DOMAIN GENERALIZATION ANALYSIS")
    print("=" * 100)
    print(f"Project root             : {project_root}")
    print(f"Step-10 manifest         : {manifest_path}")
    print(f"Step-11 folds            : {folds_path}")
    print(f"Step-5B acquisition CSV  : {acquisition_path}")
    print(f"Step-4 resampling JSON   : {resampling_path}")
    print(f"P1 shortlist             : {shortlist_path}")
    print(f"Step-20 winner           : {step20_winner_path}")
    print(f"Output                   : {output_dir}")
    print("Neural-network training  : NO")
    print("Calibration fitting      : NO")
    print("New CV folds frozen      : NO")
    print()

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"Output directory is non-empty:\n{output_dir}\n"
            "Use --overwrite to rebuild."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load canonical subject/fold table.
    # ------------------------------------------------------------------
    manifest = load_manifest(manifest_path, args.expected_subjects)
    folds = load_folds(
        folds_path,
        args.expected_subjects,
        args.expected_folds,
    )

    base = manifest.merge(
        folds,
        on="uid",
        how="inner",
        validate="one_to_one",
    )

    if len(base) != args.expected_subjects:
        raise RuntimeError(
            "Manifest/fold UID intersection did not preserve all subjects."
        )

    # ------------------------------------------------------------------
    # Attach existing acquisition metadata + robust Step-4 fallback.
    # ------------------------------------------------------------------
    acquisition = load_acquisition_csv(acquisition_path)
    resampling = load_resampling_summary(resampling_path)

    if not acquisition.empty and len(acquisition) > 0:
        # Avoid reusing any duplicated column names from the fallback.
        overlap = (
            set(base.columns)
            & set(acquisition.columns)
        ) - {"uid"}
        if overlap:
            acquisition = acquisition.rename(
                columns={c: f"step5b_{c}" for c in overlap}
            )
        base = base.merge(
            acquisition,
            on="uid",
            how="left",
            validate="one_to_one",
        )

    if not resampling.empty and len(resampling) > 0:
        overlap = (
            set(base.columns)
            & set(resampling.columns)
        ) - {"uid"}
        if overlap:
            resampling = resampling.rename(
                columns={c: f"step4_{c}" for c in overlap}
            )
        base = base.merge(
            resampling,
            on="uid",
            how="left",
            validate="one_to_one",
        )

    base = derive_geometry_features(
        base,
        spacing_decimals=args.spacing_round_decimals,
    )

    # ------------------------------------------------------------------
    # Detect candidate grouping variables.
    # ------------------------------------------------------------------
    group_columns = candidate_group_columns(base)

    print("Candidate acquisition/domain grouping variables:")
    if group_columns:
        for col in group_columns:
            source_type, priority = classify_group_column(col)
            print(f"  - {col:<36s} type={source_type} priority={priority}")
    else:
        print("  NONE")

    diagnostics_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []

    for col in group_columns:
        result, details = analyze_group_variable(
            base,
            column=col,
            expected_folds=args.expected_folds,
            min_domain_size=args.min_domain_size,
        )
        diagnostics_rows.append(result)
        detail_frames.append(details)

    diagnostics = pd.DataFrame(diagnostics_rows)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values(
            ["selection_priority", "coverage_fraction", "number_of_groups"],
            ascending=[True, False, False],
            kind="stable",
        ).reset_index(drop=True)

    group_details = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames
        else pd.DataFrame()
    )

    primary_group, primary_reason = choose_primary_group(diagnostics)

    # ------------------------------------------------------------------
    # Load P1 + Step-20 winner OOF predictions and compute domain errors.
    # ------------------------------------------------------------------
    p1_df, p1_meta = load_p1_predictions(
        project_root,
        shortlist_path,
        args.expected_subjects,
        args.expected_folds,
    )
    ens_df, ens_meta = load_step20_winner_predictions(
        project_root,
        step20_winner_path,
        args.expected_subjects,
        args.expected_folds,
    )

    base = merge_predictions(base, p1_df, ens_df)

    p1_global = global_log_loss(
        base["is_pathologic"].to_numpy(),
        base["p1_probability"].to_numpy(),
    )
    ens_global = global_log_loss(
        base["is_pathologic"].to_numpy(),
        base["ens328_probability"].to_numpy(),
    )

    domain_perf = domain_performance_table(
        base,
        group_columns,
        args.expected_folds,
        args.min_domain_size,
    )

    # ------------------------------------------------------------------
    # Fold/class summary.
    # ------------------------------------------------------------------
    fold_summary = (
        base.groupby("fold", as_index=False)
        .agg(
            subjects=("uid", "size"),
            negatives=("is_pathologic", lambda x: int((x == 0).sum())),
            positives=("is_pathologic", lambda x: int((x == 1).sum())),
            positive_rate=("is_pathologic", "mean"),
            P1_log_loss=("p1_subject_log_loss", "mean"),
            ENS328_log_loss=("ens328_subject_log_loss", "mean"),
        )
    )
    fold_summary["ENS328_minus_P1_log_loss"] = (
        fold_summary["ENS328_log_loss"] - fold_summary["P1_log_loss"]
    )

    # ------------------------------------------------------------------
    # Primary-domain specific outputs.
    # ------------------------------------------------------------------
    primary_summary = pd.DataFrame()
    primary_crosstab = pd.DataFrame()
    primary_perf = pd.DataFrame()
    high_risk_domains = pd.DataFrame()

    primary_diag: dict[str, Any] | None = None

    if primary_group is not None:
        primary_diag = (
            diagnostics.loc[
                diagnostics["group_variable"] == primary_group
            ].iloc[0].to_dict()
        )

        primary_summary = group_details.loc[
            group_details["group_variable"] == primary_group
        ].copy()

        work = base[["uid", "fold", primary_group]].copy()
        work[primary_group] = clean_group_values(work[primary_group])
        work = work.loc[work[primary_group].notna()].copy()

        primary_crosstab = pd.crosstab(
            work[primary_group],
            work["fold"],
            margins=True,
        )

        if not domain_perf.empty:
            primary_perf = domain_perf.loc[
                domain_perf["group_variable"] == primary_group
            ].copy()

            eligible = primary_perf.loc[
                primary_perf["eligible_min_domain_size"]
            ].copy()

            high_risk_domains = eligible.sort_values(
                [
                    "ENS328_log_loss",
                    "ENS328_excess_vs_global",
                    "subjects",
                ],
                ascending=[False, False, False],
                kind="stable",
            ).head(args.top_domains)

    # ------------------------------------------------------------------
    # Interpretation.
    # ------------------------------------------------------------------
    recommendation = {
        "primary_group_variable": primary_group,
        "primary_group_reason": primary_reason,
        "step25c_action": (
            "Construct candidate domain-aware folds using this grouping "
            "variable, then inspect fold/class/domain balance before freezing."
            if primary_group is not None
            else
            "Do not construct domain-aware folds yet; no sufficiently "
            "supported grouping variable was identified."
        ),
        "important_caution": (
            "Only an actual site/center/hospital variable should be called "
            "a true site split. Geometry/scanner/cohort variables are "
            "acquisition-domain proxies."
        ),
    }

    if primary_diag is not None:
        overlap_level = str(
            primary_diag["current_cv_domain_overlap_level"]
        )
        label_v = float(
            primary_diag["label_association_cramers_v"]
        )
        recommendation["current_cv_domain_overlap_level"] = overlap_level
        recommendation["label_domain_association_level"] = (
            label_association_level(label_v)
        )
        recommendation["label_domain_cramers_v"] = label_v

        if (
            primary_diag["source_type"] == "actual_site_or_center"
            and overlap_level == "HIGH"
        ):
            interpretation = (
                "The current Step-11 folds strongly mix the same true "
                "site/center groups across folds. This is direct evidence "
                "that validation is not site-held-out."
            )
        elif overlap_level == "HIGH":
            interpretation = (
                "The current Step-11 folds strongly mix the same acquisition/"
                "geometry proxy groups across folds. This supports a domain-"
                "shift concern, but it is not proof of hospital leakage."
            )
        elif overlap_level == "MODERATE":
            interpretation = (
                "The current folds show moderate acquisition-domain overlap."
            )
        else:
            interpretation = (
                "The chosen acquisition/domain groups are not strongly mixed "
                "across current folds."
            )

        recommendation["interpretation"] = interpretation

    # ------------------------------------------------------------------
    # Save outputs.
    # ------------------------------------------------------------------
    subject_path = output_dir / "subject_acquisition_domain_table.csv"
    diagnostics_path = output_dir / "group_variable_diagnostics.csv"
    group_details_path = output_dir / "domain_group_details_long.csv"
    fold_summary_path = output_dir / "current_fold_label_and_oof_summary.csv"
    perf_path = output_dir / "domain_oof_performance_long.csv"
    primary_summary_path = output_dir / "primary_domain_summary.csv"
    primary_crosstab_path = output_dir / "primary_domain_by_current_fold.csv"
    primary_perf_path = output_dir / "primary_domain_oof_performance.csv"
    high_risk_path = output_dir / "primary_domain_high_error_groups.csv"
    summary_path = output_dir / "step25b_summary.json"
    report_path = output_dir / "step25b_report.md"

    # Keep all relevant subject-level acquisition variables, but avoid
    # accidental huge object/path columns when possible.
    subject_cols = [
        "uid",
        "is_pathologic",
        "fold",
    ]
    for col in group_columns:
        if col not in subject_cols:
            subject_cols.append(col)

    for col in (
        "derived_voxel_volume_mm3",
        "derived_fov_x_mm",
        "derived_fov_y_mm",
        "derived_fov_z_mm",
        "p1_probability",
        "ens328_probability",
        "p1_subject_log_loss",
        "ens328_subject_log_loss",
        "ens328_minus_p1_subject_log_loss",
    ):
        if col in base.columns:
            subject_cols.append(col)

    base[subject_cols].to_csv(
        subject_path,
        index=False,
        float_format="%.9f",
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
        float_format="%.9f",
    )
    group_details.to_csv(
        group_details_path,
        index=False,
        float_format="%.9f",
    )
    fold_summary.to_csv(
        fold_summary_path,
        index=False,
        float_format="%.9f",
    )
    domain_perf.to_csv(
        perf_path,
        index=False,
        float_format="%.9f",
    )

    primary_summary.to_csv(
        primary_summary_path,
        index=False,
        float_format="%.9f",
    )
    primary_crosstab.to_csv(primary_crosstab_path)
    primary_perf.to_csv(
        primary_perf_path,
        index=False,
        float_format="%.9f",
    )
    high_risk_domains.to_csv(
        high_risk_path,
        index=False,
        float_format="%.9f",
    )

    summary_payload: dict[str, Any] = {
        "status": "PASS",
        "step": "25B",
        "analysis_type": "acquisition_domain_generalization_analysis",
        "subjects": int(len(base)),
        "folds": int(args.expected_folds),
        "class_counts": {
            "normal": int((base["is_pathologic"] == 0).sum()),
            "pathologic": int((base["is_pathologic"] == 1).sum()),
        },
        "input_artifacts": {
            "manifest": str(manifest_path),
            "folds": str(folds_path),
            "acquisition_csv": (
                str(acquisition_path)
                if acquisition_path.is_file()
                else None
            ),
            "resampling_summary": (
                str(resampling_path)
                if resampling_path.is_file()
                else None
            ),
            "shortlist": str(shortlist_path),
            "step20_winner": str(step20_winner_path),
            "P1_prediction_file": p1_meta["prediction_file"],
            "ENS328_prediction_file": ens_meta["prediction_file"],
        },
        "oof_performance_integrity": {
            "P1_recomputed_global_log_loss": p1_global,
            "ENS328_recomputed_global_log_loss": ens_global,
            "ENS328_minus_P1_global_log_loss": ens_global - p1_global,
            "P1_stored_log_loss": p1_meta.get("stored_log_loss"),
            "ENS328_stored_log_loss": ens_meta.get("stored_log_loss"),
            "ENS328_id": ens_meta.get("ensemble_id"),
            "ENS328_members": ens_meta.get("members"),
            "ENS328_calibration": ens_meta.get("calibration_method"),
        },
        "candidate_group_variables": group_columns,
        "number_candidate_group_variables": len(group_columns),
        "primary_group_diagnostic": primary_diag,
        "recommendation": recommendation,
        "methodological_limits": [
            (
                "Step 25B is observational analysis of training metadata and "
                "OOF predictions; it does not estimate a new unbiased test score."
            ),
            (
                "A geometry/acquisition proxy cannot be assumed to equal the "
                "true hospital/site identity unless explicit site metadata says so."
            ),
            (
                "Association between acquisition domain and pathology label "
                "does not by itself prove confounding, but strong association "
                "raises concern for random subject-level CV."
            ),
            (
                "No new fold assignment is frozen in Step 25B. Step 25C should "
                "construct and validate candidate group-aware folds."
            ),
        ],
    }
    write_json(summary_path, summary_payload)

    # ------------------------------------------------------------------
    # Human-readable report.
    # ------------------------------------------------------------------
    report_lines: list[str] = []
    report_lines.append("# Step 25B — Acquisition / Domain Generalization Analysis")
    report_lines.append("")
    report_lines.append("## Status")
    report_lines.append("")
    report_lines.append("**PASS — analysis completed. No training was performed.**")
    report_lines.append("")
    report_lines.append("## Dataset")
    report_lines.append("")
    report_lines.append(f"- Subjects: **{len(base)}**")
    report_lines.append(
        f"- Normal: **{int((base['is_pathologic'] == 0).sum())}**"
    )
    report_lines.append(
        f"- Pathologic: **{int((base['is_pathologic'] == 1).sum())}**"
    )
    report_lines.append(f"- Existing folds: **{args.expected_folds}**")
    report_lines.append("")
    report_lines.append("## OOF integrity")
    report_lines.append("")
    report_lines.append(f"- P1 global OOF Log Loss: **{p1_global:.6f}**")
    report_lines.append(
        f"- ENS328 global CF OOF Log Loss: **{ens_global:.6f}**"
    )
    report_lines.append(
        f"- ENS328 − P1: **{ens_global - p1_global:+.6f}**"
    )
    report_lines.append("")
    report_lines.append("## Candidate domain variables")
    report_lines.append("")

    if diagnostics.empty:
        report_lines.append("No usable acquisition/domain grouping variables were found.")
    else:
        display_cols = [
            "group_variable",
            "source_type",
            "coverage_fraction",
            "number_of_groups",
            "largest_group_fraction",
            "subject_fraction_in_groups_present_all_folds",
            "weighted_mean_normalized_fold_entropy",
            "label_association_cramers_v",
            "current_cv_domain_overlap_level",
            "basic_5fold_group_cv_feasible",
        ]
        report_lines.append(
            diagnostics[display_cols]
            .to_markdown(index=False, floatfmt=".4f")
        )

    report_lines.append("")
    report_lines.append("## Recommended grouping variable for Step 25C")
    report_lines.append("")

    if primary_group is None:
        report_lines.append(
            "**None identified with sufficient support for a 5-fold "
            "group-aware split.**"
        )
        report_lines.append("")
        report_lines.append(f"Reason: `{primary_reason}`")
    else:
        report_lines.append(f"**{primary_group}**")
        report_lines.append("")
        report_lines.append(f"Reason: `{primary_reason}`")

        if primary_diag is not None:
            report_lines.append("")
            report_lines.append(
                f"- Source type: **{primary_diag['source_type']}**"
            )
            report_lines.append(
                "- Current-CV domain overlap: "
                f"**{primary_diag['current_cv_domain_overlap_level']}**"
            )
            report_lines.append(
                "- Subjects in domains present in all current folds: "
                f"**{100.0 * primary_diag['subject_fraction_in_groups_present_all_folds']:.1f}%**"
            )
            report_lines.append(
                "- Weighted normalized fold entropy: "
                f"**{primary_diag['weighted_mean_normalized_fold_entropy']:.3f}**"
            )
            report_lines.append(
                "- Label/domain Cramer's V: "
                f"**{primary_diag['label_association_cramers_v']:.3f}**"
            )
            report_lines.append(
                "- Label/domain association level: "
                f"**{label_association_level(float(primary_diag['label_association_cramers_v']))}**"
            )

    report_lines.append("")
    report_lines.append("## Important interpretation")
    report_lines.append("")
    report_lines.append(
        recommendation.get(
            "interpretation",
            "No primary-domain interpretation available.",
        )
    )
    report_lines.append("")
    report_lines.append(
        "A geometry/scanner/acquisition cohort is a **proxy domain** unless "
        "the metadata explicitly identifies the hospital/site."
    )

    if not high_risk_domains.empty:
        report_lines.append("")
        report_lines.append("## Highest-error primary domains")
        report_lines.append("")
        display = high_risk_domains[
            [
                "group_value",
                "subjects",
                "positive_rate",
                "P1_log_loss",
                "ENS328_log_loss",
                "ENS328_minus_P1_log_loss",
                "ENS328_excess_vs_global",
            ]
        ].head(15)
        report_lines.append(
            display.to_markdown(index=False, floatfmt=".4f")
        )

    report_lines.append("")
    report_lines.append("## Next step")
    report_lines.append("")
    report_lines.append(
        "Step 25C should construct candidate **domain-aware CV folds** "
        "using the recommended grouping variable, check class/group balance, "
        "and freeze nothing until those diagnostics pass."
    )
    report_lines.append("")
    report_lines.append(
        "**Do not retrain the full 60-experiment grid before inspecting this report.**"
    )

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("STEP 25B SUMMARY")
    print("=" * 100)
    print(f"Status                         : PASS")
    print(f"Subjects                       : {len(base)}")
    print(f"Candidate domain variables     : {len(group_columns)}")
    print(f"P1 global OOF Log Loss         : {p1_global:.6f}")
    print(f"ENS328 global CF OOF Log Loss  : {ens_global:.6f}")
    print(f"Recommended Step-25C grouping  : {primary_group}")
    print(f"Recommendation reason          : {primary_reason}")

    if primary_diag is not None:
        print(
            "Current-CV domain overlap      : "
            f"{primary_diag['current_cv_domain_overlap_level']}"
        )
        print(
            "Domains-present-all-folds subj : "
            f"{100.0 * primary_diag['subject_fraction_in_groups_present_all_folds']:.1f}%"
        )
        print(
            "Label/domain Cramer's V        : "
            f"{primary_diag['label_association_cramers_v']:.4f}"
        )

    print(f"Saved                          : {output_dir}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
