#!/usr/bin/env python3
"""
STEP 33 — HIDDEN-TRANSFER / ACQUISITION-STRATIFIED OOF DIAGNOSIS

Purpose
-------
Diagnose why the frozen competition predictors generalize worse to the hidden
competition distribution than expected from OOF validation.

This step:
  * performs NO neural-network training;
  * loads NO checkpoints;
  * changes NO model/preprocessing source;
  * never uses hidden-test labels or hidden-test per-subject predictions;
  * reuses the authoritative ProbabilityCalibrator for cross-fitted final
    temperature calibration;
  * compares ENS328 and ENS328R globally, by original CV fold, and across
    available acquisition/preprocessing metadata subgroups.

Expected OOF schema for each predictor:
    uid, fold, is_pathologic, probability

The input `probability` is expected to be the ensemble probability BEFORE the
final deployment temperature. Step 33 cross-fits the final temperature:
for held-out fold f, fit temperature on OOF folds != f, apply to fold f.

Outputs
-------
  step33_subject_level_predictions.csv
  step33_overall_metrics.csv
  step33_fold_metrics.csv
  step33_subgroup_metrics.csv
  step33_feature_summary.csv
  step33_calibration_bins.csv
  step33_metadata_sources.csv
  step33_crossfit_calibrators.json
  step33_bootstrap_delta.json
  step33_summary.json
  step33_report.txt
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


EPS = 1e-6
RNG_SEED = 2033

# Metadata columns that would create leakage or are just prediction outputs.
EXCLUDED_METADATA_TOKENS = (
    "probability",
    "prediction",
    "predicted",
    "log_loss",
    "logloss",
    "auroc",
    "auc",
    "auprc",
    "brier",
    "ece",
    "is_pathologic",
    "label",
    "target",
    "class",
)

# Search is deliberately biased toward the diagnostics already created in
# Steps 25B and preprocessing/QC outputs.
METADATA_PATH_KEYWORDS = {
    "step25b": 120,
    "acquisition": 100,
    "spacing": 55,
    "geometry": 55,
    "registration": 45,
    "reference": 45,
    "localization": 40,
    "resampl": 35,
    "crop": 30,
    "metadata": 25,
    "manifest": 10,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 33 acquisition-stratified OOF diagnosis for ENS328 vs ENS328R."
        )
    )
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--ens328-oof", type=Path, default=None)
    p.add_argument("--ens328r-oof", type=Path, default=None)
    p.add_argument(
        "--fold-assignments",
        type=Path,
        default=None,
        help=(
            "Frozen Step-11 fold assignment CSV. Required only when an OOF file "
            "does not already contain a fold column. Expected columns: uid, fold."
        ),
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--metadata-csv",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional metadata CSV to join by uid. Repeat the argument for multiple "
            "files. If omitted, Step 33 auto-discovers likely acquisition/QC CSVs."
        ),
    )
    p.add_argument(
        "--no-auto-metadata",
        action="store_true",
        help="Disable recursive metadata discovery when --metadata-csv is omitted.",
    )
    p.add_argument("--max-auto-metadata-files", type=int, default=12)
    p.add_argument("--expected-subjects", type=int, default=1362)
    p.add_argument("--expected-folds", type=int, default=5)
    p.add_argument("--min-group-size", type=int, default=20)
    p.add_argument("--max-categorical-levels", type=int, default=30)
    p.add_argument("--continuous-quantile-bins", type=int, default=5)
    p.add_argument("--ece-bins", type=int, default=10)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--hidden-ens328-logloss", type=float, default=None)
    p.add_argument("--hidden-ens328r-logloss", type=float, default=None)
    return p.parse_args()


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def add_project_to_path(root: Path) -> None:
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def load_fold_assignments(
    path: Path,
    expected_subjects: int,
    expected_folds: int,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Fold-assignment file not found: {path}")

    df = pd.read_csv(path)
    required = {"uid", "fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    df = df[["uid", "fold"]].copy()
    df["uid"] = df["uid"].astype(str)
    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)

    if len(df) != expected_subjects:
        raise ValueError(
            f"{path}: expected {expected_subjects} subjects, found {len(df)}"
        )
    if df["uid"].duplicated().any():
        duplicates = df.loc[df["uid"].duplicated(), "uid"].head(10).tolist()
        raise ValueError(f"{path}: duplicate UIDs, examples={duplicates}")

    observed_folds = sorted(df["fold"].unique().tolist())
    expected = list(range(expected_folds))
    if observed_folds != expected:
        raise ValueError(f"{path}: folds={observed_folds}, expected={expected}")

    return df.sort_values("uid", kind="stable").reset_index(drop=True)


def load_oof(
    path: Path,
    expected_subjects: int,
    expected_folds: int,
    fold_assignments: pd.DataFrame,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"OOF file not found: {path}")

    df = pd.read_csv(path)
    # The frozen ensemble OOF files from Steps 20/31 do not necessarily carry
    # the Step-11 fold column. uid + label + probability are authoritative in
    # those files; fold is reconstructed from the frozen Step-11 assignment.
    required = {"uid", "is_pathologic", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    keep = ["uid", "is_pathologic", "probability"]
    has_fold = "fold" in df.columns
    if has_fold:
        keep.insert(1, "fold")
    df = df[keep].copy()

    df["uid"] = df["uid"].astype(str)
    df["is_pathologic"] = pd.to_numeric(
        df["is_pathologic"], errors="raise"
    ).astype(int)
    df["probability"] = pd.to_numeric(df["probability"], errors="raise").astype(float)

    if len(df) != expected_subjects:
        raise ValueError(
            f"{path}: expected {expected_subjects} subjects, found {len(df)}"
        )
    if df["uid"].duplicated().any():
        duplicates = df.loc[df["uid"].duplicated(), "uid"].head(10).tolist()
        raise ValueError(f"{path}: duplicate UIDs, examples={duplicates}")
    if not np.isin(df["is_pathologic"].to_numpy(), [0, 1]).all():
        raise ValueError(f"{path}: labels must be 0/1")
    if not np.isfinite(df["probability"].to_numpy()).all():
        raise ValueError(f"{path}: probability contains NaN/Inf")
    if not ((df["probability"] >= 0.0) & (df["probability"] <= 1.0)).all():
        raise ValueError(f"{path}: probability must be in [0,1]")

    if has_fold:
        df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
        check = df[["uid", "fold"]].merge(
            fold_assignments.rename(columns={"fold": "step11_fold"}),
            on="uid",
            how="left",
            validate="one_to_one",
        )
        if check["step11_fold"].isna().any():
            missing_uids = check.loc[check["step11_fold"].isna(), "uid"].head(10).tolist()
            raise ValueError(
                f"{path}: UIDs missing from Step-11 fold assignments, examples={missing_uids}"
            )
        mismatch = check["fold"].to_numpy() != check["step11_fold"].to_numpy()
        if mismatch.any():
            examples = check.loc[mismatch, ["uid", "fold", "step11_fold"]].head(10)
            raise ValueError(
                f"{path}: embedded folds disagree with frozen Step-11 assignments. "
                f"Examples:\n{examples.to_string(index=False)}"
            )
    else:
        df = df.merge(
            fold_assignments,
            on="uid",
            how="left",
            validate="one_to_one",
        )
        if df["fold"].isna().any():
            missing_uids = df.loc[df["fold"].isna(), "uid"].head(10).tolist()
            raise ValueError(
                f"{path}: UIDs missing from Step-11 fold assignments, examples={missing_uids}"
            )
        df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
        print(f"  fold column absent -> reconstructed from Step-11 assignments: {path.name}")

    observed_folds = sorted(df["fold"].unique().tolist())
    expected = list(range(expected_folds))
    if observed_folds != expected:
        raise ValueError(f"{path}: folds={observed_folds}, expected={expected}")

    return df[["uid", "fold", "is_pathologic", "probability"]].sort_values(
        "uid", kind="stable"
    ).reset_index(drop=True)


def cross_fit_temperature(
    df: pd.DataFrame,
    *,
    expected_folds: int,
    ProbabilityCalibrator: Any,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    y = df["is_pathologic"].to_numpy(dtype=np.int64)
    p = df["probability"].to_numpy(dtype=np.float64)
    folds = df["fold"].to_numpy(dtype=np.int64)

    result = np.full(len(df), np.nan, dtype=np.float64)
    states: list[dict[str, Any]] = []

    for heldout in range(expected_folds):
        fit_mask = folds != heldout
        apply_mask = folds == heldout

        calibrator = ProbabilityCalibrator(
            method="temperature",
            output_epsilon=EPS,
        )
        calibrator.fit(
            labels=y[fit_mask],
            probabilities=p[fit_mask],
            max_iter=150,
        )
        result[apply_mask] = calibrator.predict_proba(p[apply_mask])
        states.append(
            {
                "heldout_fold": int(heldout),
                "fit_subjects": int(fit_mask.sum()),
                "apply_subjects": int(apply_mask.sum()),
                "calibrator": calibrator.to_dict(),
            }
        )

    if not np.isfinite(result).all():
        raise RuntimeError("Cross-fitted final temperature produced NaN/Inf")

    return np.clip(result, EPS, 1.0 - EPS), states


def binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def per_subject_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def safe_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    ece_bins: int,
    compute_binary_metrics: Any,
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)

    base = {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "mean_confidence": float(np.mean(np.abs(p - 0.5) * 2.0)),
        "log_loss": binary_log_loss(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece_score(y, p, ece_bins),
        "auroc": float("nan"),
        "auprc": float("nan"),
        "balanced_accuracy": float("nan"),
    }

    if len(np.unique(y)) == 2:
        try:
            m = compute_binary_metrics(y, p, threshold=0.5, ece_bins=ece_bins)
            base["log_loss"] = float(m["log_loss"])
            base["brier"] = float(m["brier_score"])
            base["ece"] = float(m["ece"])
            base["auroc"] = float(m["auroc"])
            base["auprc"] = float(m["auprc"])
            base["balanced_accuracy"] = float(m["balanced_accuracy"])
        except Exception:
            pass

    return base


def comparison_row(
    subset: pd.DataFrame,
    *,
    feature: str,
    value: str,
    ece_bins: int,
    compute_binary_metrics: Any,
) -> dict[str, Any]:
    y = subset["is_pathologic"].to_numpy(dtype=np.int64)
    p_old = subset["ens328_cf_probability"].to_numpy(dtype=float)
    p_new = subset["ens328r_cf_probability"].to_numpy(dtype=float)

    m_old = safe_metrics(y, p_old, ece_bins=ece_bins, compute_binary_metrics=compute_binary_metrics)
    m_new = safe_metrics(y, p_new, ece_bins=ece_bins, compute_binary_metrics=compute_binary_metrics)

    return {
        "feature": feature,
        "group": value,
        "n": int(len(subset)),
        "positive_rate": float(np.mean(y)),
        "ens328_log_loss": m_old["log_loss"],
        "ens328r_log_loss": m_new["log_loss"],
        "delta_log_loss_r_minus_old": m_new["log_loss"] - m_old["log_loss"],
        "ens328_brier": m_old["brier"],
        "ens328r_brier": m_new["brier"],
        "delta_brier_r_minus_old": m_new["brier"] - m_old["brier"],
        "ens328_ece": m_old["ece"],
        "ens328r_ece": m_new["ece"],
        "ens328_auroc": m_old["auroc"],
        "ens328r_auroc": m_new["auroc"],
        "ens328_auprc": m_old["auprc"],
        "ens328r_auprc": m_new["auprc"],
        "ens328_mean_probability": m_old["mean_probability"],
        "ens328r_mean_probability": m_new["mean_probability"],
        "ens328_mean_confidence": m_old["mean_confidence"],
        "ens328r_mean_confidence": m_new["mean_confidence"],
        "ens328r_better_log_loss": bool(m_new["log_loss"] < m_old["log_loss"]),
    }


def sanitize_prefix(path: Path) -> str:
    name = path.stem.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name[:60] or "metadata"


def metadata_file_score(path: Path, columns: Iterable[str]) -> int:
    low_path = str(path).lower()
    score = 0
    for key, weight in METADATA_PATH_KEYWORDS.items():
        if key in low_path:
            score += weight
    cols = [str(c).lower() for c in columns]
    for key in ("spacing", "shape", "geometry", "registration", "reference", "center", "padding"):
        if any(key in c for c in cols):
            score += 15
    return score


def useful_metadata_columns(columns: Iterable[str]) -> list[str]:
    useful: list[str] = []
    for col in columns:
        c = str(col)
        low = c.lower()
        if low == "uid":
            continue
        if any(tok in low for tok in EXCLUDED_METADATA_TOKENS):
            continue
        useful.append(c)
    return useful


def discover_metadata_files(root: Path, max_files: int) -> list[Path]:
    candidates: list[tuple[int, Path, list[str]]] = []
    for path in root.glob("data/**/*.csv"):
        low = str(path).lower()
        if any(token in low for token in ("final_predictor", "oof", "ensemble_calibration")):
            continue
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        if "uid" not in header.columns:
            continue
        useful = useful_metadata_columns(header.columns)
        if not useful:
            continue
        score = metadata_file_score(path, header.columns)
        if score <= 0:
            continue
        candidates.append((score, path.resolve(), useful))

    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    selected: list[Path] = []
    for _, path, _ in candidates:
        if path not in selected:
            selected.append(path)
        if len(selected) >= max_files:
            break
    return selected


def merge_metadata(
    base: pd.DataFrame,
    paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = base.copy()
    source_rows: list[dict[str, Any]] = []

    for path in paths:
        if not path.is_file():
            source_rows.append(
                {
                    "path": str(path),
                    "status": "MISSING",
                    "rows": 0,
                    "uid_coverage": 0.0,
                    "columns_added": 0,
                    "added_columns": "",
                }
            )
            continue

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            source_rows.append(
                {
                    "path": str(path),
                    "status": f"READ_ERROR:{type(exc).__name__}",
                    "rows": 0,
                    "uid_coverage": 0.0,
                    "columns_added": 0,
                    "added_columns": "",
                }
            )
            continue

        if "uid" not in df.columns:
            source_rows.append(
                {
                    "path": str(path),
                    "status": "NO_UID",
                    "rows": len(df),
                    "uid_coverage": 0.0,
                    "columns_added": 0,
                    "added_columns": "",
                }
            )
            continue

        df = df.copy()
        df["uid"] = df["uid"].astype(str)
        if df["uid"].duplicated().any():
            # A repeated-measure / candidate-level table is not safe for a 1:1 join.
            source_rows.append(
                {
                    "path": str(path),
                    "status": "SKIP_DUPLICATE_UID",
                    "rows": len(df),
                    "uid_coverage": float(df["uid"].isin(base["uid"]).mean()),
                    "columns_added": 0,
                    "added_columns": "",
                }
            )
            continue

        useful = useful_metadata_columns(df.columns)
        if not useful:
            source_rows.append(
                {
                    "path": str(path),
                    "status": "NO_USEFUL_COLUMNS",
                    "rows": len(df),
                    "uid_coverage": float(df["uid"].isin(base["uid"]).mean()),
                    "columns_added": 0,
                    "added_columns": "",
                }
            )
            continue

        coverage = float(base["uid"].isin(df["uid"]).mean())
        if coverage < 0.50:
            source_rows.append(
                {
                    "path": str(path),
                    "status": "SKIP_LOW_COVERAGE",
                    "rows": len(df),
                    "uid_coverage": coverage,
                    "columns_added": 0,
                    "added_columns": "",
                }
            )
            continue

        prefix = sanitize_prefix(path)
        rename = {c: f"{prefix}__{c}" for c in useful}
        tmp = df[["uid"] + useful].rename(columns=rename)
        merged = merged.merge(tmp, on="uid", how="left", validate="one_to_one")
        source_rows.append(
            {
                "path": str(path),
                "status": "USED",
                "rows": len(df),
                "uid_coverage": coverage,
                "columns_added": len(rename),
                "added_columns": "|".join(rename.values()),
            }
        )

    return merged, pd.DataFrame(source_rows)


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    columns = list(out.columns)

    # Generic XYZ signature discovery for spacing/shape/dimension columns.
    for token, out_name in (("spacing", "derived_spacing_signature"), ("shape", "derived_shape_signature")):
        if out_name in out.columns:
            continue
        candidates = [c for c in columns if token in c.lower()]
        triples: dict[str, dict[str, str]] = {}
        for c in candidates:
            low = c.lower()
            axis = None
            for a in ("x", "y", "z"):
                if re.search(rf"(?:^|[_\-]){a}(?:[_\-]|$)", low) or low.endswith(a):
                    axis = a
                    break
            if axis is None:
                continue
            stem = re.sub(r"(?:^|[_\-])[xyz](?:[_\-]|$)", "_", low)
            stem = re.sub(r"[xyz]$", "", stem)
            triples.setdefault(stem, {})[axis] = c

        valid = [v for v in triples.values() if set(v) == {"x", "y", "z"}]
        if valid:
            triple = valid[0]
            vals = []
            for _, row in out.iterrows():
                parts = []
                ok = True
                for axis in ("x", "y", "z"):
                    val = pd.to_numeric(pd.Series([row[triple[axis]]]), errors="coerce").iloc[0]
                    if pd.isna(val):
                        ok = False
                        break
                    parts.append(f"{float(val):.3f}")
                vals.append("x".join(parts) if ok else np.nan)
            out[out_name] = vals

    return out


def candidate_group_features(
    df: pd.DataFrame,
    *,
    max_levels: int,
    quantile_bins: int,
    min_group_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    excluded = {
        "uid",
        "fold",
        "is_pathologic",
        "ens328_raw_probability",
        "ens328r_raw_probability",
        "ens328_cf_probability",
        "ens328r_cf_probability",
        "ens328_loss",
        "ens328r_loss",
        "delta_loss_r_minus_old",
    }

    features: list[str] = ["fold"]

    for c in list(work.columns):
        if c in excluded or c == "fold":
            continue
        s = work[c]
        nonnull = s.dropna()
        if nonnull.empty:
            continue

        n_unique = int(nonnull.nunique(dropna=True))
        if n_unique < 2:
            continue

        # Direct categorical/low-cardinality grouping.
        if (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_bool_dtype(s)
            or n_unique <= max_levels
        ):
            counts = s.astype("string").value_counts(dropna=True)
            if int((counts >= min_group_size).sum()) >= 2:
                features.append(c)
            continue

        # Numeric continuous variable -> quantile bins.
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().sum() < min_group_size * 2:
            continue
        if numeric.nunique(dropna=True) <= max_levels:
            continue
        qcol = f"{c}__q{quantile_bins}"
        try:
            work[qcol] = pd.qcut(numeric, q=quantile_bins, duplicates="drop").astype("string")
        except Exception:
            continue
        counts = work[qcol].value_counts(dropna=True)
        if int((counts >= min_group_size).sum()) >= 2:
            features.append(qcol)

    # Stable de-duplicate, prevent an explosion if auto-discovery found many files.
    deduped: list[str] = []
    for f in features:
        if f not in deduped:
            deduped.append(f)
    return work, deduped[:120]


def calibration_bins(
    y: np.ndarray,
    p: np.ndarray,
    *,
    predictor: str,
    n_bins: int,
) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        rows.append(
            {
                "predictor": predictor,
                "bin": i,
                "lower": lo,
                "upper": hi,
                "n": int(mask.sum()),
                "mean_probability": float(np.mean(p[mask])),
                "observed_positive_rate": float(np.mean(y[mask])),
                "calibration_gap_observed_minus_predicted": float(
                    np.mean(y[mask]) - np.mean(p[mask])
                ),
            }
        )
    return rows


def paired_bootstrap_delta(
    y: np.ndarray,
    p_old: np.ndarray,
    p_new: np.ndarray,
    *,
    replicates: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(RNG_SEED)
    n = len(y)
    observed = binary_log_loss(y, p_new) - binary_log_loss(y, p_old)
    if replicates <= 0:
        return {"replicates": 0, "observed_delta": observed}

    deltas = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        idx = rng.integers(0, n, size=n)
        deltas[i] = binary_log_loss(y[idx], p_new[idx]) - binary_log_loss(
            y[idx], p_old[idx]
        )

    return {
        "replicates": int(replicates),
        "seed": RNG_SEED,
        "observed_delta_log_loss_r_minus_old": float(observed),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "fraction_ens328r_better": float(np.mean(deltas < 0.0)),
    }


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = (
        args.project_root or script_path.parents[2]
    ).expanduser().resolve()
    add_project_to_path(project_root)

    try:
        from src.calibration.probability_calibration import ProbabilityCalibrator
        from src.evaluation.metrics import compute_binary_metrics
    except Exception as exc:
        raise RuntimeError(
            "Could not import authoritative project calibration/evaluation modules. "
            f"Project root resolved to {project_root}. Original error: {exc}"
        ) from exc

    ens328_path = resolve(
        project_root,
        args.ens328_oof
        or Path("data/final_predictor_data/ENS328/final_ensemble_oof_for_calibration.csv"),
    )
    ens328r_path = resolve(
        project_root,
        args.ens328r_oof
        or Path("data/final_predictor_data/ENS328R/final_ensemble_oof_for_calibration.csv"),
    )
    fold_assignments_path = resolve(
        project_root,
        args.fold_assignments
        or Path(
            "data/preprocessing_supervised_data/step11_create_freeze_cv_splits_data/"
            "fold_assignments.csv"
        ),
    )
    output_dir = resolve(
        project_root,
        args.output_dir
        or Path("data/generalization_validation_data/step33_hidden_transfer_diagnosis"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 92)
    print("STEP 33 — HIDDEN-TRANSFER / ACQUISITION-STRATIFIED OOF DIAGNOSIS")
    print("=" * 92)
    print(f"Project root            : {project_root}")
    print(f"ENS328 OOF              : {ens328_path}")
    print(f"ENS328R OOF             : {ens328r_path}")
    print(f"Step-11 fold assignments: {fold_assignments_path}")
    print(f"Output                  : {output_dir}")
    print("Training                : NONE")
    print("Checkpoint loading      : NONE")
    print("Hidden per-case labels  : NEVER USED")
    print("Final calibration       : cross-fitted temperature via authoritative module")
    print()

    fold_assignments = load_fold_assignments(
        fold_assignments_path,
        args.expected_subjects,
        args.expected_folds,
    )
    old = load_oof(
        ens328_path,
        args.expected_subjects,
        args.expected_folds,
        fold_assignments,
    )
    new = load_oof(
        ens328r_path,
        args.expected_subjects,
        args.expected_folds,
        fold_assignments,
    )

    merged = old.rename(columns={"probability": "ens328_raw_probability"}).merge(
        new.rename(columns={"probability": "ens328r_raw_probability"}),
        on="uid",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )

    for col in ("fold", "is_pathologic"):
        a = merged[f"{col}_old"].to_numpy()
        b = merged[f"{col}_new"].to_numpy()
        if not np.array_equal(a, b):
            raise RuntimeError(f"ENS328 and ENS328R disagree on {col}")
        merged[col] = a
        merged.drop(columns=[f"{col}_old", f"{col}_new"], inplace=True)

    old_for_cf = merged[["uid", "fold", "is_pathologic", "ens328_raw_probability"]].rename(
        columns={"ens328_raw_probability": "probability"}
    )
    new_for_cf = merged[["uid", "fold", "is_pathologic", "ens328r_raw_probability"]].rename(
        columns={"ens328r_raw_probability": "probability"}
    )

    old_cf, old_states = cross_fit_temperature(
        old_for_cf,
        expected_folds=args.expected_folds,
        ProbabilityCalibrator=ProbabilityCalibrator,
    )
    new_cf, new_states = cross_fit_temperature(
        new_for_cf,
        expected_folds=args.expected_folds,
        ProbabilityCalibrator=ProbabilityCalibrator,
    )
    merged["ens328_cf_probability"] = old_cf
    merged["ens328r_cf_probability"] = new_cf

    y = merged["is_pathologic"].to_numpy(dtype=np.int64)
    merged["ens328_loss"] = per_subject_log_loss(y, old_cf)
    merged["ens328r_loss"] = per_subject_log_loss(y, new_cf)
    merged["delta_loss_r_minus_old"] = merged["ens328r_loss"] - merged["ens328_loss"]

    calibrator_payload = {
        "policy": "For held-out fold f, fit final temperature on OOF folds != f and apply to f.",
        "ENS328": old_states,
        "ENS328R": new_states,
    }
    (output_dir / "step33_crossfit_calibrators.json").write_text(
        json.dumps(calibrator_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Metadata discovery / join.
    if args.metadata_csv:
        metadata_paths = [resolve(project_root, p) for p in args.metadata_csv]
    elif args.no_auto_metadata:
        metadata_paths = []
    else:
        metadata_paths = discover_metadata_files(
            project_root,
            max_files=args.max_auto_metadata_files,
        )

    merged, source_df = merge_metadata(merged, metadata_paths)
    merged = add_composite_features(merged)
    source_df.to_csv(output_dir / "step33_metadata_sources.csv", index=False)

    # Overall metrics.
    overall_old = safe_metrics(
        y,
        merged["ens328_cf_probability"].to_numpy(),
        ece_bins=args.ece_bins,
        compute_binary_metrics=compute_binary_metrics,
    )
    overall_new = safe_metrics(
        y,
        merged["ens328r_cf_probability"].to_numpy(),
        ece_bins=args.ece_bins,
        compute_binary_metrics=compute_binary_metrics,
    )
    overall_rows = []
    for predictor, metrics in (("ENS328", overall_old), ("ENS328R", overall_new)):
        row = {"predictor": predictor}
        row.update(metrics)
        overall_rows.append(row)
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(output_dir / "step33_overall_metrics.csv", index=False, float_format="%.9f")

    # Fold metrics are always included.
    fold_rows = []
    for fold, subset in merged.groupby("fold", sort=True):
        fold_rows.append(
            comparison_row(
                subset,
                feature="fold",
                value=str(fold),
                ece_bins=args.ece_bins,
                compute_binary_metrics=compute_binary_metrics,
            )
        )
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "step33_fold_metrics.csv", index=False, float_format="%.9f")

    # Build acquisition/preprocessing subgroup features.
    grouped_df, features = candidate_group_features(
        merged,
        max_levels=args.max_categorical_levels,
        quantile_bins=args.continuous_quantile_bins,
        min_group_size=args.min_group_size,
    )

    subgroup_rows: list[dict[str, Any]] = []
    for feature in features:
        values = grouped_df[feature].astype("string")
        counts = values.value_counts(dropna=True)
        valid_values = counts[counts >= args.min_group_size].index.tolist()
        if len(valid_values) < 2 and feature != "fold":
            continue
        for value in valid_values:
            mask = values == value
            subset = grouped_df.loc[mask]
            subgroup_rows.append(
                comparison_row(
                    subset,
                    feature=feature,
                    value=str(value),
                    ece_bins=args.ece_bins,
                    compute_binary_metrics=compute_binary_metrics,
                )
            )

    subgroup_df = pd.DataFrame(subgroup_rows)
    if not subgroup_df.empty:
        subgroup_df = subgroup_df.sort_values(
            ["ens328r_log_loss", "n"], ascending=[False, False], kind="stable"
        ).reset_index(drop=True)
    subgroup_df.to_csv(
        output_dir / "step33_subgroup_metrics.csv",
        index=False,
        float_format="%.9f",
    )

    # Feature-level robustness summary.
    feature_rows: list[dict[str, Any]] = []
    if not subgroup_df.empty:
        for feature, part in subgroup_df.groupby("feature", sort=False):
            if len(part) < 2:
                continue
            weights = part["n"].to_numpy(dtype=float)
            weights = weights / weights.sum()
            feature_rows.append(
                {
                    "feature": feature,
                    "groups_evaluated": int(len(part)),
                    "subjects_represented": int(part["n"].sum()),
                    "weighted_ens328_log_loss": float(np.sum(weights * part["ens328_log_loss"])),
                    "weighted_ens328r_log_loss": float(np.sum(weights * part["ens328r_log_loss"])),
                    "weighted_delta_r_minus_old": float(
                        np.sum(weights * part["delta_log_loss_r_minus_old"])
                    ),
                    "worst_group_ens328_log_loss": float(part["ens328_log_loss"].max()),
                    "worst_group_ens328r_log_loss": float(part["ens328r_log_loss"].max()),
                    "worst_group_delta_r_minus_old": float(
                        part.loc[part["ens328r_log_loss"].idxmax(), "delta_log_loss_r_minus_old"]
                    ),
                    "group_ll_sd_ens328": float(part["ens328_log_loss"].std(ddof=0)),
                    "group_ll_sd_ens328r": float(part["ens328r_log_loss"].std(ddof=0)),
                    "groups_where_ens328r_better": int(part["ens328r_better_log_loss"].sum()),
                    "fraction_groups_ens328r_better": float(part["ens328r_better_log_loss"].mean()),
                }
            )
    feature_df = pd.DataFrame(feature_rows)
    if not feature_df.empty:
        feature_df = feature_df.sort_values(
            ["worst_group_ens328r_log_loss", "weighted_delta_r_minus_old"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
    feature_df.to_csv(
        output_dir / "step33_feature_summary.csv",
        index=False,
        float_format="%.9f",
    )

    # Calibration bins.
    cal_rows = []
    cal_rows.extend(
        calibration_bins(
            y,
            merged["ens328_cf_probability"].to_numpy(),
            predictor="ENS328",
            n_bins=args.ece_bins,
        )
    )
    cal_rows.extend(
        calibration_bins(
            y,
            merged["ens328r_cf_probability"].to_numpy(),
            predictor="ENS328R",
            n_bins=args.ece_bins,
        )
    )
    pd.DataFrame(cal_rows).to_csv(
        output_dir / "step33_calibration_bins.csv",
        index=False,
        float_format="%.9f",
    )

    # Paired bootstrap is diagnostic only; it does not remove the known post-hoc
    # development dependence of ENS328R.
    bootstrap = paired_bootstrap_delta(
        y,
        merged["ens328_cf_probability"].to_numpy(),
        merged["ens328r_cf_probability"].to_numpy(),
        replicates=args.bootstrap_replicates,
    )
    (output_dir / "step33_bootstrap_delta.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Subject-level file: prediction/loss columns plus joined metadata.
    grouped_df.sort_values("uid", kind="stable").to_csv(
        output_dir / "step33_subject_level_predictions.csv",
        index=False,
        float_format="%.12f",
    )

    old_ll = float(overall_old["log_loss"])
    new_ll = float(overall_new["log_loss"])
    summary: dict[str, Any] = {
        "status": "PASS",
        "step": 33,
        "description": "Hidden-transfer / acquisition-stratified OOF diagnosis",
        "subjects": int(len(merged)),
        "folds": int(args.expected_folds),
        "neural_network_training": False,
        "checkpoint_loading": False,
        "hidden_subject_labels_used": False,
        "cross_fitted_final_temperature": True,
        "authoritative_calibrator": "src.calibration.probability_calibration.ProbabilityCalibrator",
        "ens328_crossfitted_log_loss": old_ll,
        "ens328r_crossfitted_log_loss": new_ll,
        "delta_log_loss_r_minus_old": new_ll - old_ll,
        "metadata_files_requested_or_discovered": len(metadata_paths),
        "metadata_files_used": int((source_df.get("status", pd.Series(dtype=str)) == "USED").sum()),
        "group_features_evaluated": int(subgroup_df["feature"].nunique()) if not subgroup_df.empty else 0,
        "subgroups_evaluated": int(len(subgroup_df)),
        "bootstrap": bootstrap,
        "important_interpretation": (
            "This is a diagnostic analysis on training OOF data. It must not be treated "
            "as independent proof that a change will improve the hidden competition set."
        ),
    }

    if args.hidden_ens328_logloss is not None:
        summary["hidden_ens328_log_loss"] = float(args.hidden_ens328_logloss)
        summary["ens328_hidden_minus_oof_gap"] = float(args.hidden_ens328_logloss - old_ll)
    if args.hidden_ens328r_logloss is not None:
        summary["hidden_ens328r_log_loss"] = float(args.hidden_ens328r_logloss)
        summary["ens328r_hidden_minus_oof_gap"] = float(args.hidden_ens328r_logloss - new_ll)
    if args.hidden_ens328_logloss is not None and args.hidden_ens328r_logloss is not None:
        summary["hidden_delta_r_minus_old"] = float(
            args.hidden_ens328r_logloss - args.hidden_ens328_logloss
        )

    (output_dir / "step33_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Human-readable ranked report.
    report_lines = [
        "STEP 33 — HIDDEN-TRANSFER / ACQUISITION-STRATIFIED OOF DIAGNOSIS",
        "=" * 78,
        f"Subjects: {len(merged)}",
        f"ENS328  cross-fitted final-calibration LL: {old_ll:.6f}",
        f"ENS328R cross-fitted final-calibration LL: {new_ll:.6f}",
        f"Delta ENS328R - ENS328: {new_ll - old_ll:+.6f}",
        f"Metadata files used: {summary['metadata_files_used']}",
        f"Group features evaluated: {summary['group_features_evaluated']}",
        f"Subgroups evaluated: {summary['subgroups_evaluated']}",
        "",
        "Bootstrap diagnostic:",
        json.dumps(bootstrap, indent=2),
        "",
    ]
    if args.hidden_ens328_logloss is not None and args.hidden_ens328r_logloss is not None:
        report_lines.extend(
            [
                "Observed competition aggregate scores (for context only; not used for fitting):",
                f"ENS328  hidden LL: {args.hidden_ens328_logloss:.6f}",
                f"ENS328R hidden LL: {args.hidden_ens328r_logloss:.6f}",
                f"Hidden delta: {args.hidden_ens328r_logloss - args.hidden_ens328_logloss:+.6f}",
                "",
            ]
        )

    if not feature_df.empty:
        report_lines.extend(
            [
                "TOP FEATURES BY WORST ENS328R SUBGROUP LOG LOSS",
                "-" * 78,
                feature_df.head(20).to_string(index=False),
                "",
            ]
        )
    if not subgroup_df.empty:
        report_lines.extend(
            [
                "WORST ENS328R SUBGROUPS",
                "-" * 78,
                subgroup_df.head(30).to_string(index=False),
                "",
                "SUBGROUPS WHERE ENS328R IMPROVED MOST",
                "-" * 78,
                subgroup_df.sort_values("delta_log_loss_r_minus_old", ascending=True)
                .head(30)
                .to_string(index=False),
                "",
                "SUBGROUPS WHERE ENS328R WORSENED MOST",
                "-" * 78,
                subgroup_df.sort_values("delta_log_loss_r_minus_old", ascending=False)
                .head(30)
                .to_string(index=False),
                "",
            ]
        )

    report_lines.extend(
        [
            "INTERPRETATION RULE",
            "-" * 78,
            "Use this step to identify systematic acquisition/calibration weaknesses.",
            "Do NOT choose a new submission merely because one subgroup or one temperature",
            "looks favorable in these same OOF subjects.",
            "",
            "STATUS: PASS",
            "No neural network was trained or modified.",
        ]
    )
    (output_dir / "step33_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("OVERALL")
    print(f"ENS328  CF final-temp LL : {old_ll:.6f}")
    print(f"ENS328R CF final-temp LL : {new_ll:.6f}")
    print(f"Delta R - old            : {new_ll - old_ll:+.6f}")
    print(f"Metadata files used      : {summary['metadata_files_used']}")
    print(f"Group features evaluated : {summary['group_features_evaluated']}")
    print(f"Subgroups evaluated      : {summary['subgroups_evaluated']}")
    if bootstrap.get("replicates", 0):
        ci = bootstrap["ci95"]
        print(
            "Paired bootstrap delta CI : "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}]  "
            f"P(R better)={bootstrap['fraction_ens328r_better']:.4f}"
        )
    print()
    print(f"Saved: {output_dir / 'step33_report.txt'}")
    print(f"Saved: {output_dir / 'step33_subgroup_metrics.csv'}")
    print(f"Saved: {output_dir / 'step33_feature_summary.csv'}")
    print("\nSTEP 33 STATUS: PASS")
    print("No neural network was trained or modified.\n")


if __name__ == "__main__":
    main()
