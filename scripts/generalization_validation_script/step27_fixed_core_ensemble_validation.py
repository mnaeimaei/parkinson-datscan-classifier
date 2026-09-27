#!/usr/bin/env python3
"""
STEP 27 — FIXED ROBUST-CORE ENSEMBLE VALIDATION

Purpose
-------
Compare three FIXED ensemble structures suggested by Step 26:

    CORE3   = P1 + P2 + P3
    CORE4   = P1 + P2 + P3 + P8
    ENS328  = P1 + P2 + P3 + P7 + P8

All three use:
    - nested member-calibration selection inside OUTER TRAIN only;
    - equal-weight LOGIT MEAN;
    - final TEMPERATURE calibration;
    - one untouched original Step-11 outer fold for final evaluation.

There is NO 494-ensemble search in Step 27.
There is NO neural-network retraining.
There is NO modification of src/, Step 11, ENS328, or submission code.

Why this step?
--------------
Step 26 showed:
    P1, P2, P3, P8 selected in 5/5 outer folds
    P7 selected in 3/5 outer folds
    logit_mean selected in 5/5 outer folds
    temperature selected in 5/5 outer folds

Step 27 therefore asks a focused robustness question:

    Does the fixed CORE4 structure
        P1+P2+P3+P8

    preserve most of the ensemble benefit while improving fold robustness
    relative to the current ENS328 structure?

Methodological status
---------------------
This is a POST-HOC ROBUSTNESS / SENSITIVITY ANALYSIS.

The candidate structures were motivated by Step 26 results from the same
1,362 OOF subjects. Therefore Step 27 must NOT be described as a brand-new
unbiased model-selection estimate.

Its value is comparative:
    - fixed membership
    - fixed aggregation rule
    - fixed final calibration family
    - identical outer folds
    - identical nested member-calibration process

Thus differences among CORE3 / CORE4 / ENS328 are much cleaner than another
large search.

Primary comparisons
-------------------
1) CORE3 vs P1
2) CORE4 vs P1
3) fixed ENS328 vs P1
4) CORE4 vs fixed ENS328

Metrics
-------
- global Log Loss (primary)
- per-fold Log Loss
- worst-fold Log Loss
- fold Log-Loss SD/range
- AUROC / AUPRC / Brier / ECE
- paired class-stratified bootstrap confidence intervals
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    auc,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


EPS = 1e-6
METHODS = ("none", "temperature", "platt", "isotonic")
EXPECTED_IDS = tuple(f"P{i}" for i in range(1, 9))

FIXED_STRUCTURES: dict[str, tuple[str, ...]] = {
    "CORE3": ("P1", "P2", "P3"),
    "CORE4": ("P1", "P2", "P3", "P8"),
    "ENS328_FIXED": ("P1", "P2", "P3", "P7", "P8"),
}

PRIMARY_CANDIDATE = "CORE4"
CURRENT_FROZEN_STRUCTURE = "ENS328_FIXED"

STEP20_ENS328_OOF_LL = 0.255582
STEP26_NESTED_SELECTED_LL = 0.2650832308787487
P1_REFERENCE_LL = 0.29712211698055785
COMPETITION_LL = 0.2997


# =============================================================================
# CLI / filesystem
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 27: fixed CORE3/CORE4/ENS328 nested robustness comparison."
    )
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--shortlist", type=Path, default=None)
    p.add_argument("--step26-stability", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)

    p.add_argument("--outer-fold", type=int, choices=(0, 1, 2, 3, 4), default=None)
    p.add_argument("--aggregate-only", action="store_true")

    p.add_argument("--expected-subjects", type=int, default=1362)
    p.add_argument("--expected-folds", type=int, default=5)
    p.add_argument("--calibration-max-iter", type=int, default=150)
    p.add_argument("--ece-bins", type=int, default=10)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=2027)

    p.add_argument("--overwrite-outer-fold", action="store_true")
    p.add_argument("--overwrite-aggregate", action="store_true")
    return p.parse_args()


def resolve(project_root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def resolve_stored_path(project_root: Path, value: Any) -> Path:
    raw = str(value).strip()
    path = Path(raw).expanduser()

    if path.is_file():
        return path.resolve()

    if not path.is_absolute():
        candidate = (project_root / path).resolve()
        if candidate.is_file():
            return candidate

    normalized = raw.replace("\\", "/")
    if "/data/" in normalized:
        rel = normalized.split("/data/", 1)[1]
        candidate = (project_root / "data" / rel).resolve()
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve prediction file:\n  stored={raw}\n  project={project_root}"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_builtin(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# Probability / metrics
# =============================================================================

def clip_probability(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip_probability(p)
    return np.log(p) - np.log1p(-p)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit_mean(matrix: np.ndarray) -> np.ndarray:
    return sigmoid(np.mean(logit(matrix), axis=1))


def subject_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    p = clip_probability(p)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(subject_log_loss(y, p).mean())


def expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    bins: int,
) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0

    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / total) * abs(float(p[mask].mean()) - float(y[mask].mean()))

    return float(ece)


def compute_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    ece_bins: int,
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)

    if len(np.unique(y)) < 2:
        auroc = float("nan")
        auprc = float("nan")
        ap = float("nan")
    else:
        auroc = float(roc_auc_score(y, p))
        precision, recall, _ = precision_recall_curve(y, p)
        auprc = float(auc(recall, precision))
        ap = float(average_precision_score(y, p))

    return {
        "log_loss": binary_log_loss(y, p),
        "auroc": auroc,
        "auprc": auprc,
        "average_precision": ap,
        "brier_score": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, bins=ece_bins),
    }


# =============================================================================
# Input loading
# =============================================================================

def shortlist_sort_key(value: str) -> int:
    text = str(value).strip()
    if text.startswith("P") and text[1:].isdigit():
        return int(text[1:])
    return 999


def load_raw_p1_p8(
    *,
    project_root: Path,
    shortlist_path: Path,
    expected_subjects: int,
    expected_folds: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    shortlist = pd.read_csv(shortlist_path)

    required = {
        "Shortlist ID",
        "Calibration Method",
        "Cross-Fitted Prediction File",
    }
    missing = required - set(shortlist.columns)
    if missing:
        raise RuntimeError(f"Shortlist missing columns: {sorted(missing)}")

    shortlist = shortlist.copy()
    shortlist["_sort"] = shortlist["Shortlist ID"].map(shortlist_sort_key)
    shortlist = shortlist.sort_values("_sort", kind="stable").drop(columns="_sort")

    ids = shortlist["Shortlist ID"].astype(str).tolist()
    if tuple(ids) != EXPECTED_IDS:
        raise RuntimeError(f"Expected {EXPECTED_IDS}, found {ids}.")

    canonical_uid = None
    canonical_y = None
    canonical_fold = None
    raw_cols = []
    provenance = []

    for _, row in shortlist.iterrows():
        pid = str(row["Shortlist ID"])
        path = resolve_stored_path(project_root, row["Cross-Fitted Prediction File"])
        df = pd.read_csv(path)

        req = {"uid", "fold", "is_pathologic", "raw_probability"}
        miss = req - set(df.columns)
        if miss:
            raise RuntimeError(f"{pid}: missing {sorted(miss)} in {path}")

        df = df.copy()
        df["uid"] = df["uid"].astype(str)
        df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(np.int64)
        df["is_pathologic"] = pd.to_numeric(
            df["is_pathologic"], errors="raise"
        ).astype(np.int64)
        df["raw_probability"] = pd.to_numeric(
            df["raw_probability"], errors="raise"
        ).astype(np.float64)
        df = df.sort_values("uid", kind="stable").reset_index(drop=True)

        if len(df) != expected_subjects:
            raise RuntimeError(f"{pid}: expected {expected_subjects}, found {len(df)}")
        if df["uid"].duplicated().any():
            raise RuntimeError(f"{pid}: duplicate UID")
        if not np.isfinite(df["raw_probability"]).all():
            raise RuntimeError(f"{pid}: invalid raw_probability")
        if ((df["raw_probability"] < 0) | (df["raw_probability"] > 1)).any():
            raise RuntimeError(f"{pid}: probability outside [0,1]")

        if canonical_uid is None:
            canonical_uid = df["uid"].to_numpy(dtype=str)
            canonical_y = df["is_pathologic"].to_numpy(dtype=np.int64)
            canonical_fold = df["fold"].to_numpy(dtype=np.int64)
            if set(canonical_fold.tolist()) != set(range(expected_folds)):
                raise RuntimeError("Unexpected original fold IDs.")
        else:
            if not np.array_equal(canonical_uid, df["uid"].to_numpy(dtype=str)):
                raise RuntimeError(f"{pid}: UID alignment mismatch.")
            if not np.array_equal(canonical_y, df["is_pathologic"].to_numpy(dtype=np.int64)):
                raise RuntimeError(f"{pid}: label mismatch.")
            if not np.array_equal(canonical_fold, df["fold"].to_numpy(dtype=np.int64)):
                raise RuntimeError(f"{pid}: fold mismatch.")

        raw_cols.append(df["raw_probability"].to_numpy(dtype=np.float64))
        provenance.append(
            {
                "Shortlist ID": pid,
                "Original Step-16 Method": str(row["Calibration Method"]),
                "Prediction File": str(path),
                "Prediction File SHA256": sha256_file(path),
            }
        )

    return (
        ids,
        canonical_uid,
        canonical_y,
        canonical_fold,
        np.column_stack(raw_cols),
        provenance,
    )


def audit_step26_stability(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "available": False,
            "message": "Step-26 stability file not supplied/found; audit skipped.",
        }

    df = pd.read_csv(path)
    required = {
        "outer_fold",
        "selected_members",
        "selected_rule",
        "selected_final_calibration",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Step-26 stability file missing columns: {sorted(missing)}"
        )

    if len(df) != 5:
        raise RuntimeError("Expected five Step-26 outer-fold stability rows.")

    rule_ok = bool((df["selected_rule"].astype(str) == "logit_mean").all())
    temp_ok = bool(
        (df["selected_final_calibration"].astype(str) == "temperature").all()
    )

    counts = {pid: 0 for pid in EXPECTED_IDS}
    for text in df["selected_members"].astype(str):
        members = set(text.split("+"))
        for pid in EXPECTED_IDS:
            counts[pid] += int(pid in members)

    expected_core = all(counts[pid] == 5 for pid in ("P1", "P2", "P3", "P8"))
    p7_count = counts["P7"]

    if not rule_ok or not temp_ok or not expected_core:
        raise RuntimeError(
            "Step-26 stability audit does not support Step-27 fixed assumptions."
        )

    return {
        "available": True,
        "path": str(path),
        "logit_mean_selected_5_of_5": rule_ok,
        "temperature_selected_5_of_5": temp_ok,
        "member_selection_counts": counts,
        "P1_P2_P3_P8_selected_5_of_5": expected_core,
        "P7_selection_count": p7_count,
    }


# =============================================================================
# Calibration
# =============================================================================

def fit_calibrator(
    ProbabilityCalibrator,
    *,
    method: str,
    y: np.ndarray,
    p: np.ndarray,
    max_iter: int,
):
    if method == "none":
        return None

    calibrator = ProbabilityCalibrator(method=method, output_epsilon=EPS)
    calibrator.fit(
        labels=np.asarray(y, dtype=np.int64),
        probabilities=np.asarray(p, dtype=np.float64),
        max_iter=max_iter,
    )
    return calibrator


def apply_calibrator(calibrator, method: str, p: np.ndarray) -> np.ndarray:
    if method == "none":
        return np.asarray(p, dtype=np.float64).copy()
    return np.asarray(calibrator.predict_proba(p), dtype=np.float64)


def cross_fit_calibration(
    ProbabilityCalibrator,
    *,
    y: np.ndarray,
    p: np.ndarray,
    folds: np.ndarray,
    method: str,
    max_iter: int,
) -> np.ndarray:
    if method == "none":
        return np.asarray(p, dtype=np.float64).copy()

    out = np.full(len(y), np.nan, dtype=np.float64)
    for heldout in sorted(np.unique(folds).tolist()):
        fit_mask = folds != heldout
        apply_mask = folds == heldout

        if len(np.unique(y[fit_mask])) < 2:
            raise RuntimeError(
                f"Calibration fit has one class only: heldout={heldout}, method={method}"
            )

        calibrator = fit_calibrator(
            ProbabilityCalibrator,
            method=method,
            y=y[fit_mask],
            p=p[fit_mask],
            max_iter=max_iter,
        )
        out[apply_mask] = apply_calibrator(
            calibrator, method, p[apply_mask]
        )

    if not np.isfinite(out).all():
        raise RuntimeError(f"Cross-fitted calibration failed: {method}")
    return out


def select_member_calibration(
    ProbabilityCalibrator,
    *,
    ids: list[str],
    y_train: np.ndarray,
    folds_train: np.ndarray,
    raw_train: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, str], pd.DataFrame]:
    selected_cf = np.empty_like(raw_train, dtype=np.float64)
    selected_methods: dict[str, str] = {}
    rows = []

    method_order = {m: i for i, m in enumerate(METHODS)}

    for j, pid in enumerate(ids):
        candidate_rows = []
        predictions: dict[str, np.ndarray] = {}

        for method in METHODS:
            p_cf = cross_fit_calibration(
                ProbabilityCalibrator,
                y=y_train,
                p=raw_train[:, j],
                folds=folds_train,
                method=method,
                max_iter=max_iter,
            )
            predictions[method] = p_cf
            candidate_rows.append(
                {
                    "Shortlist ID": pid,
                    "Calibration Method": method,
                    "Inner Cross-Fitted Log Loss": binary_log_loss(y_train, p_cf),
                }
            )

        candidate_rows.sort(
            key=lambda r: (
                r["Inner Cross-Fitted Log Loss"],
                method_order[r["Calibration Method"]],
            )
        )
        best_method = str(candidate_rows[0]["Calibration Method"])
        selected_methods[pid] = best_method
        selected_cf[:, j] = predictions[best_method]

        for row in candidate_rows:
            row["Selected"] = row["Calibration Method"] == best_method
            rows.append(row)

    return selected_cf, selected_methods, pd.DataFrame(rows)


def fit_members_full_outer_train(
    ProbabilityCalibrator,
    *,
    ids: list[str],
    selected_methods: dict[str, str],
    y_train: np.ndarray,
    raw_train: np.ndarray,
    raw_test: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_out = np.empty_like(raw_train, dtype=np.float64)
    test_out = np.empty_like(raw_test, dtype=np.float64)
    states = {}

    for j, pid in enumerate(ids):
        method = selected_methods[pid]
        calibrator = fit_calibrator(
            ProbabilityCalibrator,
            method=method,
            y=y_train,
            p=raw_train[:, j],
            max_iter=max_iter,
        )
        train_out[:, j] = apply_calibrator(calibrator, method, raw_train[:, j])
        test_out[:, j] = apply_calibrator(calibrator, method, raw_test[:, j])

        states[pid] = {
            "method": method,
            "calibrator": (
                calibrator.to_dict()
                if calibrator is not None
                else {"method": "none"}
            ),
        }

    return train_out, test_out, states


# =============================================================================
# Fixed ensembles
# =============================================================================

def structure_indices(ids: list[str], members: tuple[str, ...]) -> list[int]:
    return [ids.index(pid) for pid in members]


def evaluate_fixed_structure(
    ProbabilityCalibrator,
    *,
    structure_name: str,
    members: tuple[str, ...],
    ids: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    folds_train: np.ndarray,
    member_cf_train: np.ndarray,
    member_fullfit_train: np.ndarray,
    member_fullfit_test: np.ndarray,
    max_iter: int,
    ece_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    idx = structure_indices(ids, members)

    # Inner-CF members provide an outer-training diagnostic that avoids using
    # full-fit member calibrators for the inner estimate.
    raw_inner_cf_ensemble = logit_mean(member_cf_train[:, idx])

    # Final calibration family is FIXED to temperature, not searched.
    inner_cf_temperature = cross_fit_calibration(
        ProbabilityCalibrator,
        y=y_train,
        p=raw_inner_cf_ensemble,
        folds=folds_train,
        method="temperature",
        max_iter=max_iter,
    )

    inner_metrics = compute_metrics(
        y_train,
        inner_cf_temperature,
        ece_bins=ece_bins,
    )

    # Refit selected member calibrators on all outer train, construct fixed
    # logit-mean ensemble, fit one temperature on all outer train, apply once
    # to untouched outer test.
    raw_fullfit_train_ensemble = logit_mean(member_fullfit_train[:, idx])
    raw_outer_test_ensemble = logit_mean(member_fullfit_test[:, idx])

    final_temperature = fit_calibrator(
        ProbabilityCalibrator,
        method="temperature",
        y=y_train,
        p=raw_fullfit_train_ensemble,
        max_iter=max_iter,
    )
    calibrated_outer_test = apply_calibrator(
        final_temperature,
        "temperature",
        raw_outer_test_ensemble,
    )

    outer_test_metrics = compute_metrics(
        y_test,
        calibrated_outer_test,
        ece_bins=ece_bins,
    )
    raw_outer_test_metrics = compute_metrics(
        y_test,
        raw_outer_test_ensemble,
        ece_bins=ece_bins,
    )

    result = {
        "structure": structure_name,
        "members": "+".join(members),
        "member_count": len(members),
        "aggregation_rule": "logit_mean",
        "final_calibration": "temperature",
        "inner_cross_fitted_temperature_metrics": inner_metrics,
        "outer_test_raw_logit_mean_metrics": raw_outer_test_metrics,
        "outer_test_temperature_metrics": outer_test_metrics,
        "final_temperature_state": final_temperature.to_dict(),
    }

    return calibrated_outer_test, result


# =============================================================================
# Outer fold
# =============================================================================

def run_outer_fold(
    ProbabilityCalibrator,
    *,
    outer_fold: int,
    output_dir: Path,
    ids: list[str],
    uids: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    raw_matrix: np.ndarray,
    provenance: list[dict[str, Any]],
    step26_audit: dict[str, Any],
    max_iter: int,
    ece_bins: int,
    overwrite: bool,
) -> None:
    fold_dir = output_dir / f"outer_fold_{outer_fold}"

    if fold_dir.exists() and any(fold_dir.iterdir()):
        if overwrite:
            shutil.rmtree(fold_dir)
        else:
            raise FileExistsError(
                f"Outer-fold output exists: {fold_dir}\n"
                "Use --overwrite-outer-fold to rebuild."
            )
    fold_dir.mkdir(parents=True, exist_ok=True)

    test_mask = folds == outer_fold
    train_mask = ~test_mask

    y_train = y[train_mask]
    y_test = y[test_mask]
    folds_train = folds[train_mask]
    raw_train = raw_matrix[train_mask]
    raw_test = raw_matrix[test_mask]

    started = time.monotonic()

    member_cf_train, selected_methods, member_results = (
        select_member_calibration(
            ProbabilityCalibrator,
            ids=ids,
            y_train=y_train,
            folds_train=folds_train,
            raw_train=raw_train,
            max_iter=max_iter,
        )
    )
    member_results.to_csv(
        fold_dir / "nested_member_calibration_results.csv",
        index=False,
        float_format="%.9f",
    )

    member_fullfit_train, member_fullfit_test, member_states = (
        fit_members_full_outer_train(
            ProbabilityCalibrator,
            ids=ids,
            selected_methods=selected_methods,
            y_train=y_train,
            raw_train=raw_train,
            raw_test=raw_test,
            max_iter=max_iter,
        )
    )

    # P1 reference remains its raw OOF probability.
    p1_index = ids.index("P1")
    p1_test = raw_test[:, p1_index]
    p1_metrics = compute_metrics(y_test, p1_test, ece_bins=ece_bins)

    prediction_table = pd.DataFrame(
        {
            "uid": uids[test_mask],
            "fold": folds[test_mask],
            "is_pathologic": y_test,
            "P1_raw_probability": p1_test,
        }
    )

    structure_results = {}
    for structure_name, members in FIXED_STRUCTURES.items():
        p_test, result = evaluate_fixed_structure(
            ProbabilityCalibrator,
            structure_name=structure_name,
            members=members,
            ids=ids,
            y_train=y_train,
            y_test=y_test,
            folds_train=folds_train,
            member_cf_train=member_cf_train,
            member_fullfit_train=member_fullfit_train,
            member_fullfit_test=member_fullfit_test,
            max_iter=max_iter,
            ece_bins=ece_bins,
        )
        structure_results[structure_name] = result
        prediction_table[f"{structure_name}_probability"] = p_test

    prediction_table["P1_subject_log_loss"] = subject_log_loss(
        y_test, p1_test
    )
    for structure_name in FIXED_STRUCTURES:
        prediction_table[f"{structure_name}_subject_log_loss"] = (
            subject_log_loss(
                y_test,
                prediction_table[f"{structure_name}_probability"].to_numpy(),
            )
        )

    prediction_table.to_csv(
        fold_dir / "outer_test_fixed_ensemble_predictions.csv",
        index=False,
        float_format="%.9f",
    )

    result_payload = {
        "status": "PASS",
        "outer_fold": outer_fold,
        "outer_train_subjects": int(train_mask.sum()),
        "outer_test_subjects": int(test_mask.sum()),
        "outer_train_folds": sorted(np.unique(folds_train).tolist()),
        "selected_member_calibration_methods": selected_methods,
        "member_fullfit_calibrators": member_states,
        "P1_outer_test_metrics": p1_metrics,
        "fixed_structure_results": structure_results,
        "step26_assumption_audit": step26_audit,
        "runtime_seconds": time.monotonic() - started,
        "no_structure_search": True,
        "aggregation_rule_fixed": "logit_mean",
        "final_calibration_fixed": "temperature",
        "methodological_status": (
            "Post-hoc robustness/sensitivity analysis. Candidate structures "
            "were motivated by Step 26 on the same OOF subjects."
        ),
    }
    write_json(
        fold_dir / "outer_fold_fixed_ensemble_result.json",
        result_payload,
    )

    print("=" * 108)
    print(f"STEP 27 — OUTER FOLD {outer_fold}")
    print("=" * 108)
    print(f"Outer train subjects : {train_mask.sum()}")
    print(f"Outer test subjects  : {test_mask.sum()}")
    print(f"P1 LL                : {p1_metrics['log_loss']:.6f}")
    for name in FIXED_STRUCTURES:
        ll = structure_results[name]["outer_test_temperature_metrics"]["log_loss"]
        print(f"{name:<20s}: {ll:.6f}")
    print(f"Saved                : {fold_dir}")
    print("=" * 108)


# =============================================================================
# Bootstrap / aggregation
# =============================================================================

def paired_class_stratified_bootstrap(
    *,
    y: np.ndarray,
    delta_subject_loss: np.ndarray,
    n_bootstrap: int,
    seed: int,
    chunk_size: int = 250,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    delta = np.asarray(delta_subject_loss, dtype=np.float64)

    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y == 1)

    if len(neg) == 0 or len(pos) == 0:
        raise RuntimeError("Bootstrap requires both classes.")

    rng = np.random.default_rng(seed)
    out = np.empty(n_bootstrap, dtype=np.float64)

    cursor = 0
    while cursor < n_bootstrap:
        b = min(chunk_size, n_bootstrap - cursor)
        neg_sample = rng.choice(neg, size=(b, len(neg)), replace=True)
        pos_sample = rng.choice(pos, size=(b, len(pos)), replace=True)

        out[cursor:cursor+b] = (
            len(neg) * delta[neg_sample].mean(axis=1)
            + len(pos) * delta[pos_sample].mean(axis=1)
        ) / len(y)

        cursor += b

    return out


def aggregate(
    *,
    output_dir: Path,
    expected_subjects: int,
    expected_folds: int,
    ece_bins: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    overwrite: bool,
) -> None:
    aggregate_targets = [
        output_dir / "step27_fixed_ensemble_oof_predictions.csv",
        output_dir / "step27_global_comparison.csv",
        output_dir / "step27_fold_comparison.csv",
        output_dir / "step27_pairwise_bootstrap.csv",
        output_dir / "step27_summary.json",
        output_dir / "step27_report.md",
    ]
    if any(p.exists() for p in aggregate_targets) and not overwrite:
        raise FileExistsError(
            "Aggregate Step-27 outputs already exist. "
            "Use --overwrite-aggregate to rebuild."
        )

    frames = []
    for fold_id in range(expected_folds):
        path = (
            output_dir
            / f"outer_fold_{fold_id}"
            / "outer_test_fixed_ensemble_predictions.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing outer-fold predictions: {path}")
        frames.append(pd.read_csv(path))

    oof = pd.concat(frames, ignore_index=True)
    oof["uid"] = oof["uid"].astype(str)

    if len(oof) != expected_subjects:
        raise RuntimeError(
            f"Expected {expected_subjects} OOF rows, found {len(oof)}."
        )
    if oof["uid"].duplicated().any():
        raise RuntimeError("Duplicate UID across Step-27 outer folds.")
    if set(oof["fold"].astype(int).unique()) != set(range(expected_folds)):
        raise RuntimeError("Step-27 OOF does not cover all original folds.")

    y = oof["is_pathologic"].to_numpy(dtype=np.int64)

    predictors = {
        "P1": oof["P1_raw_probability"].to_numpy(dtype=np.float64),
        **{
            name: oof[f"{name}_probability"].to_numpy(dtype=np.float64)
            for name in FIXED_STRUCTURES
        },
    }

    # Global metrics + fold robustness.
    global_rows = []
    fold_rows = []

    for name, p in predictors.items():
        metrics = compute_metrics(y, p, ece_bins=ece_bins)

        fold_lls = []
        for fold_id in range(expected_folds):
            mask = oof["fold"].to_numpy(dtype=np.int64) == fold_id
            fm = compute_metrics(
                y[mask], p[mask], ece_bins=ece_bins
            )
            fold_lls.append(fm["log_loss"])
            fold_rows.append(
                {
                    "predictor": name,
                    "fold": fold_id,
                    "subjects": int(mask.sum()),
                    **fm,
                }
            )

        global_rows.append(
            {
                "predictor": name,
                **metrics,
                "fold_log_loss_mean": float(np.mean(fold_lls)),
                "fold_log_loss_sd": float(np.std(fold_lls, ddof=1)),
                "fold_log_loss_min": float(np.min(fold_lls)),
                "fold_log_loss_max": float(np.max(fold_lls)),
                "fold_log_loss_range": float(np.max(fold_lls) - np.min(fold_lls)),
            }
        )

    global_df = pd.DataFrame(global_rows).sort_values(
        ["log_loss", "fold_log_loss_max"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)

    fold_df = pd.DataFrame(fold_rows).sort_values(
        ["fold", "log_loss", "predictor"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    # Prespecified pairwise comparisons.
    comparisons = [
        ("CORE3", "P1"),
        ("CORE4", "P1"),
        ("ENS328_FIXED", "P1"),
        ("CORE4", "ENS328_FIXED"),
    ]

    bootstrap_rows = []
    bootstrap_payload = {}
    n_tests = len(comparisons)
    bonf_alpha = 0.05 / n_tests
    bonf_low_pct = 100.0 * bonf_alpha / 2.0
    bonf_high_pct = 100.0 * (1.0 - bonf_alpha / 2.0)

    for i, (candidate, reference) in enumerate(comparisons):
        cand_loss = subject_log_loss(y, predictors[candidate])
        ref_loss = subject_log_loss(y, predictors[reference])
        delta = cand_loss - ref_loss

        boot = paired_class_stratified_bootstrap(
            y=y,
            delta_subject_loss=delta,
            n_bootstrap=bootstrap_replicates,
            seed=bootstrap_seed + i,
        )

        nominal_low, nominal_high = np.percentile(boot, [2.5, 97.5])
        bonf_low, bonf_high = np.percentile(
            boot, [bonf_low_pct, bonf_high_pct]
        )

        observed = float(delta.mean())

        if bonf_high < 0:
            familywise_conclusion = f"{candidate} BETTER"
        elif bonf_low > 0:
            familywise_conclusion = f"{reference} BETTER"
        else:
            familywise_conclusion = "INCONCLUSIVE"

        bootstrap_rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "observed_delta_log_loss_candidate_minus_reference": observed,
                "nominal_95_ci_lower": float(nominal_low),
                "nominal_95_ci_upper": float(nominal_high),
                "bonferroni_familywise_ci_level_percent": float(
                    100.0 * (1.0 - bonf_alpha)
                ),
                "bonferroni_ci_lower": float(bonf_low),
                "bonferroni_ci_upper": float(bonf_high),
                "familywise_conclusion": familywise_conclusion,
            }
        )
        bootstrap_payload[f"{candidate}_vs_{reference}"] = boot

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    np.savez_compressed(
        output_dir / "step27_pairwise_bootstrap_distributions.npz",
        **bootstrap_payload,
    )

    # Descriptive CORE4 vs ENS328 decision.
    lookup = global_df.set_index("predictor")
    core4_ll = float(lookup.loc["CORE4", "log_loss"])
    ens_ll = float(lookup.loc["ENS328_FIXED", "log_loss"])
    core4_worst = float(lookup.loc["CORE4", "fold_log_loss_max"])
    ens_worst = float(lookup.loc["ENS328_FIXED", "fold_log_loss_max"])
    core4_sd = float(lookup.loc["CORE4", "fold_log_loss_sd"])
    ens_sd = float(lookup.loc["ENS328_FIXED", "fold_log_loss_sd"])

    if core4_ll <= ens_ll and core4_worst <= ens_worst:
        comparison_label = "CORE4_DESCRIPTIVELY_DOMINATES_ENS328"
    elif ens_ll <= core4_ll and ens_worst <= core4_worst:
        comparison_label = "ENS328_DESCRIPTIVELY_DOMINATES_CORE4"
    else:
        comparison_label = "GLOBAL_VS_WORST_FOLD_TRADEOFF"

    oof = oof.sort_values("uid", kind="stable").reset_index(drop=True)
    oof.to_csv(
        output_dir / "step27_fixed_ensemble_oof_predictions.csv",
        index=False,
        float_format="%.9f",
    )
    global_df.to_csv(
        output_dir / "step27_global_comparison.csv",
        index=False,
        float_format="%.9f",
    )
    fold_df.to_csv(
        output_dir / "step27_fold_comparison.csv",
        index=False,
        float_format="%.9f",
    )
    bootstrap_df.to_csv(
        output_dir / "step27_pairwise_bootstrap.csv",
        index=False,
        float_format="%.9f",
    )

    best_global = str(global_df.iloc[0]["predictor"])
    best_worst_fold = str(
        global_df.sort_values(
            ["fold_log_loss_max", "log_loss"],
            ascending=[True, True],
            kind="stable",
        ).iloc[0]["predictor"]
    )

    summary = {
        "status": "PASS",
        "step": "27",
        "analysis": "fixed_robust_core_ensemble_sensitivity",
        "methodological_status": (
            "Post-hoc sensitivity analysis. CORE3/CORE4/ENS328 structures were "
            "motivated using Step-26 results from the same OOF subjects."
        ),
        "subjects": expected_subjects,
        "outer_folds": expected_folds,
        "fixed_structures": {
            name: list(members)
            for name, members in FIXED_STRUCTURES.items()
        },
        "aggregation_rule": "logit_mean_fixed",
        "final_calibration": "temperature_fixed",
        "global_results": global_df.to_dict(orient="records"),
        "best_global_log_loss_predictor": best_global,
        "best_worst_fold_predictor": best_worst_fold,
        "CORE4_vs_ENS328": {
            "comparison_label": comparison_label,
            "CORE4_log_loss": core4_ll,
            "ENS328_log_loss": ens_ll,
            "CORE4_minus_ENS328_log_loss": core4_ll - ens_ll,
            "CORE4_worst_fold_log_loss": core4_worst,
            "ENS328_worst_fold_log_loss": ens_worst,
            "CORE4_minus_ENS328_worst_fold_log_loss": core4_worst - ens_worst,
            "CORE4_fold_log_loss_sd": core4_sd,
            "ENS328_fold_log_loss_sd": ens_sd,
        },
        "pairwise_bootstrap": bootstrap_df.to_dict(orient="records"),
        "reference_values": {
            "Step20_ENS328_OOF_LL": STEP20_ENS328_OOF_LL,
            "Step26_nested_selected_LL": STEP26_NESTED_SELECTED_LL,
            "P1_OOF_LL": P1_REFERENCE_LL,
            "competition_LL": COMPETITION_LL,
        },
        "critical_limit": (
            "Do not treat the best Step-27 structure as a new unbiased winner. "
            "These structures were chosen after observing Step-26 selection "
            "patterns on the same subjects."
        ),
    }
    write_json(output_dir / "step27_summary.json", summary)

    report_lines = [
        "# Step 27 — Fixed Robust-Core Ensemble Validation",
        "",
        "## Status",
        "",
        "**PASS — CORE3, CORE4 and fixed ENS328 evaluated on the same five outer folds.**",
        "",
        "## Methodological status",
        "",
        (
            "**Post-hoc robustness/sensitivity analysis.** The structures were "
            "motivated by Step 26, which used the same 1,362 OOF subjects. "
            "Therefore this is not a new unbiased model-selection estimate."
        ),
        "",
        "What is fixed for all three ensembles:",
        "",
        "- aggregation: **equal-weight logit mean**",
        "- final calibration: **temperature**",
        "- outer folds: **original Step-11 folds**",
        "- member-calibration selection: **nested inside outer train only**",
        "- neural-network predictions: **existing raw P1–P8 OOF predictions**",
        "",
        "## Fixed structures",
        "",
        "- CORE3 = **P1 + P2 + P3**",
        "- CORE4 = **P1 + P2 + P3 + P8**",
        "- ENS328_FIXED = **P1 + P2 + P3 + P7 + P8**",
        "",
        "## Global comparison",
        "",
        global_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Per-fold comparison",
        "",
        fold_df[
            ["fold", "predictor", "subjects", "log_loss", "auroc", "auprc"]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Paired bootstrap",
        "",
        bootstrap_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## CORE4 vs current ENS328",
        "",
        f"- CORE4 global LL: **{core4_ll:.6f}**",
        f"- ENS328 global LL: **{ens_ll:.6f}**",
        f"- CORE4 − ENS328 global LL: **{core4_ll - ens_ll:+.6f}**",
        f"- CORE4 worst-fold LL: **{core4_worst:.6f}**",
        f"- ENS328 worst-fold LL: **{ens_worst:.6f}**",
        f"- CORE4 fold-LL SD: **{core4_sd:.6f}**",
        f"- ENS328 fold-LL SD: **{ens_sd:.6f}**",
        f"- Descriptive comparison: **{comparison_label}**",
        "",
        "## Important interpretation",
        "",
        (
            "If CORE4 is similar or better in global Log Loss while improving "
            "the worst fold and fold-to-fold SD, that supports the hypothesis "
            "that P7 adds unstable benefit."
        ),
        "",
        (
            "If ENS328 remains better globally and is no worse on the worst "
            "fold, there is no OOF evidence that removing P7 improves robustness."
        ),
        "",
        "## Critical limitation",
        "",
        (
            "Do **not** freeze a new competition predictor solely because one "
            "structure wins Step 27. The candidate structures themselves were "
            "defined after seeing Step-26 behavior on these same subjects."
        ),
    ]

    (output_dir / "step27_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("STEP 27 — FINAL FIXED-ENSEMBLE SUMMARY")
    print("=" * 110)
    print("Status                         : PASS")
    for _, row in global_df.iterrows():
        print(
            f"{row['predictor']:<20s} "
            f"LL={row['log_loss']:.6f} "
            f"worst-fold={row['fold_log_loss_max']:.6f} "
            f"fold-SD={row['fold_log_loss_sd']:.6f}"
        )
    print(f"Best global LL                 : {best_global}")
    print(f"Best worst-fold LL             : {best_worst_fold}")
    print(f"CORE4 vs ENS328                : {comparison_label}")
    print(f"Saved                          : {output_dir}")
    print("=" * 110)


# =============================================================================
# Main
# =============================================================================

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

    from src.calibration.probability_calibration import ProbabilityCalibrator

    shortlist_path = resolve(
        project_root,
        args.shortlist
        or Path(
            "data/calibration_validation_data/"
            "competition_shortlist_data/"
            "competition_shortlist_8.csv"
        ),
    )
    output_dir = resolve(
        project_root,
        args.output_dir
        or Path(
            "data/generalization_validation_data/"
            "step27_fixed_core_ensemble_validation"
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    step26_path = None
    if args.step26_stability is not None:
        step26_path = resolve(project_root, args.step26_stability)
    else:
        default_step26 = (
            project_root
            / "data/generalization_validation_data/"
            "step26_nested_ensemble_validation/"
            "nested_selection_stability.csv"
        )
        if default_step26.is_file():
            step26_path = default_step26

    step26_audit = audit_step26_stability(step26_path)

    ids, uids, y, folds, raw_matrix, provenance = load_raw_p1_p8(
        project_root=project_root,
        shortlist_path=shortlist_path,
        expected_subjects=args.expected_subjects,
        expected_folds=args.expected_folds,
    )

    if args.aggregate_only:
        aggregate(
            output_dir=output_dir,
            expected_subjects=args.expected_subjects,
            expected_folds=args.expected_folds,
            ece_bins=args.ece_bins,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            overwrite=args.overwrite_aggregate,
        )
        return 0

    if args.outer_fold is None:
        raise ValueError(
            "Provide --outer-fold 0..4 or use --aggregate-only."
        )

    run_outer_fold(
        ProbabilityCalibrator,
        outer_fold=int(args.outer_fold),
        output_dir=output_dir,
        ids=ids,
        uids=uids,
        y=y,
        folds=folds,
        raw_matrix=raw_matrix,
        provenance=provenance,
        step26_audit=step26_audit,
        max_iter=args.calibration_max_iter,
        ece_bins=args.ece_bins,
        overwrite=args.overwrite_outer_fold,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
