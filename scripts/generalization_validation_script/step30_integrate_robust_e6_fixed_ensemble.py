#!/usr/bin/env python3
"""
STEP 30 — FIXED INTEGRATION TEST FOR ROBUST E6

Compare three fixed ensemble structures:

CURRENT_ENS328
    P1 + P2 + P3 + P7 + P8

ROBUST_REPLACE_P1
    R_E6 + P2 + P3 + P7 + P8

ROBUST_ADD
    P1 + R_E6 + P2 + P3 + P7 + P8

where R_E6 is the Step-29 robust E6-S5 model.

All three structures use the same procedure:
- raw OOF member probabilities as inputs;
- outer original Step-11 fold held untouched;
- member calibration method selected inside OUTER TRAIN only from:
      none / temperature / platt / isotonic
- selected member calibrators refit on all OUTER TRAIN;
- fixed equal-weight logit mean;
- fixed final temperature calibration;
- one prediction on OUTER TEST.

No neural-network training is performed.
No 494-ensemble search is performed.
No src/ file is modified.
No submission artifact is modified.

IMPORTANT METHODOLOGICAL STATUS
-------------------------------
Step 30 is a POST-HOC INTEGRATION / SENSITIVITY ANALYSIS.

The robust-E6 training strategy was accepted after observing Step-28/29
results from these same subjects. Therefore Step 30 is NOT a fresh unbiased
estimate of a newly selected final model.

Its purpose is narrower:
- determine whether robust E6 appears complementary to the current ensemble;
- compare replacement vs addition under a tightly controlled fixed design;
- decide whether a more expensive deployment/finalization experiment is
  justified.

Primary pairwise comparisons
----------------------------
1. ROBUST_REPLACE_P1 vs CURRENT_ENS328
2. ROBUST_ADD        vs CURRENT_ENS328
3. ROBUST_ADD        vs ROBUST_REPLACE_P1

Supporting references
---------------------
- raw P1 individual
- raw robust E6 individual
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
EXPECTED_P_IDS = tuple(f"P{i}" for i in range(1, 9))
ROBUST_ID = "R_E6"

FIXED_STRUCTURES: dict[str, tuple[str, ...]] = {
    "CURRENT_ENS328": ("P1", "P2", "P3", "P7", "P8"),
    "ROBUST_REPLACE_P1": (ROBUST_ID, "P2", "P3", "P7", "P8"),
    "ROBUST_ADD": ("P1", ROBUST_ID, "P2", "P3", "P7", "P8"),
}

DEFAULT_SHORTLIST = Path(
    "data/calibration_validation_data/"
    "competition_shortlist_data/"
    "competition_shortlist_8.csv"
)

DEFAULT_ROBUST_OOF = Path(
    "data/generalization_validation_data/"
    "step29_e6s5_robust_step11_confirmation/"
    "step29_robust_oof_predictions.csv"
)

DEFAULT_STEP27_GLOBAL = Path(
    "data/generalization_validation_data/"
    "step27_fixed_core_ensemble_validation/"
    "step27_global_comparison.csv"
)

DEFAULT_OUTPUT = Path(
    "data/generalization_validation_data/"
    "step30_robust_e6_ensemble_integration"
)


# =============================================================================
# CLI / filesystem
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 30: fixed robust-E6 integration into ENS328."
    )
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--shortlist", type=Path, default=None)
    p.add_argument("--robust-oof", type=Path, default=None)
    p.add_argument("--step27-global", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)

    p.add_argument(
        "--outer-fold",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=None,
    )
    p.add_argument("--aggregate-only", action="store_true")

    p.add_argument("--expected-subjects", type=int, default=1362)
    p.add_argument("--expected-folds", type=int, default=5)
    p.add_argument("--calibration-max-iter", type=int, default=150)
    p.add_argument("--ece-bins", type=int, default=10)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=2030)

    p.add_argument("--overwrite-outer-fold", action="store_true")
    p.add_argument("--overwrite-aggregate", action="store_true")
    return p.parse_args()


def resolve(root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def resolve_stored_path(root: Path, value: Any) -> Path:
    raw = str(value).strip()
    path = Path(raw).expanduser()

    if path.is_file():
        return path.resolve()

    if not path.is_absolute():
        candidate = (root / path).resolve()
        if candidate.is_file():
            return candidate

    normalized = raw.replace("\\", "/")
    if "/data/" in normalized:
        rel = normalized.split("/data/", 1)[1]
        candidate = (root / "data" / rel).resolve()
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve stored path:\n"
        f"  stored={raw}\n"
        f"  project={root}"
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
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
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
    value = 0.0

    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])

        n = int(mask.sum())
        if n == 0:
            continue

        value += (n / total) * abs(
            float(p[mask].mean()) - float(y[mask].mean())
        )

    return float(value)


def compute_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    ece_bins: int,
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)

    precision, recall, _ = precision_recall_curve(y, p)

    return {
        "log_loss": binary_log_loss(y, p),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p, bins=ece_bins),
    }


# =============================================================================
# Load P1-P8 + robust E6
# =============================================================================

def shortlist_sort_key(value: str) -> int:
    text = str(value).strip()
    if text.startswith("P") and text[1:].isdigit():
        return int(text[1:])
    return 999


def load_member_matrix(
    *,
    project_root: Path,
    shortlist_path: Path,
    robust_oof_path: Path,
    expected_subjects: int,
    expected_folds: int,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    if not shortlist_path.is_file():
        raise FileNotFoundError(f"Shortlist missing: {shortlist_path}")
    if not robust_oof_path.is_file():
        raise FileNotFoundError(f"Step-29 robust OOF missing: {robust_oof_path}")

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

    p_ids = shortlist["Shortlist ID"].astype(str).tolist()
    if tuple(p_ids) != EXPECTED_P_IDS:
        raise RuntimeError(f"Expected {EXPECTED_P_IDS}, found {p_ids}.")

    canonical_uid = None
    canonical_y = None
    canonical_fold = None
    columns = []
    provenance: list[dict[str, Any]] = []

    for _, row in shortlist.iterrows():
        pid = str(row["Shortlist ID"])
        path = resolve_stored_path(
            project_root,
            row["Cross-Fitted Prediction File"],
        )
        df = pd.read_csv(path)

        required_pred = {
            "uid",
            "fold",
            "is_pathologic",
            "raw_probability",
        }
        miss = required_pred - set(df.columns)
        if miss:
            raise RuntimeError(
                f"{pid}: missing {sorted(miss)} in {path}"
            )

        df = df.copy()
        df["uid"] = df["uid"].astype(str)
        df["fold"] = pd.to_numeric(
            df["fold"], errors="raise"
        ).astype(np.int64)
        df["is_pathologic"] = pd.to_numeric(
            df["is_pathologic"], errors="raise"
        ).astype(np.int64)
        df["raw_probability"] = pd.to_numeric(
            df["raw_probability"], errors="raise"
        ).astype(np.float64)

        df = df.sort_values("uid", kind="stable").reset_index(drop=True)

        if len(df) != expected_subjects:
            raise RuntimeError(
                f"{pid}: expected {expected_subjects} rows, found {len(df)}."
            )
        if df["uid"].duplicated().any():
            raise RuntimeError(f"{pid}: duplicate UID.")
        if not np.isfinite(df["raw_probability"]).all():
            raise RuntimeError(f"{pid}: invalid probabilities.")
        if ((df["raw_probability"] < 0) | (df["raw_probability"] > 1)).any():
            raise RuntimeError(f"{pid}: probabilities outside [0,1].")

        if canonical_uid is None:
            canonical_uid = df["uid"].to_numpy(dtype=str)
            canonical_y = df["is_pathologic"].to_numpy(dtype=np.int64)
            canonical_fold = df["fold"].to_numpy(dtype=np.int64)

            if set(canonical_fold.tolist()) != set(range(expected_folds)):
                raise RuntimeError("Original P1-P8 folds are not 0..4.")
        else:
            if not np.array_equal(
                canonical_uid,
                df["uid"].to_numpy(dtype=str),
            ):
                raise RuntimeError(f"{pid}: UID mismatch.")
            if not np.array_equal(
                canonical_y,
                df["is_pathologic"].to_numpy(dtype=np.int64),
            ):
                raise RuntimeError(f"{pid}: label mismatch.")
            if not np.array_equal(
                canonical_fold,
                df["fold"].to_numpy(dtype=np.int64),
            ):
                raise RuntimeError(f"{pid}: fold mismatch.")

        columns.append(
            df["raw_probability"].to_numpy(dtype=np.float64)
        )
        provenance.append(
            {
                "member_id": pid,
                "source": "Step16 shortlist raw OOF",
                "original_step16_calibration_method":
                    str(row["Calibration Method"]),
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )

    robust = pd.read_csv(robust_oof_path)
    required_robust = {
        "uid",
        "is_pathologic",
        "robust_probability",
        "fold",
    }
    missing_robust = required_robust - set(robust.columns)
    if missing_robust:
        raise RuntimeError(
            f"Step-29 robust OOF missing columns: {sorted(missing_robust)}"
        )

    robust = robust.copy()
    robust["uid"] = robust["uid"].astype(str)
    robust["is_pathologic"] = pd.to_numeric(
        robust["is_pathologic"], errors="raise"
    ).astype(np.int64)
    robust["fold"] = pd.to_numeric(
        robust["fold"], errors="raise"
    ).astype(np.int64)
    robust["robust_probability"] = pd.to_numeric(
        robust["robust_probability"], errors="raise"
    ).astype(np.float64)
    robust = robust.sort_values("uid", kind="stable").reset_index(drop=True)

    if len(robust) != expected_subjects:
        raise RuntimeError(
            f"Robust E6 expected {expected_subjects} rows, found {len(robust)}."
        )
    if robust["uid"].duplicated().any():
        raise RuntimeError("Robust E6 has duplicate UID.")
    if not np.isfinite(robust["robust_probability"]).all():
        raise RuntimeError("Robust E6 contains invalid probabilities.")
    if ((robust["robust_probability"] < 0) | (robust["robust_probability"] > 1)).any():
        raise RuntimeError("Robust E6 probabilities outside [0,1].")

    if not np.array_equal(
        canonical_uid,
        robust["uid"].to_numpy(dtype=str),
    ):
        raise RuntimeError("Robust E6 UID alignment differs from P1-P8.")
    if not np.array_equal(
        canonical_y,
        robust["is_pathologic"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Robust E6 labels differ from P1-P8.")
    if not np.array_equal(
        canonical_fold,
        robust["fold"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError(
            "Robust E6 fold assignment differs from original Step-11 P1-P8."
        )

    columns.append(
        robust["robust_probability"].to_numpy(dtype=np.float64)
    )
    provenance.append(
        {
            "member_id": ROBUST_ID,
            "source": "Step29 robust E6 raw OOF",
            "path": str(robust_oof_path),
            "sha256": sha256_file(robust_oof_path),
        }
    )

    ids = p_ids + [ROBUST_ID]
    matrix = np.column_stack(columns)

    return (
        ids,
        canonical_uid,
        canonical_y,
        canonical_fold,
        matrix,
        provenance,
    )


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

    calibrator = ProbabilityCalibrator(
        method=method,
        output_epsilon=EPS,
    )
    calibrator.fit(
        labels=np.asarray(y, dtype=np.int64),
        probabilities=np.asarray(p, dtype=np.float64),
        max_iter=max_iter,
    )
    return calibrator


def apply_calibrator(
    calibrator,
    method: str,
    p: np.ndarray,
) -> np.ndarray:
    if method == "none":
        return np.asarray(p, dtype=np.float64).copy()

    return np.asarray(
        calibrator.predict_proba(p),
        dtype=np.float64,
    )


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
                f"Calibration fit has one class only: "
                f"heldout={heldout}, method={method}"
            )

        calibrator = fit_calibrator(
            ProbabilityCalibrator,
            method=method,
            y=y[fit_mask],
            p=p[fit_mask],
            max_iter=max_iter,
        )
        out[apply_mask] = apply_calibrator(
            calibrator,
            method,
            p[apply_mask],
        )

    if not np.isfinite(out).all():
        raise RuntimeError(
            f"Cross-fitted calibration failed for {method}."
        )

    return out


def select_member_calibration(
    ProbabilityCalibrator,
    *,
    ids: list[str],
    y_train: np.ndarray,
    folds_train: np.ndarray,
    raw_train: np.ndarray,
    max_iter: int,
) -> tuple[
    np.ndarray,
    dict[str, str],
    pd.DataFrame,
]:
    selected_cf = np.empty_like(raw_train, dtype=np.float64)
    selected_methods: dict[str, str] = {}
    rows = []

    method_order = {
        method: i for i, method in enumerate(METHODS)
    }

    for j, member_id in enumerate(ids):
        candidates = []
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
            candidates.append(
                {
                    "member_id": member_id,
                    "calibration_method": method,
                    "inner_cross_fitted_log_loss":
                        binary_log_loss(y_train, p_cf),
                }
            )

        candidates.sort(
            key=lambda row: (
                row["inner_cross_fitted_log_loss"],
                method_order[row["calibration_method"]],
            )
        )

        best_method = str(candidates[0]["calibration_method"])
        selected_methods[member_id] = best_method
        selected_cf[:, j] = predictions[best_method]

        for row in candidates:
            row["selected"] = (
                row["calibration_method"] == best_method
            )
            rows.append(row)

    return (
        selected_cf,
        selected_methods,
        pd.DataFrame(rows),
    )


def fit_members_on_all_outer_train(
    ProbabilityCalibrator,
    *,
    ids: list[str],
    selected_methods: dict[str, str],
    y_train: np.ndarray,
    raw_train: np.ndarray,
    raw_test: np.ndarray,
    max_iter: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    train_out = np.empty_like(raw_train, dtype=np.float64)
    test_out = np.empty_like(raw_test, dtype=np.float64)
    states: dict[str, Any] = {}

    for j, member_id in enumerate(ids):
        method = selected_methods[member_id]

        calibrator = fit_calibrator(
            ProbabilityCalibrator,
            method=method,
            y=y_train,
            p=raw_train[:, j],
            max_iter=max_iter,
        )

        train_out[:, j] = apply_calibrator(
            calibrator,
            method,
            raw_train[:, j],
        )
        test_out[:, j] = apply_calibrator(
            calibrator,
            method,
            raw_test[:, j],
        )

        states[member_id] = {
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

def member_indices(
    ids: list[str],
    members: tuple[str, ...],
) -> list[int]:
    return [ids.index(member) for member in members]


def evaluate_fixed_structure(
    ProbabilityCalibrator,
    *,
    name: str,
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
    idx = member_indices(ids, members)

    # Inner diagnostic using cross-fitted member calibrations.
    raw_inner_cf = logit_mean(
        member_cf_train[:, idx]
    )

    # Final calibration family fixed to temperature.
    inner_temp_cf = cross_fit_calibration(
        ProbabilityCalibrator,
        y=y_train,
        p=raw_inner_cf,
        folds=folds_train,
        method="temperature",
        max_iter=max_iter,
    )
    inner_metrics = compute_metrics(
        y_train,
        inner_temp_cf,
        ece_bins=ece_bins,
    )

    # Full OUTER TRAIN fit -> untouched OUTER TEST.
    raw_train_ensemble = logit_mean(
        member_fullfit_train[:, idx]
    )
    raw_test_ensemble = logit_mean(
        member_fullfit_test[:, idx]
    )

    final_temperature = fit_calibrator(
        ProbabilityCalibrator,
        method="temperature",
        y=y_train,
        p=raw_train_ensemble,
        max_iter=max_iter,
    )

    final_test = apply_calibrator(
        final_temperature,
        "temperature",
        raw_test_ensemble,
    )

    result = {
        "structure": name,
        "members": "+".join(members),
        "member_count": len(members),
        "aggregation_rule": "logit_mean",
        "final_calibration": "temperature",
        "inner_cross_fitted_metrics": inner_metrics,
        "outer_test_raw_metrics": compute_metrics(
            y_test,
            raw_test_ensemble,
            ece_bins=ece_bins,
        ),
        "outer_test_final_metrics": compute_metrics(
            y_test,
            final_test,
            ece_bins=ece_bins,
        ),
        "final_temperature_state": final_temperature.to_dict(),
    }

    return final_test, result


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
                f"Outer-fold output already exists: {fold_dir}\n"
                "Use --overwrite-outer-fold for an intentional rerun."
            )

    fold_dir.mkdir(parents=True, exist_ok=True)

    test_mask = folds == outer_fold
    train_mask = ~test_mask

    y_train = y[train_mask]
    y_test = y[test_mask]
    folds_train = folds[train_mask]
    raw_train = raw_matrix[train_mask]
    raw_test = raw_matrix[test_mask]

    if set(folds_train.tolist()) != (
        set(range(5)) - {outer_fold}
    ):
        raise RuntimeError(
            f"Unexpected inner folds for outer fold {outer_fold}."
        )

    started = time.monotonic()

    # The member calibration procedure is identical for all three structures.
    member_cf_train, selected_methods, calibration_results = (
        select_member_calibration(
            ProbabilityCalibrator,
            ids=ids,
            y_train=y_train,
            folds_train=folds_train,
            raw_train=raw_train,
            max_iter=max_iter,
        )
    )

    calibration_results.to_csv(
        fold_dir / "nested_member_calibration_results.csv",
        index=False,
        float_format="%.9f",
    )

    (
        member_fullfit_train,
        member_fullfit_test,
        member_states,
    ) = fit_members_on_all_outer_train(
        ProbabilityCalibrator,
        ids=ids,
        selected_methods=selected_methods,
        y_train=y_train,
        raw_train=raw_train,
        raw_test=raw_test,
        max_iter=max_iter,
    )

    prediction_table = pd.DataFrame(
        {
            "uid": uids[test_mask],
            "fold": folds[test_mask],
            "is_pathologic": y_test,
            "P1_raw_probability": raw_test[:, ids.index("P1")],
            "R_E6_raw_probability": raw_test[:, ids.index(ROBUST_ID)],
        }
    )

    structure_results: dict[str, Any] = {}

    for name, members in FIXED_STRUCTURES.items():
        p_test, result = evaluate_fixed_structure(
            ProbabilityCalibrator,
            name=name,
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
        structure_results[name] = result
        prediction_table[f"{name}_probability"] = p_test

    # Subject-level losses for later paired aggregation.
    prediction_table["P1_subject_log_loss"] = subject_log_loss(
        y_test,
        prediction_table["P1_raw_probability"].to_numpy(),
    )
    prediction_table["R_E6_subject_log_loss"] = subject_log_loss(
        y_test,
        prediction_table["R_E6_raw_probability"].to_numpy(),
    )

    for name in FIXED_STRUCTURES:
        prediction_table[f"{name}_subject_log_loss"] = subject_log_loss(
            y_test,
            prediction_table[f"{name}_probability"].to_numpy(),
        )

    prediction_table.to_csv(
        fold_dir / "outer_test_predictions.csv",
        index=False,
        float_format="%.9f",
    )

    payload = {
        "status": "PASS",
        "step": "30",
        "outer_fold": outer_fold,
        "outer_train_subjects": int(train_mask.sum()),
        "outer_test_subjects": int(test_mask.sum()),
        "outer_train_folds": sorted(
            np.unique(folds_train).tolist()
        ),
        "selected_member_calibration_methods": selected_methods,
        "member_calibrator_states": member_states,
        "fixed_structure_results": structure_results,
        "provenance": provenance,
        "runtime_seconds": time.monotonic() - started,
        "neural_network_training": False,
        "ensemble_search": False,
        "aggregation_rule_fixed": "logit_mean",
        "final_calibration_fixed": "temperature",
        "methodological_status": (
            "Post-hoc integration/sensitivity analysis. Robust E6 was "
            "validated using Step-28/29 results from these same subjects."
        ),
    }

    write_json(
        fold_dir / "outer_fold_result.json",
        payload,
    )

    print("=" * 110)
    print(f"STEP 30 — OUTER FOLD {outer_fold}")
    print("=" * 110)
    print(f"Outer train subjects : {train_mask.sum()}")
    print(f"Outer test subjects  : {test_mask.sum()}")

    p1_m = compute_metrics(
        y_test,
        prediction_table["P1_raw_probability"].to_numpy(),
        ece_bins=ece_bins,
    )
    r_m = compute_metrics(
        y_test,
        prediction_table["R_E6_raw_probability"].to_numpy(),
        ece_bins=ece_bins,
    )
    print(f"P1 raw LL           : {p1_m['log_loss']:.6f}")
    print(f"R_E6 raw LL         : {r_m['log_loss']:.6f}")

    for name in FIXED_STRUCTURES:
        ll = structure_results[name]["outer_test_final_metrics"]["log_loss"]
        print(f"{name:<22s}: {ll:.6f}")

    print(f"Saved                : {fold_dir}")
    print("=" * 110)


# =============================================================================
# Bootstrap / aggregation
# =============================================================================

def paired_class_stratified_bootstrap(
    *,
    y: np.ndarray,
    delta_subject_loss: np.ndarray,
    n_bootstrap: int,
    seed: int,
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
        b = min(250, n_bootstrap - cursor)

        neg_sample = rng.choice(
            neg,
            size=(b, len(neg)),
            replace=True,
        )
        pos_sample = rng.choice(
            pos,
            size=(b, len(pos)),
            replace=True,
        )

        out[cursor : cursor + b] = (
            len(neg) * delta[neg_sample].mean(axis=1)
            + len(pos) * delta[pos_sample].mean(axis=1)
        ) / len(y)

        cursor += b

    return out


def optional_step27_audit(
    *,
    step27_path: Path | None,
    current_ll: float,
) -> dict[str, Any]:
    if step27_path is None or not step27_path.is_file():
        return {
            "available": False,
            "message": "Step-27 comparison file not supplied/found.",
        }

    df = pd.read_csv(step27_path)
    if "predictor" not in df.columns or "log_loss" not in df.columns:
        return {
            "available": False,
            "message": "Step-27 file has unexpected columns.",
        }

    rows = df.loc[
        df["predictor"].astype(str) == "ENS328_FIXED"
    ]
    if len(rows) != 1:
        return {
            "available": False,
            "message": "Could not identify ENS328_FIXED in Step-27 file.",
        }

    prior = float(rows.iloc[0]["log_loss"])
    delta = current_ll - prior

    return {
        "available": True,
        "step27_ENS328_FIXED_log_loss": prior,
        "step30_CURRENT_ENS328_log_loss": current_ll,
        "difference": delta,
        "reproduction_close": abs(delta) <= 1e-6,
    }


def aggregate(
    *,
    output_dir: Path,
    expected_subjects: int,
    expected_folds: int,
    ece_bins: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    step27_global_path: Path | None,
    overwrite: bool,
) -> None:
    targets = [
        output_dir / "step30_oof_predictions.csv",
        output_dir / "step30_global_comparison.csv",
        output_dir / "step30_fold_comparison.csv",
        output_dir / "step30_pairwise_bootstrap.csv",
        output_dir / "step30_summary.json",
        output_dir / "step30_report.md",
    ]

    if any(path.exists() for path in targets) and not overwrite:
        raise FileExistsError(
            "Step-30 aggregate outputs already exist. "
            "Use --overwrite-aggregate to rebuild."
        )

    frames = []

    for fold in range(expected_folds):
        path = (
            output_dir
            / f"outer_fold_{fold}"
            / "outer_test_predictions.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing outer-fold prediction file: {path}"
            )
        frames.append(pd.read_csv(path))

    oof = pd.concat(frames, ignore_index=True)
    oof["uid"] = oof["uid"].astype(str)

    if len(oof) != expected_subjects:
        raise RuntimeError(
            f"Expected {expected_subjects} OOF rows, found {len(oof)}."
        )
    if oof["uid"].duplicated().any():
        raise RuntimeError("Duplicate UID across Step-30 outer folds.")
    if set(oof["fold"].astype(int).unique()) != set(range(expected_folds)):
        raise RuntimeError("Step-30 does not cover all original folds.")

    y = oof["is_pathologic"].to_numpy(dtype=np.int64)

    predictors: dict[str, np.ndarray] = {
        "P1_RAW": oof["P1_raw_probability"].to_numpy(dtype=np.float64),
        "ROBUST_E6_RAW":
            oof["R_E6_raw_probability"].to_numpy(dtype=np.float64),
    }

    for name in FIXED_STRUCTURES:
        predictors[name] = oof[
            f"{name}_probability"
        ].to_numpy(dtype=np.float64)

    global_rows = []
    fold_rows = []

    for name, p in predictors.items():
        global_metrics = compute_metrics(
            y,
            p,
            ece_bins=ece_bins,
        )

        fold_lls = []

        for fold in range(expected_folds):
            mask = (
                oof["fold"].to_numpy(dtype=np.int64)
                == fold
            )

            fold_metrics = compute_metrics(
                y[mask],
                p[mask],
                ece_bins=ece_bins,
            )
            fold_lls.append(fold_metrics["log_loss"])

            fold_rows.append(
                {
                    "predictor": name,
                    "fold": fold,
                    "subjects": int(mask.sum()),
                    **fold_metrics,
                }
            )

        global_rows.append(
            {
                "predictor": name,
                **global_metrics,
                "fold_log_loss_sd":
                    float(np.std(fold_lls, ddof=1)),
                "fold_log_loss_min": float(np.min(fold_lls)),
                "fold_log_loss_max": float(np.max(fold_lls)),
                "fold_log_loss_range":
                    float(np.max(fold_lls) - np.min(fold_lls)),
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

    # Three prespecified structural comparisons.
    comparisons = [
        ("ROBUST_REPLACE_P1", "CURRENT_ENS328"),
        ("ROBUST_ADD", "CURRENT_ENS328"),
        ("ROBUST_ADD", "ROBUST_REPLACE_P1"),
    ]

    bootstrap_rows = []
    distributions: dict[str, np.ndarray] = {}

    # Bonferroni familywise correction across the three fixed comparisons.
    n_tests = len(comparisons)
    alpha_each = 0.05 / n_tests
    lower_pct = 100.0 * alpha_each / 2.0
    upper_pct = 100.0 * (1.0 - alpha_each / 2.0)

    for i, (candidate, reference) in enumerate(comparisons):
        candidate_loss = subject_log_loss(
            y,
            predictors[candidate],
        )
        reference_loss = subject_log_loss(
            y,
            predictors[reference],
        )
        delta = candidate_loss - reference_loss

        boot = paired_class_stratified_bootstrap(
            y=y,
            delta_subject_loss=delta,
            n_bootstrap=bootstrap_replicates,
            seed=bootstrap_seed + i,
        )

        nominal_low, nominal_high = np.percentile(
            boot,
            [2.5, 97.5],
        )
        family_low, family_high = np.percentile(
            boot,
            [lower_pct, upper_pct],
        )

        observed = float(delta.mean())

        if family_high < 0:
            conclusion = f"{candidate}_BETTER"
        elif family_low > 0:
            conclusion = f"{reference}_BETTER"
        else:
            conclusion = "INCONCLUSIVE"

        bootstrap_rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "observed_delta_log_loss_candidate_minus_reference":
                    observed,
                "nominal_95_ci_lower": float(nominal_low),
                "nominal_95_ci_upper": float(nominal_high),
                "bonferroni_familywise_ci_level_percent":
                    float(100.0 * (1.0 - alpha_each)),
                "bonferroni_ci_lower": float(family_low),
                "bonferroni_ci_upper": float(family_high),
                "familywise_conclusion": conclusion,
            }
        )

        distributions[
            f"{candidate}_vs_{reference}"
        ] = boot

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    np.savez_compressed(
        output_dir / "step30_pairwise_bootstrap_distributions.npz",
        **distributions,
    )

    lookup = global_df.set_index("predictor")

    current_ll = float(
        lookup.loc["CURRENT_ENS328", "log_loss"]
    )
    replace_ll = float(
        lookup.loc["ROBUST_REPLACE_P1", "log_loss"]
    )
    add_ll = float(
        lookup.loc["ROBUST_ADD", "log_loss"]
    )

    current_worst = float(
        lookup.loc["CURRENT_ENS328", "fold_log_loss_max"]
    )
    replace_worst = float(
        lookup.loc["ROBUST_REPLACE_P1", "fold_log_loss_max"]
    )
    add_worst = float(
        lookup.loc["ROBUST_ADD", "fold_log_loss_max"]
    )

    current_sd = float(
        lookup.loc["CURRENT_ENS328", "fold_log_loss_sd"]
    )
    replace_sd = float(
        lookup.loc["ROBUST_REPLACE_P1", "fold_log_loss_sd"]
    )
    add_sd = float(
        lookup.loc["ROBUST_ADD", "fold_log_loss_sd"]
    )

    # Descriptive next-step label only.
    if add_ll < current_ll and add_worst <= current_worst:
        descriptive_decision = "ROBUST_ADD_PROMISING"
    elif replace_ll < current_ll and replace_worst <= current_worst:
        descriptive_decision = "ROBUST_REPLACE_PROMISING"
    elif min(add_ll, replace_ll) < current_ll:
        descriptive_decision = "AVERAGE_VS_WORST_FOLD_TRADEOFF"
    else:
        descriptive_decision = "KEEP_CURRENT_ENS328"

    step27_audit = optional_step27_audit(
        step27_path=step27_global_path,
        current_ll=current_ll,
    )

    oof = oof.sort_values(
        "uid",
        kind="stable",
    ).reset_index(drop=True)

    oof.to_csv(
        output_dir / "step30_oof_predictions.csv",
        index=False,
        float_format="%.9f",
    )
    global_df.to_csv(
        output_dir / "step30_global_comparison.csv",
        index=False,
        float_format="%.9f",
    )
    fold_df.to_csv(
        output_dir / "step30_fold_comparison.csv",
        index=False,
        float_format="%.9f",
    )
    bootstrap_df.to_csv(
        output_dir / "step30_pairwise_bootstrap.csv",
        index=False,
        float_format="%.9f",
    )

    summary = {
        "status": "PASS",
        "step": "30",
        "analysis": "fixed_robust_E6_ensemble_integration",
        "methodological_status": (
            "Post-hoc integration/sensitivity analysis. Robust E6 was "
            "developed/validated after observing Step-28/29 results on "
            "these same OOF subjects."
        ),
        "fixed_structures": {
            name: list(members)
            for name, members in FIXED_STRUCTURES.items()
        },
        "aggregation_rule": "logit_mean_fixed",
        "final_calibration": "temperature_fixed",
        "member_calibration": (
            "none/temperature/platt/isotonic selected inside each outer "
            "training set only"
        ),
        "global_results": global_df.to_dict(orient="records"),
        "pairwise_bootstrap": bootstrap_df.to_dict(orient="records"),
        "current_vs_robust": {
            "CURRENT_ENS328_log_loss": current_ll,
            "ROBUST_REPLACE_P1_log_loss": replace_ll,
            "ROBUST_ADD_log_loss": add_ll,
            "replace_minus_current_log_loss": replace_ll - current_ll,
            "add_minus_current_log_loss": add_ll - current_ll,
            "CURRENT_ENS328_worst_fold": current_worst,
            "ROBUST_REPLACE_P1_worst_fold": replace_worst,
            "ROBUST_ADD_worst_fold": add_worst,
            "CURRENT_ENS328_fold_sd": current_sd,
            "ROBUST_REPLACE_P1_fold_sd": replace_sd,
            "ROBUST_ADD_fold_sd": add_sd,
        },
        "step27_reproduction_audit": step27_audit,
        "descriptive_decision": descriptive_decision,
        "critical_limit": (
            "Do not freeze or submit a new predictor solely from Step 30. "
            "If robust replacement/addition is promising, deployment-style "
            "five-checkpoint averaging and calibration must be rebuilt and "
            "validated before changing the competition bundle."
        ),
    }

    write_json(
        output_dir / "step30_summary.json",
        summary,
    )

    lines = [
        "# Step 30 — Robust E6 Integration into ENS328",
        "",
        "## Status",
        "",
        "**PASS — three fixed structures evaluated on the same five outer folds.**",
        "",
        "## Methodological status",
        "",
        (
            "**Post-hoc integration/sensitivity analysis.** Robust E6 was "
            "developed after seeing Step-28/29 results from these same "
            "subjects, so this is not a fresh unbiased final-model estimate."
        ),
        "",
        "## Structures",
        "",
        "- CURRENT_ENS328 = **P1 + P2 + P3 + P7 + P8**",
        "- ROBUST_REPLACE_P1 = **R_E6 + P2 + P3 + P7 + P8**",
        "- ROBUST_ADD = **P1 + R_E6 + P2 + P3 + P7 + P8**",
        "",
        "All three use:",
        "",
        "- nested member-calibration selection inside outer train",
        "- fixed equal-weight **logit mean**",
        "- fixed final **temperature calibration**",
        "- original Step-11 folds",
        "- no neural-network retraining",
        "",
        "## Global comparison",
        "",
        global_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Fold comparison",
        "",
        fold_df[
            [
                "fold",
                "predictor",
                "subjects",
                "log_loss",
                "auroc",
                "auprc",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Prespecified paired bootstrap comparisons",
        "",
        bootstrap_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Current vs robust structures",
        "",
        f"- CURRENT_ENS328 LL: **{current_ll:.6f}**",
        f"- ROBUST_REPLACE_P1 LL: **{replace_ll:.6f}**",
        f"- ROBUST_ADD LL: **{add_ll:.6f}**",
        f"- Replace − current: **{replace_ll - current_ll:+.6f}**",
        f"- Add − current: **{add_ll - current_ll:+.6f}**",
        "",
        f"- CURRENT worst fold: **{current_worst:.6f}**",
        f"- Replace worst fold: **{replace_worst:.6f}**",
        f"- Add worst fold: **{add_worst:.6f}**",
        "",
        f"- CURRENT fold SD: **{current_sd:.6f}**",
        f"- Replace fold SD: **{replace_sd:.6f}**",
        f"- Add fold SD: **{add_sd:.6f}**",
        "",
        "## Descriptive decision",
        "",
        f"**{descriptive_decision}**",
        "",
        "## Critical limitation",
        "",
        (
            "Do not overwrite the frozen ENS328 submission solely from this "
            "result. Step 30 uses OOF single-fold-model probabilities. "
            "Deployment averages five checkpoints per member before member "
            "calibration, so a promising structure still requires a separate "
            "deployment-style freeze/calibration/runtime validation."
        ),
    ]

    (output_dir / "step30_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("STEP 30 — FINAL ROBUST-E6 INTEGRATION SUMMARY")
    print("=" * 110)
    print(f"CURRENT_ENS328 LL       : {current_ll:.6f}")
    print(f"ROBUST_REPLACE_P1 LL    : {replace_ll:.6f}")
    print(f"ROBUST_ADD LL           : {add_ll:.6f}")
    print(f"Replace - current       : {replace_ll - current_ll:+.6f}")
    print(f"Add - current           : {add_ll - current_ll:+.6f}")
    print(f"CURRENT worst fold      : {current_worst:.6f}")
    print(f"Replace worst fold      : {replace_worst:.6f}")
    print(f"Add worst fold          : {add_worst:.6f}")
    print(f"Decision                : {descriptive_decision}")
    print(f"Saved                   : {output_dir}")
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
        args.shortlist or DEFAULT_SHORTLIST,
    )
    robust_oof_path = resolve(
        project_root,
        args.robust_oof or DEFAULT_ROBUST_OOF,
    )
    output_dir = resolve(
        project_root,
        args.output_dir or DEFAULT_OUTPUT,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.step27_global is not None:
        step27_global_path = resolve(
            project_root,
            args.step27_global,
        )
    else:
        candidate = resolve(
            project_root,
            DEFAULT_STEP27_GLOBAL,
        )
        step27_global_path = (
            candidate if candidate.is_file() else None
        )

    (
        ids,
        uids,
        y,
        folds,
        raw_matrix,
        provenance,
    ) = load_member_matrix(
        project_root=project_root,
        shortlist_path=shortlist_path,
        robust_oof_path=robust_oof_path,
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
            step27_global_path=step27_global_path,
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
        max_iter=args.calibration_max_iter,
        ece_bins=args.ece_bins,
        overwrite=args.overwrite_outer_fold,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
