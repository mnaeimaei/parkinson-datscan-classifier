#!/usr/bin/env python3
"""
STEP 26 — NESTED ENSEMBLE-SELECTION VALIDATION

Goal
----
Test whether the apparent Step-19/20 ensemble gain survives when ensemble
selection is performed WITHOUT using the labels of the fold on which it is
finally evaluated.

No neural network is retrained.

Outer loop
----------
For each original Step-11 fold k:

    OUTER TEST  = fold k
    OUTER TRAIN = other four folds

Only OUTER TRAIN labels may be used for:

1. member-calibration method selection for each fixed P1-P8 identity;
2. fitting/cross-fitting member calibrators;
3. exhaustive 494 equal-weight ensemble search;
4. selecting the best ensemble per size 2..8;
5. selecting none/temperature/platt/isotonic calibration for those
   seven representative ensembles;
6. selecting the final ensemble configuration.

Then the selected pipeline is refitted on ALL OUTER TRAIN subjects and applied
exactly once to OUTER TEST.

The five outer-test predictions are concatenated to produce the Step-26 nested
OOF estimate.

IMPORTANT SCOPE
---------------
This is conditional nested validation of the already chosen P1-P8 identities.

It DOES nest:
- member calibration METHOD selection within P1-P8;
- member calibrator fitting;
- Step-19 494 equal-weight ensemble search;
- Step-20 representative-by-size selection;
- final ensemble calibration method selection and fitting.

It DOES NOT nest:
- the earlier 60-experiment -> P1-P8 shortlist identity selection.

Therefore Step 26 is specifically a test of whether Steps 19-20 produced
selection optimism after the shortlist had already been defined.

Critical leakage prevention
---------------------------
Step 26 starts from `raw_probability` in each Step-16 P1-P8 prediction file.

It intentionally DOES NOT use Step-16 `calibrated_probability` as the nested
selection input, because those probabilities were cross-fitted under the
original global calibration procedure and can indirectly depend on labels
from the current outer fold through calibrators fitted for other subjects.

Reference
---------
P1 uses its raw OOF probability as the outer-fold reference because P1's
authoritative Step-16 method is none.

Primary metric
--------------
Global nested OOF Log Loss (lower is better).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    auc,
)


EPS = 1e-6
METHODS = ("none", "temperature", "platt", "isotonic")
RULES = ("probability_mean", "logit_mean")
EXPECTED_IDS = tuple(f"P{i}" for i in range(1, 9))
EXPECTED_ENSEMBLES = 494
EXPECTED_SUBSETS = 247
REFERENCE_ID = "P1"


# =============================================================================
# CLI / filesystem
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 26: nested validation of P1-P8 ensemble selection."
    )
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--shortlist", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)

    p.add_argument(
        "--outer-fold",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=None,
        help="Run one outer fold. Normally supplied by the Slurm array.",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Aggregate outer-fold results after all five jobs finish.",
    )

    p.add_argument("--expected-subjects", type=int, default=1362)
    p.add_argument("--expected-folds", type=int, default=5)
    p.add_argument("--expected-candidates", type=int, default=8)
    p.add_argument("--calibration-max-iter", type=int, default=150)
    p.add_argument("--ece-bins", type=int, default=10)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)

    p.add_argument(
        "--overwrite-outer-fold",
        action="store_true",
        help="Delete/rebuild the selected outer-fold directory.",
    )
    p.add_argument(
        "--overwrite-aggregate",
        action="store_true",
        help="Allow rebuilding aggregate Step-26 outputs.",
    )

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
    marker = "/data/"
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
        candidate = (project_root / "data" / rel).resolve()
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not resolve stored prediction path:\n"
        f"  stored : {raw}\n"
        f"  project: {project_root}"
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
        if np.isnan(value):
            return None
        return float(value)
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
# Metrics / transforms
# =============================================================================

def clip_probability(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.clip(p, EPS, 1.0 - EPS)


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
    ece = 0.0
    n_total = len(y)

    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / n_total) * abs(float(p[mask].mean()) - float(y[mask].mean()))

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
# Data loading
# =============================================================================

def shortlist_sort_key(value: str) -> int:
    text = str(value).strip()
    if text.startswith("P") and text[1:].isdigit():
        return int(text[1:])
    return 10000


def load_raw_p1_p8(
    *,
    project_root: Path,
    shortlist_path: Path,
    expected_subjects: int,
    expected_folds: int,
    expected_candidates: int,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if not shortlist_path.is_file():
        raise FileNotFoundError(f"Competition shortlist not found: {shortlist_path}")

    shortlist = pd.read_csv(shortlist_path)
    required = {
        "Shortlist ID",
        "Calibration Method",
        "Cross-Fitted Prediction File",
    }
    missing = required - set(shortlist.columns)
    if missing:
        raise RuntimeError(f"Shortlist missing columns: {sorted(missing)}")

    if len(shortlist) != expected_candidates:
        raise RuntimeError(
            f"Expected {expected_candidates} shortlist rows, found {len(shortlist)}."
        )

    shortlist = shortlist.copy()
    shortlist["_sort"] = shortlist["Shortlist ID"].map(shortlist_sort_key)
    shortlist = shortlist.sort_values("_sort", kind="stable").drop(columns="_sort")

    ids = shortlist["Shortlist ID"].astype(str).tolist()
    if tuple(ids) != EXPECTED_IDS:
        raise RuntimeError(f"Expected shortlist IDs {EXPECTED_IDS}, found {ids}.")

    canonical_uid = None
    canonical_y = None
    canonical_fold = None
    raw_columns = []
    provenance: list[dict[str, Any]] = []

    for _, row in shortlist.iterrows():
        pid = str(row["Shortlist ID"])
        pred_path = resolve_stored_path(
            project_root,
            row["Cross-Fitted Prediction File"],
        )

        df = pd.read_csv(pred_path)
        required_pred = {
            "uid",
            "fold",
            "is_pathologic",
            "raw_probability",
            "calibrated_probability",
        }
        missing_pred = required_pred - set(df.columns)
        if missing_pred:
            raise RuntimeError(
                f"{pid} prediction file missing {sorted(missing_pred)}:\n{pred_path}"
            )

        df = df.copy()
        df["uid"] = df["uid"].astype(str)
        df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(np.int64)
        df["is_pathologic"] = pd.to_numeric(
            df["is_pathologic"], errors="raise"
        ).astype(np.int64)
        df["raw_probability"] = pd.to_numeric(
            df["raw_probability"], errors="raise"
        ).astype(np.float64)

        if len(df) != expected_subjects:
            raise RuntimeError(
                f"{pid}: expected {expected_subjects} rows, found {len(df)}."
            )
        if df["uid"].duplicated().any():
            raise RuntimeError(f"{pid}: duplicate UIDs.")
        if not np.isfinite(df["raw_probability"]).all():
            raise RuntimeError(f"{pid}: raw probabilities contain NaN/Inf.")
        if ((df["raw_probability"] < 0) | (df["raw_probability"] > 1)).any():
            raise RuntimeError(f"{pid}: raw probabilities outside [0,1].")

        df = df.sort_values("uid", kind="stable").reset_index(drop=True)

        if canonical_uid is None:
            canonical_uid = df["uid"].to_numpy(dtype=str)
            canonical_y = df["is_pathologic"].to_numpy(dtype=np.int64)
            canonical_fold = df["fold"].to_numpy(dtype=np.int64)

            expected_fold_ids = set(range(expected_folds))
            if set(canonical_fold.tolist()) != expected_fold_ids:
                raise RuntimeError(
                    f"Expected original folds {sorted(expected_fold_ids)}, "
                    f"found {sorted(set(canonical_fold.tolist()))}."
                )
        else:
            if not np.array_equal(canonical_uid, df["uid"].to_numpy(dtype=str)):
                raise RuntimeError(f"{pid}: UID alignment differs from P1.")
            if not np.array_equal(
                canonical_y, df["is_pathologic"].to_numpy(dtype=np.int64)
            ):
                raise RuntimeError(f"{pid}: labels differ from P1.")
            if not np.array_equal(
                canonical_fold, df["fold"].to_numpy(dtype=np.int64)
            ):
                raise RuntimeError(f"{pid}: fold assignments differ from P1.")

        raw_columns.append(df["raw_probability"].to_numpy(dtype=np.float64))

        provenance.append(
            {
                "Shortlist ID": pid,
                "Original Step-16 Calibration Method": str(row["Calibration Method"]),
                "Prediction File": str(pred_path),
                "Prediction File SHA256": sha256_file(pred_path),
            }
        )

    raw_matrix = np.column_stack(raw_columns)
    return (
        shortlist,
        ids,
        canonical_uid,
        canonical_y,
        canonical_fold,
        raw_matrix,
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


def cross_fit_calibration_subset(
    ProbabilityCalibrator,
    *,
    y: np.ndarray,
    raw_p: np.ndarray,
    folds: np.ndarray,
    method: str,
    max_iter: int,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    raw_p = np.asarray(raw_p, dtype=np.float64)
    folds = np.asarray(folds, dtype=np.int64)

    if method == "none":
        return raw_p.copy()

    unique_folds = sorted(np.unique(folds).tolist())
    out = np.full(len(y), np.nan, dtype=np.float64)

    for heldout in unique_folds:
        fit_mask = folds != heldout
        apply_mask = folds == heldout

        if fit_mask.sum() == 0 or apply_mask.sum() == 0:
            raise RuntimeError("Empty inner calibration train/apply partition.")
        if len(np.unique(y[fit_mask])) < 2:
            raise RuntimeError(
                f"Calibration training data has one class only; method={method}."
            )

        calibrator = fit_calibrator(
            ProbabilityCalibrator,
            method=method,
            y=y[fit_mask],
            p=raw_p[fit_mask],
            max_iter=max_iter,
        )
        out[apply_mask] = apply_calibrator(
            calibrator,
            method,
            raw_p[apply_mask],
        )

    if not np.isfinite(out).all():
        raise RuntimeError(
            f"Inner cross-fitted calibration produced NaN/Inf: {method}"
        )
    return out


def select_member_calibration_methods(
    ProbabilityCalibrator,
    *,
    ids: list[str],
    y_train: np.ndarray,
    folds_train: np.ndarray,
    raw_train: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, str]]:
    """
    Within one outer-training set:
    - evaluate none/temp/platt/isotonic by inner cross-fitting;
    - choose the lowest inner-CF Log Loss separately per P1-P8;
    - return the selected inner-CF calibrated member matrix used for ensemble search.
    """
    selected_matrix = np.empty_like(raw_train, dtype=np.float64)
    rows = []
    selected_methods: dict[str, str] = {}

    method_order = {m: i for i, m in enumerate(METHODS)}

    for j, pid in enumerate(ids):
        candidate_rows = []
        method_predictions = {}

        for method in METHODS:
            p_cf = cross_fit_calibration_subset(
                ProbabilityCalibrator,
                y=y_train,
                raw_p=raw_train[:, j],
                folds=folds_train,
                method=method,
                max_iter=max_iter,
            )
            ll = binary_log_loss(y_train, p_cf)
            method_predictions[method] = p_cf
            candidate_rows.append(
                {
                    "Shortlist ID": pid,
                    "Calibration Method": method,
                    "Inner Cross-Fitted Log Loss": ll,
                    "Original Step-16 Method": None,
                }
            )

        candidate_rows = sorted(
            candidate_rows,
            key=lambda r: (
                r["Inner Cross-Fitted Log Loss"],
                method_order[r["Calibration Method"]],
            ),
        )
        best_method = str(candidate_rows[0]["Calibration Method"])
        selected_methods[pid] = best_method
        selected_matrix[:, j] = method_predictions[best_method]

        for row in candidate_rows:
            row["Selected For Member"] = (
                row["Calibration Method"] == best_method
            )
            rows.append(row)

    return selected_matrix, pd.DataFrame(rows), selected_methods


def fit_selected_member_calibrators_for_outer_test(
    ProbabilityCalibrator,
    *,
    ids: list[str],
    selected_methods: dict[str, str],
    y_train: np.ndarray,
    raw_train: np.ndarray,
    raw_test: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Fit the outer-fold selected member calibration method on ALL outer-training
    observations, then transform outer-train and untouched outer-test.
    """
    train_out = np.empty_like(raw_train, dtype=np.float64)
    test_out = np.empty_like(raw_test, dtype=np.float64)
    states: dict[str, Any] = {}

    for j, pid in enumerate(ids):
        method = selected_methods[pid]
        calibrator = fit_calibrator(
            ProbabilityCalibrator,
            method=method,
            y=y_train,
            p=raw_train[:, j],
            max_iter=max_iter,
        )
        train_out[:, j] = apply_calibrator(
            calibrator, method, raw_train[:, j]
        )
        test_out[:, j] = apply_calibrator(
            calibrator, method, raw_test[:, j]
        )

        states[pid] = {
            "method": method,
            "calibrator": (
                calibrator.to_dict() if calibrator is not None else {"method": "none"}
            ),
        }

    return train_out, test_out, states


# =============================================================================
# Ensemble search
# =============================================================================

def make_ensemble_probability(
    matrix: np.ndarray,
    member_indices: Iterable[int],
    rule: str,
) -> np.ndarray:
    idx = list(member_indices)
    selected = matrix[:, idx]

    if rule == "probability_mean":
        return np.mean(selected, axis=1)

    if rule == "logit_mean":
        return sigmoid(np.mean(logit(selected), axis=1))

    raise ValueError(f"Unknown aggregation rule: {rule}")


def enumerate_all_ensembles(
    *,
    ids: list[str],
    y: np.ndarray,
    member_matrix: np.ndarray,
) -> pd.DataFrame:
    subsets = []
    for size in range(2, len(ids) + 1):
        subsets.extend(itertools.combinations(range(len(ids)), size))

    if len(subsets) != EXPECTED_SUBSETS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBSETS} subsets, found {len(subsets)}."
        )

    rows = []
    counter = 0

    for subset in subsets:
        members = [ids[i] for i in subset]
        for rule in RULES:
            counter += 1
            p = make_ensemble_probability(
                member_matrix,
                subset,
                rule,
            )
            rows.append(
                {
                    "Ensemble ID": f"ENS{counter:03d}",
                    "Members": "+".join(members),
                    "Member Indices": ",".join(str(i) for i in subset),
                    "Number of Members": len(subset),
                    "Contains P1": REFERENCE_ID in members,
                    "Aggregation Rule": rule,
                    "Inner Raw Ensemble Log Loss": binary_log_loss(y, p),
                }
            )

    if counter != EXPECTED_ENSEMBLES:
        raise RuntimeError(
            f"Expected {EXPECTED_ENSEMBLES} ensembles, built {counter}."
        )

    return pd.DataFrame(rows).sort_values(
        ["Inner Raw Ensemble Log Loss", "Number of Members", "Ensemble ID"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def representative_ensembles_by_size(
    search_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for size in range(2, 9):
        candidates = search_results.loc[
            search_results["Number of Members"] == size
        ].sort_values(
            ["Inner Raw Ensemble Log Loss", "Ensemble ID"],
            ascending=[True, True],
            kind="stable",
        )
        if candidates.empty:
            raise RuntimeError(f"No ensembles found for size {size}.")
        rows.append(candidates.iloc[0].to_dict())

    reps = pd.DataFrame(rows)
    reps.insert(0, "Representative ID", [f"R{i}" for i in range(1, 8)])
    return reps


def parse_member_indices(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if str(x).strip()]


def select_final_representative_and_calibration(
    ProbabilityCalibrator,
    *,
    y_train: np.ndarray,
    folds_train: np.ndarray,
    member_cf_matrix: np.ndarray,
    representatives: pd.DataFrame,
    max_iter: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    method_order = {m: i for i, m in enumerate(METHODS)}

    for _, rep in representatives.iterrows():
        indices = parse_member_indices(rep["Member Indices"])
        p_raw = make_ensemble_probability(
            member_cf_matrix,
            indices,
            str(rep["Aggregation Rule"]),
        )

        for method in METHODS:
            p_cf = cross_fit_calibration_subset(
                ProbabilityCalibrator,
                y=y_train,
                raw_p=p_raw,
                folds=folds_train,
                method=method,
                max_iter=max_iter,
            )

            rows.append(
                {
                    "Representative ID": str(rep["Representative ID"]),
                    "Ensemble ID": str(rep["Ensemble ID"]),
                    "Members": str(rep["Members"]),
                    "Member Indices": str(rep["Member Indices"]),
                    "Number of Members": int(rep["Number of Members"]),
                    "Aggregation Rule": str(rep["Aggregation Rule"]),
                    "Calibration Method": method,
                    "Inner Raw Ensemble Log Loss": float(
                        rep["Inner Raw Ensemble Log Loss"]
                    ),
                    "Inner Cross-Fitted Calibrated Log Loss":
                        binary_log_loss(y_train, p_cf),
                }
            )

    results = pd.DataFrame(rows)

    ranked = results.copy()
    ranked["_method_order"] = ranked["Calibration Method"].map(method_order)
    ranked = ranked.sort_values(
        [
            "Inner Cross-Fitted Calibrated Log Loss",
            "Number of Members",
            "_method_order",
            "Ensemble ID",
        ],
        ascending=[True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    selected = ranked.iloc[0].drop(labels=["_method_order"]).to_dict()
    return results, selected


def apply_selected_pipeline_to_outer_test(
    ProbabilityCalibrator,
    *,
    selected: dict[str, Any],
    y_train: np.ndarray,
    member_train_fullfit: np.ndarray,
    member_test_fullfit: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices = parse_member_indices(selected["Member Indices"])
    rule = str(selected["Aggregation Rule"])
    method = str(selected["Calibration Method"])

    raw_ens_train = make_ensemble_probability(
        member_train_fullfit,
        indices,
        rule,
    )
    raw_ens_test = make_ensemble_probability(
        member_test_fullfit,
        indices,
        rule,
    )

    calibrator = fit_calibrator(
        ProbabilityCalibrator,
        method=method,
        y=y_train,
        p=raw_ens_train,
        max_iter=max_iter,
    )

    final_train = apply_calibrator(calibrator, method, raw_ens_train)
    final_test = apply_calibrator(calibrator, method, raw_ens_test)

    state = {
        "method": method,
        "calibrator": (
            calibrator.to_dict() if calibrator is not None else {"method": "none"}
        ),
        "outer_train_fit_log_loss": binary_log_loss(y_train, final_train),
    }

    return final_train, final_test, state


# =============================================================================
# Bootstrap
# =============================================================================

def paired_class_stratified_bootstrap(
    *,
    y: np.ndarray,
    delta: np.ndarray,
    n_bootstrap: int,
    seed: int,
    chunk_size: int = 250,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    delta = np.asarray(delta, dtype=np.float64)

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


# =============================================================================
# Outer-fold execution
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
    calibration_max_iter: int,
    ece_bins: int,
    overwrite: bool,
) -> None:
    fold_dir = output_dir / f"outer_fold_{outer_fold}"

    if fold_dir.exists() and any(fold_dir.iterdir()):
        if overwrite:
            import shutil
            shutil.rmtree(fold_dir)
        else:
            raise FileExistsError(
                f"Outer-fold output exists: {fold_dir}\n"
                "Use --overwrite-outer-fold for an intentional rerun."
            )
    fold_dir.mkdir(parents=True, exist_ok=True)

    outer_test = folds == outer_fold
    outer_train = ~outer_test

    y_train = y[outer_train]
    y_test = y[outer_test]
    folds_train = folds[outer_train]
    raw_train = raw_matrix[outer_train]
    raw_test = raw_matrix[outer_test]

    expected_inner_folds = set(range(5)) - {outer_fold}
    if set(folds_train.tolist()) != expected_inner_folds:
        raise RuntimeError(
            f"Outer fold {outer_fold}: unexpected inner fold IDs."
        )

    print("=" * 110)
    print(f"STEP 26 — OUTER FOLD {outer_fold}")
    print("=" * 110)
    print(f"Outer train subjects       : {outer_train.sum()}")
    print(f"Outer test subjects        : {outer_test.sum()}")
    print(f"Outer train folds          : {sorted(np.unique(folds_train).tolist())}")
    print("Outer-test labels used in selection: NO")
    print()

    started = time.monotonic()

    # ------------------------------------------------------------------
    # 1) Nested member-calibration METHOD selection on outer-train only.
    # ------------------------------------------------------------------
    member_cf, member_calib_results, selected_member_methods = (
        select_member_calibration_methods(
            ProbabilityCalibrator,
            ids=ids,
            y_train=y_train,
            folds_train=folds_train,
            raw_train=raw_train,
            max_iter=calibration_max_iter,
        )
    )

    original_method_lookup = {
        row["Shortlist ID"]: row["Original Step-16 Calibration Method"]
        for row in provenance
    }
    member_calib_results["Original Step-16 Method"] = (
        member_calib_results["Shortlist ID"].map(original_method_lookup)
    )

    member_calib_results.to_csv(
        fold_dir / "inner_member_calibration_results.csv",
        index=False,
        float_format="%.9f",
    )

    print("Selected member calibration methods:")
    for pid in ids:
        print(
            f"  {pid}: {selected_member_methods[pid]} "
            f"(Step16={original_method_lookup[pid]})"
        )

    # ------------------------------------------------------------------
    # 2) Exhaustive Step-19-style 494 ensemble search on outer-train only.
    # ------------------------------------------------------------------
    search = enumerate_all_ensembles(
        ids=ids,
        y=y_train,
        member_matrix=member_cf,
    )
    search.to_csv(
        fold_dir / "inner_all_494_equal_weight_ensemble_results.csv",
        index=False,
        float_format="%.9f",
    )

    reps = representative_ensembles_by_size(search)
    reps.to_csv(
        fold_dir / "inner_best_ensemble_per_size.csv",
        index=False,
        float_format="%.9f",
    )

    # ------------------------------------------------------------------
    # 3) Step-20-style calibration of seven representatives.
    # ------------------------------------------------------------------
    rep_calib_results, selected = (
        select_final_representative_and_calibration(
            ProbabilityCalibrator,
            y_train=y_train,
            folds_train=folds_train,
            member_cf_matrix=member_cf,
            representatives=reps,
            max_iter=calibration_max_iter,
        )
    )
    rep_calib_results.to_csv(
        fold_dir / "inner_28_representative_calibration_results.csv",
        index=False,
        float_format="%.9f",
    )

    print()
    print("Selected outer-fold pipeline:")
    print(f"  Ensemble ID     : {selected['Ensemble ID']}")
    print(f"  Members         : {selected['Members']}")
    print(f"  Size            : {selected['Number of Members']}")
    print(f"  Rule            : {selected['Aggregation Rule']}")
    print(f"  Final calibration: {selected['Calibration Method']}")
    print(
        "  Inner selected LL: "
        f"{selected['Inner Cross-Fitted Calibrated Log Loss']:.6f}"
    )

    # ------------------------------------------------------------------
    # 4) Refit selected member calibrators on ALL outer-train,
    #    then apply once to untouched outer-test.
    # ------------------------------------------------------------------
    member_train_fullfit, member_test_fullfit, member_states = (
        fit_selected_member_calibrators_for_outer_test(
            ProbabilityCalibrator,
            ids=ids,
            selected_methods=selected_member_methods,
            y_train=y_train,
            raw_train=raw_train,
            raw_test=raw_test,
            max_iter=calibration_max_iter,
        )
    )

    _, final_test, final_calibrator_state = (
        apply_selected_pipeline_to_outer_test(
            ProbabilityCalibrator,
            selected=selected,
            y_train=y_train,
            member_train_fullfit=member_train_fullfit,
            member_test_fullfit=member_test_fullfit,
            max_iter=calibration_max_iter,
        )
    )

    # P1 reference is raw probability (Step-16 P1 method = none).
    p1_index = ids.index(REFERENCE_ID)
    p1_test = raw_test[:, p1_index]

    nested_metrics = compute_metrics(y_test, final_test, ece_bins=ece_bins)
    p1_metrics = compute_metrics(y_test, p1_test, ece_bins=ece_bins)

    subject_df = pd.DataFrame(
        {
            "uid": uids[outer_test],
            "fold": folds[outer_test],
            "is_pathologic": y_test,
            "P1_raw_probability": p1_test,
            "nested_selected_probability": final_test,
        }
    )
    subject_df["P1_subject_log_loss"] = subject_log_loss(
        y_test, p1_test
    )
    subject_df["nested_subject_log_loss"] = subject_log_loss(
        y_test, final_test
    )
    subject_df["nested_minus_P1_subject_log_loss"] = (
        subject_df["nested_subject_log_loss"]
        - subject_df["P1_subject_log_loss"]
    )
    subject_df.to_csv(
        fold_dir / "outer_test_predictions.csv",
        index=False,
        float_format="%.9f",
    )

    selection_payload = {
        "status": "PASS",
        "outer_fold": outer_fold,
        "outer_train_subjects": int(outer_train.sum()),
        "outer_test_subjects": int(outer_test.sum()),
        "outer_train_original_folds": sorted(np.unique(folds_train).tolist()),
        "selected_member_calibration_methods": selected_member_methods,
        "selected_ensemble": selected,
        "outer_test_nested_metrics": nested_metrics,
        "outer_test_P1_metrics": p1_metrics,
        "outer_test_delta_log_loss_nested_minus_P1": (
            nested_metrics["log_loss"] - p1_metrics["log_loss"]
        ),
        "member_fullfit_calibrators": member_states,
        "final_ensemble_calibrator": final_calibrator_state,
        "runtime_seconds": time.monotonic() - started,
        "leakage_guarantee": (
            "No outer-test labels were used for member calibration selection, "
            "ensemble search, representative selection, or calibrator fitting."
        ),
        "scope_limit": (
            "P1-P8 identities are treated as fixed. The earlier 60->P1-P8 "
            "shortlist identity selection is not nested in Step 26."
        ),
    }
    write_json(
        fold_dir / "outer_fold_selection_and_result.json",
        selection_payload,
    )

    print()
    print(f"Outer-test nested LL : {nested_metrics['log_loss']:.6f}")
    print(f"Outer-test P1 LL     : {p1_metrics['log_loss']:.6f}")
    print(
        "Delta nested-P1     : "
        f"{nested_metrics['log_loss'] - p1_metrics['log_loss']:+.6f}"
    )
    print(f"Saved                 : {fold_dir}")


# =============================================================================
# Aggregate five untouched outer folds
# =============================================================================

def aggregate_outer_folds(
    *,
    output_dir: Path,
    expected_subjects: int,
    expected_folds: int,
    ece_bins: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    overwrite: bool,
) -> None:
    aggregate_files = [
        output_dir / "nested_selected_oof_predictions.csv",
        output_dir / "nested_outer_fold_results.csv",
        output_dir / "nested_selection_stability.csv",
        output_dir / "step26_summary.json",
        output_dir / "step26_report.md",
    ]
    if any(p.exists() for p in aggregate_files) and not overwrite:
        raise FileExistsError(
            "Aggregate Step-26 outputs already exist. "
            "Use --overwrite-aggregate for an intentional rebuild."
        )

    prediction_frames = []
    fold_rows = []
    selection_rows = []

    for outer_fold in range(expected_folds):
        fold_dir = output_dir / f"outer_fold_{outer_fold}"
        pred_path = fold_dir / "outer_test_predictions.csv"
        result_path = fold_dir / "outer_fold_selection_and_result.json"

        if not pred_path.is_file() or not result_path.is_file():
            raise FileNotFoundError(
                f"Outer fold {outer_fold} incomplete:\n"
                f"  {pred_path}\n  {result_path}"
            )

        pred = pd.read_csv(pred_path)
        prediction_frames.append(pred)

        result = json.loads(result_path.read_text(encoding="utf-8"))
        selected = result["selected_ensemble"]

        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "subjects": len(pred),
                "P1_log_loss": result["outer_test_P1_metrics"]["log_loss"],
                "nested_selected_log_loss":
                    result["outer_test_nested_metrics"]["log_loss"],
                "delta_log_loss_nested_minus_P1":
                    result["outer_test_delta_log_loss_nested_minus_P1"],
                "nested_AUROC": result["outer_test_nested_metrics"]["auroc"],
                "nested_AUPRC": result["outer_test_nested_metrics"]["auprc"],
                "nested_Brier": result["outer_test_nested_metrics"]["brier_score"],
                "nested_ECE": result["outer_test_nested_metrics"]["ece"],
                "selected_ensemble_id": selected["Ensemble ID"],
                "selected_members": selected["Members"],
                "selected_size": selected["Number of Members"],
                "selected_rule": selected["Aggregation Rule"],
                "selected_final_calibration": selected["Calibration Method"],
                "inner_selected_log_loss":
                    selected["Inner Cross-Fitted Calibrated Log Loss"],
            }
        )

        member_methods = result["selected_member_calibration_methods"]
        row = {
            "outer_fold": outer_fold,
            "selected_ensemble_id": selected["Ensemble ID"],
            "selected_members": selected["Members"],
            "selected_size": selected["Number of Members"],
            "selected_rule": selected["Aggregation Rule"],
            "selected_final_calibration": selected["Calibration Method"],
        }
        for pid in EXPECTED_IDS:
            row[f"{pid}_selected_member_calibration"] = member_methods[pid]
        selection_rows.append(row)

    oof = pd.concat(prediction_frames, ignore_index=True)
    oof["uid"] = oof["uid"].astype(str)

    if len(oof) != expected_subjects:
        raise RuntimeError(
            f"Expected {expected_subjects} nested OOF rows, found {len(oof)}."
        )
    if oof["uid"].duplicated().any():
        raise RuntimeError("Duplicate UID across outer-test predictions.")
    if set(oof["fold"].astype(int).unique()) != set(range(expected_folds)):
        raise RuntimeError("Nested OOF does not cover all original folds.")

    y = oof["is_pathologic"].to_numpy(dtype=np.int64)
    p1 = oof["P1_raw_probability"].to_numpy(dtype=np.float64)
    nested = oof["nested_selected_probability"].to_numpy(dtype=np.float64)

    p1_metrics = compute_metrics(y, p1, ece_bins=ece_bins)
    nested_metrics = compute_metrics(y, nested, ece_bins=ece_bins)

    observed_delta = nested_metrics["log_loss"] - p1_metrics["log_loss"]
    delta_subject = subject_log_loss(y, nested) - subject_log_loss(y, p1)

    bootstrap = paired_class_stratified_bootstrap(
        y=y,
        delta=delta_subject,
        n_bootstrap=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])

    if ci_high < 0:
        conclusion = "NESTED ENSEMBLE BETTER"
    elif ci_low > 0:
        conclusion = "P1 BETTER"
    else:
        conclusion = "INCONCLUSIVE"

    fold_results = pd.DataFrame(fold_rows)
    selections = pd.DataFrame(selection_rows)

    oof = oof.sort_values("uid", kind="stable").reset_index(drop=True)
    oof.to_csv(
        output_dir / "nested_selected_oof_predictions.csv",
        index=False,
        float_format="%.9f",
    )
    fold_results.to_csv(
        output_dir / "nested_outer_fold_results.csv",
        index=False,
        float_format="%.9f",
    )
    selections.to_csv(
        output_dir / "nested_selection_stability.csv",
        index=False,
    )

    np.savez_compressed(
        output_dir / "nested_vs_p1_bootstrap_distribution.npz",
        delta_log_loss=bootstrap,
    )

    # Compact stability summary.
    member_counts = {pid: 0 for pid in EXPECTED_IDS}
    for members in selections["selected_members"]:
        selected_set = set(str(members).split("+"))
        for pid in EXPECTED_IDS:
            member_counts[pid] += int(pid in selected_set)

    stability = {
        "unique_selected_ensemble_structures":
            int(selections["selected_members"].nunique()),
        "unique_selected_rules":
            int(selections["selected_rule"].nunique()),
        "unique_final_calibration_methods":
            int(selections["selected_final_calibration"].nunique()),
        "member_selection_counts_across_5_outer_folds": member_counts,
        "selected_members_by_outer_fold":
            selections["selected_members"].tolist(),
        "selected_rule_by_outer_fold":
            selections["selected_rule"].tolist(),
        "selected_final_calibration_by_outer_fold":
            selections["selected_final_calibration"].tolist(),
    }

    summary = {
        "status": "PASS",
        "step": "26",
        "analysis": "nested_ensemble_selection_validation",
        "scope": (
            "Conditional on fixed P1-P8 identities; member calibration method, "
            "ensemble search, representative selection, and final ensemble "
            "calibration are nested inside each outer training set."
        ),
        "subjects": expected_subjects,
        "outer_folds": expected_folds,
        "outer_test_predictions_per_subject": 1,
        "P1_metrics": p1_metrics,
        "nested_selected_metrics": nested_metrics,
        "observed_delta_log_loss_nested_minus_P1": observed_delta,
        "relative_log_loss_improvement_vs_P1_percent":
            float((p1_metrics["log_loss"] - nested_metrics["log_loss"])
                  / p1_metrics["log_loss"] * 100.0),
        "paired_bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "ci_95_lower": float(ci_low),
            "ci_95_upper": float(ci_high),
            "conclusion": conclusion,
        },
        "selection_stability": stability,
        "interpretation_reference": {
            "Step20_ENS328_cross_fitted_OOF_log_loss": 0.255582,
            "P1_original_OOF_log_loss": 0.297122,
            "competition_full_test_log_loss": 0.2997,
        },
        "methodological_limit": (
            "The P1-P8 candidate identities themselves were selected earlier "
            "using the full OOF dataset. Step 26 does not re-run the 60-model "
            "shortlist selection inside each outer fold."
        ),
    }
    write_json(output_dir / "step26_summary.json", summary)

    lines = [
        "# Step 26 — Nested Ensemble-Selection Validation",
        "",
        "## Status",
        "",
        "**PASS — all five outer folds were selected and evaluated without using their labels during selection.**",
        "",
        "## Scope",
        "",
        (
            "This is nested validation of Steps 19–20 **conditional on the "
            "already chosen P1–P8 identities**. Member calibration method "
            "selection, the 494-ensemble search, representative selection, "
            "and final ensemble calibration are repeated using only each "
            "outer-training set."
        ),
        "",
        "It does **not** re-run the earlier 60-experiment → P1–P8 shortlist identity selection.",
        "",
        "## Global nested result",
        "",
        f"- P1 Log Loss: **{p1_metrics['log_loss']:.6f}**",
        f"- Nested-selected ensemble Log Loss: **{nested_metrics['log_loss']:.6f}**",
        f"- ΔLogLoss (nested − P1): **{observed_delta:+.6f}**",
        (
            "- Relative improvement vs P1: "
            f"**{(p1_metrics['log_loss'] - nested_metrics['log_loss']) / p1_metrics['log_loss'] * 100.0:.2f}%**"
        ),
        f"- Nested AUROC: **{nested_metrics['auroc']:.6f}**",
        f"- Nested AUPRC: **{nested_metrics['auprc']:.6f}**",
        "",
        "## Paired bootstrap",
        "",
        f"- Replicates: **{bootstrap_replicates:,}**",
        f"- 95% CI for ΔLogLoss: **[{ci_low:.6f}, {ci_high:.6f}]**",
        f"- Conclusion: **{conclusion}**",
        "",
        "## Outer-fold results",
        "",
        fold_results.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Selection stability",
        "",
        selections.to_markdown(index=False),
        "",
        "Member appearance counts across the five independently selected outer-fold pipelines:",
        "",
        "| Member | Outer folds selected |",
        "|---|---:|",
    ]
    for pid in EXPECTED_IDS:
        lines.append(f"| {pid} | {member_counts[pid]}/5 |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Compare the nested-selected Log Loss primarily with P1 "
                "(~0.2971), not with the already-selected Step-20 value "
                "(0.255582)."
            ),
            "",
            "- If nested LL remains near ~0.26, the ensemble gain is robust conditional on P1–P8.",
            "- If nested LL moves toward ~0.29–0.30, much of the Step-19/20 gain was selection optimism.",
            "- If nested LL is worse than P1, the ensemble-search procedure was actively overfitting these OOF labels.",
            "",
            "## Methodological limit",
            "",
            (
                "The identities P1–P8 were themselves obtained from an earlier "
                "full-data shortlist-selection stage. A completely nested "
                "validation of the entire project would also have to repeat the "
                "60-model calibration/ranking/shortlisting process inside each "
                "outer training set. Step 26 intentionally does not do that."
            ),
        ]
    )

    (output_dir / "step26_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("STEP 26 — FINAL NESTED VALIDATION SUMMARY")
    print("=" * 110)
    print("Status                         : PASS")
    print(f"P1 nested-reference Log Loss   : {p1_metrics['log_loss']:.6f}")
    print(f"Nested-selected Log Loss       : {nested_metrics['log_loss']:.6f}")
    print(f"Delta nested - P1              : {observed_delta:+.6f}")
    print(
        "Relative improvement vs P1     : "
        f"{(p1_metrics['log_loss'] - nested_metrics['log_loss']) / p1_metrics['log_loss'] * 100.0:.2f}%"
    )
    print(
        f"Paired bootstrap 95% CI        : [{ci_low:.6f}, {ci_high:.6f}]"
    )
    print(f"Bootstrap conclusion           : {conclusion}")
    print(
        "Unique ensemble structures     : "
        f"{stability['unique_selected_ensemble_structures']}"
    )
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
            "step26_nested_ensemble_validation"
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        shortlist,
        ids,
        uids,
        y,
        folds,
        raw_matrix,
        provenance,
    ) = load_raw_p1_p8(
        project_root=project_root,
        shortlist_path=shortlist_path,
        expected_subjects=args.expected_subjects,
        expected_folds=args.expected_folds,
        expected_candidates=args.expected_candidates,
    )

    if args.aggregate_only:
        aggregate_outer_folds(
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
            "Provide --outer-fold 0..4 or use --aggregate-only. "
            "The supplied Slurm array wrapper runs all five outer folds."
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
        calibration_max_iter=args.calibration_max_iter,
        ece_bins=args.ece_bins,
        overwrite=args.overwrite_outer_fold,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
