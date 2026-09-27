#!/usr/bin/env python3
"""
STEP 34 — DOMAIN-ROBUST FINAL-CALIBRATION ANALYSIS FOR ENS328R

CPU-only diagnostic. No neural-network training, no checkpoint loading, no hidden labels.

Purpose
-------
Evaluate whether a slightly more conservative final temperature than the current
ENS328R deployment value improves acquisition-domain robustness without paying
an unacceptable ordinary-OOF log-loss penalty.

Key safeguards
--------------
1. Uses only Step-33 training OOF predictions + training metadata.
2. Never fits to competition/test labels.
3. Performs OUTER-FOLD nested temperature selection:
   - ordinary rule: minimize training-fold overall LL.
   - robust rule: allow temperatures within `overall_slack` of best overall LL,
     then choose the one with the smallest training-domain p90 LL.
4. Reports a full-OOF deployment candidate only as a POST-HOC candidate. It is
   not claimed to be independent validation.
5. Exact spacing signatures are explicitly included (Step 33 intentionally
   skipped them from its generic low-cardinality subgroup table).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

EPS = 1e-6
CURRENT_DEPLOYMENT_T = 0.8105797487349038

PRIMARY_DOMAIN_FEATURES = [
    "derived_spacing_signature",
    "subject_acquisition_domain_table__geometry_cohort",
    "subject_acquisition_domain_table__scale_decade",
    "subject_acquisition_domain_table__derived_voxel_volume_mm3__q5",
    "subject_acquisition_domain_table__derived_fov_z_mm__q5",
    "step7c_l1_localization__transform_type",
]

KNOWN_METADATA_BASENAMES = [
    "final_registration_manifest.csv",
    "final_reference_values.csv",
    "final_localization.csv",
    "final_striatal_crop_manifest.csv",
    "fixed_whole_volume_manifest.csv",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def apply_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Invalid temperature: {temperature}")
    p = np.clip(np.asarray(prob, dtype=np.float64), EPS, 1.0 - EPS)
    z = np.log(p) - np.log1p(-p)
    return np.clip(sigmoid(z / float(temperature)), EPS, 1.0 - EPS)


def binary_ll(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
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
        if n:
            value += (n / total) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(value)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "mean_confidence": float(np.mean(np.maximum(p, 1 - p))),
        "log_loss": binary_ll(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p),
        "auroc": safe_auc(y, p),
        "auprc": safe_auprc(y, p),
    }


def normalize_uid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def find_step33_dir(project_root: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(p)
    p = project_root / "data/generalization_validation_data/step33_hidden_transfer_diagnosis"
    if not p.exists():
        raise FileNotFoundError(f"Step33 output directory not found: {p}")
    return p


def load_step33(step33_dir: Path, expected_subjects: int) -> pd.DataFrame:
    path = step33_dir / "step33_subject_level_predictions.csv"
    df = pd.read_csv(path)
    required = {"uid", "fold", "is_pathologic", "ens328r_raw_probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["uid"] = normalize_uid(df["uid"])
    if df["uid"].duplicated().any():
        raise ValueError("Step33 subject table contains duplicate UIDs")
    if len(df) != expected_subjects:
        raise ValueError(f"Expected {expected_subjects} subjects, found {len(df)}")
    if set(df["is_pathologic"].unique()) - {0, 1}:
        raise ValueError("is_pathologic must be binary 0/1")
    return df


def choose_uid_col(df: pd.DataFrame) -> str | None:
    candidates = ["uid", "subject_uid", "scan_uid", "id"]
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def merge_known_metadata(df: pd.DataFrame, project_root: Path, log_rows: list[dict]) -> pd.DataFrame:
    """Best-effort merge of known final preprocessing manifests with high UID coverage."""
    data_root = project_root / "data"
    if not data_root.exists():
        return df
    base_uids = set(df["uid"])
    out = df.copy()
    for basename in KNOWN_METADATA_BASENAMES:
        candidates = sorted(data_root.rglob(basename))
        if not candidates:
            log_rows.append({"basename": basename, "path": "", "status": "NOT_FOUND", "coverage": 0.0, "columns_added": 0})
            continue
        best = None
        best_cov = -1.0
        best_df = None
        best_uid_col = None
        for path in candidates:
            try:
                m = pd.read_csv(path)
            except Exception:
                continue
            uid_col = choose_uid_col(m)
            if uid_col is None:
                continue
            uids = normalize_uid(m[uid_col])
            cov = len(base_uids.intersection(set(uids))) / len(base_uids)
            if cov > best_cov:
                best, best_cov, best_df, best_uid_col = path, cov, m, uid_col
        if best is None:
            log_rows.append({"basename": basename, "path": "", "status": "NO_UID_COLUMN", "coverage": 0.0, "columns_added": 0})
            continue
        if best_cov < 0.90:
            log_rows.append({"basename": basename, "path": str(best), "status": "SKIP_LOW_COVERAGE", "coverage": best_cov, "columns_added": 0})
            continue
        m = best_df.copy()
        m[best_uid_col] = normalize_uid(m[best_uid_col])
        m = m[m[best_uid_col].isin(base_uids)].copy()
        if m[best_uid_col].duplicated().any():
            log_rows.append({"basename": basename, "path": str(best), "status": "SKIP_DUPLICATE_UID", "coverage": best_cov, "columns_added": 0})
            continue
        prefix = best.stem + "__"
        keep = [c for c in m.columns if c != best_uid_col]
        # Exclude path-heavy columns from calibration-domain selection but keep useful scalars/categories.
        rename = {c: prefix + c for c in keep if prefix + c not in out.columns}
        m = m[[best_uid_col] + list(rename)].rename(columns={best_uid_col: "uid", **rename})
        before = set(out.columns)
        out = out.merge(m, on="uid", how="left", validate="one_to_one")
        added = len(set(out.columns) - before)
        log_rows.append({"basename": basename, "path": str(best), "status": "USED", "coverage": best_cov, "columns_added": added})
    return out


def infer_extra_domain_features(df: pd.DataFrame) -> list[str]:
    extras = []
    preferred_tokens = [
        "reference_method", "reference_rescued", "final_source_type", "final_candidate",
        "final_qc", "localization_method", "fallback_used", "localization_confidence",
        "qc_status", "final_registration_qc", "geometry_cohort", "scale_decade",
    ]
    for col in df.columns:
        lc = col.lower()
        if col in PRIMARY_DOMAIN_FEATURES:
            continue
        if any(tok in lc for tok in preferred_tokens):
            nunique = df[col].nunique(dropna=True)
            if 2 <= nunique <= 30:
                extras.append(col)
    return extras


def eligible_features(df: pd.DataFrame) -> list[str]:
    features = [f for f in PRIMARY_DOMAIN_FEATURES if f in df.columns]
    for f in infer_extra_domain_features(df):
        if f not in features:
            features.append(f)
    return features


def group_loss_rows(df: pd.DataFrame, p: np.ndarray, features: Iterable[str], min_group_n: int) -> pd.DataFrame:
    rows = []
    y_all = df["is_pathologic"].to_numpy(dtype=int)
    tmp = df.copy()
    tmp["__p"] = p
    for feature in features:
        if feature not in tmp.columns:
            continue
        s = tmp[feature]
        # stringify but preserve missing marker
        vals = s.where(s.notna(), "<MISSING>").astype(str)
        for group, idx in vals.groupby(vals).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            if len(idx) < min_group_n:
                continue
            y = y_all[idx]
            pp = p[idx]
            rows.append({
                "feature": feature,
                "group": str(group),
                "n": int(len(idx)),
                "positive_rate": float(np.mean(y)),
                "log_loss": binary_ll(y, pp),
                "brier": brier(y, pp),
                "ece": ece(y, pp, bins=10),
            })
    return pd.DataFrame(rows)


def domain_summary(group_df: pd.DataFrame) -> dict:
    if group_df.empty:
        return {
            "groups": 0, "domain_mean_ll": float("nan"), "domain_p90_ll": float("nan"),
            "domain_worst_ll": float("nan"), "domain_ll_sd": float("nan")
        }
    x = group_df["log_loss"].to_numpy(dtype=float)
    return {
        "groups": int(len(x)),
        "domain_mean_ll": float(np.mean(x)),
        "domain_p90_ll": float(np.quantile(x, 0.90)),
        "domain_worst_ll": float(np.max(x)),
        "domain_ll_sd": float(np.std(x, ddof=0)),
    }


def evaluate_temperature(df: pd.DataFrame, temperature: float, features: list[str], min_group_n: int) -> tuple[dict, pd.DataFrame]:
    p = apply_temperature(df["ens328r_raw_probability"].to_numpy(dtype=float), temperature)
    y = df["is_pathologic"].to_numpy(dtype=int)
    m = metrics(y, p)
    g = group_loss_rows(df.reset_index(drop=True), p, features, min_group_n)
    d = domain_summary(g)
    return {"temperature": float(temperature), **m, **d}, g


def make_grid(t_min: float, t_max: float, t_step: float, anchors: list[float]) -> list[float]:
    vals = list(np.arange(t_min, t_max + t_step / 2, t_step)) + anchors
    vals = sorted({round(float(v), 12) for v in vals if v > 0})
    return vals


def select_ordinary(scan: pd.DataFrame) -> pd.Series:
    return scan.sort_values(["log_loss", "temperature"], ascending=[True, True]).iloc[0]


def select_robust(scan: pd.DataFrame, slack: float) -> pd.Series:
    best_ll = float(scan["log_loss"].min())
    eligible = scan[scan["log_loss"] <= best_ll + slack + 1e-15].copy()
    # If no domain groups exist, fall back to ordinary.
    if eligible["domain_p90_ll"].notna().sum() == 0:
        return select_ordinary(scan)
    return eligible.sort_values(
        ["domain_p90_ll", "domain_worst_ll", "log_loss", "temperature"],
        ascending=[True, True, True, True]
    ).iloc[0]


def nested_selection(df: pd.DataFrame, grid: list[float], features: list[str], train_min_group_n: int, eval_min_group_n: int, slack: float):
    pred_rows = []
    fold_rows = []
    scan_rows = []
    folds = sorted(df["fold"].astype(int).unique())
    for heldout in folds:
        train = df[df["fold"].astype(int) != heldout].reset_index(drop=True)
        val = df[df["fold"].astype(int) == heldout].reset_index(drop=True)
        scans = []
        for t in grid:
            r, _ = evaluate_temperature(train, t, features, train_min_group_n)
            r["heldout_fold"] = int(heldout)
            scans.append(r)
        scan = pd.DataFrame(scans)
        scan_rows.append(scan)
        ordinary = select_ordinary(scan)
        robust = select_robust(scan, slack)

        raw = val["ens328r_raw_probability"].to_numpy(dtype=float)
        y = val["is_pathologic"].to_numpy(dtype=int)
        p_ord = apply_temperature(raw, float(ordinary["temperature"]))
        p_rob = apply_temperature(raw, float(robust["temperature"]))
        p_current = apply_temperature(raw, CURRENT_DEPLOYMENT_T)
        p_none = raw.copy()

        for rule, temp, pp in [
            ("ordinary_nested", float(ordinary["temperature"]), p_ord),
            ("robust_nested", float(robust["temperature"]), p_rob),
            ("current_fixed_T", CURRENT_DEPLOYMENT_T, p_current),
            ("no_final_temperature", 1.0, p_none),
        ]:
            mm = metrics(y, pp)
            gg = group_loss_rows(val.reset_index(drop=True), pp, features, eval_min_group_n)
            dd = domain_summary(gg)
            fold_rows.append({
                "heldout_fold": int(heldout), "rule": rule, "selected_temperature": float(temp),
                **mm, **dd,
            })
        for i, row in val.iterrows():
            pred_rows.append({
                "uid": row["uid"], "fold": int(heldout), "is_pathologic": int(row["is_pathologic"]),
                "raw_probability": float(raw[i]),
                "ordinary_nested_temperature": float(ordinary["temperature"]),
                "ordinary_nested_probability": float(p_ord[i]),
                "robust_nested_temperature": float(robust["temperature"]),
                "robust_nested_probability": float(p_rob[i]),
                "current_fixed_probability": float(p_current[i]),
            })
    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows), pd.concat(scan_rows, ignore_index=True)


def bootstrap_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, reps: int, seed: int) -> dict:
    # delta = B - A; negative means B better.
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(reps, dtype=float)
    for r in range(reps):
        idx = rng.integers(0, n, size=n)
        deltas[r] = binary_ll(y[idx], p_b[idx]) - binary_ll(y[idx], p_a[idx])
    return {
        "replicates": int(reps), "seed": int(seed),
        "observed_delta_B_minus_A": float(binary_ll(y, p_b) - binary_ll(y, p_a)),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "fraction_B_better": float(np.mean(deltas < 0)),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--step33-dir", default=None)
    ap.add_argument("--output-dir", default="data/generalization_validation_data/step34_domain_robust_calibration")
    ap.add_argument("--expected-subjects", type=int, default=1362)
    ap.add_argument("--t-min", type=float, default=0.70)
    ap.add_argument("--t-max", type=float, default=1.05)
    ap.add_argument("--t-step", type=float, default=0.01)
    ap.add_argument("--overall-slack", type=float, default=0.0015,
                    help="Max training overall-LL cost above ordinary optimum allowed before optimizing domain p90.")
    ap.add_argument("--train-min-group-n", type=int, default=20)
    ap.add_argument("--eval-min-group-n", type=int, default=8)
    ap.add_argument("--bootstrap-reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2034)
    ap.add_argument("--hidden-ens328r-log-loss", type=float, default=0.2950,
                    help="Context only; never used for fitting/selection.")
    return ap.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    step33_dir = find_step33_dir(project_root, args.step33_dir)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("STEP 34 — DOMAIN-ROBUST FINAL-CALIBRATION ANALYSIS")
    print("=" * 96)
    print(f"Project root             : {project_root}")
    print(f"Step33 input             : {step33_dir}")
    print(f"Output                   : {output_dir}")
    print(f"Current ENS328R final T  : {CURRENT_DEPLOYMENT_T:.9f}")
    print(f"Temperature grid         : {args.t_min:.2f}..{args.t_max:.2f} step {args.t_step:.2f}")
    print(f"Robust LL slack          : {args.overall_slack:.6f}")
    print("Training                 : NONE")
    print("Checkpoint loading       : NONE")
    print("Hidden labels            : NEVER USED")
    print()

    df = load_step33(step33_dir, args.expected_subjects)
    metadata_log: list[dict] = []
    df = merge_known_metadata(df, project_root, metadata_log)
    pd.DataFrame(metadata_log).to_csv(output_dir / "step34_metadata_sources.csv", index=False)

    features = eligible_features(df)
    if not features:
        raise RuntimeError("No domain features available for Step34")
    print(f"Domain features          : {len(features)}")
    for f in features:
        print(f"  - {f}")
    print()

    grid = make_grid(args.t_min, args.t_max, args.t_step, [CURRENT_DEPLOYMENT_T, 0.85, 0.90, 0.95, 1.0])

    # ------------------------------------------------------------------
    # Full-OOF fixed-temperature scan (diagnostic, not independent)
    # ------------------------------------------------------------------
    full_rows = []
    all_group_rows = []
    for t in grid:
        r, g = evaluate_temperature(df.reset_index(drop=True), t, features, args.train_min_group_n)
        full_rows.append(r)
        if not g.empty:
            g = g.copy()
            g.insert(0, "temperature", float(t))
            all_group_rows.append(g)
    full_scan = pd.DataFrame(full_rows)
    full_scan.to_csv(output_dir / "step34_full_oof_temperature_scan.csv", index=False)
    if all_group_rows:
        pd.concat(all_group_rows, ignore_index=True).to_csv(output_dir / "step34_full_oof_domain_scan.csv", index=False)

    full_ordinary = select_ordinary(full_scan)
    full_robust = select_robust(full_scan, args.overall_slack)

    # ------------------------------------------------------------------
    # Nested outer-fold selection
    # ------------------------------------------------------------------
    nested_pred, nested_fold, nested_train_scan = nested_selection(
        df.reset_index(drop=True), grid, features,
        args.train_min_group_n, args.eval_min_group_n, args.overall_slack,
    )
    nested_pred.to_csv(output_dir / "step34_nested_predictions.csv", index=False)
    nested_fold.to_csv(output_dir / "step34_nested_fold_metrics.csv", index=False)
    nested_train_scan.to_csv(output_dir / "step34_nested_training_temperature_scan.csv", index=False)

    y = nested_pred["is_pathologic"].to_numpy(dtype=int)
    rule_cols = {
        "ordinary_nested": "ordinary_nested_probability",
        "robust_nested": "robust_nested_probability",
        "current_fixed_T": "current_fixed_probability",
        "no_final_temperature": "raw_probability",
    }
    overall_rows = []
    for rule, col in rule_cols.items():
        mm = metrics(y, nested_pred[col].to_numpy(dtype=float))
        overall_rows.append({"rule": rule, **mm})
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(output_dir / "step34_nested_overall_metrics.csv", index=False)

    boot = bootstrap_delta(
        y,
        nested_pred["ordinary_nested_probability"].to_numpy(dtype=float),
        nested_pred["robust_nested_probability"].to_numpy(dtype=float),
        args.bootstrap_reps, args.seed,
    )
    (output_dir / "step34_nested_robust_vs_ordinary_bootstrap.json").write_text(json.dumps(boot, indent=2))

    # ------------------------------------------------------------------
    # Exact spacing report at important temperatures
    # ------------------------------------------------------------------
    spacing_feature = "derived_spacing_signature" if "derived_spacing_signature" in df.columns else None
    important_temps = sorted({CURRENT_DEPLOYMENT_T, float(full_ordinary["temperature"]), float(full_robust["temperature"]), 0.85, 0.90, 0.95, 1.0})
    spacing_rows = []
    if spacing_feature:
        for t in important_temps:
            p = apply_temperature(df["ens328r_raw_probability"].to_numpy(dtype=float), t)
            tmp = df[["uid", "is_pathologic", spacing_feature]].copy().reset_index(drop=True)
            g = group_loss_rows(tmp, p, [spacing_feature], min_group_n=10)
            if not g.empty:
                g.insert(0, "temperature", t)
                spacing_rows.append(g)
    if spacing_rows:
        pd.concat(spacing_rows, ignore_index=True).to_csv(output_dir / "step34_exact_spacing_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # Decision / report
    # ------------------------------------------------------------------
    ord_row = overall[overall["rule"] == "ordinary_nested"].iloc[0]
    rob_row = overall[overall["rule"] == "robust_nested"].iloc[0]
    ord_fold = nested_fold[nested_fold["rule"] == "ordinary_nested"]
    rob_fold = nested_fold[nested_fold["rule"] == "robust_nested"]

    nested_delta = float(rob_row["log_loss"] - ord_row["log_loss"])
    ord_p90 = float(ord_fold["domain_p90_ll"].mean())
    rob_p90 = float(rob_fold["domain_p90_ll"].mean())
    p90_delta = rob_p90 - ord_p90
    robust_temps = rob_fold["selected_temperature"].to_numpy(dtype=float)
    ordinary_temps = ord_fold["selected_temperature"].to_numpy(dtype=float)

    # Conservative decision: only promote a new temperature if nested robust selection
    # improves average heldout domain p90 and keeps overall nested LL within the same slack.
    if p90_delta < -1e-6 and nested_delta <= args.overall_slack:
        status = "PROMISING_ROBUST_TEMPERATURE"
        recommendation = (
            "Robust temperature selection improves held-out domain-tail calibration while keeping "
            "overall nested OOF log loss within the predefined slack. Treat the full-OOF robust T "
            "as a candidate for a separately frozen submission; do not overwrite ENS328R yet."
        )
    else:
        status = "KEEP_CURRENT_CALIBRATION"
        recommendation = (
            "Robust temperature selection does not show a sufficiently favorable held-out domain-tail/overall "
            "trade-off. Keep the current ENS328R final calibration and move to robustifying another complementary member."
        )

    summary = {
        "step": 34,
        "status": status,
        "subjects": int(len(df)),
        "neural_network_training": False,
        "checkpoint_loading": False,
        "hidden_subject_labels_used": False,
        "hidden_ens328r_log_loss_context_only": float(args.hidden_ens328r_log_loss),
        "current_deployment_temperature": CURRENT_DEPLOYMENT_T,
        "temperature_grid": {"min": args.t_min, "max": args.t_max, "step": args.t_step},
        "robust_selection_rule": {
            "stage1": "find minimum training overall log loss",
            "stage2": f"retain temperatures within +{args.overall_slack:.6f} LL",
            "stage3": "among retained temperatures minimize training acquisition-domain p90 log loss",
        },
        "domain_features": features,
        "full_oof_diagnostic": {
            "ordinary_optimal_temperature": float(full_ordinary["temperature"]),
            "ordinary_optimal_log_loss": float(full_ordinary["log_loss"]),
            "robust_candidate_temperature": float(full_robust["temperature"]),
            "robust_candidate_log_loss": float(full_robust["log_loss"]),
            "robust_candidate_domain_p90_ll": float(full_robust["domain_p90_ll"]),
            "warning": "post-hoc on all OOF; not independent validation",
        },
        "nested_outer_fold": {
            "ordinary_selected_temperatures": ordinary_temps.tolist(),
            "robust_selected_temperatures": robust_temps.tolist(),
            "ordinary_global_log_loss": float(ord_row["log_loss"]),
            "robust_global_log_loss": float(rob_row["log_loss"]),
            "delta_robust_minus_ordinary_log_loss": nested_delta,
            "ordinary_mean_fold_domain_p90_ll": ord_p90,
            "robust_mean_fold_domain_p90_ll": rob_p90,
            "delta_robust_minus_ordinary_domain_p90_ll": p90_delta,
            "bootstrap": boot,
        },
        "recommendation": recommendation,
        "important_interpretation": (
            "Step34 validates a calibration-selection RULE on training OOF folds. It cannot prove improvement on the hidden competition set. "
            "Competition score is context only and never participates in selection."
        ),
    }
    (output_dir / "step34_summary.json").write_text(json.dumps(summary, indent=2))

    lines = []
    lines.append("STEP 34 — DOMAIN-ROBUST FINAL-CALIBRATION ANALYSIS")
    lines.append("=" * 78)
    lines.append(f"Status: {status}")
    lines.append("")
    lines.append(f"Current deployment T: {CURRENT_DEPLOYMENT_T:.9f}")
    lines.append(f"Full-OOF ordinary optimum T: {float(full_ordinary['temperature']):.6f}  LL={float(full_ordinary['log_loss']):.6f}")
    lines.append(f"Full-OOF robust candidate T: {float(full_robust['temperature']):.6f}  LL={float(full_robust['log_loss']):.6f}  domain-p90={float(full_robust['domain_p90_ll']):.6f}")
    lines.append("  NOTE: full-OOF values are post-hoc diagnostics, not independent validation.")
    lines.append("")
    lines.append("Nested outer-fold selection:")
    lines.append(f"  Ordinary selected T by fold: {', '.join(f'{x:.3f}' for x in ordinary_temps)}")
    lines.append(f"  Robust   selected T by fold: {', '.join(f'{x:.3f}' for x in robust_temps)}")
    lines.append(f"  Ordinary global LL: {float(ord_row['log_loss']):.6f}")
    lines.append(f"  Robust   global LL: {float(rob_row['log_loss']):.6f}")
    lines.append(f"  Delta robust-ordinary LL: {nested_delta:+.6f}")
    lines.append(f"  Mean fold domain-p90: {ord_p90:.6f} -> {rob_p90:.6f} ({p90_delta:+.6f})")
    lines.append(f"  Bootstrap 95% CI for LL delta: [{boot['ci95'][0]:+.6f}, {boot['ci95'][1]:+.6f}]")
    lines.append(f"  Fraction robust better: {boot['fraction_B_better']:.4f}")
    lines.append("")
    lines.append("Recommendation:")
    lines.append(recommendation)
    lines.append("")
    lines.append("No NN training. No checkpoint loading. No hidden labels used.")
    (output_dir / "step34_report.txt").write_text("\n".join(lines) + "\n")

    print("STEP 34 COMPLETE")
    print(f"Status                   : {status}")
    print(f"Full OOF ordinary T      : {float(full_ordinary['temperature']):.6f}")
    print(f"Full OOF robust T        : {float(full_robust['temperature']):.6f}")
    print(f"Nested ordinary LL       : {float(ord_row['log_loss']):.6f}")
    print(f"Nested robust LL         : {float(rob_row['log_loss']):.6f}")
    print(f"Nested domain-p90 delta  : {p90_delta:+.6f}")
    print(f"Report                   : {output_dir / 'step34_report.txt'}")


if __name__ == "__main__":
    main()
