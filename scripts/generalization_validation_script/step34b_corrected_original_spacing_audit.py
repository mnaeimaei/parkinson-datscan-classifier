#!/usr/bin/env python3
"""
STEP 34B — CORRECTED ORIGINAL-ACQUISITION SPACING AUDIT FOR ENS328R

Purpose
-------
Correct one diagnostic problem in Step 34:

Step 34 accidentally used the post-resampling/localization spacing (2.460 mm)
as "exact spacing", collapsing all 1,362 subjects into one spacing group.

Step 34B explicitly uses the Step-25B authoritative acquisition-domain table:
    data/generalization_validation_data/
    step25b_acquisition_domain_analysis/
    subject_acquisition_domain_table.csv

That table contains the ORIGINAL pre-resampling acquisition signatures.

Step 34B:
  * performs NO neural-network training,
  * loads NO checkpoints,
  * changes NO predictor/configuration,
  * uses NO hidden/test labels,
  * verifies UID / label / fold identity across Step 25B and Step 33,
  * verifies the expected original spacing cardinality (default 52),
  * ignores the post-resampling 2.460x2.460x2.460 field,
  * evaluates the current final temperature and Step-34 candidate temperatures
    across the true original spacing families,
  * quantifies overall LL versus exact-spacing tail LL,
  * does NOT overwrite ENS328R.

This is a diagnostic correction only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

EPS = 1e-6
CURRENT_DEPLOYMENT_T = 0.8105797487349038


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--step33-dir", type=Path, default=None)
    p.add_argument("--step25b-dir", type=Path, default=None)
    p.add_argument("--step34-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--expected-subjects", type=int, default=1362)
    p.add_argument("--expected-spacing-signatures", type=int, default=52)
    p.add_argument("--min-group-n", type=int, default=10)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--hidden-ens328-log-loss", type=float, default=0.2997)
    p.add_argument("--hidden-ens328r-log-loss", type=float, default=0.2950)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


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
        raise ValueError(f"Invalid temperature {temperature}")
    p = np.clip(np.asarray(prob, dtype=np.float64), EPS, 1.0 - EPS)
    z = np.log(p) - np.log1p(-p)
    return np.clip(sigmoid(z / float(temperature)), EPS, 1.0 - EPS)


def subject_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def binary_ll(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, np.clip(p, EPS, 1.0 - EPS), labels=[0, 1]))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    n_total = len(y)
    for i in range(bins):
        if i == bins - 1:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            m = (p >= edges[i]) & (p < edges[i + 1])
        n = int(m.sum())
        if n:
            out += (n / n_total) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(out)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "log_loss": binary_ll(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p),
        "auroc": safe_auc(y, p),
        "auprc": safe_auprc(y, p),
    }


def normalize_uid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def resolve_dir(project_root: Path, explicit: Path | None, relative: str) -> Path:
    p = explicit if explicit is not None else project_root / relative
    p = p.expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Output directory is non-empty: {path}\n"
            "Use --overwrite to replace Step34B outputs."
        )
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                import shutil
                shutil.rmtree(child)


def load_step33(path: Path, expected_subjects: int) -> pd.DataFrame:
    f = path / "step33_subject_level_predictions.csv"
    df = pd.read_csv(f)
    required = {"uid", "fold", "is_pathologic", "ens328r_raw_probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{f}: missing required columns {sorted(missing)}")
    df["uid"] = normalize_uid(df["uid"])
    if df["uid"].duplicated().any():
        raise ValueError("Step33 subject table contains duplicate UIDs")
    if len(df) != expected_subjects:
        raise ValueError(f"Step33 expected {expected_subjects} subjects, found {len(df)}")
    return df


def load_step25b(path: Path, expected_subjects: int) -> pd.DataFrame:
    f = path / "subject_acquisition_domain_table.csv"
    df = pd.read_csv(f)
    required = {
        "uid", "fold", "is_pathologic",
        "derived_spacing_signature",
        "derived_shape_signature",
        "geometry_cohort",
        "derived_voxel_volume_mm3",
        "derived_fov_x_mm",
        "derived_fov_y_mm",
        "derived_fov_z_mm",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{f}: missing required Step25B columns {sorted(missing)}")
    df["uid"] = normalize_uid(df["uid"])
    if df["uid"].duplicated().any():
        raise ValueError("Step25B acquisition table contains duplicate UIDs")
    if len(df) != expected_subjects:
        raise ValueError(f"Step25B expected {expected_subjects} subjects, found {len(df)}")
    return df


def verify_and_merge(step33: pd.DataFrame, step25b: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    u33 = set(step33["uid"])
    u25 = set(step25b["uid"])
    missing_in_25 = sorted(u33 - u25)
    missing_in_33 = sorted(u25 - u33)
    if missing_in_25 or missing_in_33:
        raise ValueError(
            f"UID mismatch: missing_in_step25b={len(missing_in_25)}, "
            f"missing_in_step33={len(missing_in_33)}"
        )

    acq_cols = [
        "uid", "fold", "is_pathologic",
        "derived_spacing_signature",
        "derived_shape_signature",
        "geometry_cohort",
        "scale_decade",
        "derived_voxel_volume_mm3",
        "derived_fov_x_mm",
        "derived_fov_y_mm",
        "derived_fov_z_mm",
    ]
    acq_cols = [c for c in acq_cols if c in step25b.columns]
    acq = step25b[acq_cols].copy()
    acq = acq.rename(columns={c: f"orig_{c}" for c in acq.columns if c != "uid"})
    merged = step33.merge(acq, on="uid", how="inner", validate="one_to_one")

    label_mismatch = int((merged["is_pathologic"].astype(int) != merged["orig_is_pathologic"].astype(int)).sum())
    fold_mismatch = int((merged["fold"].astype(int) != merged["orig_fold"].astype(int)).sum())
    if label_mismatch or fold_mismatch:
        raise ValueError(
            f"Step33 vs Step25B mismatch: labels={label_mismatch}, folds={fold_mismatch}"
        )

    wrong_spacing_unique = None
    wrong_spacing_example = None
    if "derived_spacing_signature" in merged.columns:
        wrong_spacing_unique = int(merged["derived_spacing_signature"].nunique(dropna=True))
        vals = merged["derived_spacing_signature"].dropna().astype(str).value_counts()
        wrong_spacing_example = vals.index[0] if len(vals) else None

    integrity = {
        "subjects_step33": int(len(step33)),
        "subjects_step25b": int(len(step25b)),
        "subjects_merged": int(len(merged)),
        "uid_sets_identical": True,
        "label_mismatches": label_mismatch,
        "fold_mismatches": fold_mismatch,
        "authoritative_original_spacing_column": "Step25B: derived_spacing_signature",
        "merged_original_spacing_column": "orig_derived_spacing_signature",
        "ignored_step33_post_resampling_spacing_column": (
            "derived_spacing_signature" if "derived_spacing_signature" in merged.columns else None
        ),
        "ignored_step33_spacing_unique_count": wrong_spacing_unique,
        "ignored_step33_spacing_most_common_value": wrong_spacing_example,
    }
    return merged, integrity


def load_step34_context(step34_dir: Path | None) -> dict:
    out = {
        "found": False,
        "current_temperature": CURRENT_DEPLOYMENT_T,
        "ordinary_temperature": CURRENT_DEPLOYMENT_T,
        "robust_candidate_temperature": 0.89,
        "step34_status": None,
        "nested_robust_minus_ordinary_log_loss": None,
        "nested_bootstrap_ci95": None,
    }
    if step34_dir is None:
        return out
    f = step34_dir / "step34_summary.json"
    if not f.exists():
        return out
    s = json.loads(f.read_text())
    out["found"] = True
    out["step34_status"] = s.get("status")
    out["current_temperature"] = float(s.get("current_deployment_temperature", CURRENT_DEPLOYMENT_T))
    fd = s.get("full_oof_diagnostic", {})
    out["ordinary_temperature"] = float(fd.get("ordinary_optimal_temperature", CURRENT_DEPLOYMENT_T))
    out["robust_candidate_temperature"] = float(fd.get("robust_candidate_temperature", 0.89))
    nested = s.get("nested_outer_fold", {})
    if "delta_robust_minus_ordinary_log_loss" in nested:
        out["nested_robust_minus_ordinary_log_loss"] = float(
            nested["delta_robust_minus_ordinary_log_loss"]
        )
    boot = nested.get("bootstrap", {})
    if "ci95" in boot:
        out["nested_bootstrap_ci95"] = [float(x) for x in boot["ci95"]]
    return out


def exact_spacing_group_metrics(
    df: pd.DataFrame,
    probability: np.ndarray,
    temperature: float,
) -> pd.DataFrame:
    y = df["is_pathologic"].to_numpy(dtype=int)
    space = df["orig_derived_spacing_signature"].astype(str)
    rows = []
    for group, idx0 in space.groupby(space).groups.items():
        idx = np.asarray(list(idx0), dtype=int)
        yy = y[idx]
        pp = probability[idx]
        mm = metrics(yy, pp)
        rows.append({
            "temperature": float(temperature),
            "spacing_signature": str(group),
            **mm,
        })
    return pd.DataFrame(rows).sort_values(
        ["n", "spacing_signature"], ascending=[False, True]
    ).reset_index(drop=True)


def temperature_summary(
    df: pd.DataFrame,
    temperature: float,
    min_group_n: int,
) -> tuple[dict, pd.DataFrame]:
    y = df["is_pathologic"].to_numpy(dtype=int)
    raw = df["ens328r_raw_probability"].to_numpy(dtype=float)
    p = apply_temperature(raw, temperature)
    groups = exact_spacing_group_metrics(df, p, temperature)
    eligible = groups[groups["n"] >= min_group_n].copy()
    if eligible.empty:
        raise RuntimeError("No exact-spacing groups satisfy min_group_n")
    overall = metrics(y, p)
    overall.update({
        "temperature": float(temperature),
        "exact_spacing_total_groups": int(len(groups)),
        "exact_spacing_eligible_groups": int(len(eligible)),
        "exact_spacing_min_group_n": int(min_group_n),
        "exact_spacing_p90_log_loss": float(np.quantile(eligible["log_loss"], 0.90)),
        "exact_spacing_worst_log_loss": float(eligible["log_loss"].max()),
        "exact_spacing_mean_unweighted_log_loss": float(eligible["log_loss"].mean()),
        "exact_spacing_median_log_loss": float(eligible["log_loss"].median()),
    })
    return overall, groups


def bootstrap_current_vs_candidate(
    df: pd.DataFrame,
    current_t: float,
    candidate_t: float,
    min_group_n: int,
    reps: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    y = df["is_pathologic"].to_numpy(dtype=int)
    raw = df["ens328r_raw_probability"].to_numpy(dtype=float)

    p_cur = apply_temperature(raw, current_t)
    p_can = apply_temperature(raw, candidate_t)
    loss_cur = subject_log_loss(y, p_cur)
    loss_can = subject_log_loss(y, p_can)

    spacing = df["orig_derived_spacing_signature"].astype(str)
    eligible_indices = []
    for _, idx0 in spacing.groupby(spacing).groups.items():
        idx = np.asarray(list(idx0), dtype=int)
        if len(idx) >= min_group_n:
            eligible_indices.append(idx)

    n = len(df)
    overall_delta = np.empty(reps, dtype=np.float64)
    p90_delta = np.empty(reps, dtype=np.float64)

    for r in range(reps):
        sample = rng.integers(0, n, size=n)
        overall_delta[r] = float(np.mean(loss_can[sample] - loss_cur[sample]))

        cur_group = []
        can_group = []
        for idx in eligible_indices:
            boot_idx = idx[rng.integers(0, len(idx), size=len(idx))]
            cur_group.append(float(np.mean(loss_cur[boot_idx])))
            can_group.append(float(np.mean(loss_can[boot_idx])))
        p90_delta[r] = float(np.quantile(can_group, 0.90) - np.quantile(cur_group, 0.90))

    return {
        "repetitions": int(reps),
        "seed": int(seed),
        "delta_definition": "candidate_minus_current",
        "overall_log_loss_delta": {
            "mean": float(overall_delta.mean()),
            "ci95": [
                float(np.quantile(overall_delta, 0.025)),
                float(np.quantile(overall_delta, 0.975)),
            ],
            "fraction_candidate_better": float(np.mean(overall_delta < 0)),
        },
        "exact_spacing_p90_delta": {
            "mean": float(p90_delta.mean()),
            "ci95": [
                float(np.quantile(p90_delta, 0.025)),
                float(np.quantile(p90_delta, 0.975)),
            ],
            "fraction_candidate_better": float(np.mean(p90_delta < 0)),
        },
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.expanduser().resolve()

    step33_dir = resolve_dir(
        root, args.step33_dir,
        "data/generalization_validation_data/step33_hidden_transfer_diagnosis",
    )
    step25b_dir = resolve_dir(
        root, args.step25b_dir,
        "data/generalization_validation_data/step25b_acquisition_domain_analysis",
    )

    step34_default = root / "data/generalization_validation_data/step34_domain_robust_calibration"
    step34_dir = args.step34_dir.expanduser().resolve() if args.step34_dir else (
        step34_default.resolve() if step34_default.exists() else None
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (root / "data/generalization_validation_data/step34b_corrected_original_spacing_audit").resolve()
    )
    prepare_output(output_dir, args.overwrite)

    print("=" * 92)
    print("STEP 34B — CORRECTED ORIGINAL-ACQUISITION SPACING AUDIT")
    print("=" * 92)
    print(f"Project root                 : {root}")
    print(f"Step33                       : {step33_dir}")
    print(f"Step25B ORIGINAL acquisition : {step25b_dir}")
    print(f"Step34 context               : {step34_dir if step34_dir else 'NOT FOUND'}")
    print(f"Output                       : {output_dir}")
    print("Training                     : NONE")
    print("Checkpoint loading           : NONE")
    print("Hidden labels                : NEVER USED")
    print()

    s33 = load_step33(step33_dir, args.expected_subjects)
    s25 = load_step25b(step25b_dir, args.expected_subjects)
    df, integrity = verify_and_merge(s33, s25)

    n_spacing = int(df["orig_derived_spacing_signature"].nunique(dropna=True))
    integrity["original_spacing_signature_count"] = n_spacing
    integrity["expected_original_spacing_signature_count"] = int(args.expected_spacing_signatures)

    if n_spacing != args.expected_spacing_signatures:
        raise ValueError(
            f"Expected {args.expected_spacing_signatures} ORIGINAL spacing signatures "
            f"from Step25B, found {n_spacing}. Refusing to continue."
        )

    # Explicitly verify that the wrong Step33 field is not accidentally being used.
    if "derived_spacing_signature" in df.columns:
        wrong_n = int(df["derived_spacing_signature"].nunique(dropna=True))
        integrity["post_resampling_spacing_unique_count"] = wrong_n
        if wrong_n == 1:
            integrity["post_resampling_spacing_field_status"] = "DETECTED_AND_IGNORED"
        else:
            integrity["post_resampling_spacing_field_status"] = "IGNORED_REGARDLESS"
    else:
        integrity["post_resampling_spacing_field_status"] = "NOT_PRESENT"

    (output_dir / "step34b_source_integrity.json").write_text(
        json.dumps(integrity, indent=2)
    )

    context = load_step34_context(step34_dir)
    current_t = float(context["current_temperature"])
    ordinary_t = float(context["ordinary_temperature"])
    candidate_t = float(context["robust_candidate_temperature"])

    # Round temperature keys so numerically identical values such as
    # 0.8105797487349038 and 0.810579748735 do not create duplicate
    # near-identical rows that later collide during group comparison.
    temperatures = sorted({
        round(float(x), 12)
        for x in [
            current_t,
            ordinary_t,
            candidate_t,
            0.85,
            0.90,
            0.95,
            1.00,
        ]
    })

    scan_rows = []
    group_tables = []
    for t in temperatures:
        row, groups = temperature_summary(df.reset_index(drop=True), t, args.min_group_n)
        scan_rows.append(row)
        group_tables.append(groups)

    scan = pd.DataFrame(scan_rows).sort_values("temperature").reset_index(drop=True)
    groups_long = pd.concat(group_tables, ignore_index=True)

    scan.to_csv(output_dir / "step34b_exact_spacing_temperature_scan.csv", index=False)
    groups_long.to_csv(output_dir / "step34b_exact_spacing_group_metrics.csv", index=False)

    current_eval_t = float(scan.iloc[(scan["temperature"] - current_t).abs().argmin()]["temperature"])
    candidate_eval_t = float(scan.iloc[(scan["temperature"] - candidate_t).abs().argmin()]["temperature"])

    cur_groups = groups_long[groups_long["temperature"] == current_eval_t].copy()
    can_groups = groups_long[groups_long["temperature"] == candidate_eval_t].copy()

    comparison = cur_groups.merge(
        can_groups,
        on="spacing_signature",
        how="inner",
        suffixes=("_current", "_candidate"),
        validate="one_to_one",
    )
    comparison["delta_candidate_minus_current_log_loss"] = (
        comparison["log_loss_candidate"] - comparison["log_loss_current"]
    )
    comparison["candidate_better"] = (
        comparison["delta_candidate_minus_current_log_loss"] < 0
    )
    comparison = comparison.sort_values(
        ["n_current", "spacing_signature"], ascending=[False, True]
    )
    comparison.to_csv(
        output_dir / "step34b_current_vs_candidate_by_spacing.csv", index=False
    )

    boot = bootstrap_current_vs_candidate(
        df.reset_index(drop=True),
        current_t=current_t,
        candidate_t=candidate_t,
        min_group_n=args.min_group_n,
        reps=args.bootstrap_reps,
        seed=args.seed,
    )
    (output_dir / "step34b_bootstrap.json").write_text(json.dumps(boot, indent=2))

    cur_row = scan.iloc[(scan["temperature"] - current_t).abs().argmin()]
    can_row = scan.iloc[(scan["temperature"] - candidate_t).abs().argmin()]

    overall_delta = float(can_row["log_loss"] - cur_row["log_loss"])
    p90_delta = float(
        can_row["exact_spacing_p90_log_loss"] - cur_row["exact_spacing_p90_log_loss"]
    )
    worst_delta = float(
        can_row["exact_spacing_worst_log_loss"] - cur_row["exact_spacing_worst_log_loss"]
    )

    # Step34B is intentionally conservative:
    # corrected spacing-tail improvement alone is insufficient to overwrite the
    # current predictor because the Step34 nested overall comparison did not
    # establish that the robust rule was globally superior.
    prior_nested_delta = context.get("nested_robust_minus_ordinary_log_loss")
    prior_ci = context.get("nested_bootstrap_ci95")
    prior_supports_replacement = False
    if prior_nested_delta is not None and prior_ci is not None:
        prior_supports_replacement = (
            prior_nested_delta < 0 and float(prior_ci[1]) < 0
        )

    if prior_supports_replacement and overall_delta <= 0:
        decision = "REVIEW_ROBUST_TEMPERATURE_FOR_SEPARATE_FREEZE"
        decision_reason = (
            "Corrected exact-spacing analysis and prior nested evidence both favor "
            "the robust candidate. Freeze separately before any deployment change."
        )
    else:
        decision = "KEEP_CURRENT_CALIBRATION"
        decision_reason = (
            "The corrected original-spacing analysis may favor a more conservative "
            "temperature in the spacing tail, but prior nested Step34 evidence did "
            "not establish a global held-out log-loss improvement. Keep the current "
            "ENS328R final temperature; do not overwrite the predictor."
        )

    summary = {
        "step": "34B",
        "status": "PASS",
        "decision": decision,
        "decision_reason": decision_reason,
        "subjects": int(len(df)),
        "neural_network_training": False,
        "checkpoint_loading": False,
        "hidden_subject_labels_used": False,
        "authoritative_acquisition_source": str(
            step25b_dir / "subject_acquisition_domain_table.csv"
        ),
        "original_spacing_signature_count": n_spacing,
        "expected_original_spacing_signature_count": int(args.expected_spacing_signatures),
        "eligible_spacing_groups_min_n": int(
            (cur_groups["n"] >= args.min_group_n).sum()
        ),
        "min_group_n_for_tail_metric": int(args.min_group_n),
        "current_temperature": current_t,
        "step34_robust_candidate_temperature": candidate_t,
        "current": {
            "overall_log_loss": float(cur_row["log_loss"]),
            "brier": float(cur_row["brier"]),
            "ece": float(cur_row["ece"]),
            "exact_spacing_p90_log_loss": float(cur_row["exact_spacing_p90_log_loss"]),
            "exact_spacing_worst_log_loss": float(cur_row["exact_spacing_worst_log_loss"]),
        },
        "candidate": {
            "overall_log_loss": float(can_row["log_loss"]),
            "brier": float(can_row["brier"]),
            "ece": float(can_row["ece"]),
            "exact_spacing_p90_log_loss": float(can_row["exact_spacing_p90_log_loss"]),
            "exact_spacing_worst_log_loss": float(can_row["exact_spacing_worst_log_loss"]),
        },
        "delta_candidate_minus_current": {
            "overall_log_loss": overall_delta,
            "exact_spacing_p90_log_loss": p90_delta,
            "exact_spacing_worst_log_loss": worst_delta,
        },
        "step34_prior_context": context,
        "bootstrap": boot,
        "competition_context_only": {
            "ens328_hidden_log_loss": float(args.hidden_ens328_log_loss),
            "ens328r_hidden_log_loss": float(args.hidden_ens328r_log_loss),
            "used_for_selection": False,
        },
        "interpretation": (
            "Step34B corrects only the exact ORIGINAL acquisition-spacing diagnostic. "
            "It does not retrain, recalibrate, or modify ENS328R."
        ),
    }
    (output_dir / "step34b_summary.json").write_text(json.dumps(summary, indent=2))

    eligible_comp = comparison[comparison["n_current"] >= args.min_group_n].copy()
    improved_n = int(eligible_comp["candidate_better"].sum())
    worsened_n = int((~eligible_comp["candidate_better"]).sum())

    lines = [
        "STEP 34B — CORRECTED ORIGINAL-ACQUISITION SPACING AUDIT",
        "=" * 82,
        "Status: PASS",
        f"Decision: {decision}",
        "",
        "SOURCE INTEGRITY",
        f"  Subjects matched: {len(df)}/{args.expected_subjects}",
        f"  Label mismatches: {integrity['label_mismatches']}",
        f"  Fold mismatches: {integrity['fold_mismatches']}",
        f"  ORIGINAL spacing signatures from Step25B: {n_spacing}",
        f"  Expected original spacing signatures: {args.expected_spacing_signatures}",
        f"  Wrong/post-resampling Step33 spacing field: {integrity['post_resampling_spacing_field_status']}",
        "",
        "CALIBRATION COMPARISON",
        f"  Current T:   {current_t:.9f}",
        f"  Candidate T: {candidate_t:.9f}",
        f"  Overall LL: {float(cur_row['log_loss']):.6f} -> {float(can_row['log_loss']):.6f} ({overall_delta:+.6f})",
        f"  Exact-spacing p90 LL: {float(cur_row['exact_spacing_p90_log_loss']):.6f} -> {float(can_row['exact_spacing_p90_log_loss']):.6f} ({p90_delta:+.6f})",
        f"  Exact-spacing worst LL: {float(cur_row['exact_spacing_worst_log_loss']):.6f} -> {float(can_row['exact_spacing_worst_log_loss']):.6f} ({worst_delta:+.6f})",
        f"  Eligible exact-spacing groups (N>={args.min_group_n}): {len(eligible_comp)}",
        f"    candidate better: {improved_n}",
        f"    candidate worse/equal: {worsened_n}",
        "",
        "BOOTSTRAP — candidate minus current",
        f"  Overall LL delta 95% CI: [{boot['overall_log_loss_delta']['ci95'][0]:+.6f}, {boot['overall_log_loss_delta']['ci95'][1]:+.6f}]",
        f"  Candidate better overall fraction: {boot['overall_log_loss_delta']['fraction_candidate_better']:.4f}",
        f"  Exact-spacing p90 delta 95% CI: [{boot['exact_spacing_p90_delta']['ci95'][0]:+.6f}, {boot['exact_spacing_p90_delta']['ci95'][1]:+.6f}]",
        f"  Candidate better spacing-p90 fraction: {boot['exact_spacing_p90_delta']['fraction_candidate_better']:.4f}",
        "",
        "DECISION",
        f"  {decision_reason}",
        "",
        "No NN training. No checkpoint loading. No hidden labels used.",
    ]
    (output_dir / "step34b_report.txt").write_text("\n".join(lines) + "\n")

    print("SOURCE CHECK")
    print(f"  Subjects                  : {len(df)}/{args.expected_subjects}")
    print(f"  Original spacing groups   : {n_spacing}")
    print(f"  Expected                  : {args.expected_spacing_signatures}")
    print(f"  Post-resampling field     : {integrity['post_resampling_spacing_field_status']}")
    print()
    print("RESULT")
    print(f"  Current T                 : {current_t:.9f}")
    print(f"  Candidate T               : {candidate_t:.9f}")
    print(f"  Overall LL                : {float(cur_row['log_loss']):.6f} -> {float(can_row['log_loss']):.6f}")
    print(f"  Exact-spacing p90 LL      : {float(cur_row['exact_spacing_p90_log_loss']):.6f} -> {float(can_row['exact_spacing_p90_log_loss']):.6f}")
    print(f"  Exact-spacing worst LL    : {float(cur_row['exact_spacing_worst_log_loss']):.6f} -> {float(can_row['exact_spacing_worst_log_loss']):.6f}")
    print(f"  Decision                  : {decision}")
    print()
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
