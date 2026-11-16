#!/usr/bin/env python3
"""Aggregate Step35C matched P2/E7 baseline-vs-robust stress predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)

EPS = 1e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "data/generalization_validation_data/"
            "step35c_e7p2_acquisition_robust_domain_stress"
        ),
    )
    p.add_argument("--bootstrap-reps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=3503)
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def find_col(df, names, label):
    for n in names:
        if n in df.columns:
            return n
    raise ValueError(f"Missing {label}; columns={list(df.columns)}")


def load_arm(base: Path, arm: str) -> pd.DataFrame:
    parts = []
    for fold in range(3):
        path = base / arm / f"fold_{fold}" / "predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(path)

        d = pd.read_csv(path)
        uid = find_col(d, ("uid", "UID", "subject_uid", "subject_id"), "UID")
        label = find_col(
            d, ("is_pathologic", "label", "target", "y_true"), "label"
        )
        prob = find_col(
            d,
            ("probability", "prob", "y_prob", "positive_probability"),
            "probability",
        )

        parts.append(
            pd.DataFrame(
                {
                    "uid": d[uid].astype(str),
                    "is_pathologic": pd.to_numeric(
                        d[label], errors="raise"
                    ).astype(int),
                    "probability": pd.to_numeric(
                        d[prob], errors="raise"
                    ).astype(float),
                    "fold": fold,
                }
            )
        )

    out = pd.concat(parts, ignore_index=True)
    if len(out) != 1362:
        raise ValueError(
            f"{arm}: expected 1362 OOF rows, found {len(out)}."
        )
    if out["uid"].duplicated().any():
        raise ValueError(f"{arm}: duplicate OOF UIDs.")
    return out


def ece(y, p, bins=15):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.sum():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def metrics(df):
    y = df["is_pathologic"].to_numpy(dtype=int)
    p = np.clip(df["probability"].to_numpy(dtype=float), EPS, 1 - EPS)
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(df)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece(y, p),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def subject_ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    base = resolve(root, args.input_dir)

    baseline = load_arm(base, "baseline")
    robust = load_arm(base, "robust")

    paired = baseline.merge(
        robust,
        on="uid",
        suffixes=("_baseline", "_robust"),
        validate="one_to_one",
    )

    if not np.array_equal(
        paired["is_pathologic_baseline"].values,
        paired["is_pathologic_robust"].values,
    ):
        raise ValueError("Label mismatch baseline vs robust.")

    if not np.array_equal(
        paired["fold_baseline"].values,
        paired["fold_robust"].values,
    ):
        raise ValueError("Fold mismatch baseline vs robust.")

    mb = metrics(baseline)
    mr = metrics(robust)

    fold_rows = []
    for fold in range(3):
        b = metrics(baseline[baseline["fold"] == fold])
        r = metrics(robust[robust["fold"] == fold])
        fold_rows.append(
            {
                "fold": fold,
                "n": b["n"],
                "baseline_log_loss": b["log_loss"],
                "robust_log_loss": r["log_loss"],
                "delta_robust_minus_baseline_log_loss": (
                    r["log_loss"] - b["log_loss"]
                ),
                "baseline_auroc": b["auroc"],
                "robust_auroc": r["auroc"],
                "baseline_auprc": b["auprc"],
                "robust_auprc": r["auprc"],
                "baseline_brier": b["brier"],
                "robust_brier": r["brier"],
                "baseline_ece": b["ece"],
                "robust_ece": r["ece"],
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(base / "step35c_fold_comparison.csv", index=False)

    y = paired["is_pathologic_baseline"].to_numpy(dtype=int)
    loss_b = subject_ll(
        y, paired["probability_baseline"].to_numpy(dtype=float)
    )
    loss_r = subject_ll(
        y, paired["probability_robust"].to_numpy(dtype=float)
    )
    delta = loss_r - loss_b

    rng = np.random.default_rng(args.seed)
    n = len(paired)
    boot = np.empty(args.bootstrap_reps, dtype=np.float64)
    for i in range(args.bootstrap_reps):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(delta[idx].mean())

    ci = [
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    ]
    observed = float(delta.mean())
    wins = int(
        (fold_df["delta_robust_minus_baseline_log_loss"] < 0).sum()
    )

    # Same screening gate used for Step35A.
    if observed < 0 and wins >= 2:
        status = "PROMISING_PASS_TO_STEP35D"
    else:
        status = "DO_NOT_ADVANCE"

    summary = {
        "step": "35C",
        "status": status,
        "target_member": "P2",
        "comparison": (
            "E7-S4 standard inplane_2d augmentation baseline vs "
            "same E7-S4 + acquisition-robust perturbations"
        ),
        "baseline": mb,
        "robust": mr,
        "delta_robust_minus_baseline": {
            "log_loss": mr["log_loss"] - mb["log_loss"],
            "auroc": mr["auroc"] - mb["auroc"],
            "auprc": mr["auprc"] - mb["auprc"],
            "brier": mr["brier"] - mb["brier"],
            "ece": mr["ece"] - mb["ece"],
            "balanced_accuracy": (
                mr["balanced_accuracy"] - mb["balanced_accuracy"]
            ),
        },
        "folds_robust_better_log_loss": wins,
        "paired_bootstrap": {
            "repetitions": int(args.bootstrap_reps),
            "observed_delta_log_loss": observed,
            "ci95": ci,
            "fraction_robust_better": float(np.mean(boot < 0)),
        },
        "decision_rule": (
            "Advance only if robust global stress LL is lower and robust "
            "wins >=2/3 stress folds. Original Step11 confirmation remains "
            "mandatory before integration."
        ),
        "hidden_competition_data_used": False,
    }

    (base / "step35c_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    report = [
        "STEP 35C — P2/E7 ACQUISITION-ROBUST DOMAIN-STRESS COMPARISON",
        "=" * 88,
        f"Status: {status}",
        "",
        "Scientific comparison:",
        "  baseline = E7-S4 standard inplane_2d augmentation",
        "  robust   = same E7-S4 + Step28 acquisition perturbations",
        "",
        f"Baseline LL    : {mb['log_loss']:.6f}",
        f"Robust LL      : {mr['log_loss']:.6f}",
        f"Delta          : {mr['log_loss'] - mb['log_loss']:+.6f}",
        f"Baseline AUROC : {mb['auroc']:.6f}",
        f"Robust AUROC   : {mr['auroc']:.6f}",
        f"Baseline AUPRC : {mb['auprc']:.6f}",
        f"Robust AUPRC   : {mr['auprc']:.6f}",
        f"Baseline Brier : {mb['brier']:.6f}",
        f"Robust Brier   : {mr['brier']:.6f}",
        f"Baseline ECE   : {mb['ece']:.6f}",
        f"Robust ECE     : {mr['ece']:.6f}",
        "",
        f"Stress folds improved: {wins}/3",
        (
            "Bootstrap 95% CI robust-baseline LL: "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        ),
        (
            "Bootstrap fraction robust better: "
            f"{float(np.mean(boot < 0)):.4f}"
        ),
        "",
        "Decision rule:",
        "  global robust LL < baseline AND >=2/3 folds improve",
        "",
        f"Decision: {status}",
        "",
        "No hidden competition labels were used.",
    ]

    (base / "step35c_report.txt").write_text("\n".join(report) + "\n")

    paired["subject_log_loss_baseline"] = loss_b
    paired["subject_log_loss_robust"] = loss_r
    paired["delta_robust_minus_baseline_log_loss"] = delta
    paired.to_csv(
        base / "step35c_paired_oof_predictions.csv", index=False
    )

    print("\n".join(report))


if __name__ == "__main__":
    main()
