#!/usr/bin/env python3
"""
STEP 29 — ORDINARY STEP-11 5-FOLD CONFIRMATION OF ROBUST E6-S5

Purpose
-------
Confirm whether the Step-28 acquisition-robust augmentation improves domain
robustness WITHOUT materially sacrificing ordinary in-distribution performance.

This experiment returns to the ORIGINAL Step-11 five-fold CV.

Controlled comparison
---------------------
Original P1 / E6-S5:
    E6 R3D-18 Kinetics-400
    ROI
    original S5 training augmentation
    original Step-11 folds

Step 29:
    E6 R3D-18 Kinetics-400
    ROI
    original S5 training augmentation
    + Step-28 acquisition-robust augmentation
    original Step-11 folds

UNCHANGED
---------
- src/
- E6 architecture
- Kinetics-400 initialization
- ROI preprocessing / frozen input files
- original Step-11 fold membership
- loss
- optimizer
- scheduler
- early stopping
- checkpoint selection
- fold seed convention
- AMP
- validation pipeline
- evaluation implementation

ONLY INTENDED CHANGE
--------------------
Training augmentation:
    original S5 augmentation
        +
    Step-28 acquisition-robust augmentation

Validation remains completely unaugmented.

Primary comparison
------------------
New Step-29 robust 5-fold OOF probabilities
vs
Original P1 raw 5-fold OOF probabilities

P1's authoritative Step-16 calibration method is NONE, therefore the fair
comparison is against P1 `raw_probability`.

No ensemble construction is performed in Step 29.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
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

EXPECTED_SUBJECTS = 1362
EXPECTED_FOLDS = 5
MODEL_NAME = "model06_r3d18_kinetics400_pretrained"
SCENARIO_ID = 5
INPUT_TYPE = "roi"

DEFAULT_MANIFEST = Path(
    "data/preprocessing_supervised_data/"
    "step10_supervised_dataset_manifest_data/"
    "supervised_dataset/supervised_dataset_manifest.csv"
)

DEFAULT_FOLDS = Path(
    "data/preprocessing_supervised_data/"
    "step11_create_freeze_cv_splits_data/"
    "fold_assignments.csv"
)

DEFAULT_SHORTLIST = Path(
    "data/calibration_validation_data/"
    "competition_shortlist_data/"
    "competition_shortlist_8.csv"
)

DEFAULT_OUTPUT = Path(
    "data/generalization_validation_data/"
    "step29_e6s5_robust_step11_confirmation"
)


# =============================================================================
# CLI / filesystem
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 29: confirm Step-28 robust E6-S5 under original "
            "Step-11 5-fold CV."
        )
    )

    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--folds", type=Path, default=None)
    p.add_argument("--shortlist", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)

    p.add_argument(
        "--fold",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=None,
        help="Train one original Step-11 fold.",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Aggregate all five completed Step-29 folds.",
    )
    p.add_argument(
        "--augmentation-self-test-only",
        action="store_true",
        help="Run only the Step-28 augmentation contract self-test.",
    )

    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=2029)

    p.add_argument("--overwrite-fold", action="store_true")
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
        "Could not resolve stored path:\n"
        f"  stored : {raw}\n"
        f"  project: {root}"
    )


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n",
        encoding="utf-8",
    )


def first_existing_column(df: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise RuntimeError(
        f"None of columns {names} found. Available columns: {list(df.columns)}"
    )


# =============================================================================
# Input validation
# =============================================================================

def load_manifest_and_original_folds(
    *,
    manifest_path: Path,
    folds_path: Path,
    resolve_manifest_columns,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")
    if not folds_path.is_file():
        raise FileNotFoundError(f"Step-11 folds missing: {folds_path}")

    manifest = pd.read_csv(manifest_path)
    folds = pd.read_csv(folds_path)

    columns = resolve_manifest_columns(manifest, INPUT_TYPE)
    uid_col = columns["uid"]
    label_col = columns["label"]

    manifest = manifest.copy()
    manifest[uid_col] = manifest[uid_col].astype(str)
    manifest[label_col] = pd.to_numeric(
        manifest[label_col], errors="raise"
    ).astype(np.int64)

    if len(manifest) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} manifest rows, found {len(manifest)}."
        )
    if manifest[uid_col].duplicated().any():
        raise RuntimeError("Duplicate UID in Step-10 manifest.")
    if not np.isin(manifest[label_col].to_numpy(), [0, 1]).all():
        raise RuntimeError("Manifest labels must be binary 0/1.")

    fold_uid_col = first_existing_column(
        folds, ("uid", "UID", "subject_uid", "subject_id")
    )
    fold_col = first_existing_column(
        folds, ("fold", "Fold", "cv_fold")
    )

    folds = folds.copy()
    folds[fold_uid_col] = folds[fold_uid_col].astype(str)
    folds[fold_col] = pd.to_numeric(
        folds[fold_col], errors="raise"
    ).astype(np.int64)

    if len(folds) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} Step-11 rows, found {len(folds)}."
        )
    if folds[fold_uid_col].duplicated().any():
        raise RuntimeError("Duplicate UID in Step-11 folds.")
    if set(folds[fold_col].unique()) != set(range(EXPECTED_FOLDS)):
        raise RuntimeError(
            f"Expected Step-11 folds 0..4, found "
            f"{sorted(folds[fold_col].unique().tolist())}."
        )

    if set(manifest[uid_col]) != set(folds[fold_uid_col]):
        raise RuntimeError("Step-10 and Step-11 UID sets differ.")

    # If the original fold file also stores labels, verify them.
    fold_label_candidates = (
        "is_pathologic",
        "label",
        "target",
    )
    fold_label_col = next(
        (c for c in fold_label_candidates if c in folds.columns),
        None,
    )
    if fold_label_col is not None:
        folds[fold_label_col] = pd.to_numeric(
            folds[fold_label_col], errors="raise"
        ).astype(np.int64)

        manifest_labels = manifest.set_index(uid_col)[label_col]
        fold_labels = folds.set_index(fold_uid_col)[fold_label_col]
        if not np.array_equal(
            manifest_labels.loc[fold_labels.index].to_numpy(),
            fold_labels.to_numpy(),
        ):
            raise RuntimeError("Step-10 and Step-11 labels differ.")

    fold_attach = folds[
        [fold_uid_col, fold_col]
    ].rename(
        columns={
            fold_uid_col: "_step11_uid",
            fold_col: "_step11_fold",
        }
    )

    merged = manifest.merge(
        fold_attach,
        left_on=uid_col,
        right_on="_step11_uid",
        how="left",
        validate="one_to_one",
    )

    if merged["_step11_fold"].isna().any():
        raise RuntimeError("Missing Step-11 fold after merge.")

    return manifest, merged, columns


# =============================================================================
# Training config
# =============================================================================

def fold_shared_config(
    *,
    fold: int,
    num_workers: int,
    max_epochs: int | None,
    amp: bool | None,
    DEFAULT_TRAINING_CONFIG,
):
    shared = DEFAULT_TRAINING_CONFIG

    # Preserve the original experiment convention:
    # base reproducibility seed + fold ID.
    reproducibility = replace(
        shared.reproducibility,
        seed=int(shared.reproducibility.seed) + int(fold),
    )

    if num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")

    dataloader = replace(
        shared.dataloader,
        num_workers=int(num_workers),
    )

    training = shared.training
    if max_epochs is not None:
        if max_epochs < 1:
            raise ValueError("--max-epochs must be >= 1.")
        training = replace(
            training,
            max_epochs=int(max_epochs),
        )

    if amp is not None:
        training = replace(
            training,
            use_amp=bool(amp),
        )

    shared = replace(
        shared,
        reproducibility=reproducibility,
        dataloader=dataloader,
        training=training,
    )
    shared.validate()
    return shared


# =============================================================================
# Metrics
# =============================================================================

def subject_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(
        np.asarray(p, dtype=np.float64),
        1e-12,
        1.0 - 1e-12,
    )
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    bins: int = 10,
) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0

    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])

        n = int(mask.sum())
        if n == 0:
            continue

        value += (n / len(y)) * abs(
            float(p[mask].mean()) - float(y[mask].mean())
        )

    return float(value)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)

    precision, recall, _ = precision_recall_curve(y, p)

    return {
        "log_loss": float(subject_log_loss(y, p).mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10bin": expected_calibration_error(y, p, bins=10),
    }


# =============================================================================
# Aggregation helpers
# =============================================================================

def collect_step29_predictions(output_dir: Path) -> pd.DataFrame:
    frames = []

    for fold in range(EXPECTED_FOLDS):
        path = output_dir / f"fold_{fold}" / "predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Step-29 prediction file: {path}")

        df = pd.read_csv(path)
        uid_col = first_existing_column(
            df, ("uid", "UID", "subject_uid", "subject_id")
        )
        label_col = first_existing_column(
            df, ("label", "is_pathologic", "target")
        )
        prob_col = first_existing_column(
            df,
            (
                "probability",
                "probabilities",
                "prediction_probability",
                "y_prob",
            ),
        )

        frames.append(
            pd.DataFrame(
                {
                    "uid": df[uid_col].astype(str),
                    "is_pathologic": pd.to_numeric(
                        df[label_col], errors="raise"
                    ).astype(np.int64),
                    "robust_probability": pd.to_numeric(
                        df[prob_col], errors="raise"
                    ).astype(np.float64),
                    "fold": int(fold),
                }
            )
        )

    oof = pd.concat(frames, ignore_index=True)

    if len(oof) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} Step-29 OOF rows, found {len(oof)}."
        )
    if oof["uid"].duplicated().any():
        raise RuntimeError("Duplicate UID across Step-29 folds.")
    if not np.isfinite(oof["robust_probability"]).all():
        raise RuntimeError("Invalid Step-29 probabilities.")

    return oof


def load_original_p1(
    *,
    project_root: Path,
    shortlist_path: Path,
) -> pd.DataFrame:
    if not shortlist_path.is_file():
        raise FileNotFoundError(f"Shortlist missing: {shortlist_path}")

    shortlist = pd.read_csv(shortlist_path)
    required = {
        "Shortlist ID",
        "Calibration Method",
        "Cross-Fitted Prediction File",
    }
    missing = required - set(shortlist.columns)
    if missing:
        raise RuntimeError(
            f"Shortlist missing columns: {sorted(missing)}"
        )

    p1_rows = shortlist.loc[
        shortlist["Shortlist ID"].astype(str) == "P1"
    ]
    if len(p1_rows) != 1:
        raise RuntimeError(
            f"Expected exactly one P1 shortlist row, found {len(p1_rows)}."
        )

    p1_row = p1_rows.iloc[0]
    p1_method = str(p1_row["Calibration Method"]).strip().lower()
    if p1_method != "none":
        raise RuntimeError(
            f"Expected authoritative P1 method='none', found {p1_method!r}."
        )

    p1_path = resolve_stored_path(
        project_root,
        p1_row["Cross-Fitted Prediction File"],
    )

    df = pd.read_csv(p1_path)
    required_pred = {
        "uid",
        "fold",
        "is_pathologic",
        "raw_probability",
    }
    missing_pred = required_pred - set(df.columns)
    if missing_pred:
        raise RuntimeError(
            f"P1 prediction file missing {sorted(missing_pred)}:\n{p1_path}"
        )

    out = pd.DataFrame(
        {
            "uid": df["uid"].astype(str),
            "is_pathologic": pd.to_numeric(
                df["is_pathologic"], errors="raise"
            ).astype(np.int64),
            "fold": pd.to_numeric(
                df["fold"], errors="raise"
            ).astype(np.int64),
            "P1_raw_probability": pd.to_numeric(
                df["raw_probability"], errors="raise"
            ).astype(np.float64),
        }
    )

    if len(out) != EXPECTED_SUBJECTS:
        raise RuntimeError("Unexpected P1 OOF row count.")
    if out["uid"].duplicated().any():
        raise RuntimeError("Duplicate UID in P1 predictions.")
    if set(out["fold"].unique()) != set(range(EXPECTED_FOLDS)):
        raise RuntimeError("P1 predictions do not cover folds 0..4.")

    return out


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


def aggregate_step29(
    *,
    project_root: Path,
    output_dir: Path,
    shortlist_path: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    overwrite: bool,
    aggregate_cv_results,
    threshold: float,
) -> None:
    aggregate_outputs = [
        output_dir / "step29_robust_oof_predictions.csv",
        output_dir / "step29_vs_original_p1_fold_comparison.csv",
        output_dir / "step29_summary.json",
        output_dir / "step29_report.md",
    ]

    if any(p.exists() for p in aggregate_outputs) and not overwrite:
        raise FileExistsError(
            "Step-29 aggregate outputs already exist. "
            "Use --overwrite-aggregate for an intentional rebuild."
        )

    prediction_paths = [
        output_dir / f"fold_{fold}" / "predictions.csv"
        for fold in range(EXPECTED_FOLDS)
    ]

    # Project-standard aggregation.
    aggregate_cv_results(
        prediction_paths=prediction_paths,
        output_dir=output_dir,
        threshold=threshold,
        expected_folds=EXPECTED_FOLDS,
        expected_subjects=EXPECTED_SUBJECTS,
    )

    robust = collect_step29_predictions(output_dir)
    p1 = load_original_p1(
        project_root=project_root,
        shortlist_path=shortlist_path,
    )

    merged = robust.merge(
        p1,
        on=["uid", "is_pathologic", "fold"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            "Step-29 robust and original P1 OOF predictions did not align "
            "for all 1,362 subjects."
        )

    y = merged["is_pathologic"].to_numpy(dtype=np.int64)
    robust_p = merged["robust_probability"].to_numpy(dtype=np.float64)
    p1_p = merged["P1_raw_probability"].to_numpy(dtype=np.float64)

    robust_global = metrics(y, robust_p)
    p1_global = metrics(y, p1_p)

    fold_rows = []
    robust_fold_lls = []
    p1_fold_lls = []

    for fold in range(EXPECTED_FOLDS):
        mask = merged["fold"].to_numpy(dtype=np.int64) == fold
        rm = metrics(y[mask], robust_p[mask])
        pm = metrics(y[mask], p1_p[mask])

        robust_fold_lls.append(rm["log_loss"])
        p1_fold_lls.append(pm["log_loss"])

        fold_rows.append(
            {
                "fold": fold,
                "subjects": int(mask.sum()),
                "P1_log_loss": pm["log_loss"],
                "robust_log_loss": rm["log_loss"],
                "delta_robust_minus_P1_log_loss":
                    rm["log_loss"] - pm["log_loss"],
                "P1_auroc": pm["auroc"],
                "robust_auroc": rm["auroc"],
                "P1_auprc": pm["auprc"],
                "robust_auprc": rm["auprc"],
                "P1_brier": pm["brier_score"],
                "robust_brier": rm["brier_score"],
                "P1_ece": pm["ece_10bin"],
                "robust_ece": rm["ece_10bin"],
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    delta_subject = (
        subject_log_loss(y, robust_p)
        - subject_log_loss(y, p1_p)
    )
    observed_delta = float(delta_subject.mean())

    bootstrap = paired_class_stratified_bootstrap(
        y=y,
        delta_subject_loss=delta_subject,
        n_bootstrap=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    ci_low, ci_high = np.percentile(
        bootstrap,
        [2.5, 97.5],
    )

    if ci_high < 0:
        bootstrap_conclusion = "ROBUST_E6_BETTER"
    elif ci_low > 0:
        bootstrap_conclusion = "ORIGINAL_P1_BETTER"
    else:
        bootstrap_conclusion = "INCONCLUSIVE"

    robust_fold_sd = float(np.std(robust_fold_lls, ddof=1))
    p1_fold_sd = float(np.std(p1_fold_lls, ddof=1))
    robust_worst = float(np.max(robust_fold_lls))
    p1_worst = float(np.max(p1_fold_lls))

    # Descriptive decision, not a statistical claim.
    if (
        robust_global["log_loss"] <= p1_global["log_loss"]
        and robust_global["auroc"] >= p1_global["auroc"] - 0.002
    ):
        decision = "STRONG_CONFIRMATION"
    elif (
        robust_global["log_loss"]
        <= p1_global["log_loss"] + 0.005
    ):
        decision = "SMALL_IN_DISTRIBUTION_TRADEOFF"
    else:
        decision = "MEANINGFUL_IN_DISTRIBUTION_COST"

    merged["P1_subject_log_loss"] = subject_log_loss(y, p1_p)
    merged["robust_subject_log_loss"] = subject_log_loss(y, robust_p)
    merged["robust_minus_P1_subject_log_loss"] = delta_subject

    merged.to_csv(
        output_dir / "step29_robust_oof_predictions.csv",
        index=False,
        float_format="%.9f",
    )
    fold_df.to_csv(
        output_dir / "step29_vs_original_p1_fold_comparison.csv",
        index=False,
        float_format="%.9f",
    )
    np.savez_compressed(
        output_dir / "step29_vs_original_p1_bootstrap_distribution.npz",
        delta_log_loss=bootstrap,
    )

    summary = {
        "status": "PASS",
        "step": "29",
        "experiment": (
            "E6-S5 Step-28 acquisition-robust augmentation "
            "confirmed under original Step-11 five-fold CV"
        ),
        "controlled_change": "TRAINING_AUGMENTATION_ONLY",
        "src_modified": False,
        "original_step11_folds_used": True,
        "original_P1_metrics": p1_global,
        "robust_E6_metrics": robust_global,
        "delta_robust_minus_P1_log_loss": observed_delta,
        "relative_log_loss_change_percent": float(
            observed_delta / p1_global["log_loss"] * 100.0
        ),
        "paired_bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "ci_95_lower": float(ci_low),
            "ci_95_upper": float(ci_high),
            "conclusion": bootstrap_conclusion,
        },
        "fold_robustness": {
            "P1_fold_log_loss_sd": p1_fold_sd,
            "robust_fold_log_loss_sd": robust_fold_sd,
            "P1_worst_fold_log_loss": p1_worst,
            "robust_worst_fold_log_loss": robust_worst,
        },
        "fold_comparison": fold_df.to_dict(orient="records"),
        "decision": decision,
        "step28_context": {
            "stress_baseline_step25d_log_loss": 0.335003,
            "stress_robust_step28_log_loss": 0.307685,
            "stress_fold0_step25d_log_loss": 0.366938,
            "stress_fold0_step28_log_loss": 0.307388,
        },
        "next_rule": (
            "Do not modify ENS328 solely from Step 29. If ordinary-CV "
            "performance is preserved, the robust E6 should next be evaluated "
            "as a replacement/additional ensemble member using predictions "
            "generated under a controlled validation design."
        ),
    }
    save_json(
        summary,
        output_dir / "step29_summary.json",
    )

    lines = [
        "# Step 29 — Robust E6-S5 Ordinary Step-11 Confirmation",
        "",
        "## Status",
        "",
        "**PASS — robust E6-S5 trained on all five original Step-11 folds.**",
        "",
        "## Controlled comparison",
        "",
        "- Original P1: E6-S5 + original S5 augmentation",
        "- Step 29: E6-S5 + original S5 augmentation + Step-28 acquisition augmentation",
        "- Fold assignments: identical original Step-11 folds",
        "- Validation: unaugmented",
        "- `src/`: unchanged",
        "",
        "## Global result",
        "",
        f"- Original P1 Log Loss: **{p1_global['log_loss']:.6f}**",
        f"- Robust E6 Log Loss: **{robust_global['log_loss']:.6f}**",
        f"- ΔLL robust − P1: **{observed_delta:+.6f}**",
        f"- Original P1 AUROC: **{p1_global['auroc']:.6f}**",
        f"- Robust E6 AUROC: **{robust_global['auroc']:.6f}**",
        f"- Original P1 AUPRC: **{p1_global['auprc']:.6f}**",
        f"- Robust E6 AUPRC: **{robust_global['auprc']:.6f}**",
        "",
        "## Paired bootstrap",
        "",
        f"- Replicates: **{bootstrap_replicates:,}**",
        f"- 95% CI for ΔLL: **[{ci_low:.6f}, {ci_high:.6f}]**",
        f"- Conclusion: **{bootstrap_conclusion}**",
        "",
        "## Fold comparison",
        "",
        fold_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Fold robustness",
        "",
        f"- P1 fold-LL SD: **{p1_fold_sd:.6f}**",
        f"- Robust fold-LL SD: **{robust_fold_sd:.6f}**",
        f"- P1 worst-fold LL: **{p1_worst:.6f}**",
        f"- Robust worst-fold LL: **{robust_worst:.6f}**",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        "## Step-28 context",
        "",
        "- Step-25D stress LL: **0.335003**",
        "- Step-28 robust stress LL: **0.307685**",
        "- Step-25D Fold-0 LL: **0.366938**",
        "- Step-28 Fold-0 LL: **0.307388**",
        "",
        "## Interpretation",
        "",
        (
            "If Step 29 preserves or improves ordinary-CV performance while "
            "Step 28 already improved the domain-stress result, the robust "
            "augmentation has passed both sides of the validation test."
        ),
        "",
        (
            "Do not yet overwrite the frozen ENS328 predictor. The next step "
            "would be to determine whether robust-E6 predictions improve a "
            "fixed ensemble under a controlled validation procedure."
        ),
    ]

    (output_dir / "step29_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print("STEP 29 — FINAL ORDINARY-CV CONFIRMATION")
    print("=" * 108)
    print(f"Original P1 Log Loss          : {p1_global['log_loss']:.6f}")
    print(f"Robust E6 Log Loss            : {robust_global['log_loss']:.6f}")
    print(f"Delta robust - P1             : {observed_delta:+.6f}")
    print(f"Original P1 AUROC             : {p1_global['auroc']:.6f}")
    print(f"Robust E6 AUROC               : {robust_global['auroc']:.6f}")
    print(f"95% paired bootstrap CI       : [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Bootstrap conclusion          : {bootstrap_conclusion}")
    print(f"P1 worst-fold LL              : {p1_worst:.6f}")
    print(f"Robust worst-fold LL          : {robust_worst:.6f}")
    print(f"Decision                      : {decision}")
    print(f"Saved                         : {output_dir}")
    print("=" * 108)


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

    from scripts.experiments_script.common_cv_experiment import (
        FrozenDATScanDataset,
        make_loader,
        resolve_device,
        resolve_manifest_columns,
    )
    from scripts.experiments_script.model06_r3d18_kinetics400_pretrained_exp_script.kinetics_pretrained_factory import (
        build_single_e6,
    )
    from scripts.generalization_validation_script.step28_acquisition_robust_augmentation import (
        AcquisitionRobustAugmenter,
        AcquisitionRobustConfig,
        self_test as augmentation_self_test,
    )
    from src.augmentation.augmentations import build_augmentation
    from src.configs.training_config import (
        DEFAULT_TRAINING_CONFIG,
        ExperimentConfig,
        SCENARIOS,
        save_experiment_config,
        seed_everything,
        suggested_batch_size,
    )
    from src.evaluation.aggregate_cv_results import aggregate_cv_results
    from src.evaluation.evaluate import (
        evaluate_predictions,
        save_evaluation_result,
    )
    from src.training.checkpointing import load_checkpoint
    from src.training.factory import build_trainer_from_config

    if args.augmentation_self_test_only:
        augmentation_self_test()
        print("Step-29 reused Step-28 augmentation self-test: PASS")
        return 0

    manifest_path = resolve(
        project_root,
        args.manifest or DEFAULT_MANIFEST,
    )
    folds_path = resolve(
        project_root,
        args.folds or DEFAULT_FOLDS,
    )
    shortlist_path = resolve(
        project_root,
        args.shortlist or DEFAULT_SHORTLIST,
    )
    output_dir = resolve(
        project_root,
        args.output_dir or DEFAULT_OUTPUT,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = SCENARIOS[SCENARIO_ID]
    if scenario.input_type != INPUT_TYPE or not scenario.augmentation:
        raise RuntimeError(
            "Central Scenario-5 contract changed unexpectedly: "
            f"input_type={scenario.input_type}, "
            f"augmentation={scenario.augmentation}"
        )

    if args.aggregate_only:
        aggregate_step29(
            project_root=project_root,
            output_dir=output_dir,
            shortlist_path=shortlist_path,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            overwrite=args.overwrite_aggregate,
            aggregate_cv_results=aggregate_cv_results,
            threshold=DEFAULT_TRAINING_CONFIG.training.classification_threshold,
        )
        return 0

    if args.fold is None:
        raise ValueError(
            "Training mode requires --fold 0..4. "
            "Use the supplied Slurm array wrapper."
        )

    manifest, merged, columns = load_manifest_and_original_folds(
        manifest_path=manifest_path,
        folds_path=folds_path,
        resolve_manifest_columns=resolve_manifest_columns,
    )

    fold = int(args.fold)
    uid_col = columns["uid"]

    val_uids = set(
        merged.loc[
            merged["_step11_fold"] == fold,
            uid_col,
        ].astype(str)
    )

    train_rows = manifest.loc[
        ~manifest[uid_col].astype(str).isin(val_uids)
    ].reset_index(drop=True)

    val_rows = manifest.loc[
        manifest[uid_col].astype(str).isin(val_uids)
    ].reset_index(drop=True)

    if len(train_rows) + len(val_rows) != EXPECTED_SUBJECTS:
        raise RuntimeError("Step-11 fold does not partition all subjects.")

    if set(train_rows[uid_col].astype(str)) & set(
        val_rows[uid_col].astype(str)
    ):
        raise RuntimeError("Train/validation UID leakage.")

    # Strong membership check against original Step-11 file.
    expected_val_uids = set(
        merged.loc[
            merged["_step11_fold"] == fold,
            uid_col,
        ].astype(str)
    )
    if set(val_rows[uid_col].astype(str)) != expected_val_uids:
        raise RuntimeError("Step-11 validation membership mismatch.")

    shared = fold_shared_config(
        fold=fold,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        amp=args.amp,
        DEFAULT_TRAINING_CONFIG=DEFAULT_TRAINING_CONFIG,
    )
    seed_everything(shared.reproducibility)

    device = resolve_device(args.device)

    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else suggested_batch_size(MODEL_NAME, INPUT_TYPE)
    )
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    fold_dir = output_dir / f"fold_{fold}"

    if fold_dir.exists() and any(fold_dir.iterdir()):
        if args.overwrite_fold:
            shutil.rmtree(fold_dir)
        else:
            raise FileExistsError(
                f"Fold directory already exists and is non-empty: {fold_dir}\n"
                "Use --overwrite-fold only for an intentional rerun."
            )

    fold_dir.mkdir(parents=True, exist_ok=True)

    fold_config = ExperimentConfig(
        model_name=MODEL_NAME,
        scenario_id=SCENARIO_ID,
        fold=fold,
        batch_size=batch_size,
        output_dir=str(fold_dir),
        shared=shared,
    )
    fold_config.validate()
    save_experiment_config(
        fold_config,
        fold_dir / "experiment_config.json",
    )

    # Authoritative original S5 augmentation.
    base_s5_augmenter = build_augmentation(
        enabled=True,
        spatial_mode="3d",
        config=shared.augmentation,
    )

    # ONLY new ingredient relative to original E6-S5.
    robust_config = AcquisitionRobustConfig()
    robust_augmenter = AcquisitionRobustAugmenter(
        base_augmenter=base_s5_augmenter,
        config=robust_config,
    )

    train_dataset = FrozenDATScanDataset(
        rows=train_rows,
        columns=columns,
        project_root=project_root,
        input_type=INPUT_TYPE,
        augmenter=robust_augmenter,
        sample_transform=None,
    )

    # Validation must stay frozen/unaugmented.
    val_dataset = FrozenDATScanDataset(
        rows=val_rows,
        columns=columns,
        project_root=project_root,
        input_type=INPUT_TYPE,
        augmenter=None,
        sample_transform=None,
    )

    fold_seed = int(shared.reproducibility.seed)

    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        training=True,
        shared_config=shared,
        seed=fold_seed,
        device=device,
    )

    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        training=False,
        shared_config=shared,
        seed=fold_seed,
        device=device,
    )

    # Fresh authoritative E6 Kinetics initialization.
    model = build_single_e6()

    metadata = {
        "step": "29",
        "purpose": "ordinary_step11_confirmation_of_step28_robust_E6",
        "experiment_id": "E6",
        "model_name": MODEL_NAME,
        "scenario_id": SCENARIO_ID,
        "scenario_name": scenario.name,
        "input_type": INPUT_TYPE,
        "pretraining": "Kinetics-400",
        "original_step11_fold": fold,
        "train_subjects": len(train_rows),
        "validation_subjects": len(val_rows),
        "fold_seed": fold_seed,
        "baseline_S5_augmentation": asdict(shared.augmentation),
        "additional_step28_augmentation": robust_config.to_dict(),
        "augmentation_scope": "training_only",
        "validation_augmentation": False,
        "controlled_change": "TRAINING_AUGMENTATION_ONLY",
        "src_modified": False,
    }

    trainer = build_trainer_from_config(
        model=model,
        config=shared,
        device=device,
        checkpoint_dir=fold_dir,
        metadata=metadata,
    )

    print("=" * 108)
    print("STEP 29 — ROBUST E6-S5 ORIGINAL STEP-11 CONFIRMATION")
    print("=" * 108)
    print(f"Original Step-11 fold       : {fold}")
    print(f"Train subjects              : {len(train_rows)}")
    print(f"Validation subjects         : {len(val_rows)}")
    print(f"Model                       : E6 / R3D-18 Kinetics-400")
    print(f"Scenario                    : S5 / ROI")
    print(f"Original S5 augmentation    : ON")
    print(f"Step-28 robust augmentation : ON")
    print(f"Validation augmentation     : OFF")
    print(f"Controlled change           : TRAINING AUGMENTATION ONLY")
    print(f"Device                      : {device}")
    print(f"Batch size                  : {batch_size}")
    print(f"Workers                     : {shared.dataloader.num_workers}")
    print(f"AMP                         : {trainer.amp_enabled}")
    print(f"Seed                        : {fold_seed}")
    print(f"Output                      : {fold_dir}")
    print("=" * 108)

    start = time.monotonic()

    training_result = trainer.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        max_epochs=shared.training.max_epochs,
        verbose=True,
    )

    if training_result.best_checkpoint_path is None:
        raise RuntimeError("Trainer did not create a best checkpoint.")

    load_checkpoint(
        training_result.best_checkpoint_path,
        model=trainer.model,
        map_location=device,
        strict_model=True,
    )

    best_validation = trainer.validate(val_loader)

    if not best_validation.uids:
        raise RuntimeError("Validation UIDs are required for OOF evaluation.")

    evaluation = evaluate_predictions(
        uids=best_validation.uids,
        labels=best_validation.targets.numpy(),
        probabilities=best_validation.probabilities.numpy(),
        logits=best_validation.logits.numpy(),
        threshold=shared.training.classification_threshold,
        fold=fold,
    )

    save_evaluation_result(
        evaluation,
        output_dir=fold_dir,
    )

    pd.DataFrame(training_result.history).to_csv(
        fold_dir / "history.csv",
        index=False,
    )

    save_json(
        {
            **metadata,
            "best_epoch": training_result.best_epoch,
            "best_selection_metric": training_result.best_metric,
            "selection_metric_name": shared.training.checkpoint_metric,
            "selection_mode": shared.training.checkpoint_mode,
            "best_checkpoint": str(training_result.best_checkpoint_path),
            "last_checkpoint": (
                str(training_result.last_checkpoint_path)
                if training_result.last_checkpoint_path is not None
                else None
            ),
            "epochs_completed": training_result.epochs_completed,
            "stopped_early": training_result.stopped_early,
            "best_checkpoint_validation_loss": best_validation.loss,
            "best_checkpoint_metrics": evaluation.metrics,
            "runtime_seconds": time.monotonic() - start,
        },
        fold_dir / "training_summary.json",
    )

    print()
    print("=" * 108)
    print(f"STEP 29 FOLD {fold} COMPLETE")
    print("=" * 108)
    print(f"Best epoch          : {training_result.best_epoch}")
    print(f"Validation loss     : {best_validation.loss:.6f}")
    print(f"AUROC               : {evaluation.metrics['auroc']:.6f}")
    print(f"Balanced accuracy   : {evaluation.metrics['balanced_accuracy']:.6f}")
    print(f"Predictions         : {fold_dir / 'predictions.csv'}")
    print("=" * 108)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
