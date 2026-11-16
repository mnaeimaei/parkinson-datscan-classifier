#!/usr/bin/env python3
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
import torch
from sklearn.metrics import (
    average_precision_score,
    auc,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

EXPECTED_SUBJECTS = 1362
EXPECTED_FOLDS = 3
MODEL_NAME = "model06_r3d18_kinetics400_pretrained"
SCENARIO_ID = 5
INPUT_TYPE = "roi"
GROUP_COLUMN = "derived_spacing_signature"
FOLD_COLUMN = "candidate_fold"
EXPECTED_SCHEME = "spacing_sgkf3"

STEP25D_DEFAULT = Path(
    "data/generalization_validation_data/"
    "step25d_e6s5_domain_stress_training/"
    "step25d_domain_stress_oof_predictions.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 28: E6-S5 acquisition-robust domain-stress training."
    )
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--assignments", type=Path, default=None)
    p.add_argument("--step25c-summary", type=Path, default=None)
    p.add_argument("--step25d-baseline-oof", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--fold", type=int, choices=(0, 1, 2), default=None)
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--augmentation-self-test-only", action="store_true")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--overwrite-fold", action="store_true")
    p.add_argument("--overwrite-aggregate", action="store_true")
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=2028)
    return p.parse_args()


def resolve(root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(v: Any):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n",
        encoding="utf-8",
    )


def load_tables(
    *,
    manifest_path: Path,
    assignments_path: Path,
    step25c_summary_path: Path,
    resolve_manifest_columns,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    manifest = pd.read_csv(manifest_path)
    assignments = pd.read_csv(assignments_path)
    summary = json.loads(step25c_summary_path.read_text(encoding="utf-8"))

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
        raise RuntimeError("Duplicate manifest UIDs.")

    required = {"uid", "is_pathologic", FOLD_COLUMN, GROUP_COLUMN, "scheme"}
    missing = required - set(assignments.columns)
    if missing:
        raise RuntimeError(f"Assignments missing: {sorted(missing)}")

    assignments = assignments.copy()
    assignments["uid"] = assignments["uid"].astype(str)
    assignments["is_pathologic"] = pd.to_numeric(
        assignments["is_pathologic"], errors="raise"
    ).astype(np.int64)
    assignments[FOLD_COLUMN] = pd.to_numeric(
        assignments[FOLD_COLUMN], errors="raise"
    ).astype(np.int64)
    assignments[GROUP_COLUMN] = assignments[GROUP_COLUMN].astype(str)

    if len(assignments) != EXPECTED_SUBJECTS:
        raise RuntimeError("Unexpected Step-25C assignment row count.")
    if assignments["uid"].duplicated().any():
        raise RuntimeError("Duplicate Step-25C UIDs.")
    if set(assignments[FOLD_COLUMN].unique()) != {0, 1, 2}:
        raise RuntimeError("Step-25C assignments must contain folds 0,1,2.")
    if set(assignments["scheme"].astype(str).unique()) != {EXPECTED_SCHEME}:
        raise RuntimeError("Unexpected Step-25C split scheme.")
    if summary.get("recommended_scheme") != EXPECTED_SCHEME:
        raise RuntimeError("Step-25C summary does not recommend spacing_sgkf3.")
    if set(manifest[uid_col]) != set(assignments["uid"]):
        raise RuntimeError("Manifest/Step-25C UID sets differ.")

    label_lookup = manifest.set_index(uid_col)[label_col]
    assigned = assignments.set_index("uid")["is_pathologic"]
    if not np.array_equal(
        label_lookup.loc[assigned.index].to_numpy(),
        assigned.to_numpy(),
    ):
        raise RuntimeError("Step-10 / Step-25C label mismatch.")

    folds_per_group = assignments.groupby(GROUP_COLUMN)[FOLD_COLUMN].nunique()
    if (folds_per_group != 1).any():
        raise RuntimeError("A spacing family crosses Step-25C folds.")

    merged = manifest.merge(
        assignments[["uid", FOLD_COLUMN, GROUP_COLUMN, "scheme"]],
        left_on=uid_col,
        right_on="uid",
        validate="one_to_one",
        how="left",
    )
    return manifest, merged, columns


def fold_shared_config(
    *,
    fold: int,
    num_workers: int,
    max_epochs: int | None,
    amp: bool | None,
    DEFAULT_TRAINING_CONFIG,
):
    shared = DEFAULT_TRAINING_CONFIG

    reproducibility = replace(
        shared.reproducibility,
        seed=int(shared.reproducibility.seed) + int(fold),
    )
    dataloader = replace(
        shared.dataloader,
        num_workers=int(num_workers),
    )
    training = shared.training

    if max_epochs is not None:
        if max_epochs < 1:
            raise ValueError("--max-epochs must be >=1.")
        training = replace(training, max_epochs=int(max_epochs))

    if amp is not None:
        training = replace(training, use_amp=bool(amp))

    shared = replace(
        shared,
        reproducibility=reproducibility,
        dataloader=dataloader,
        training=training,
    )
    shared.validate()
    return shared


def subject_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for i in range(bins):
        mask = (
            (p >= edges[i]) & (p <= edges[i + 1])
            if i == bins - 1
            else (p >= edges[i]) & (p < edges[i + 1])
        )
        n = int(mask.sum())
        if n:
            value += (n / len(y)) * abs(
                float(p[mask].mean()) - float(y[mask].mean())
            )
    return float(value)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(y, p)
    return {
        "log_loss": float(subject_log_loss(y, p).mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10bin": ece(y, p),
    }


def find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not find any of {candidates}; available={list(df.columns)}"
    )


def collect_step28_predictions(output_dir: Path) -> pd.DataFrame:
    frames = []
    for fold in range(EXPECTED_FOLDS):
        path = output_dir / f"fold_{fold}" / "predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)

        frames.append(
            pd.DataFrame(
                {
                    "uid": df[find_col(df, ("uid", "UID", "subject_id"))].astype(str),
                    "is_pathologic": pd.to_numeric(
                        df[find_col(df, ("label", "is_pathologic", "target"))],
                        errors="raise",
                    ).astype(np.int64),
                    "probability": pd.to_numeric(
                        df[
                            find_col(
                                df,
                                (
                                    "probability",
                                    "probabilities",
                                    "prediction_probability",
                                    "y_prob",
                                ),
                            )
                        ],
                        errors="raise",
                    ).astype(np.float64),
                    "candidate_fold": fold,
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)
    if len(out) != EXPECTED_SUBJECTS or out["uid"].duplicated().any():
        raise RuntimeError("Invalid Step-28 OOF prediction coverage.")
    return out


def load_step25d_baseline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "uid": df[find_col(df, ("uid", "UID", "subject_id"))].astype(str),
            "is_pathologic": pd.to_numeric(
                df[find_col(df, ("is_pathologic", "label", "target"))],
                errors="raise",
            ).astype(np.int64),
            "baseline_probability": pd.to_numeric(
                df[
                    find_col(
                        df,
                        (
                            "probability",
                            "P1_probability",
                            "nested_selected_probability",
                        ),
                    )
                ],
                errors="raise",
            ).astype(np.float64),
            "candidate_fold": pd.to_numeric(
                df[find_col(df, ("candidate_fold", "fold"))],
                errors="raise",
            ).astype(np.int64),
        }
    )
    if len(out) != EXPECTED_SUBJECTS or out["uid"].duplicated().any():
        raise RuntimeError("Invalid Step-25D baseline OOF file.")
    return out


def paired_bootstrap(
    *,
    y: np.ndarray,
    delta_subject_loss: np.ndarray,
    reps: int,
    seed: int,
) -> np.ndarray:
    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    out = np.empty(reps, dtype=np.float64)

    cursor = 0
    while cursor < reps:
        b = min(250, reps - cursor)
        ni = rng.choice(neg, size=(b, len(neg)), replace=True)
        pi = rng.choice(pos, size=(b, len(pos)), replace=True)
        out[cursor : cursor + b] = (
            len(neg) * delta_subject_loss[ni].mean(axis=1)
            + len(pos) * delta_subject_loss[pi].mean(axis=1)
        ) / len(y)
        cursor += b

    return out


def aggregate(
    *,
    output_dir: Path,
    baseline_path: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    overwrite: bool,
    aggregate_cv_results,
    threshold: float,
) -> None:
    if (
        (output_dir / "step28_summary.json").exists()
        or (output_dir / "step28_report.md").exists()
    ) and not overwrite:
        raise FileExistsError(
            "Step-28 aggregate output exists. Use --overwrite-aggregate."
        )

    prediction_paths = [
        output_dir / f"fold_{f}" / "predictions.csv"
        for f in range(EXPECTED_FOLDS)
    ]
    aggregate_cv_results(
        prediction_paths=prediction_paths,
        output_dir=output_dir,
        threshold=threshold,
        expected_folds=EXPECTED_FOLDS,
        expected_subjects=EXPECTED_SUBJECTS,
    )

    robust = collect_step28_predictions(output_dir)
    baseline = load_step25d_baseline(baseline_path)

    merged = robust.merge(
        baseline,
        on=["uid", "is_pathologic", "candidate_fold"],
        validate="one_to_one",
        how="inner",
    )
    if len(merged) != EXPECTED_SUBJECTS:
        raise RuntimeError("Step25D/Step28 alignment failed.")

    y = merged["is_pathologic"].to_numpy(dtype=np.int64)
    robust_p = merged["probability"].to_numpy(dtype=np.float64)
    base_p = merged["baseline_probability"].to_numpy(dtype=np.float64)

    robust_global = metrics(y, robust_p)
    base_global = metrics(y, base_p)

    fold_rows = []
    for fold in range(EXPECTED_FOLDS):
        mask = merged["candidate_fold"].to_numpy(dtype=np.int64) == fold
        rm = metrics(y[mask], robust_p[mask])
        bm = metrics(y[mask], base_p[mask])
        fold_rows.append(
            {
                "candidate_fold": fold,
                "subjects": int(mask.sum()),
                "baseline_log_loss": bm["log_loss"],
                "robust_log_loss": rm["log_loss"],
                "delta_robust_minus_baseline_log_loss":
                    rm["log_loss"] - bm["log_loss"],
                "baseline_auroc": bm["auroc"],
                "robust_auroc": rm["auroc"],
                "baseline_auprc": bm["auprc"],
                "robust_auprc": rm["auprc"],
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    delta_subject = subject_log_loss(y, robust_p) - subject_log_loss(y, base_p)
    boot = paired_bootstrap(
        y=y,
        delta_subject_loss=delta_subject,
        reps=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    observed = float(delta_subject.mean())

    if ci_high < 0:
        bootstrap_conclusion = "ROBUST_AUGMENTATION_BETTER"
    elif ci_low > 0:
        bootstrap_conclusion = "BASELINE_BETTER"
    else:
        bootstrap_conclusion = "INCONCLUSIVE"

    fold0 = fold_df.loc[fold_df["candidate_fold"] == 0].iloc[0]
    fold0_delta = float(fold0["delta_robust_minus_baseline_log_loss"])

    if observed < 0 and fold0_delta < 0:
        decision = "PROMISING"
    elif observed < 0:
        decision = "GLOBAL_IMPROVEMENT_BUT_FOLD0_NOT_IMPROVED"
    elif fold0_delta < 0:
        decision = "FOLD0_IMPROVED_BUT_GLOBAL_NOT_IMPROVED"
    else:
        decision = "NOT_PROMISING"

    merged["baseline_subject_log_loss"] = subject_log_loss(y, base_p)
    merged["robust_subject_log_loss"] = subject_log_loss(y, robust_p)
    merged["robust_minus_baseline_subject_log_loss"] = delta_subject
    merged.to_csv(
        output_dir / "step28_domain_robust_oof_predictions.csv",
        index=False,
        float_format="%.9f",
    )
    fold_df.to_csv(
        output_dir / "step28_vs_step25d_fold_comparison.csv",
        index=False,
        float_format="%.9f",
    )
    np.savez_compressed(
        output_dir / "step28_vs_step25d_bootstrap_distribution.npz",
        delta_log_loss=boot,
    )

    summary = {
        "status": "PASS",
        "step": "28",
        "controlled_change": "TRAINING_AUGMENTATION_ONLY",
        "src_modified": False,
        "baseline_global_metrics": base_global,
        "robust_global_metrics": robust_global,
        "delta_robust_minus_baseline_log_loss": observed,
        "paired_bootstrap_95_ci": [float(ci_low), float(ci_high)],
        "bootstrap_conclusion": bootstrap_conclusion,
        "fold0_baseline_log_loss": float(fold0["baseline_log_loss"]),
        "fold0_robust_log_loss": float(fold0["robust_log_loss"]),
        "fold0_delta_robust_minus_baseline": fold0_delta,
        "decision": decision,
        "fold_comparison": fold_df.to_dict(orient="records"),
        "critical_caveat": (
            "Positive Step-28 stress-fold results must be confirmed under "
            "ordinary Step-11 CV before changing the competition predictor."
        ),
    }
    save_json(summary, output_dir / "step28_summary.json")

    lines = [
        "# Step 28 — E6-S5 Acquisition-Robust Training",
        "",
        "## Status",
        "",
        "**PASS — three Step-25C stress folds trained and compared directly with Step 25D.**",
        "",
        "## Controlled change",
        "",
        "Only **training augmentation** changed.",
        "",
        f"- Step-25D baseline global LL: **{base_global['log_loss']:.6f}**",
        f"- Step-28 robust global LL: **{robust_global['log_loss']:.6f}**",
        f"- ΔLL robust − baseline: **{observed:+.6f}**",
        f"- paired 95% CI: **[{ci_low:.6f}, {ci_high:.6f}]**",
        f"- conclusion: **{bootstrap_conclusion}**",
        "",
        "## Fold comparison",
        "",
        fold_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Fold-0 target",
        "",
        f"- baseline: **{float(fold0['baseline_log_loss']):.6f}**",
        f"- robust: **{float(fold0['robust_log_loss']):.6f}**",
        f"- Δ robust − baseline: **{fold0_delta:+.6f}**",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        "If promising, next confirm under ordinary Step-11 CV before changing the competition model.",
    ]
    (output_dir / "step28_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("STEP 28 — FINAL COMPARISON")
    print("=" * 100)
    print(f"Baseline global LL  : {base_global['log_loss']:.6f}")
    print(f"Robust global LL    : {robust_global['log_loss']:.6f}")
    print(f"Delta               : {observed:+.6f}")
    print(f"95% CI              : [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Fold0 baseline LL   : {float(fold0['baseline_log_loss']):.6f}")
    print(f"Fold0 robust LL     : {float(fold0['robust_log_loss']):.6f}")
    print(f"Decision            : {decision}")
    print("=" * 100)


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
    from scripts.generalization_validation_script.step28_acquisition_robust_augmentation import (
        AcquisitionRobustAugmenter,
        AcquisitionRobustConfig,
        self_test as augmentation_self_test,
    )

    if args.augmentation_self_test_only:
        augmentation_self_test()
        print("Step-28 augmentation self-test: PASS")
        return 0

    manifest_path = resolve(
        project_root,
        args.manifest
        or Path(
            "data/preprocessing_supervised_data/"
            "step10_supervised_dataset_manifest_data/"
            "supervised_dataset/supervised_dataset_manifest.csv"
        ),
    )
    assignments_path = resolve(
        project_root,
        args.assignments
        or Path(
            "data/generalization_validation_data/"
            "step25c_domain_stress_split_design/"
            "recommended_step25d_fold_assignments.csv"
        ),
    )
    step25c_summary_path = resolve(
        project_root,
        args.step25c_summary
        or Path(
            "data/generalization_validation_data/"
            "step25c_domain_stress_split_design/step25c_summary.json"
        ),
    )
    baseline_path = resolve(
        project_root,
        args.step25d_baseline_oof or STEP25D_DEFAULT,
    )
    output_dir = resolve(
        project_root,
        args.output_dir
        or Path(
            "data/generalization_validation_data/"
            "step28_e6s5_acquisition_robust_training"
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = SCENARIOS[SCENARIO_ID]
    if scenario.input_type != "roi" or not scenario.augmentation:
        raise RuntimeError("Central Scenario-5 contract changed unexpectedly.")

    if args.aggregate_only:
        if not baseline_path.is_file():
            raise FileNotFoundError(
                f"Step-25D baseline OOF missing: {baseline_path}"
            )
        aggregate(
            output_dir=output_dir,
            baseline_path=baseline_path,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            overwrite=args.overwrite_aggregate,
            aggregate_cv_results=aggregate_cv_results,
            threshold=DEFAULT_TRAINING_CONFIG.training.classification_threshold,
        )
        return 0

    if args.fold is None:
        raise ValueError("Training requires --fold 0/1/2.")

    manifest, merged, columns = load_tables(
        manifest_path=manifest_path,
        assignments_path=assignments_path,
        step25c_summary_path=step25c_summary_path,
        resolve_manifest_columns=resolve_manifest_columns,
    )

    fold = int(args.fold)
    uid_col = columns["uid"]

    val_uids = set(
        merged.loc[merged[FOLD_COLUMN] == fold, "uid"].astype(str)
    )
    train_rows = manifest.loc[
        ~manifest[uid_col].astype(str).isin(val_uids)
    ].reset_index(drop=True)
    val_rows = manifest.loc[
        manifest[uid_col].astype(str).isin(val_uids)
    ].reset_index(drop=True)

    group_lookup = merged.set_index("uid")[GROUP_COLUMN]
    train_groups = set(
        group_lookup.loc[train_rows[uid_col].astype(str)].astype(str)
    )
    val_groups = set(
        group_lookup.loc[val_rows[uid_col].astype(str)].astype(str)
    )
    if train_groups & val_groups:
        raise RuntimeError("Spacing-family leakage detected.")

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

    fold_dir = output_dir / f"fold_{fold}"
    if fold_dir.exists() and any(fold_dir.iterdir()):
        if args.overwrite_fold:
            shutil.rmtree(fold_dir)
        else:
            raise FileExistsError(
                f"{fold_dir} exists. Use --overwrite-fold to rerun."
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
    save_experiment_config(fold_config, fold_dir / "experiment_config.json")

    base_s5_augmenter = build_augmentation(
        enabled=True,
        spatial_mode="3d",
        config=shared.augmentation,
    )
    robust_cfg = AcquisitionRobustConfig()
    robust_augmenter = AcquisitionRobustAugmenter(
        base_augmenter=base_s5_augmenter,
        config=robust_cfg,
    )

    train_dataset = FrozenDATScanDataset(
        rows=train_rows,
        columns=columns,
        project_root=project_root,
        input_type="roi",
        augmenter=robust_augmenter,
        sample_transform=None,
    )
    val_dataset = FrozenDATScanDataset(
        rows=val_rows,
        columns=columns,
        project_root=project_root,
        input_type="roi",
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

    model = build_single_e6()

    metadata = {
        "step": "28",
        "experiment_id": "E6",
        "model_name": MODEL_NAME,
        "scenario_id": SCENARIO_ID,
        "scenario_name": scenario.name,
        "input_type": "roi",
        "pretraining": "Kinetics-400",
        "baseline_augmentation": asdict(shared.augmentation),
        "additional_acquisition_robust_augmentation": robust_cfg.to_dict(),
        "augmentation_scope": "training_only",
        "validation_augmentation": False,
        "stress_split": EXPECTED_SCHEME,
        "candidate_fold": fold,
        "train_subjects": len(train_rows),
        "validation_subjects": len(val_rows),
        "train_spacing_families": len(train_groups),
        "validation_spacing_families": len(val_groups),
        "shared_spacing_families": 0,
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

    print("=" * 100)
    print("STEP 28 — E6-S5 ACQUISITION-ROBUST TRAINING")
    print("=" * 100)
    print(f"Fold                    : {fold}")
    print(f"Train subjects          : {len(train_rows)}")
    print(f"Validation subjects     : {len(val_rows)}")
    print(f"Shared spacing families : 0")
    print(f"Model                   : E6 / R3D-18 Kinetics-400")
    print(f"Base S5 augmentation    : ON")
    print(f"Added robust augmentation: ON")
    print(f"Validation augmentation : OFF")
    print(f"Device                  : {device}")
    print(f"AMP                     : {trainer.amp_enabled}")
    print("=" * 100)

    start = time.monotonic()
    result = trainer.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        max_epochs=shared.training.max_epochs,
        verbose=True,
    )
    if result.best_checkpoint_path is None:
        raise RuntimeError("Best checkpoint was not produced.")

    load_checkpoint(
        result.best_checkpoint_path,
        model=trainer.model,
        map_location=device,
        strict_model=True,
    )
    validation = trainer.validate(val_loader)
    if not validation.uids:
        raise RuntimeError("Validation UIDs are missing.")

    evaluation = evaluate_predictions(
        uids=validation.uids,
        labels=validation.targets.numpy(),
        probabilities=validation.probabilities.numpy(),
        logits=validation.logits.numpy(),
        threshold=shared.training.classification_threshold,
        fold=fold,
    )
    save_evaluation_result(evaluation, output_dir=fold_dir)
    pd.DataFrame(result.history).to_csv(
        fold_dir / "history.csv", index=False
    )

    save_json(
        {
            **metadata,
            "fold_seed": fold_seed,
            "best_epoch": result.best_epoch,
            "best_selection_metric": result.best_metric,
            "selection_metric_name": shared.training.checkpoint_metric,
            "selection_mode": shared.training.checkpoint_mode,
            "best_checkpoint": str(result.best_checkpoint_path),
            "epochs_completed": result.epochs_completed,
            "stopped_early": result.stopped_early,
            "best_checkpoint_validation_loss": validation.loss,
            "best_checkpoint_metrics": evaluation.metrics,
            "runtime_seconds": time.monotonic() - start,
        },
        fold_dir / "training_summary.json",
    )

    print(f"STEP 28 fold {fold}: PASS")
    print(f"Best epoch : {result.best_epoch}")
    print(f"Val loss   : {validation.loss:.6f}")
    print(f"AUROC      : {evaluation.metrics['auroc']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
