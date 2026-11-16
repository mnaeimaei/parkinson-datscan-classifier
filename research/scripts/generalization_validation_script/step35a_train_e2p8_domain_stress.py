#!/usr/bin/env python3
"""
STEP 35A — E2/P8 ACQUISITION-ROBUST DOMAIN-STRESS TRAINING

Controlled comparison on the frozen Step25C 3-fold acquisition stress split.

Arms
----
baseline:
    E2 / P8-like
    ROI only
    NO stochastic augmentation

robust:
    exact same E2/P8-like setup
    + Step28 acquisition perturbations ONLY during training
    + NO standard S5 augmenter

This isolates acquisition robustness as the only intended training difference.

No preprocessing is changed. Validation is always unaugmented.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments_script.common_cv_experiment import (
    FrozenDATScanDataset,
    _find_column,
    _shared_config_for_fold,
    make_loader,
    normalize_fold_value,
    resolve_device,
    resolve_manifest_columns,
    save_json,
)
from src.configs.training_config import ExperimentConfig, save_experiment_config, seed_everything
from src.evaluation.evaluate import evaluate_predictions, save_evaluation_result
from src.models.model02_resnet18_3d_scratch import build_model
from src.training.checkpointing import load_checkpoint
from src.training.factory import build_trainer_from_config

from scripts.generalization_validation_script.step35a_acquisition_robust_augmentation import (
    AcquisitionRobustAugmenter,
    AcquisitionRobustConfig,
    Identity,
)

DEFAULT_MANIFEST = Path(
    "data/preprocessing_supervised_data/"
    "step10_supervised_dataset_manifest_data/"
    "supervised_dataset/supervised_dataset_manifest.csv"
)
DEFAULT_OUTPUT = Path(
    "data/generalization_validation_data/"
    "step35a_e2p8_acquisition_robust_domain_stress"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--stress-folds", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--arm", choices=("baseline", "robust"), required=True)
    p.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite-fold", action="store_true")
    return p.parse_args()


def resolve_path(root: Path, p: Path) -> Path:
    return p.expanduser().resolve() if p.is_absolute() else (root / p).resolve()


def discover_stress_folds(root: Path) -> Path:
    base = root / "data/generalization_validation_data"
    if not base.exists():
        raise FileNotFoundError(base)

    candidates = []
    for path in base.rglob("*.csv"):
        name = str(path).lower()
        score = 0
        if "step25c" in name:
            score += 8
        if "spacing_sgkf3" in name:
            score += 10
        if "stress" in name:
            score += 5
        if "fold" in path.name.lower():
            score += 4
        if score == 0:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        uid_col = next((c for c in ("uid","UID","subject_uid","subject_id") if c in df.columns), None)
        fold_col = next((c for c in ("fold","fold_id","cv_fold","fold_index","stress_fold") if c in df.columns), None)
        if uid_col is None or fold_col is None:
            continue
        try:
            folds = sorted(pd.Series(df[fold_col]).map(normalize_fold_value).unique().tolist())
        except Exception:
            continue
        if folds == [0, 1, 2] and len(df) == 1362:
            candidates.append((score, path))

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover the frozen Step25C 3-fold stress assignment CSV. "
            "Pass it explicitly with --stress-folds."
        )

    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    best_score = candidates[0][0]
    tied = [p for s, p in candidates if s == best_score]
    if len(tied) > 1:
        raise RuntimeError(
            "Multiple equally plausible Step25C stress fold files found:\n"
            + "\n".join(f"  - {p}" for p in tied)
            + "\nPass the authoritative file explicitly with --stress-folds."
        )
    return tied[0].resolve()


def load_tables(manifest_path: Path, folds_path: Path):
    manifest = pd.read_csv(manifest_path)
    columns = resolve_manifest_columns(manifest, "roi")
    uid_col = columns["uid"]
    label_col = columns["label"]

    manifest = manifest.copy()
    manifest[uid_col] = manifest[uid_col].astype(str)
    manifest[label_col] = pd.to_numeric(manifest[label_col], errors="raise").astype(int)
    if len(manifest) != 1362:
        raise ValueError(f"Expected 1362 manifest subjects, found {len(manifest)}")
    if manifest[uid_col].duplicated().any():
        raise ValueError("Duplicate UID in manifest.")
    if not manifest[label_col].isin([0,1]).all():
        raise ValueError("Labels must be binary 0/1.")

    folds = pd.read_csv(folds_path)
    fold_uid_col = _find_column(
        folds, ("uid","UID","subject_uid","subject_id"), "stress fold UID"
    )
    fold_col = _find_column(
        folds, ("fold","fold_id","cv_fold","fold_index","stress_fold"), "stress fold"
    )
    folds = folds.copy()
    folds[fold_uid_col] = folds[fold_uid_col].astype(str)
    folds["_fold"] = folds[fold_col].map(normalize_fold_value)

    if folds[fold_uid_col].duplicated().any():
        raise ValueError("Duplicate UID in stress folds.")
    if sorted(folds["_fold"].unique().tolist()) != [0,1,2]:
        raise ValueError(
            f"Expected Step25C stress folds [0,1,2], found {sorted(folds['_fold'].unique().tolist())}"
        )
    if len(folds) != 1362:
        raise ValueError(f"Expected 1362 stress assignments, found {len(folds)}")

    m = set(manifest[uid_col])
    f = set(folds[fold_uid_col])
    if m != f:
        raise ValueError(
            f"Manifest/stress-fold UID mismatch: missing={len(m-f)}, unknown={len(f-m)}"
        )
    return manifest, folds, columns, fold_uid_col


def split_rows(manifest, folds, columns, fold_uid_col, fold_id):
    uid_col = columns["uid"]
    val_uids = set(folds.loc[folds["_fold"] == fold_id, fold_uid_col].astype(str))
    train = manifest[~manifest[uid_col].isin(val_uids)].reset_index(drop=True)
    val = manifest[manifest[uid_col].isin(val_uids)].reset_index(drop=True)
    if set(train[uid_col]) & set(val[uid_col]):
        raise RuntimeError("Train/validation leakage.")
    if set(val[uid_col]) != val_uids:
        raise RuntimeError("Validation membership mismatch.")
    if len(train) + len(val) != len(manifest):
        raise RuntimeError("Split does not partition dataset.")
    return train, val


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    manifest_path = resolve_path(root, args.manifest)
    stress_path = (
        resolve_path(root, args.stress_folds)
        if args.stress_folds is not None
        else discover_stress_folds(root)
    )
    output_root = resolve_path(root, args.output_dir)
    fold_dir = output_root / args.arm / f"fold_{args.fold}"

    manifest, folds, columns, fold_uid_col = load_tables(manifest_path, stress_path)
    train_rows, val_rows = split_rows(
        manifest, folds, columns, fold_uid_col, args.fold
    )

    if fold_dir.exists() and any(fold_dir.iterdir()):
        if args.overwrite_fold:
            shutil.rmtree(fold_dir)
        else:
            raise FileExistsError(
                f"Non-empty output exists: {fold_dir}\nUse --overwrite-fold to rerun intentionally."
            )
    fold_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    # Reuse the authoritative central training policy with the same fold-specific
    # seed logic as common_cv_experiment.
    shared_args = argparse.Namespace(
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        amp=args.amp,
    )
    shared = _shared_config_for_fold(shared_args, args.fold)
    seed_everything(shared.reproducibility)

    device = resolve_device(args.device)

    robust_cfg = AcquisitionRobustConfig()
    train_augmenter = None
    if args.arm == "robust":
        train_augmenter = AcquisitionRobustAugmenter(
            base_augmenter=Identity(),
            config=robust_cfg,
        )

    train_ds = FrozenDATScanDataset(
        rows=train_rows,
        columns=columns,
        project_root=root,
        input_type="roi",
        augmenter=train_augmenter,
        sample_transform=None,
    )
    val_ds = FrozenDATScanDataset(
        rows=val_rows,
        columns=columns,
        project_root=root,
        input_type="roi",
        augmenter=None,
        sample_transform=None,
    )

    fold_seed = shared.reproducibility.seed
    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        training=True,
        shared_config=shared,
        seed=fold_seed,
        device=device,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=args.batch_size,
        training=False,
        shared_config=shared,
        seed=fold_seed,
        device=device,
    )

    model = build_model()

    # Scenario ID deliberately remains 2: this is a P8-like ROI/no-standard-
    # augmentation experiment. Acquisition perturbation is recorded separately.
    fold_config = ExperimentConfig(
        model_name="model02_resnet18_3d_scratch",
        scenario_id=2,
        fold=args.fold,
        batch_size=args.batch_size,
        output_dir=str(fold_dir),
        shared=shared,
    )
    fold_config.validate()
    save_experiment_config(fold_config, fold_dir / "experiment_config.json")

    definition = {
        "step": "35A",
        "purpose": "P8/E2 acquisition robustness under frozen Step25C stress splits",
        "arm": args.arm,
        "model": "E2 / model02_resnet18_3d_scratch",
        "p8_reference_scenario": "S2 ROI no standard augmentation",
        "input": "ROI [1,36,44,44]",
        "stress_folds_source": str(stress_path),
        "fold": int(args.fold),
        "train_subjects": int(len(train_rows)),
        "validation_subjects": int(len(val_rows)),
        "standard_s5_augmentation": False,
        "acquisition_robust_augmentation": args.arm == "robust",
        "acquisition_robust_config": robust_cfg.to_dict() if args.arm == "robust" else None,
        "validation_augmentation": False,
        "seed": int(fold_seed),
        "training_policy_source": "src.configs.training_config.DEFAULT_TRAINING_CONFIG",
        "preprocessing_changed": False,
    }
    save_json(definition, fold_dir / "step35a_definition.json")

    metadata = {
        "step": "35A",
        "arm": args.arm,
        "model": "E2",
        "member_target": "P8",
        "scenario": "P8-like S2 ROI",
        "stress_split": "Step25C spacing_sgkf3",
        "acquisition_robust": args.arm == "robust",
        "standard_s5_augmentation": False,
        "fold": int(args.fold),
        "fold_seed": int(fold_seed),
    }

    trainer = build_trainer_from_config(
        model=model,
        config=shared,
        device=device,
        checkpoint_dir=fold_dir,
        metadata=metadata,
    )

    print("=" * 88)
    print("STEP 35A — E2/P8 ACQUISITION-ROBUST DOMAIN STRESS")
    print("=" * 88)
    print(f"Arm            : {args.arm}")
    print(f"Fold           : {args.fold}")
    print(f"Stress folds   : {stress_path}")
    print(f"Train          : {len(train_rows)}")
    print(f"Validation     : {len(val_rows)}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Workers        : {args.num_workers}")
    print(f"Device         : {device}")
    print(f"AMP            : {trainer.amp_enabled}")
    print(f"Fold seed      : {fold_seed}")
    print(f"Robust aug     : {args.arm == 'robust'}")
    print("Standard S5 aug: False")
    print()

    started = time.monotonic()
    result = trainer.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        max_epochs=shared.training.max_epochs,
        verbose=True,
    )
    if result.best_checkpoint_path is None:
        raise RuntimeError("No best checkpoint created.")

    load_checkpoint(
        result.best_checkpoint_path,
        model=trainer.model,
        map_location=device,
        strict_model=True,
    )
    val_result = trainer.validate(val_loader)
    if not val_result.uids:
        raise RuntimeError("Validation UIDs missing.")

    evaluation = evaluate_predictions(
        uids=val_result.uids,
        labels=val_result.targets.numpy(),
        probabilities=val_result.probabilities.numpy(),
        logits=val_result.logits.numpy(),
        threshold=shared.training.classification_threshold,
        fold=args.fold,
    )
    save_evaluation_result(evaluation, output_dir=fold_dir)
    pd.DataFrame(result.history).to_csv(fold_dir / "history.csv", index=False)

    summary = {
        "step": "35A",
        "arm": args.arm,
        "fold": int(args.fold),
        "train_subjects": int(len(train_rows)),
        "validation_subjects": int(len(val_rows)),
        "seed": int(fold_seed),
        "best_epoch": int(result.best_epoch),
        "best_selection_metric": float(result.best_metric),
        "best_checkpoint": str(result.best_checkpoint_path),
        "epochs_completed": int(result.epochs_completed),
        "stopped_early": bool(result.stopped_early),
        "best_checkpoint_validation_loss": float(val_result.loss),
        "best_checkpoint_metrics": evaluation.metrics,
        "runtime_seconds": float(time.monotonic() - started),
    }
    save_json(summary, fold_dir / "training_summary.json")

    print()
    print("STEP35A FOLD COMPLETE")
    print(f"Arm      : {args.arm}")
    print(f"Fold     : {args.fold}")
    print(f"Val loss : {val_result.loss:.6f}")
    print(f"Metrics  : {evaluation.metrics}")
    print(f"Saved    : {fold_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
