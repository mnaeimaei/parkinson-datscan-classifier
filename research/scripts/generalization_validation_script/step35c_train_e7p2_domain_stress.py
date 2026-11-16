#!/usr/bin/env python3
"""
STEP 35C — P2/E7 ACQUISITION-ROBUST DOMAIN-STRESS TRAINING

Target frozen member
--------------------
P2 = E7-S4
    model07_resnet18_2p5d_attention_scratch
    whole-volume input
    deterministic frozen 40-slice 2.5D selector
    standard training augmentation ON
    augmentation spatial mode = "inplane_2d"

Controlled arms
---------------
baseline:
    exact P2-like E7-S4 training
    = standard inplane_2d augmentation only

robust:
    exact same P2-like E7-S4 training
    = same standard inplane_2d augmentation
    + frozen Step-28 acquisition-robust perturbations

Validation is deterministic/unaugmented in both arms.

No preprocessing changes.
No hidden competition labels.
No ENS328R changes.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The frozen E4/E7 selector resolves subject-specific ROI information from the
# authoritative supervised manifest.
os.environ.setdefault(
    "DATSCAN_SUPERVISED_MANIFEST",
    str(
        PROJECT_ROOT
        / "data"
        / "preprocessing_supervised_data"
        / "step10_supervised_dataset_manifest_data"
        / "supervised_dataset"
        / "supervised_dataset_manifest.csv"
    ),
)

from scripts.experiments_script.common_cv_experiment import (
    FrozenDATScanDataset,
    _find_column,
    make_loader,
    normalize_fold_value,
    resolve_device,
    resolve_manifest_columns,
    save_json,
)
from scripts.experiments_script.model07_resnet18_2p5d_attention_scratch_exp_script.scratch_2p5d_factory import (
    build_single_e7,
)
from scripts.experiments_script.two_point_five_d_input import (
    select_2p5d_slices,
    slice_selection_metadata,
)
from src.augmentation.augmentations import build_augmentation
from src.configs.training_config import (
    DEFAULT_TRAINING_CONFIG,
    ExperimentConfig,
    save_experiment_config,
    seed_everything,
    suggested_batch_size,
)
from src.evaluation.evaluate import evaluate_predictions, save_evaluation_result
from src.training.checkpointing import load_checkpoint
from src.training.factory import build_trainer_from_config

# IMPORTANT: reuse the frozen Step35A/Step28 acquisition policy unchanged.
from scripts.generalization_validation_script.step35a_acquisition_robust_augmentation import (
    AcquisitionRobustAugmenter,
    AcquisitionRobustConfig,
)

DEFAULT_MANIFEST = Path(
    "data/preprocessing_supervised_data/"
    "step10_supervised_dataset_manifest_data/"
    "supervised_dataset/supervised_dataset_manifest.csv"
)
DEFAULT_OUTPUT = Path(
    "data/generalization_validation_data/"
    "step35c_e7p2_acquisition_robust_domain_stress"
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
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite-fold", action="store_true")
    return p.parse_args()


def resolve_path(root: Path, p: Path) -> Path:
    return p.expanduser().resolve() if p.is_absolute() else (root / p).resolve()


def discover_stress_folds(root: Path) -> Path:
    """
    Locate the frozen Step25C spacing_sgkf3 assignment table.

    Deliberately fail on an equally plausible tie instead of guessing.
    """
    base = root / "data/generalization_validation_data"
    if not base.exists():
        raise FileNotFoundError(base)

    candidates: list[tuple[int, Path]] = []

    for path in base.rglob("*.csv"):
        lname = str(path).lower()
        score = 0
        if "step25c" in lname:
            score += 8
        if "spacing_sgkf3" in lname:
            score += 10
        if "stress" in lname:
            score += 5
        if "fold" in path.name.lower():
            score += 4
        if score == 0:
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        uid_col = next(
            (c for c in ("uid", "UID", "subject_uid", "subject_id") if c in df.columns),
            None,
        )
        fold_col = next(
            (c for c in ("fold", "fold_id", "cv_fold", "fold_index", "stress_fold")
             if c in df.columns),
            None,
        )
        if uid_col is None or fold_col is None:
            continue

        try:
            observed = sorted(
                pd.Series(df[fold_col]).map(normalize_fold_value).unique().tolist()
            )
        except Exception:
            continue

        if observed == [0, 1, 2] and len(df) == 1362:
            candidates.append((score, path.resolve()))

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover the frozen Step25C 3-fold stress assignment CSV. "
            "Pass the authoritative file with --stress-folds."
        )

    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    best_score = candidates[0][0]
    tied = [p for s, p in candidates if s == best_score]

    if len(tied) > 1:
        raise RuntimeError(
            "Multiple equally plausible Step25C fold files found:\n"
            + "\n".join(f"  - {p}" for p in tied)
            + "\nPass the authoritative file explicitly with --stress-folds."
        )

    return tied[0]


def load_tables(manifest_path: Path, folds_path: Path):
    manifest = pd.read_csv(manifest_path)
    columns = resolve_manifest_columns(manifest, "whole")

    uid_col = columns["uid"]
    label_col = columns["label"]

    manifest = manifest.copy()
    manifest[uid_col] = manifest[uid_col].astype(str)
    manifest[label_col] = pd.to_numeric(
        manifest[label_col], errors="raise"
    ).astype(int)

    if len(manifest) != 1362:
        raise ValueError(f"Expected 1362 subjects, found {len(manifest)}.")
    if manifest[uid_col].duplicated().any():
        raise ValueError("Duplicate UID in manifest.")
    if not manifest[label_col].isin([0, 1]).all():
        raise ValueError("Labels must be 0/1.")

    folds = pd.read_csv(folds_path).copy()
    fold_uid_col = _find_column(
        folds, ("uid", "UID", "subject_uid", "subject_id"), "stress UID"
    )
    fold_col = _find_column(
        folds,
        ("fold", "fold_id", "cv_fold", "fold_index", "stress_fold"),
        "stress fold",
    )
    folds[fold_uid_col] = folds[fold_uid_col].astype(str)
    folds["_fold"] = folds[fold_col].map(normalize_fold_value)

    if len(folds) != 1362:
        raise ValueError(f"Expected 1362 stress assignments, found {len(folds)}.")
    if folds[fold_uid_col].duplicated().any():
        raise ValueError("Duplicate UID in stress assignments.")
    if sorted(folds["_fold"].unique().tolist()) != [0, 1, 2]:
        raise ValueError(
            f"Expected stress folds [0,1,2], found "
            f"{sorted(folds['_fold'].unique().tolist())}"
        )

    mset = set(manifest[uid_col])
    fset = set(folds[fold_uid_col])
    if mset != fset:
        raise ValueError(
            f"Manifest/stress UID mismatch: missing={len(mset-fset)}, "
            f"unknown={len(fset-mset)}."
        )

    return manifest, folds, columns, fold_uid_col


def split_rows(manifest, folds, columns, fold_uid_col, fold_id):
    uid_col = columns["uid"]
    val_uids = set(
        folds.loc[folds["_fold"] == fold_id, fold_uid_col].astype(str)
    )
    train = manifest[
        ~manifest[uid_col].astype(str).isin(val_uids)
    ].reset_index(drop=True)
    val = manifest[
        manifest[uid_col].astype(str).isin(val_uids)
    ].reset_index(drop=True)

    if set(train[uid_col]) & set(val[uid_col]):
        raise RuntimeError("Train/validation UID leakage.")
    if set(val[uid_col]) != val_uids:
        raise RuntimeError("Validation membership mismatch.")
    if len(train) + len(val) != len(manifest):
        raise RuntimeError("Split does not partition dataset.")

    return train, val


def build_train_augmenter(shared, arm: str):
    """
    Baseline = exact P2 standard augmentation.
    Robust   = exact same P2 augmentation + frozen acquisition perturbations.
    """
    standard = build_augmentation(
        enabled=True,
        spatial_mode="inplane_2d",
        config=shared.augmentation,
    )

    if arm == "baseline":
        return standard

    return AcquisitionRobustAugmenter(
        base_augmenter=standard,
        config=AcquisitionRobustConfig(),
    )



def build_step35c_shared_config(args, fold_id: int):
    """
    Build the fold-specific shared configuration without depending on the
    private CLI helper _shared_config_for_fold().

    This intentionally preserves the central DEFAULT_TRAINING_CONFIG and only
    applies the same experiment-layer overrides needed by Step35C:
      - deterministic fold seed = base seed + fold_id
      - optional num_workers override
      - optional max_epochs override
      - optional AMP override

    DataLoader policies such as drop_last_train/drop_last_validation remain
    exactly as defined by the central config. This makes Step35C robust to
    future additions to common_cv_experiment.py's argparse Namespace.
    """
    shared = DEFAULT_TRAINING_CONFIG

    fold_seed = int(shared.reproducibility.seed) + int(fold_id)
    reproducibility = replace(
        shared.reproducibility,
        seed=fold_seed,
    )

    dataloader = shared.dataloader
    if args.num_workers is not None:
        if int(args.num_workers) < 0:
            raise ValueError("--num-workers must be >= 0.")
        dataloader = replace(
            dataloader,
            num_workers=int(args.num_workers),
        )

    training = shared.training
    if args.max_epochs is not None:
        if int(args.max_epochs) < 1:
            raise ValueError("--max-epochs must be >= 1.")
        training = replace(
            training,
            max_epochs=int(args.max_epochs),
        )

    if args.amp is not None:
        training = replace(
            training,
            use_amp=bool(args.amp),
        )

    shared = replace(
        shared,
        reproducibility=reproducibility,
        dataloader=dataloader,
        training=training,
    )
    shared.validate()
    return shared

def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()

    # Rebind the manifest variable in case --project-root differs from import-time root.
    os.environ["DATSCAN_SUPERVISED_MANIFEST"] = str(
        root
        / "data"
        / "preprocessing_supervised_data"
        / "step10_supervised_dataset_manifest_data"
        / "supervised_dataset"
        / "supervised_dataset_manifest.csv"
    )

    manifest_path = resolve_path(root, args.manifest)
    stress_path = (
        resolve_path(root, args.stress_folds)
        if args.stress_folds is not None
        else discover_stress_folds(root)
    )
    output_root = resolve_path(root, args.output_dir)
    fold_dir = output_root / args.arm / f"fold_{args.fold}"

    manifest, folds, columns, fold_uid_col = load_tables(
        manifest_path, stress_path
    )
    train_rows, val_rows = split_rows(
        manifest, folds, columns, fold_uid_col, args.fold
    )

    if fold_dir.exists() and any(fold_dir.iterdir()):
        if args.overwrite_fold:
            shutil.rmtree(fold_dir)
        else:
            raise FileExistsError(
                f"Non-empty output exists: {fold_dir}\n"
                "Use --overwrite-fold only for an intentional rerun."
            )

    fold_dir.mkdir(parents=True, exist_ok=True)

    shared = build_step35c_shared_config(args, args.fold)
    seed_everything(shared.reproducibility)

    device = resolve_device(args.device)

    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(
            suggested_batch_size(
                "model07_resnet18_2p5d_attention_scratch",
                "whole",
            )
        )
    )
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    train_augmenter = build_train_augmenter(shared, args.arm)

    train_ds = FrozenDATScanDataset(
        rows=train_rows,
        columns=columns,
        project_root=root,
        input_type="whole",
        augmenter=train_augmenter,
        sample_transform=select_2p5d_slices,
    )
    val_ds = FrozenDATScanDataset(
        rows=val_rows,
        columns=columns,
        project_root=root,
        input_type="whole",
        augmenter=None,
        sample_transform=select_2p5d_slices,
    )

    fold_seed = int(shared.reproducibility.seed)
    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        training=True,
        shared_config=shared,
        seed=fold_seed,
        device=device,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=batch_size,
        training=False,
        shared_config=shared,
        seed=fold_seed,
        device=device,
    )

    model = build_single_e7()

    # This remains Scenario 4 because both arms preserve the original P2/S4
    # standard augmentation. Robustness is an additional controlled factor.
    fold_config = ExperimentConfig(
        model_name="model07_resnet18_2p5d_attention_scratch",
        scenario_id=4,
        fold=args.fold,
        batch_size=batch_size,
        output_dir=str(fold_dir),
        shared=shared,
    )
    fold_config.validate()
    save_experiment_config(
        fold_config, fold_dir / "experiment_config.json"
    )

    robust_cfg = AcquisitionRobustConfig()

    definition = {
        "step": "35C",
        "purpose": "P2/E7 acquisition robustness under frozen Step25C stress splits",
        "arm": args.arm,
        "target_member": "P2",
        "model": "E7 / model07_resnet18_2p5d_attention_scratch",
        "reference_scenario": "S4 whole + standard augmentation",
        "input": "whole_2p5d",
        "raw_whole_tensor": "[1,160,192,192]",
        "model_input_after_frozen_selector": "[1,40,192,192]",
        "sample_transform": "select_2p5d_slices",
        "slice_selection": slice_selection_metadata(),
        "standard_augmentation": True,
        "standard_augmentation_spatial_mode": "inplane_2d",
        "acquisition_robust_augmentation": args.arm == "robust",
        "acquisition_robust_config": (
            robust_cfg.to_dict() if args.arm == "robust" else None
        ),
        "augmentation_scope": "training_only",
        "validation_augmentation": False,
        "stress_folds_source": str(stress_path),
        "fold": int(args.fold),
        "train_subjects": int(len(train_rows)),
        "validation_subjects": int(len(val_rows)),
        "seed": fold_seed,
        "batch_size": batch_size,
        "preprocessing_changed": False,
        "hidden_competition_labels_used": False,
    }
    save_json(definition, fold_dir / "step35c_definition.json")

    metadata = {
        "step": "35C",
        "arm": args.arm,
        "target_member": "P2",
        "model": "E7",
        "scenario": "S4",
        "input": "whole_2p5d",
        "standard_augmentation": True,
        "standard_augmentation_spatial_mode": "inplane_2d",
        "acquisition_robust": args.arm == "robust",
        "stress_split": "Step25C spacing_sgkf3",
        "fold": int(args.fold),
        "fold_seed": fold_seed,
    }

    trainer = build_trainer_from_config(
        model=model,
        config=shared,
        device=device,
        checkpoint_dir=fold_dir,
        metadata=metadata,
    )

    print("=" * 92)
    print("STEP 35C — P2/E7 ACQUISITION-ROBUST DOMAIN STRESS")
    print("=" * 92)
    print(f"Arm             : {args.arm}")
    print(f"Fold            : {args.fold}")
    print(f"Stress folds    : {stress_path}")
    print(f"Train subjects  : {len(train_rows)}")
    print(f"Val subjects    : {len(val_rows)}")
    print(f"Model           : E7 scratch 2.5D attention")
    print(f"Scenario        : S4 whole + augmentation")
    print(f"Slice selector  : frozen 40-slice E4/E7 selector")
    print(f"Standard aug    : ON (inplane_2d)")
    print(f"Acquisition aug : {args.arm == 'robust'}")
    print(f"Batch size      : {batch_size}")
    print(f"Workers         : {shared.dataloader.num_workers}")
    print(f"Device          : {device}")
    print(f"AMP             : {trainer.amp_enabled}")
    print(f"Seed            : {fold_seed}")
    print()

    started = time.monotonic()

    result = trainer.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        max_epochs=shared.training.max_epochs,
        verbose=True,
    )

    if result.best_checkpoint_path is None:
        raise RuntimeError("No best checkpoint was created.")

    load_checkpoint(
        result.best_checkpoint_path,
        model=trainer.model,
        map_location=device,
        strict_model=True,
    )

    val_result = trainer.validate(val_loader)
    if not val_result.uids:
        raise RuntimeError("Validation UIDs are missing.")

    evaluation = evaluate_predictions(
        uids=val_result.uids,
        labels=val_result.targets.numpy(),
        probabilities=val_result.probabilities.numpy(),
        logits=val_result.logits.numpy(),
        threshold=shared.training.classification_threshold,
        fold=args.fold,
    )
    save_evaluation_result(evaluation, output_dir=fold_dir)

    pd.DataFrame(result.history).to_csv(
        fold_dir / "history.csv", index=False
    )

    summary = {
        "step": "35C",
        "arm": args.arm,
        "fold": int(args.fold),
        "target_member": "P2",
        "model": "E7",
        "scenario": "S4",
        "train_subjects": int(len(train_rows)),
        "validation_subjects": int(len(val_rows)),
        "seed": fold_seed,
        "batch_size": batch_size,
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
    print("STEP35C FOLD COMPLETE")
    print(f"Arm      : {args.arm}")
    print(f"Fold     : {args.fold}")
    print(f"Val loss : {val_result.loss:.6f}")
    print(f"Metrics  : {evaluation.metrics}")
    print(f"Saved    : {fold_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
