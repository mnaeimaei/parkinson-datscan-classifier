#!/usr/bin/env python3
"""
common_cv_experiment.py

Shared experiment-layer orchestration for frozen 5-fold DAT-SPECT experiments.

This module deliberately does NOT implement:
    - loss
    - optimizer
    - scheduler
    - early stopping
    - checkpoint serialization
    - epoch training
    - validation metric definitions
    - CV metric aggregation

Those responsibilities belong to src/.

This module owns only:
    - frozen manifest/fold validation
    - NIfTI -> frozen tensor loading
    - optional deterministic model-input transform
    - DataLoader construction
    - per-fold model construction
    - calls into the shared src training/evaluation engine
    - experiment-level artifact organization

Scope:
    scenarios 01-06

Scenarios 04-06 use the project's src.augmentation implementation
for TRAINING ONLY.

Validation, OOF prediction, calibration prediction, and normal
inference remain unaugmented.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Mapping

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.augmentation.augmentations import build_augmentation
from src.configs.training_config import (
    DEFAULT_TRAINING_CONFIG,
    ExperimentConfig,
    SCENARIOS,
    make_dataloader_generator,
    save_experiment_config,
    seed_everything,
    seed_worker,
    suggested_batch_size,
)
from src.evaluation.aggregate_cv_results import aggregate_cv_results
from src.evaluation.evaluate import (
    evaluate_predictions,
    save_evaluation_result,
)
from src.training.checkpointing import load_checkpoint
from src.training.factory import build_trainer_from_config


WHOLE_TENSOR_SHAPE = (1, 160, 192, 192)  # [C,D,H,W]
ROI_TENSOR_SHAPE = (1, 36, 44, 44)       # [C,D,H,W]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_name: str
    model_name: str
    scenario_id: int
    output_subdir: str
    model_builder: Callable[[], nn.Module]
    metadata: Mapping[str, object] | None = None
    augmentation_spatial_mode: str = "3d"
    sample_transform: Callable[[dict], dict] | None = None

    @property
    def scenario(self):
        if self.scenario_id not in SCENARIOS:
            raise ValueError(f"Unknown scenario_id={self.scenario_id}.")
        return SCENARIOS[self.scenario_id]


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...], logical_name: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not identify {logical_name!r} column. "
        f"Tried {candidates}. Available columns: {list(df.columns)}"
    )


def resolve_manifest_columns(df: pd.DataFrame, input_type: str) -> Dict[str, str]:
    columns = {
        "uid": _find_column(
            df,
            ("uid", "UID", "subject_uid", "subject_id"),
            "UID",
        ),
        "label": _find_column(
            df,
            ("is_pathologic", "label", "target"),
            "label",
        ),
    }

    if input_type in {"whole", "whole_roi"}:
        columns["whole"] = _find_column(
            df,
            ("whole_path", "whole_volume_path", "whole"),
            "whole-volume path",
        )

    if input_type in {"roi", "whole_roi"}:
        columns["roi"] = _find_column(
            df,
            ("roi_path", "striatal_path", "striatal_roi_path", "roi"),
            "ROI path",
        )

    return columns


def normalize_fold_value(value: object) -> int:
    if isinstance(value, str):
        value = (
            value.strip()
            .lower()
            .replace("fold_", "")
            .replace("fold", "")
            .strip()
        )
    return int(value)


def load_and_validate_frozen_tables(
    manifest_path: Path,
    folds_path: Path,
    *,
    input_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], str]:
    manifest = pd.read_csv(manifest_path)
    folds = pd.read_csv(folds_path)

    columns = resolve_manifest_columns(manifest, input_type)

    uid_col = columns["uid"]
    label_col = columns["label"]

    manifest = manifest.copy()
    manifest[uid_col] = manifest[uid_col].astype(str)

    if manifest[uid_col].duplicated().any():
        duplicates = (
            manifest.loc[manifest[uid_col].duplicated(keep=False), uid_col]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "Duplicate UIDs in supervised manifest: "
            + ", ".join(duplicates[:20])
        )

    labels = pd.to_numeric(manifest[label_col], errors="raise").astype(int)
    if not labels.isin([0, 1]).all():
        bad = sorted(labels[~labels.isin([0, 1])].unique().tolist())
        raise ValueError(f"Labels must be binary 0/1. Found: {bad}")
    manifest[label_col] = labels

    fold_uid_col = _find_column(
        folds,
        ("uid", "UID", "subject_uid", "subject_id"),
        "fold UID",
    )
    fold_col = _find_column(
        folds,
        ("fold", "fold_id", "cv_fold", "fold_index"),
        "fold assignment",
    )

    folds = folds.copy()
    folds[fold_uid_col] = folds[fold_uid_col].astype(str)
    folds["_fold"] = folds[fold_col].map(normalize_fold_value)

    if folds[fold_uid_col].duplicated().any():
        raise ValueError("A subject has more than one frozen fold assignment.")

    observed_folds = sorted(folds["_fold"].unique().tolist())
    if observed_folds != [0, 1, 2, 3, 4]:
        raise ValueError(
            "Expected frozen folds [0,1,2,3,4], "
            f"found {observed_folds}."
        )

    manifest_uids = set(manifest[uid_col])
    fold_uids = set(folds[fold_uid_col])

    missing_assignments = manifest_uids - fold_uids
    unknown_assignments = fold_uids - manifest_uids

    if missing_assignments or unknown_assignments:
        raise ValueError(
            "Manifest/fold UID mismatch: "
            f"missing_assignments={len(missing_assignments)}, "
            f"unknown_assignments={len(unknown_assignments)}."
        )

    if len(manifest) != len(folds):
        raise ValueError(
            f"Manifest/fold row-count mismatch: {len(manifest)} vs {len(folds)}."
        )

    return manifest, folds, columns, fold_uid_col


def load_nifti_tensor(path: Path, expected_shape: tuple[int, int, int, int]) -> torch.Tensor:
    """
    Frozen tensor conversion already validated by the preprocessing/model-input pipeline.

    NIfTI voxel array:
        [X,Y,Z]

    Model tensor:
        [C,D,H,W] = [1,Z,Y,X]

    No normalization, clipping, scaling, registration, resampling, or augmentation
    is performed here.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Input NIfTI not found: {path}")

    image = nib.load(str(path))
    array = np.asarray(image.dataobj, dtype=np.float32)

    if array.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got {array.shape}: {path}")

    array = np.transpose(array, (2, 1, 0))
    array = np.ascontiguousarray(array)
    tensor = torch.from_numpy(array).unsqueeze(0)

    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Unexpected frozen tensor shape for {path}: "
            f"{tuple(tensor.shape)} != {expected_shape}"
        )

    if tensor.dtype != torch.float32:
        raise TypeError(f"Expected float32 tensor, got {tensor.dtype}: {path}")

    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"NaN/Inf detected in frozen input: {path}")

    return tensor


class FrozenDATScanDataset(Dataset):
    """
    Dataset for already-frozen model inputs.

    Optional augmentation is applied only when an augmenter is explicitly
    supplied by run_experiment(). Validation datasets never receive one.
    """

    def __init__(
        self,
        *,
        rows: pd.DataFrame,
        columns: Dict[str, str],
        project_root: Path,
        input_type: str,
        augmenter=None,
        sample_transform: Callable[[dict], dict] | None = None,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.columns = dict(columns)
        self.project_root = project_root
        self.input_type = input_type
        self.augmenter = augmenter
        self.sample_transform = sample_transform

        if input_type not in {"whole", "roi", "whole_roi"}:
            raise ValueError(f"Unsupported input_type={input_type!r}.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]

        sample = {
            "uid": str(row[self.columns["uid"]]),
            "label": torch.tensor(
                int(row[self.columns["label"]]),
                dtype=torch.long,
            ),
        }

        if self.input_type in {"whole", "whole_roi"}:
            whole_path = resolve_path(
                self.project_root,
                str(row[self.columns["whole"]]),
            )
            sample["whole"] = load_nifti_tensor(
                whole_path,
                WHOLE_TENSOR_SHAPE,
            )

        if self.input_type in {"roi", "whole_roi"}:
            roi_path = resolve_path(
                self.project_root,
                str(row[self.columns["roi"]]),
            )
            sample["roi"] = load_nifti_tensor(
                roi_path,
                ROI_TENSOR_SHAPE,
            )

        # -------------------------------------------------------------
        # Optional deterministic model-input transform.
        #
        # This is used for model-family input preparation that is NOT
        # stochastic augmentation. For E4/E7 it performs the frozen
        # deterministic axial-slice selection.
        #
        # It runs for BOTH training and validation, before any training
        # augmentation, so train/validation use exactly the same
        # deterministic model-input definition.
        # -------------------------------------------------------------
        if self.sample_transform is not None:
            original_uid = sample["uid"]
            original_label = sample["label"]

            sample = self.sample_transform(sample)

            if not isinstance(sample, dict):
                raise TypeError(
                    "sample_transform must return a dictionary."
                )

            if sample.get("uid") != original_uid:
                raise RuntimeError(
                    "sample_transform changed the subject UID."
                )

            transformed_label = sample.get("label")
            if not torch.is_tensor(transformed_label):
                raise RuntimeError(
                    "sample_transform removed or invalidated the label."
                )

            if int(transformed_label.item()) != int(original_label.item()):
                raise RuntimeError(
                    "sample_transform changed the subject label."
                )

            for key in ("whole", "roi"):
                if key in sample:
                    tensor = sample[key]
                    if not torch.is_tensor(tensor):
                        raise TypeError(
                            f"sample_transform returned non-tensor {key!r}."
                        )
                    if tensor.ndim != 4:
                        raise ValueError(
                            f"sample_transform must return {key} as "
                            "[C,D,H,W], received "
                            f"{tuple(tensor.shape)}."
                        )
                    if tensor.dtype != torch.float32:
                        raise TypeError(
                            f"sample_transform changed {key} dtype to "
                            f"{tensor.dtype}; expected float32."
                        )
                    if not bool(torch.isfinite(tensor).all().item()):
                        raise FloatingPointError(
                            f"sample_transform produced NaN/Inf in {key}."
                        )

        # Shapes immediately before stochastic augmentation. Augmentation
        # must preserve these shapes, including model-specific selected-slice
        # shapes such as E4/E7 whole=[1,40,192,192].
        pre_augmentation_shapes = {
            key: tuple(sample[key].shape)
            for key in ("whole", "roi")
            if key in sample
        }

        # -------------------------------------------------------------
        # TRAINING-ONLY augmentation.
        #
        # run_experiment() passes an augmenter only to the training
        # dataset for scenarios 04-06. Validation/OOF datasets receive
        # augmenter=None and are therefore deterministic/unaugmented.
        # -------------------------------------------------------------
        if self.augmenter is not None:
            if self.input_type == "whole":
                sample["whole"] = self.augmenter(sample["whole"])

            elif self.input_type == "roi":
                sample["roi"] = self.augmenter(sample["roi"])

            else:
                sample["whole"], sample["roi"] = self.augmenter.augment_pair(
                    sample["whole"],
                    sample["roi"],
                )

            if "whole" in sample:
                if (
                    tuple(sample["whole"].shape)
                    != pre_augmentation_shapes["whole"]
                ):
                    raise RuntimeError(
                        "Whole-volume augmentation changed tensor shape: "
                        f"{pre_augmentation_shapes['whole']} -> "
                        f"{tuple(sample['whole'].shape)}"
                    )
                if not bool(torch.isfinite(sample["whole"]).all().item()):
                    raise FloatingPointError(
                        "Whole-volume augmentation produced NaN/Inf."
                    )

            if "roi" in sample:
                if (
                    tuple(sample["roi"].shape)
                    != pre_augmentation_shapes["roi"]
                ):
                    raise RuntimeError(
                        "ROI augmentation changed tensor shape: "
                        f"{pre_augmentation_shapes['roi']} -> "
                        f"{tuple(sample['roi'].shape)}"
                    )
                if not bool(torch.isfinite(sample["roi"]).all().item()):
                    raise FloatingPointError(
                        "ROI augmentation produced NaN/Inf."
                    )

        return sample


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    training: bool,
    shared_config,
    seed: int,
    device: torch.device,
) -> DataLoader:
    cfg = shared_config.dataloader
    workers = int(cfg.num_workers)

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(cfg.train_shuffle if training else cfg.validation_shuffle),
        num_workers=workers,
        pin_memory=bool(cfg.pin_memory and device.type == "cuda"),
        drop_last=(cfg.drop_last_train if training else cfg.drop_last_validation),
        worker_init_fn=(seed_worker if workers > 0 else None),
        generator=make_dataloader_generator(seed),
        persistent_workers=bool(cfg.persistent_workers and workers > 0),
    )

    if workers > 0:
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)

    return DataLoader(**kwargs)


def split_fold(
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    columns: Dict[str, str],
    fold_uid_col: str,
    fold_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    uid_col = columns["uid"]

    val_uids = set(
        folds.loc[folds["_fold"] == fold_id, fold_uid_col].astype(str)
    )

    train_rows = manifest[
        ~manifest[uid_col].astype(str).isin(val_uids)
    ].reset_index(drop=True)

    val_rows = manifest[
        manifest[uid_col].astype(str).isin(val_uids)
    ].reset_index(drop=True)

    train_uids = set(train_rows[uid_col].astype(str))
    actual_val_uids = set(val_rows[uid_col].astype(str))

    if train_uids & actual_val_uids:
        raise RuntimeError(f"Train/validation UID leakage in fold {fold_id}.")

    if actual_val_uids != val_uids:
        raise RuntimeError(
            f"Frozen validation membership mismatch in fold {fold_id}."
        )

    if len(train_rows) + len(val_rows) != len(manifest):
        raise RuntimeError(f"Fold {fold_id} does not partition the dataset.")

    return train_rows, val_rows


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False."
        )

    return device


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Not JSON serializable: {type(value).__name__}")

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=default)


def build_parser(spec: ExperimentSpec, default_project_root: Path) -> argparse.ArgumentParser:
    scenario = spec.scenario

    parser = argparse.ArgumentParser(
        description=(
            f"{spec.experiment_name}: "
            f"{scenario.input_type}, augmentation={scenario.augmentation}"
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/preprocessing_supervised_data/"
            "step10_supervised_dataset_manifest_data/"
            "supervised_dataset/supervised_dataset_manifest.csv"
        ),
    )

    parser.add_argument(
        "--folds",
        type=Path,
        default=Path(
            "data/preprocessing_supervised_data/"
            "step11_create_freeze_cv_splits_data/"
            "fold_assignments.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments_data") / spec.output_subdir,
    )

    parser.add_argument(
        "--fold",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=None,
        help="Run one fold only. Omit to run all five official folds.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Memory-dependent override. If omitted, uses the central "
            "suggested_batch_size() value."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="HPC/DataLoader override; training behavior remains in src.",
    )

    parser.add_argument(
        "--drop-last-train",
        action="store_true",
        help=(
            "Drop an incomplete final TRAINING batch. "
            "Validation batches are not affected."
        ),
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help=(
            "Optional smoke-test override. Official runs should normally "
            "use src.configs.training_config.DEFAULT_TRAINING_CONFIG."
        ),
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override central AMP setting only when needed.",
    )

    parser.add_argument(
        "--overwrite-fold",
        action="store_true",
        help="Delete the selected fold output directory before training it.",
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Do not train. Aggregate existing fold_0..fold_4/predictions.csv."
        ),
    )

    return parser


def _shared_config_for_fold(args, fold_id: int):
    shared = DEFAULT_TRAINING_CONFIG

    # Fold-specific deterministic seed. The same fold receives the same seed
    # across every model/scenario, preserving fair experiment comparisons.
    fold_seed = int(shared.reproducibility.seed) + int(fold_id)

    reproducibility = replace(
        shared.reproducibility,
        seed=fold_seed,
    )

    dataloader = shared.dataloader
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be >= 0.")
        dataloader = replace(
            dataloader,
            num_workers=int(args.num_workers),
        )

    if args.drop_last_train:
        dataloader = replace(
            dataloader,
            drop_last_train=True,
        )

    training = shared.training
    if args.max_epochs is not None:
        if args.max_epochs < 1:
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


def aggregate_existing(
    *,
    output_dir: Path,
    expected_subjects: int,
    threshold: float,
):
    prediction_paths = [
        output_dir / f"fold_{fold}" / "predictions.csv"
        for fold in range(5)
    ]

    missing = [path for path in prediction_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot aggregate: missing fold prediction files:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    return aggregate_cv_results(
        prediction_paths=prediction_paths,
        output_dir=output_dir,
        threshold=threshold,
        expected_folds=5,
        expected_subjects=expected_subjects,
    )


def run_experiment(spec: ExperimentSpec, *, default_project_root: Path) -> int:
    scenario = spec.scenario

    parser = build_parser(spec, default_project_root)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    manifest_path = resolve_path(project_root, args.manifest).resolve()
    folds_path = resolve_path(project_root, args.folds).resolve()
    output_dir = resolve_path(project_root, args.output_dir).resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not folds_path.is_file():
        raise FileNotFoundError(f"Fold assignments not found: {folds_path}")

    manifest, folds, columns, fold_uid_col = load_and_validate_frozen_tables(
        manifest_path,
        folds_path,
        input_type=scenario.input_type,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Root-level immutable experiment identity.
    save_json(
        {
            "experiment_name": spec.experiment_name,
            "model_name": spec.model_name,
            "scenario": asdict(scenario),
            "augmentation": bool(scenario.augmentation),
            "augmentation_spatial_mode": (
                spec.augmentation_spatial_mode
                if scenario.augmentation
                else None
            ),
            "augmentation_config": (
                asdict(DEFAULT_TRAINING_CONFIG.augmentation)
                if scenario.augmentation
                else None
            ),
            "augmentation_scope": (
                "training_only"
                if scenario.augmentation
                else "off"
            ),
            "sample_transform": (
                getattr(spec.sample_transform, "__name__", None)
                if spec.sample_transform is not None
                else None
            ),
            "manifest": str(manifest_path),
            "folds": str(folds_path),
            "n_subjects": int(len(manifest)),
            "whole_tensor_shape": list(WHOLE_TENSOR_SHAPE),
            "roi_tensor_shape": list(ROI_TENSOR_SHAPE),
            "additional_normalization": False,
            "additional_clipping": False,
            "additional_registration": False,
            "metadata": dict(spec.metadata or {}),
        },
        output_dir / "experiment_definition.json",
    )

    if args.aggregate_only:
        result = aggregate_existing(
            output_dir=output_dir,
            expected_subjects=len(manifest),
            threshold=DEFAULT_TRAINING_CONFIG.training.classification_threshold,
        )
        print(
            "CV aggregation complete. Global OOF AUROC = "
            f"{result['summary']['global_oof_metrics']['auroc']:.6f}"
        )
        return 0

    device = resolve_device(args.device)

    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else suggested_batch_size(spec.model_name, scenario.input_type)
    )

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    folds_to_run = [args.fold] if args.fold is not None else [0, 1, 2, 3, 4]

    print("=" * 79)
    print(spec.experiment_name)
    print("=" * 79)
    print(f"Model         : {spec.model_name}")
    print(f"Scenario      : {scenario.name}")
    print(f"Input         : {scenario.input_type}")
    print(
        "Augmentation  : "
        + (
            f"ON (training only, {spec.augmentation_spatial_mode})"
            if scenario.augmentation
            else "OFF"
        )
    )
    print(f"Subjects      : {len(manifest)}")
    print(f"Manifest      : {manifest_path}")
    print(f"Folds         : {folds_path}")
    print(f"Output        : {output_dir}")
    print(f"Device        : {device}")
    print(f"Batch size    : {batch_size}")
    print(f"Folds to run  : {folds_to_run}")

    for fold_id in folds_to_run:
        fold_start = time.monotonic()
        shared = _shared_config_for_fold(args, fold_id)
        seed_everything(shared.reproducibility)

        train_rows, val_rows = split_fold(
            manifest,
            folds,
            columns,
            fold_uid_col,
            fold_id,
        )

        fold_dir = output_dir / f"fold_{fold_id}"

        if fold_dir.exists():
            if args.overwrite_fold:
                shutil.rmtree(fold_dir)
            elif any(fold_dir.iterdir()):
                raise FileExistsError(
                    f"Fold output already exists and is non-empty: {fold_dir}\n"
                    "Use --overwrite-fold only when you intentionally want to rerun it."
                )

        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_config = ExperimentConfig(
            model_name=spec.model_name,
            scenario_id=spec.scenario_id,
            fold=fold_id,
            batch_size=batch_size,
            output_dir=str(fold_dir),
            shared=shared,
        )
        fold_config.validate()

        save_experiment_config(
            fold_config,
            fold_dir / "experiment_config.json",
        )

        # Build augmentation only for the training dataset.
        #
        # The parameter values come from shared.augmentation, which is the
        # central source of truth in src/configs/training_config.py.
        train_augmenter = (
            build_augmentation(
                enabled=True,
                spatial_mode=spec.augmentation_spatial_mode,
                config=shared.augmentation,
            )
            if scenario.augmentation
            else None
        )

        train_dataset = FrozenDATScanDataset(
            rows=train_rows,
            columns=columns,
            project_root=project_root,
            input_type=scenario.input_type,
            augmenter=train_augmenter,
            sample_transform=spec.sample_transform,
        )

        # IMPORTANT: validation/OOF data are never stochastically augmented.
        # The deterministic sample_transform, when present, is still applied
        # so training and validation have the same model-input definition.
        val_dataset = FrozenDATScanDataset(
            rows=val_rows,
            columns=columns,
            project_root=project_root,
            input_type=scenario.input_type,
            augmenter=None,
            sample_transform=spec.sample_transform,
        )

        fold_seed = shared.reproducibility.seed

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

        # Fresh model on every fold.
        model = spec.model_builder()

        metadata = {
            "experiment_name": spec.experiment_name,
            "model_name": spec.model_name,
            "scenario_id": spec.scenario_id,
            "scenario_name": scenario.name,
            "input_type": scenario.input_type,
            "augmentation": bool(scenario.augmentation),
            "augmentation_scope": (
                "training_only"
                if scenario.augmentation
                else "off"
            ),
            "augmentation_spatial_mode": (
                spec.augmentation_spatial_mode
                if scenario.augmentation
                else None
            ),
            "augmentation_config": (
                asdict(shared.augmentation)
                if scenario.augmentation
                else None
            ),
            "sample_transform": (
                getattr(spec.sample_transform, "__name__", None)
                if spec.sample_transform is not None
                else None
            ),
            "fold": fold_id,
            "train_subjects": len(train_rows),
            "validation_subjects": len(val_rows),
            "fold_seed": fold_seed,
            **dict(spec.metadata or {}),
        }

        trainer = build_trainer_from_config(
            model=model,
            config=shared,
            device=device,
            checkpoint_dir=fold_dir,
            metadata=metadata,
        )

        print()
        print("-" * 79)
        print(f"FOLD {fold_id}")
        print("-" * 79)
        print(f"Train subjects : {len(train_rows)}")
        print(f"Val subjects   : {len(val_rows)}")
        print(f"Seed           : {fold_seed}")
        print(f"AMP            : {trainer.amp_enabled}")
        print(f"Workers        : {shared.dataloader.num_workers}")
        print(
            "Checkpoint     : "
            f"{shared.training.checkpoint_metric} "
            f"({shared.training.checkpoint_mode})"
        )

        training_result = trainer.fit(
            train_loader=train_loader,
            validation_loader=val_loader,
            max_epochs=shared.training.max_epochs,
            verbose=True,
        )

        if training_result.best_checkpoint_path is None:
            raise RuntimeError("Trainer did not create a best checkpoint.")

        # Final fold evaluation must use the BEST checkpoint, not the last epoch.
        load_checkpoint(
            training_result.best_checkpoint_path,
            model=trainer.model,
            map_location=device,
            strict_model=True,
        )

        best_validation = trainer.validate(val_loader)

        if not best_validation.uids:
            raise RuntimeError(
                "Validation UIDs are required for fold/OOF evaluation."
            )

        evaluation = evaluate_predictions(
            uids=best_validation.uids,
            labels=best_validation.targets.numpy(),
            probabilities=best_validation.probabilities.numpy(),
            logits=best_validation.logits.numpy(),
            threshold=shared.training.classification_threshold,
            fold=fold_id,
        )

        save_evaluation_result(
            evaluation,
            output_dir=fold_dir,
        )

        pd.DataFrame(training_result.history).to_csv(
            fold_dir / "history.csv",
            index=False,
        )

        training_summary = {
            "fold": fold_id,
            "train_subjects": len(train_rows),
            "validation_subjects": len(val_rows),
            "fold_seed": fold_seed,
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
            "runtime_seconds": time.monotonic() - fold_start,
        }

        save_json(
            training_summary,
            fold_dir / "training_summary.json",
        )

        print(
            f"Fold {fold_id} complete | "
            f"best_epoch={training_result.best_epoch} | "
            f"val_loss={best_validation.loss:.6f} | "
            f"AUROC={evaluation.metrics['auroc']:.6f} | "
            f"balanced_accuracy={evaluation.metrics['balanced_accuracy']:.6f}"
        )

        # Explicit fold cleanup before constructing the next large 3D model.
        del trainer
        del model
        del train_loader
        del val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Only an invocation that trained all five folds automatically aggregates.
    # A single-fold invocation never mixes itself with potentially stale folds.
    if args.fold is None:
        aggregate_result = aggregate_existing(
            output_dir=output_dir,
            expected_subjects=len(manifest),
            threshold=DEFAULT_TRAINING_CONFIG.training.classification_threshold,
        )

        global_metrics = aggregate_result["summary"]["global_oof_metrics"]

        print()
        print("=" * 79)
        print("OFFICIAL 5-FOLD CV COMPLETE")
        print("=" * 79)
        print(f"Global OOF AUROC             : {global_metrics['auroc']:.6f}")
        print(f"Global OOF AUPRC             : {global_metrics['auprc']:.6f}")
        print(
            "Global OOF balanced accuracy : "
            f"{global_metrics['balanced_accuracy']:.6f}"
        )
        print(f"OOF subjects                 : {len(manifest)}")
    else:
        print(
            f"Fold {args.fold} finished. CV aggregation was intentionally skipped "
            "because this was a single-fold invocation."
        )

    return 0
