#!/usr/bin/env python3
"""
STEP 13 — FINAL PRE-TRAINING VALIDATION

Purpose
-------
Validate the complete supervised data path immediately before Step 14 training.

NO MODEL
NO OPTIMIZER
NO LOSS
NO BACKPROPAGATION
NO PARAMETER UPDATES

This script validates:
  13A subject/file accessibility
  13B tensor shapes/dtypes
  13C numerical validity
  13D labels
  13E Scenario-C pairing
  13F frozen 5-fold CV integrity
  13G augmentation separation/smoke behavior
  13H DataLoader batch loading
  13I determinism with augmentation OFF
  13J frozen final configuration + reports

Expected model tensors:
  Scenario A whole: [1,160,192,192]
  Scenario B ROI:   [1,36,44,44]
  Scenario C:       both tensors above

Important:
  - This script DOES NOT perform training.
  - It DOES NOT silently normalize, clip, or rescale images.
  - Input NIfTI arrays are converted from nibabel [X,Y,Z] to PyTorch [C,D,H,W].
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


EXPECTED_SUBJECTS = 1362
EXPECTED_NORMAL = 615
EXPECTED_PATHOLOGIC = 747

WHOLE_SHAPE = (1, 160, 192, 192)  # [C,D,H,W]
ROI_SHAPE = (1, 36, 44, 44)       # [C,D,H,W]

SCENARIOS = ("A", "B", "C")


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def resolve_path(project_root: Path, value: Any) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else project_root / p


def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def safe_scalar(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


def infer_manifest_columns(df: pd.DataFrame) -> Dict[str, str]:
    candidates = {
        "uid": ["uid", "UID", "subject_uid", "subject_id"],
        "label": ["is_pathologic", "label", "target", "y"],
        "whole": ["whole_path", "whole_volume_path", "whole", "whole_file"],
        "roi": ["roi_path", "striatal_path", "striatal_roi_path", "roi", "roi_file"],
    }
    found = {}
    for logical, names in candidates.items():
        for name in names:
            if name in df.columns:
                found[logical] = name
                break
        if logical not in found:
            raise ValueError(
                f"Could not identify required '{logical}' column. "
                f"Available columns: {list(df.columns)}"
            )
    return found


def infer_fold_columns(df: pd.DataFrame) -> Tuple[str, str]:
    uid_candidates = ["uid", "UID", "subject_uid", "subject_id"]
    fold_candidates = ["fold", "fold_id", "cv_fold", "fold_index"]

    uid_col = next((c for c in uid_candidates if c in df.columns), None)
    fold_col = next((c for c in fold_candidates if c in df.columns), None)

    if uid_col is None or fold_col is None:
        raise ValueError(
            "Could not identify UID/fold columns in fold assignments. "
            f"Available columns: {list(df.columns)}"
        )
    return uid_col, fold_col


# -------------------------------------------------------------------------
# Tensor loading — mirrors Step-12 convention:
# nibabel array [X,Y,Z] -> torch [C,D,H,W] = [1,Z,Y,X]
# -------------------------------------------------------------------------

def load_nifti_tensor(path: Path) -> torch.Tensor:
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {arr.shape} for {path}")

    # Explicitly create float32; no scaling/normalization is introduced here.
    arr = np.asarray(arr, dtype=np.float32)

    # [X,Y,Z] -> [Z,Y,X]
    arr = np.transpose(arr, (2, 1, 0))
    arr = np.ascontiguousarray(arr)

    # [D,H,W] -> [C,D,H,W]
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


def validate_tensor(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
) -> Dict[str, Any]:
    finite = torch.isfinite(tensor)
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())

    result = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "all_finite": bool(finite.all().item()) if tensor.numel() else False,
        "nonempty": bool(tensor.numel() > 0),
        "shape_ok": tuple(tensor.shape) == tuple(expected_shape),
        "dtype_ok": tensor.dtype == torch.float32,
    }

    if tensor.numel() > 0 and bool(finite.any().item()):
        vals = tensor[finite].float()
        result.update(
            min=float(vals.min().item()),
            max=float(vals.max().item()),
            mean=float(vals.mean().item()),
            std=float(vals.std(unbiased=False).item()),
            zero_fraction=float((vals == 0).float().mean().item()),
        )
    else:
        result.update(
            min=None,
            max=None,
            mean=None,
            std=None,
            zero_fraction=None,
        )

    result["valid"] = all(
        [
            result["dtype_ok"],
            result["shape_ok"],
            result["nonempty"],
            result["nan_count"] == 0,
            result["inf_count"] == 0,
        ]
    )
    return result


# -------------------------------------------------------------------------
# Minimal Step-13 Dataset.
# No model and no model-specific preprocessing.
# -------------------------------------------------------------------------

class FinalValidationDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        columns: Dict[str, str],
        project_root: Path,
        scenario: str,
        augmentation: bool = False,
        seed: int = 2026,
    ):
        self.rows = rows.reset_index(drop=True).copy()
        self.columns = columns
        self.project_root = project_root
        self.scenario = scenario.upper()
        self.augmentation = bool(augmentation)
        self.seed = int(seed)

        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.scenario}")

    def __len__(self) -> int:
        return len(self.rows)

    def _augment(self, tensor: torch.Tensor, index: int) -> torch.Tensor:
        """
        Step-13 smoke augmentation only.

        Purpose: prove augmentation ON can execute without changing shape,
        dtype, UID, or label. This is NOT intended to replace the frozen
        Step-12 augmentation policy.

        A deterministic-by-index left-right flip is used only when
        --augmentation-smoke-test is requested.
        """
        # Keep augmentation simple and shape-preserving.
        # Last dimension corresponds to X after [Z,Y,X] conversion.
        if (self.seed + index) % 2 == 0:
            return torch.flip(tensor, dims=(-1,))
        return tensor

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows.iloc[index]
        uid = str(row[self.columns["uid"]])
        label = int(row[self.columns["label"]])

        sample: Dict[str, Any] = {
            "uid": uid,
            "label": torch.tensor(label, dtype=torch.long),
        }

        if self.scenario in ("A", "C"):
            whole_path = resolve_path(self.project_root, row[self.columns["whole"]])
            whole = load_nifti_tensor(whole_path)
            if self.augmentation:
                whole = self._augment(whole, index)
            sample["whole"] = whole

        if self.scenario in ("B", "C"):
            roi_path = resolve_path(self.project_root, row[self.columns["roi"]])
            roi = load_nifti_tensor(roi_path)
            if self.augmentation:
                roi = self._augment(roi, index)
            sample["roi"] = roi

        return sample


# -------------------------------------------------------------------------
# 13A–13E
# -------------------------------------------------------------------------

def validate_subjects(
    manifest: pd.DataFrame,
    columns: Dict[str, str],
    project_root: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print_header("STEP 13A–13E — SUBJECT, FILE, TENSOR, LABEL & PAIRING VALIDATION")

    records: List[Dict[str, Any]] = []

    duplicate_uids = set(
        manifest.loc[
            manifest[columns["uid"]].astype(str).duplicated(keep=False),
            columns["uid"],
        ].astype(str)
    )

    for i, row in manifest.iterrows():
        uid = str(row[columns["uid"]])
        raw_label = row[columns["label"]]

        whole_path = resolve_path(project_root, row[columns["whole"]])
        roi_path = resolve_path(project_root, row[columns["roi"]])

        rec: Dict[str, Any] = {
            "uid": uid,
            "row_index": int(i),
            "duplicate_uid": uid in duplicate_uids,
            "whole_path": str(whole_path),
            "roi_path": str(roi_path),
            "whole_exists": whole_path.is_file(),
            "roi_exists": roi_path.is_file(),
            "label_raw": safe_scalar(raw_label),
            "label_valid": False,
            "whole_readable": False,
            "roi_readable": False,
            "whole_shape_ok": False,
            "roi_shape_ok": False,
            "whole_dtype_ok": False,
            "roi_dtype_ok": False,
            "whole_nan_count": None,
            "whole_inf_count": None,
            "roi_nan_count": None,
            "roi_inf_count": None,
            "whole_min": None,
            "whole_max": None,
            "whole_mean": None,
            "whole_std": None,
            "whole_zero_fraction": None,
            "roi_min": None,
            "roi_max": None,
            "roi_mean": None,
            "roi_std": None,
            "roi_zero_fraction": None,
            "error": "",
        }

        try:
            label_numeric = int(raw_label)
            rec["label_valid"] = label_numeric in (0, 1) and float(raw_label) == label_numeric
        except Exception:
            rec["label_valid"] = False

        try:
            if rec["whole_exists"]:
                whole = load_nifti_tensor(whole_path)
                w = validate_tensor(whole, WHOLE_SHAPE)
                rec["whole_readable"] = True
                rec["whole_shape_ok"] = w["shape_ok"]
                rec["whole_dtype_ok"] = w["dtype_ok"]
                rec["whole_nan_count"] = w["nan_count"]
                rec["whole_inf_count"] = w["inf_count"]
                rec["whole_min"] = w["min"]
                rec["whole_max"] = w["max"]
                rec["whole_mean"] = w["mean"]
                rec["whole_std"] = w["std"]
                rec["whole_zero_fraction"] = w["zero_fraction"]
        except Exception as e:
            rec["error"] += f"whole:{type(e).__name__}:{e}; "

        try:
            if rec["roi_exists"]:
                roi = load_nifti_tensor(roi_path)
                r = validate_tensor(roi, ROI_SHAPE)
                rec["roi_readable"] = True
                rec["roi_shape_ok"] = r["shape_ok"]
                rec["roi_dtype_ok"] = r["dtype_ok"]
                rec["roi_nan_count"] = r["nan_count"]
                rec["roi_inf_count"] = r["inf_count"]
                rec["roi_min"] = r["min"]
                rec["roi_max"] = r["max"]
                rec["roi_mean"] = r["mean"]
                rec["roi_std"] = r["std"]
                rec["roi_zero_fraction"] = r["zero_fraction"]
        except Exception as e:
            rec["error"] += f"roi:{type(e).__name__}:{e}; "

        rec["subject_pass"] = all(
            [
                not rec["duplicate_uid"],
                rec["whole_exists"],
                rec["roi_exists"],
                rec["whole_readable"],
                rec["roi_readable"],
                rec["label_valid"],
                rec["whole_shape_ok"],
                rec["roi_shape_ok"],
                rec["whole_dtype_ok"],
                rec["roi_dtype_ok"],
                (rec["whole_nan_count"] or 0) == 0,
                (rec["whole_inf_count"] or 0) == 0,
                (rec["roi_nan_count"] or 0) == 0,
                (rec["roi_inf_count"] or 0) == 0,
            ]
        )

        records.append(rec)

        if (i + 1) % 100 == 0 or i + 1 == len(manifest):
            print(f"Validated subjects: {i + 1}/{len(manifest)}")

    df = pd.DataFrame(records)

    labels_numeric = pd.to_numeric(manifest[columns["label"]], errors="coerce")
    label_counts = labels_numeric.value_counts(dropna=False).to_dict()

    summary = {
        "subjects": int(len(manifest)),
        "unique_uids": int(manifest[columns["uid"]].astype(str).nunique()),
        "duplicate_uid_rows": int(df["duplicate_uid"].sum()),
        "missing_whole_files": int((~df["whole_exists"]).sum()),
        "missing_roi_files": int((~df["roi_exists"]).sum()),
        "whole_read_failures": int((~df["whole_readable"]).sum()),
        "roi_read_failures": int((~df["roi_readable"]).sum()),
        "invalid_labels": int((~df["label_valid"]).sum()),
        "normal_count": int(label_counts.get(0, 0)),
        "pathologic_count": int(label_counts.get(1, 0)),
        "whole_shape_failures": int((~df["whole_shape_ok"]).sum()),
        "roi_shape_failures": int((~df["roi_shape_ok"]).sum()),
        "whole_dtype_failures": int((~df["whole_dtype_ok"]).sum()),
        "roi_dtype_failures": int((~df["roi_dtype_ok"]).sum()),
        "whole_nan_total": int(pd.to_numeric(df["whole_nan_count"], errors="coerce").fillna(0).sum()),
        "whole_inf_total": int(pd.to_numeric(df["whole_inf_count"], errors="coerce").fillna(0).sum()),
        "roi_nan_total": int(pd.to_numeric(df["roi_nan_count"], errors="coerce").fillna(0).sum()),
        "roi_inf_total": int(pd.to_numeric(df["roi_inf_count"], errors="coerce").fillna(0).sum()),
        "subject_failures": int((~df["subject_pass"]).sum()),
    }

    print(f"Subjects                  : {summary['subjects']}")
    print(f"Unique UIDs               : {summary['unique_uids']}")
    print(f"Normal / Pathologic       : {summary['normal_count']} / {summary['pathologic_count']}")
    print(f"Missing whole / ROI       : {summary['missing_whole_files']} / {summary['missing_roi_files']}")
    print(f"Whole / ROI shape failures: {summary['whole_shape_failures']} / {summary['roi_shape_failures']}")
    print(f"NaN totals whole / ROI    : {summary['whole_nan_total']} / {summary['roi_nan_total']}")
    print(f"Inf totals whole / ROI    : {summary['whole_inf_total']} / {summary['roi_inf_total']}")
    print(f"Subject failures          : {summary['subject_failures']}")

    return df, summary


# -------------------------------------------------------------------------
# 13F — folds
# -------------------------------------------------------------------------

def normalize_fold_value(v: Any) -> int:
    if isinstance(v, str):
        s = v.strip().lower().replace("fold_", "").replace("fold", "").strip()
        return int(s)
    return int(v)


def validate_folds(
    manifest: pd.DataFrame,
    columns: Dict[str, str],
    folds: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print_header("STEP 13F — VALIDATE FROZEN FIVE-FOLD CV")

    f_uid, f_fold = infer_fold_columns(folds)

    f = folds.copy()
    f[f_uid] = f[f_uid].astype(str)
    f["_fold_int"] = f[f_fold].map(normalize_fold_value)

    manifest_uids = set(manifest[columns["uid"]].astype(str))
    fold_uids = set(f[f_uid])

    unknown = sorted(fold_uids - manifest_uids)
    missing = sorted(manifest_uids - fold_uids)
    duplicate_rows = int(f[f_uid].duplicated().sum())

    observed_folds = sorted(f["_fold_int"].unique().tolist())

    # Accept either 0..4 or 1..5, but require exactly five folds.
    if len(observed_folds) != 5:
        print(f"WARNING: expected 5 unique folds, observed {observed_folds}")

    label_map = (
        manifest[[columns["uid"], columns["label"]]]
        .assign(**{columns["uid"]: lambda x: x[columns["uid"]].astype(str)})
        .set_index(columns["uid"])[columns["label"]]
        .to_dict()
    )

    records: List[Dict[str, Any]] = []

    validation_appearance = {uid: 0 for uid in manifest_uids}
    training_appearance = {uid: 0 for uid in manifest_uids}

    for fold_id in observed_folds:
        val_uids = set(f.loc[f["_fold_int"] == fold_id, f_uid])
        train_uids = fold_uids - val_uids

        for uid in val_uids & manifest_uids:
            validation_appearance[uid] += 1
        for uid in train_uids & manifest_uids:
            training_appearance[uid] += 1

        overlap = train_uids & val_uids

        train_labels = [int(label_map[u]) for u in train_uids if u in label_map]
        val_labels = [int(label_map[u]) for u in val_uids if u in label_map]

        rec = {
            "fold": int(fold_id),
            "train_n": len(train_uids),
            "val_n": len(val_uids),
            "overlap_n": len(overlap),
            "train_normal": train_labels.count(0),
            "train_pathologic": train_labels.count(1),
            "val_normal": val_labels.count(0),
            "val_pathologic": val_labels.count(1),
            "train_pathologic_fraction": float(np.mean(train_labels)) if train_labels else None,
            "val_pathologic_fraction": float(np.mean(val_labels)) if val_labels else None,
            "union_matches_all_subjects": (train_uids | val_uids) == manifest_uids,
            "fold_pass": (
                len(overlap) == 0
                and (train_uids | val_uids) == manifest_uids
                and train_uids.issubset(manifest_uids)
                and val_uids.issubset(manifest_uids)
            ),
        }
        records.append(rec)

        print(
            f"Fold {fold_id}: train={rec['train_n']} "
            f"val={rec['val_n']} overlap={rec['overlap_n']} "
            f"val 0/1={rec['val_normal']}/{rec['val_pathologic']}"
        )

    validation_once_failures = sum(v != 1 for v in validation_appearance.values())
    training_four_failures = sum(v != 4 for v in training_appearance.values())

    rdf = pd.DataFrame(records)
    summary = {
        "observed_folds": observed_folds,
        "n_folds": len(observed_folds),
        "fold_rows": int(len(f)),
        "fold_unique_uids": int(f[f_uid].nunique()),
        "unknown_cv_subjects": len(unknown),
        "missing_cv_subjects": len(missing),
        "duplicate_fold_assignments": duplicate_rows,
        "train_validation_overlap_total": int(rdf["overlap_n"].sum()) if not rdf.empty else 0,
        "validation_once_failures": int(validation_once_failures),
        "training_four_times_failures": int(training_four_failures),
        "all_fold_checks_pass": bool(
            len(observed_folds) == 5
            and not unknown
            and not missing
            and duplicate_rows == 0
            and (rdf["fold_pass"].all() if not rdf.empty else False)
            and validation_once_failures == 0
            and training_four_failures == 0
        ),
    }

    return rdf, summary


# -------------------------------------------------------------------------
# 13G–13I — augmentation / DataLoader / determinism
# -------------------------------------------------------------------------

def first_fold_split(
    manifest: pd.DataFrame,
    columns: Dict[str, str],
    folds: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    f_uid, f_fold = infer_fold_columns(folds)
    f = folds.copy()
    f[f_uid] = f[f_uid].astype(str)
    f["_fold_int"] = f[f_fold].map(normalize_fold_value)

    fold_id = sorted(f["_fold_int"].unique().tolist())[0]
    val_uids = set(f.loc[f["_fold_int"] == fold_id, f_uid])

    tmp = manifest.copy()
    tmp[columns["uid"]] = tmp[columns["uid"]].astype(str)

    val_df = tmp[tmp[columns["uid"]].isin(val_uids)].reset_index(drop=True)
    train_df = tmp[~tmp[columns["uid"]].isin(val_uids)].reset_index(drop=True)
    return train_df, val_df, fold_id


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=g,
    )


def validate_batch(
    batch: Dict[str, Any],
    scenario: str,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if "uid" not in batch or "label" not in batch:
        errors.append("batch missing uid and/or label")
        return False, errors

    labels = batch["label"]
    if not torch.is_tensor(labels):
        errors.append("labels are not tensor")
    else:
        if not bool(((labels == 0) | (labels == 1)).all().item()):
            errors.append("invalid label in batch")

    if scenario in ("A", "C"):
        if "whole" not in batch:
            errors.append("whole tensor missing")
        else:
            t = batch["whole"]
            if t.dtype != torch.float32:
                errors.append(f"whole dtype {t.dtype}")
            if tuple(t.shape[1:]) != WHOLE_SHAPE:
                errors.append(f"whole batch item shape {tuple(t.shape[1:])}")
            if not bool(torch.isfinite(t).all().item()):
                errors.append("whole has NaN/Inf")

    if scenario in ("B", "C"):
        if "roi" not in batch:
            errors.append("roi tensor missing")
        else:
            t = batch["roi"]
            if t.dtype != torch.float32:
                errors.append(f"roi dtype {t.dtype}")
            if tuple(t.shape[1:]) != ROI_SHAPE:
                errors.append(f"roi batch item shape {tuple(t.shape[1:])}")
            if not bool(torch.isfinite(t).all().item()):
                errors.append("roi has NaN/Inf")

    return len(errors) == 0, errors


def loader_smoke_tests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    columns: Dict[str, str],
    project_root: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    augmentation_smoke_test: bool,
) -> Dict[str, Any]:
    print_header("STEP 13G–13H — AUGMENTATION SEPARATION & DATALOADER SMOKE TEST")

    results: Dict[str, Any] = {}

    for scenario in SCENARIOS:
        scenario_result: Dict[str, Any] = {}

        # Train, augmentation OFF
        train_off = FinalValidationDataset(
            train_df, columns, project_root, scenario,
            augmentation=False, seed=seed
        )
        train_loader = make_loader(
            train_off, batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, seed=seed
        )

        try:
            batch = next(iter(train_loader))
            ok, errors = validate_batch(batch, scenario)
            scenario_result["train_aug_off"] = {
                "pass": ok,
                "errors": errors,
                "batch_size_observed": len(batch["uid"]),
            }
        except Exception as e:
            scenario_result["train_aug_off"] = {
                "pass": False,
                "errors": [f"{type(e).__name__}: {e}"],
            }

        # Validation, ALWAYS augmentation OFF
        val_ds = FinalValidationDataset(
            val_df, columns, project_root, scenario,
            augmentation=False, seed=seed
        )
        val_loader = make_loader(
            val_ds, batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, seed=seed
        )
        try:
            batch = next(iter(val_loader))
            ok, errors = validate_batch(batch, scenario)
            scenario_result["validation_aug_off"] = {
                "pass": ok and not val_ds.augmentation,
                "errors": errors,
                "augmentation_enabled": val_ds.augmentation,
                "batch_size_observed": len(batch["uid"]),
            }
        except Exception as e:
            scenario_result["validation_aug_off"] = {
                "pass": False,
                "errors": [f"{type(e).__name__}: {e}"],
                "augmentation_enabled": False,
            }

        # Optional smoke-only augmentation test.
        if augmentation_smoke_test:
            train_on = FinalValidationDataset(
                train_df, columns, project_root, scenario,
                augmentation=True, seed=seed
            )
            try:
                sample = train_on[0]
                errs = []
                if int(sample["label"]) not in (0, 1):
                    errs.append("label changed/invalid")
                if scenario in ("A", "C"):
                    if tuple(sample["whole"].shape) != WHOLE_SHAPE:
                        errs.append("whole shape changed")
                    if sample["whole"].dtype != torch.float32:
                        errs.append("whole dtype changed")
                    if not torch.isfinite(sample["whole"]).all():
                        errs.append("whole augmentation introduced NaN/Inf")
                if scenario in ("B", "C"):
                    if tuple(sample["roi"].shape) != ROI_SHAPE:
                        errs.append("roi shape changed")
                    if sample["roi"].dtype != torch.float32:
                        errs.append("roi dtype changed")
                    if not torch.isfinite(sample["roi"]).all():
                        errs.append("roi augmentation introduced NaN/Inf")
                scenario_result["train_aug_on_smoke"] = {
                    "pass": len(errs) == 0,
                    "errors": errs,
                    "note": "Step-13 smoke augmentation, not a replacement for frozen Step-12 augmentation definitions.",
                }
            except Exception as e:
                scenario_result["train_aug_on_smoke"] = {
                    "pass": False,
                    "errors": [f"{type(e).__name__}: {e}"],
                }
        else:
            scenario_result["train_aug_on_smoke"] = {
                "pass": True,
                "skipped": True,
                "note": "Enable with --augmentation-smoke-test. Frozen Step-12 augmentation should remain authoritative.",
            }

        results[scenario] = scenario_result
        print(
            f"Scenario {scenario}: "
            f"train_off={scenario_result['train_aug_off']['pass']} "
            f"val_off={scenario_result['validation_aug_off']['pass']} "
            f"train_aug_smoke={scenario_result['train_aug_on_smoke']['pass']}"
        )

    results["all_pass"] = all(
        entry[key]["pass"]
        for scenario, entry in results.items()
        if scenario in SCENARIOS
        for key in ("train_aug_off", "validation_aug_off", "train_aug_on_smoke")
    )
    return results


def determinism_checks(
    manifest: pd.DataFrame,
    columns: Dict[str, str],
    project_root: Path,
    seed: int,
    n_samples: int,
) -> Dict[str, Any]:
    print_header("STEP 13I — DETERMINISM CHECKS (AUGMENTATION OFF)")

    n = min(n_samples, len(manifest))
    indices = list(range(n))
    out: Dict[str, Any] = {}

    for scenario in SCENARIOS:
        ds1 = FinalValidationDataset(
            manifest, columns, project_root, scenario, augmentation=False, seed=seed
        )
        ds2 = FinalValidationDataset(
            manifest, columns, project_root, scenario, augmentation=False, seed=seed
        )

        failures = []
        for idx in indices:
            a = ds1[idx]
            b = ds2[idx]

            if a["uid"] != b["uid"]:
                failures.append({"index": idx, "reason": "UID mismatch"})
                continue
            if int(a["label"]) != int(b["label"]):
                failures.append({"index": idx, "reason": "label mismatch"})
                continue
            if scenario in ("A", "C") and not torch.equal(a["whole"], b["whole"]):
                failures.append({"index": idx, "uid": a["uid"], "reason": "whole tensor differs"})
            if scenario in ("B", "C") and not torch.equal(a["roi"], b["roi"]):
                failures.append({"index": idx, "uid": a["uid"], "reason": "ROI tensor differs"})

        out[scenario] = {
            "checked_samples": n,
            "failure_count": len(failures),
            "failures": failures[:50],
            "pass": len(failures) == 0,
        }
        print(f"Scenario {scenario}: checked={n}, failures={len(failures)}")

    out["all_pass"] = all(out[s]["pass"] for s in SCENARIOS)
    return out


# -------------------------------------------------------------------------
# 13J / final gate
# -------------------------------------------------------------------------

def numerical_dataset_summary(subject_df: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for prefix in ("whole", "roi"):
        result[prefix] = {}
        for metric in ("min", "max", "mean", "std", "zero_fraction"):
            col = f"{prefix}_{metric}"
            vals = pd.to_numeric(subject_df[col], errors="coerce").dropna()
            if vals.empty:
                result[prefix][metric] = None
            else:
                result[prefix][metric] = {
                    "min": float(vals.min()),
                    "p01": float(vals.quantile(0.01)),
                    "median": float(vals.median()),
                    "p99": float(vals.quantile(0.99)),
                    "max": float(vals.max()),
                    "mean": float(vals.mean()),
                }
    return result


def final_gate(
    subject_summary: Dict[str, Any],
    fold_summary: Dict[str, Any],
    loader_results: Dict[str, Any],
    determinism: Dict[str, Any],
) -> Dict[str, Any]:
    checks = {
        "subjects_1362": subject_summary["subjects"] == EXPECTED_SUBJECTS,
        "unique_uids_1362": subject_summary["unique_uids"] == EXPECTED_SUBJECTS,
        "normal_count_615": subject_summary["normal_count"] == EXPECTED_NORMAL,
        "pathologic_count_747": subject_summary["pathologic_count"] == EXPECTED_PATHOLOGIC,
        "duplicate_uids_0": subject_summary["duplicate_uid_rows"] == 0,
        "missing_whole_files_0": subject_summary["missing_whole_files"] == 0,
        "missing_roi_files_0": subject_summary["missing_roi_files"] == 0,
        "whole_read_failures_0": subject_summary["whole_read_failures"] == 0,
        "roi_read_failures_0": subject_summary["roi_read_failures"] == 0,
        "invalid_labels_0": subject_summary["invalid_labels"] == 0,
        "whole_shape_failures_0": subject_summary["whole_shape_failures"] == 0,
        "roi_shape_failures_0": subject_summary["roi_shape_failures"] == 0,
        "whole_dtype_failures_0": subject_summary["whole_dtype_failures"] == 0,
        "roi_dtype_failures_0": subject_summary["roi_dtype_failures"] == 0,
        "whole_nan_0": subject_summary["whole_nan_total"] == 0,
        "roi_nan_0": subject_summary["roi_nan_total"] == 0,
        "whole_inf_0": subject_summary["whole_inf_total"] == 0,
        "roi_inf_0": subject_summary["roi_inf_total"] == 0,
        "five_folds": fold_summary["n_folds"] == 5,
        "cv_unknown_subjects_0": fold_summary["unknown_cv_subjects"] == 0,
        "cv_missing_subjects_0": fold_summary["missing_cv_subjects"] == 0,
        "cv_duplicate_assignments_0": fold_summary["duplicate_fold_assignments"] == 0,
        "fold_leakage_0": fold_summary["train_validation_overlap_total"] == 0,
        "every_subject_validation_once": fold_summary["validation_once_failures"] == 0,
        "every_subject_training_four_times": fold_summary["training_four_times_failures"] == 0,
        "dataloader_checks_pass": bool(loader_results["all_pass"]),
        "determinism_checks_pass": bool(determinism["all_pass"]),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "failed_checks": [k for k, v in checks.items() if not v],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 13 final pre-training validation")

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Default: current working directory.",
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
            "step11_create_freeze_cv_splits_data/fold_assignments.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/preprocessing_supervised_data/"
            "step13_final_pretraining_validation_data"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--determinism-samples", type=int, default=16)
    parser.add_argument(
        "--augmentation-smoke-test",
        action="store_true",
        help=(
            "Run a shape/dtype-preserving augmentation smoke check. "
            "This does not replace the authoritative frozen Step-12 augmentation definition."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = args.project_root.resolve()
    manifest_path = resolve_path(project_root, args.manifest)
    folds_path = resolve_path(project_root, args.folds)
    output_dir = resolve_path(project_root, args.output_dir)

    report_dir = output_dir / "validation_reports"
    subject_dir = output_dir / "subject_validation"
    fold_dir = output_dir / "fold_validation"
    frozen_dir = output_dir / "frozen_configuration"

    for d in (report_dir, subject_dir, fold_dir, frozen_dir):
        d.mkdir(parents=True, exist_ok=True)

    print_header("STEP 13 — FINAL PRE-TRAINING VALIDATION")
    print(f"Project root             : {project_root}")
    print(f"Supervised manifest      : {manifest_path}")
    print(f"Fold assignments         : {folds_path}")
    print(f"Output directory         : {output_dir}")
    print(f"Batch size smoke test    : {args.batch_size}")
    print(f"Num workers              : {args.num_workers}")
    print(f"Pin memory               : {args.pin_memory}")
    print(f"Seed                     : {args.seed}")
    print("Model/training           : DISABLED")

    if not manifest_path.is_file():
        print(f"FATAL: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    if not folds_path.is_file():
        print(f"FATAL: fold assignments not found: {folds_path}", file=sys.stderr)
        return 2

    manifest = pd.read_csv(manifest_path)
    folds = pd.read_csv(folds_path)

    try:
        columns = infer_manifest_columns(manifest)
        print(f"Manifest column mapping  : {columns}")

        subject_df, subject_summary = validate_subjects(
            manifest, columns, project_root
        )
        subject_csv = subject_dir / "step13_subject_validation.csv"
        subject_df.to_csv(subject_csv, index=False)

        fold_df, fold_summary = validate_folds(
            manifest, columns, folds
        )
        fold_csv = fold_dir / "step13_fold_validation.csv"
        fold_df.to_csv(fold_csv, index=False)

        train_df, val_df, smoke_fold = first_fold_split(
            manifest, columns, folds
        )

        loader_results = loader_smoke_tests(
            train_df=train_df,
            val_df=val_df,
            columns=columns,
            project_root=project_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            seed=args.seed,
            augmentation_smoke_test=args.augmentation_smoke_test,
        )

        determinism = determinism_checks(
            manifest=manifest,
            columns=columns,
            project_root=project_root,
            seed=args.seed,
            n_samples=args.determinism_samples,
        )

        numerical_summary = numerical_dataset_summary(subject_df)
        gate = final_gate(
            subject_summary,
            fold_summary,
            loader_results,
            determinism,
        )

        frozen_config = {
            "step": 13,
            "name": "final_pretraining_validation",
            "created_utc": utc_now(),
            "purpose": "Final validation only; no model training.",
            "project_root": str(project_root),
            "inputs": {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "fold_assignments": str(folds_path),
                "fold_assignments_sha256": sha256_file(folds_path),
            },
            "manifest_columns": columns,
            "expected": {
                "subjects": EXPECTED_SUBJECTS,
                "normal": EXPECTED_NORMAL,
                "pathologic": EXPECTED_PATHOLOGIC,
                "scenario_A_whole_CHWD": list(WHOLE_SHAPE),
                "scenario_B_roi_CHWD": list(ROI_SHAPE),
                "scenario_C_whole_CHWD": list(WHOLE_SHAPE),
                "scenario_C_roi_CHWD": list(ROI_SHAPE),
            },
            "tensor_loading": {
                "nifti_source_order": "[X,Y,Z]",
                "model_tensor_order": "[C,D,H,W]",
                "conversion": "[X,Y,Z] -> [Z,Y,X] -> add channel",
                "dtype": "torch.float32",
                "additional_normalization": "NONE",
                "clipping": "NONE",
            },
            "dataloader_validation": {
                "smoke_fold": smoke_fold,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "pin_memory": args.pin_memory,
                "drop_last": False,
                "training_shuffle": True,
                "validation_shuffle": False,
                "validation_augmentation": False,
            },
            "augmentation": {
                "validation": "OFF",
                "train_smoke_test_enabled": args.augmentation_smoke_test,
                "note": (
                    "Step-13 smoke augmentation is not the authoritative training augmentation. "
                    "Step-12 frozen augmentation definitions remain authoritative."
                ),
            },
            "reproducibility": {
                "seed": args.seed,
                "determinism_samples": args.determinism_samples,
            },
            "final_gate_passed": gate["passed"],
        }

        full_report = {
            "step": 13,
            "created_utc": utc_now(),
            "status": "PASS" if gate["passed"] else "FAIL",
            "subject_validation": subject_summary,
            "numerical_summary": numerical_summary,
            "fold_validation": fold_summary,
            "dataloader_validation": loader_results,
            "determinism_validation": determinism,
            "final_gate": gate,
        }

        json_dump(full_report, report_dir / "step13_validation_report.json")
        json_dump(frozen_config, frozen_dir / "step13_frozen_config.json")

        summary_rows = []
        for section, values in (
            ("subject", subject_summary),
            ("fold", fold_summary),
        ):
            for key, value in values.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True)
                summary_rows.append(
                    {"section": section, "metric": key, "value": value}
                )

        for key, value in gate["checks"].items():
            summary_rows.append(
                {"section": "final_gate", "metric": key, "value": value}
            )

        pd.DataFrame(summary_rows).to_csv(
            report_dir / "step13_validation_summary.csv",
            index=False,
        )

        print_header("STEP 13 FINAL GATE")
        for name, passed in gate["checks"].items():
            print(f"{'PASS' if passed else 'FAIL':4s}  {name}")

        print()
        print(f"Subject report           : {subject_csv}")
        print(f"Fold report              : {fold_csv}")
        print(f"Validation report        : {report_dir / 'step13_validation_report.json'}")
        print(f"Validation summary       : {report_dir / 'step13_validation_summary.csv'}")
        print(f"Frozen configuration     : {frozen_dir / 'step13_frozen_config.json'}")

        if gate["passed"]:
            print()
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║                    STEP 13 — PASS                          ║")
            print("║     FINAL PRE-TRAINING VALIDATION GATE SATISFIED           ║")
            print("║       Step 14 model training may now begin.                ║")
            print("╚══════════════════════════════════════════════════════════════╝")
            return 0

        print()
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║                    STEP 13 — FAIL                          ║")
        print("║          DO NOT START STEP 14 TRAINING.                    ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print("Failed checks:")
        for x in gate["failed_checks"]:
            print(f"  - {x}")
        return 1

    except Exception as exc:
        fatal_report = {
            "step": 13,
            "created_utc": utc_now(),
            "status": "FATAL_ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        json_dump(fatal_report, report_dir / "step13_validation_report.json")
        print()
        print("STEP 13 FATAL ERROR", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
