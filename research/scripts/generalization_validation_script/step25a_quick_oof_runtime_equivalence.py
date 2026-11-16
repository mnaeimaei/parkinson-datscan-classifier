#!/usr/bin/env python3
"""
STEP 25A — QUICK HELD-OUT-FOLD RUNTIME EQUIVALENCE GATE

Purpose
-------
Before redesigning CV in Step 25B, verify on a small deterministic subset that:

1) final competition preprocessing reproduces the frozen training whole/ROI
   products, and
2) the portable ENS328 checkpoints reproduce the ORIGINAL RAW OOF
   probabilities when each subject is evaluated ONLY by its held-out fold.

This is a diagnostic gate only. It does not train, calibrate, ensemble, or
perform model selection.

Expected project layout (defaults match the DAT-scan-classifier project):

    data/raw_images_data/<uid>.nii.gz
    data/preprocessing_supervised_data/
      step10_supervised_dataset_manifest_data/supervised_dataset/
        supervised_dataset_manifest.csv
      step11_create_freeze_cv_splits_data/fold_assignments.csv
    data/calibration_validation_data/competition_shortlist_data/
      competition_shortlist_8.csv
    submission_inference/
      assets/
      pipeline_preprocessing/
      model_bundle/

The script builds an isolated temporary runtime workspace, preprocesses only
N subjects (default 25 = 5/fold), then performs one held-out-fold model pass
per selected member and subject.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping

import nibabel as nib
import numpy as np
import pandas as pd


FROZEN_MEMBERS = ("P1", "P2", "P3", "P7", "P8")
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
EXPECTED_WHOLE_SHAPE = (192, 192, 160)
EXPECTED_ROI_SHAPE = (44, 44, 36)

# Exact order used by the submitted Step-24 preprocessing runtime.
PREPROCESSING_STEPS = (
    "step2_orientation_std_local.sh",
    "step4_resampling_local.sh",
    "step6a1_full_rigid_registration_local.sh",
    "step6a4_full_registration_audit_local.sh",
    "step6b1_finalize_registration_transforms_local.sh",
    "step6b3_rescue_pnsm_similarity_local.sh",
    "step6c_finalize_all_registration_transforms_local.sh",
    "step6d1_extract_occipital_reference_local.sh",
    "step6d4_finalize_reference_values_local.sh",
    "step6e_normalize_occipital_local.sh",
    "step7b_roi_l0_fixed_center_local.sh",
    "step7c1_roi_l1_template_transform_local.sh",
    "step7c5_l1_center_consistency_local.sh",
    "step7e2_freeze_final_localization_local.sh",
    "step8c_striatal_crop_generator.sh",
    "step9c_create_fixed_whole_volumes.sh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 25A quick held-out-fold runtime equivalence gate."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("submission_inference"),
        help="Directory containing assets/, pipeline_preprocessing/, model_bundle/.",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw_images_data")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/preprocessing_supervised_data/"
            "step10_supervised_dataset_manifest_data/supervised_dataset/"
            "supervised_dataset_manifest.csv"
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
        "--shortlist",
        type=Path,
        default=Path(
            "data/calibration_validation_data/competition_shortlist_data/"
            "competition_shortlist_8.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/generalization_validation_data/step25a_quick_oof_runtime_equivalence"
        ),
    )
    parser.add_argument("--samples-per-fold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA AMP. Leave unset to mirror the portable runtime default.",
    )
    parser.add_argument("--voxel-atol", type=float, default=1e-6)
    parser.add_argument("--affine-atol", type=float, default=1e-6)
    parser.add_argument(
        "--prob-atol",
        type=float,
        default=1e-4,
        help=(
            "Raw OOF probability tolerance. 1e-4 allows small device/AMP "
            "rounding while still detecting meaningful mismatch."
        ),
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep temporary preprocessed files for manual inspection.",
    )
    return parser.parse_args()


def resolve_under_root(project_root: Path, path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def resolve_recorded_path(project_root: Path, value: object) -> Path:
    """Resolve paths stored in manifests, including stale absolute cluster paths."""
    text = str(value).strip()
    if not text:
        raise ValueError("Empty recorded path")

    p = Path(text).expanduser()
    candidates: list[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(project_root / p)

    # Step-16/Step-10 files may contain absolute paths from another machine.
    # Recover the repository-relative tail beginning at data/.
    parts = p.parts
    if "data" in parts:
        idx = parts.index("data")
        candidates.append(project_root.joinpath(*parts[idx:]))

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not resolve recorded path. Tried:\n  "
        + "\n  ".join(str(c.resolve()) for c in candidates)
    )


def require_columns(df: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def load_authoritative_tables(
    *, project_root: Path, manifest_path: Path, folds_path: Path, shortlist_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    manifest = pd.read_csv(manifest_path)
    folds = pd.read_csv(folds_path)
    shortlist = pd.read_csv(shortlist_path)

    require_columns(
        manifest, {"uid", "is_pathologic", "whole_path", "roi_path"}, "Step-10 manifest"
    )
    require_columns(folds, {"uid", "fold"}, "Step-11 folds")
    require_columns(shortlist, {"Shortlist ID"}, "Step-16 shortlist")

    manifest["uid"] = manifest["uid"].astype(str)
    folds["uid"] = folds["uid"].astype(str)
    shortlist["Shortlist ID"] = shortlist["Shortlist ID"].astype(str)

    if manifest["uid"].duplicated().any():
        raise ValueError("Step-10 manifest contains duplicate UIDs")
    if folds["uid"].duplicated().any():
        raise ValueError("Step-11 folds contain duplicate UIDs")

    selected_rows = shortlist[shortlist["Shortlist ID"].isin(FROZEN_MEMBERS)].copy()
    if set(selected_rows["Shortlist ID"]) != set(FROZEN_MEMBERS):
        raise ValueError(
            f"Shortlist does not contain exactly selected members {FROZEN_MEMBERS}"
        )

    oof_by_member: dict[str, pd.DataFrame] = {}
    for member in FROZEN_MEMBERS:
        row = selected_rows.loc[selected_rows["Shortlist ID"] == member].iloc[0]

        if "OOF Prediction File" in row.index and pd.notna(row["OOF Prediction File"]):
            oof_path = resolve_recorded_path(project_root, row["OOF Prediction File"])
            oof = pd.read_csv(oof_path)
            require_columns(
                oof, {"uid", "fold", "is_pathologic", "probability"}, f"{member} OOF"
            )
            oof = oof[["uid", "fold", "is_pathologic", "probability"]].copy()
            oof = oof.rename(columns={"probability": "raw_probability"})
        elif (
            "Cross-Fitted Prediction File" in row.index
            and pd.notna(row["Cross-Fitted Prediction File"])
        ):
            cf_path = resolve_recorded_path(
                project_root, row["Cross-Fitted Prediction File"]
            )
            oof = pd.read_csv(cf_path)
            require_columns(
                oof,
                {"uid", "fold", "is_pathologic", "raw_probability"},
                f"{member} cross-fitted prediction file",
            )
            oof = oof[["uid", "fold", "is_pathologic", "raw_probability"]].copy()
        else:
            raise ValueError(
                f"{member}: shortlist has neither OOF Prediction File nor "
                "Cross-Fitted Prediction File"
            )

        oof["uid"] = oof["uid"].astype(str)
        oof["fold"] = pd.to_numeric(oof["fold"], errors="raise").astype(int)
        oof["is_pathologic"] = pd.to_numeric(
            oof["is_pathologic"], errors="raise"
        ).astype(int)
        oof["raw_probability"] = pd.to_numeric(
            oof["raw_probability"], errors="raise"
        ).astype(float)

        if oof["uid"].duplicated().any():
            raise ValueError(f"{member}: duplicate UIDs in OOF predictions")
        if not np.isfinite(oof["raw_probability"].to_numpy()).all():
            raise ValueError(f"{member}: non-finite raw OOF probabilities")

        oof_by_member[member] = oof.sort_values("uid").reset_index(drop=True)

    # Cross-member UID/fold/label integrity.
    reference = oof_by_member["P1"][["uid", "fold", "is_pathologic"]].sort_values(
        "uid"
    ).reset_index(drop=True)
    for member in FROZEN_MEMBERS[1:]:
        current = oof_by_member[member][["uid", "fold", "is_pathologic"]].sort_values(
            "uid"
        ).reset_index(drop=True)
        if not reference.equals(current):
            raise RuntimeError(f"{member}: UID/fold/label alignment differs from P1")

    # Step-11 fold assignments must agree with OOF fold assignments.
    fold_check = reference.merge(folds[["uid", "fold"]], on="uid", suffixes=("_oof", "_step11"))
    if len(fold_check) != len(reference):
        raise RuntimeError("Step-11 folds do not cover all P1 OOF subjects")
    if not np.array_equal(
        fold_check["fold_oof"].to_numpy(), fold_check["fold_step11"].to_numpy()
    ):
        raise RuntimeError("OOF fold IDs disagree with frozen Step-11 assignments")

    return manifest, folds, shortlist, oof_by_member


def choose_subset(
    reference_oof: pd.DataFrame, *, samples_per_fold: int, seed: int
) -> pd.DataFrame:
    if samples_per_fold < 2:
        raise ValueError("--samples-per-fold must be >= 2 so both classes can be represented")

    rng = np.random.default_rng(seed)
    selected_parts: list[pd.DataFrame] = []

    for fold in EXPECTED_FOLDS:
        fold_df = reference_oof.loc[reference_oof["fold"] == fold].copy()
        if fold_df.empty:
            raise RuntimeError(f"Fold {fold} is empty")

        classes = sorted(fold_df["is_pathologic"].unique().tolist())
        if classes != [0, 1]:
            raise RuntimeError(f"Fold {fold} does not contain both classes")

        # Guarantee at least one of each class, then fill the remaining slots
        # from the rest of the fold deterministically.
        picked_indices: list[int] = []
        for label in (0, 1):
            idx = fold_df.index[fold_df["is_pathologic"] == label].to_numpy()
            picked_indices.append(int(rng.choice(idx, size=1, replace=False)[0]))

        remaining_needed = samples_per_fold - len(picked_indices)
        if remaining_needed > 0:
            remaining_pool = fold_df.index.difference(picked_indices).to_numpy()
            if remaining_needed > len(remaining_pool):
                raise RuntimeError(
                    f"Fold {fold}: requested {samples_per_fold} subjects, only {len(fold_df)} available"
                )
            extra = rng.choice(remaining_pool, size=remaining_needed, replace=False)
            picked_indices.extend(int(v) for v in extra)

        part = fold_df.loc[picked_indices].copy()
        part = part.sort_values(["is_pathologic", "uid"], kind="stable")
        selected_parts.append(part)

    selected = pd.concat(selected_parts, ignore_index=True)
    if selected["uid"].duplicated().any():
        raise RuntimeError("Subset selection produced duplicate UIDs")

    return selected[["uid", "fold", "is_pathologic"]].sort_values(
        ["fold", "is_pathologic", "uid"], kind="stable"
    ).reset_index(drop=True)


def stage_preprocessing_workspace(
    *, runtime_dir: Path, raw_dir: Path, selected_uids: list[str], keep: bool
) -> tuple[Path, Path]:
    base = Path(
        tempfile.mkdtemp(prefix="step25a_ens328_")
    ).resolve()

    data_demo = base / "data-demo"
    niftis = data_demo / "niftis"
    niftis.mkdir(parents=True, exist_ok=True)

    # The uploaded/local shell wrappers currently use submission_inference.
    # Also create a submission_src alias so the gate remains compatible if the
    # wrappers are later normalized to the official repository naming.
    work_runtime = base / "submission_inference"
    work_runtime.mkdir(parents=True, exist_ok=True)

    shutil.copytree(runtime_dir / "assets", work_runtime / "assets")
    shutil.copytree(
        runtime_dir / "pipeline_preprocessing",
        work_runtime / "pipeline_preprocessing",
    )

    submission_src_alias = base / "submission_src"
    if not submission_src_alias.exists():
        submission_src_alias.symlink_to(work_runtime, target_is_directory=True)

    for uid in selected_uids:
        source = raw_dir / f"{uid}.nii.gz"
        if not source.is_file():
            raise FileNotFoundError(f"Raw NIfTI missing for {uid}: {source}")
        (niftis / source.name).symlink_to(source.resolve())

    # Some wrappers/tools expect the template file to exist even though the
    # preprocessing steps do not use labels.
    pd.DataFrame(
        {"uid": selected_uids, "is_pathologic": np.zeros(len(selected_uids))}
    ).to_csv(data_demo / "submission_format.csv", index=False)

    if keep:
        print(f"Temporary workspace will be kept: {base}")

    return base, work_runtime


def run_preprocessing(workspace_root: Path, work_runtime: Path, output_dir: Path) -> None:
    scripts_dir = work_runtime / "pipeline_preprocessing"
    logs_dir = output_dir / "preprocessing_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(PREPROCESSING_STEPS)} frozen preprocessing steps on subset...")
    for i, step in enumerate(PREPROCESSING_STEPS, start=1):
        script = scripts_dir / step
        if not script.is_file():
            raise FileNotFoundError(f"Missing runtime preprocessing wrapper: {script}")

        print(f"  [{i:02d}/{len(PREPROCESSING_STEPS):02d}] {step} ... ", end="", flush=True)
        result = subprocess.run(
            ["bash", str(script)],
            cwd=scripts_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        (logs_dir / f"{i:02d}_{step}.log").write_text(
            result.stdout or "", encoding="utf-8"
        )
        if result.returncode != 0:
            print("FAIL")
            raise RuntimeError(
                f"Preprocessing failed at {step}. See {logs_dir / f'{i:02d}_{step}.log'}"
            )
        print("PASS")


def locate_runtime_products(workspace_root: Path, uid: str) -> tuple[Path, Path]:
    roots = [workspace_root / "submission_inference", workspace_root / "submission_src"]
    tried: list[str] = []
    for root in roots:
        roi = root / "data_preprocessing" / "step8c_striatal_crop_generator" / "crops" / f"{uid}.nii.gz"
        whole = root / "data_preprocessing" / "step9c_create_fixed_whole_volumes" / f"{uid}.nii.gz"
        tried.extend([str(roi), str(whole)])
        if roi.is_file() and whole.is_file():
            return whole.resolve(), roi.resolve()
    raise FileNotFoundError(
        f"Could not locate final runtime whole/ROI products for {uid}. Tried:\n  "
        + "\n  ".join(tried)
    )


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    affine = np.asarray(image.affine, dtype=np.float64)
    return data, affine


def compare_nifti_pair(
    runtime_path: Path,
    frozen_path: Path,
    *,
    expected_shape: tuple[int, int, int],
    voxel_atol: float,
    affine_atol: float,
) -> dict:
    runtime_data, runtime_affine = load_nifti(runtime_path)
    frozen_data, frozen_affine = load_nifti(frozen_path)

    shape_ok = tuple(runtime_data.shape) == expected_shape == tuple(frozen_data.shape)
    finite_ok = bool(np.isfinite(runtime_data).all() and np.isfinite(frozen_data).all())

    if runtime_data.shape == frozen_data.shape:
        diff = np.abs(runtime_data.astype(np.float64) - frozen_data.astype(np.float64))
        voxel_max = float(diff.max(initial=0.0))
        voxel_mean = float(diff.mean())
        voxel_exact = bool(np.array_equal(runtime_data, frozen_data))
    else:
        voxel_max = float("inf")
        voxel_mean = float("inf")
        voxel_exact = False

    if runtime_affine.shape == frozen_affine.shape:
        affine_diff = np.abs(runtime_affine - frozen_affine)
        affine_max = float(affine_diff.max(initial=0.0))
        affine_exact = bool(np.array_equal(runtime_affine, frozen_affine))
    else:
        affine_max = float("inf")
        affine_exact = False

    passed = bool(
        shape_ok
        and finite_ok
        and voxel_max <= voxel_atol
        and affine_max <= affine_atol
    )

    return {
        "shape_ok": shape_ok,
        "finite_ok": finite_ok,
        "voxel_exact": voxel_exact,
        "voxel_max_abs_diff": voxel_max,
        "voxel_mean_abs_diff": voxel_mean,
        "affine_exact": affine_exact,
        "affine_max_abs_diff": affine_max,
        "pass": passed,
    }


def build_cases_and_compare_preprocessing(
    *,
    project_root: Path,
    workspace_root: Path,
    selected: pd.DataFrame,
    manifest: pd.DataFrame,
    runtime,
    voxel_atol: float,
    affine_atol: float,
) -> tuple[list[dict], pd.DataFrame]:
    manifest_idx = manifest.set_index("uid")
    cases: list[dict] = []
    rows: list[dict] = []

    for item in selected.itertuples(index=False):
        uid = str(item.uid)
        if uid not in manifest_idx.index:
            raise KeyError(f"Selected UID missing from Step-10 manifest: {uid}")

        mrow = manifest_idx.loc[uid]
        frozen_whole = resolve_recorded_path(project_root, mrow["whole_path"])
        frozen_roi = resolve_recorded_path(project_root, mrow["roi_path"])
        runtime_whole, runtime_roi = locate_runtime_products(workspace_root, uid)

        whole_cmp = compare_nifti_pair(
            runtime_whole,
            frozen_whole,
            expected_shape=EXPECTED_WHOLE_SHAPE,
            voxel_atol=voxel_atol,
            affine_atol=affine_atol,
        )
        roi_cmp = compare_nifti_pair(
            runtime_roi,
            frozen_roi,
            expected_shape=EXPECTED_ROI_SHAPE,
            voxel_atol=voxel_atol,
            affine_atol=affine_atol,
        )

        runtime_case = {"uid": uid, "whole_path": runtime_whole, "roi_path": runtime_roi}
        frozen_case = {"uid": uid, "whole_path": frozen_whole, "roi_path": frozen_roi}

        # Extra check of the two special 2.5D paths.
        p2_runtime, p2_runtime_diag = runtime._prepare_case_input(runtime_case, "P2")
        p2_frozen, p2_frozen_diag = runtime._prepare_case_input(frozen_case, "P2")
        p2_diff = torch_max_abs(p2_runtime, p2_frozen)
        p2_indices_match = (
            p2_runtime_diag.get("selected_depth_indices")
            == p2_frozen_diag.get("selected_depth_indices")
        )

        p7_runtime, _ = runtime._prepare_case_input(runtime_case, "P7")
        p7_frozen, _ = runtime._prepare_case_input(frozen_case, "P7")
        p7_diff = torch_max_abs(p7_runtime, p7_frozen)

        input_2p5d_pass = bool(
            p2_indices_match and p2_diff <= voxel_atol and p7_diff <= voxel_atol
        )

        preprocess_pass = bool(whole_cmp["pass"] and roi_cmp["pass"] and input_2p5d_pass)

        rows.append(
            {
                "uid": uid,
                "fold": int(item.fold),
                "is_pathologic": int(item.is_pathologic),
                "runtime_whole_path": str(runtime_whole),
                "frozen_whole_path": str(frozen_whole),
                "whole_voxel_exact": whole_cmp["voxel_exact"],
                "whole_voxel_max_abs_diff": whole_cmp["voxel_max_abs_diff"],
                "whole_voxel_mean_abs_diff": whole_cmp["voxel_mean_abs_diff"],
                "whole_affine_exact": whole_cmp["affine_exact"],
                "whole_affine_max_abs_diff": whole_cmp["affine_max_abs_diff"],
                "runtime_roi_path": str(runtime_roi),
                "frozen_roi_path": str(frozen_roi),
                "roi_voxel_exact": roi_cmp["voxel_exact"],
                "roi_voxel_max_abs_diff": roi_cmp["voxel_max_abs_diff"],
                "roi_voxel_mean_abs_diff": roi_cmp["voxel_mean_abs_diff"],
                "roi_affine_exact": roi_cmp["affine_exact"],
                "roi_affine_max_abs_diff": roi_cmp["affine_max_abs_diff"],
                "p2_slice_indices_match": p2_indices_match,
                "p2_input_max_abs_diff": p2_diff,
                "p7_input_max_abs_diff": p7_diff,
                "preprocessing_pass": preprocess_pass,
            }
        )
        cases.append(runtime_case)

    return cases, pd.DataFrame(rows)


def torch_max_abs(a, b) -> float:
    # Avoid importing torch globally before model_bundle has configured imports.
    if tuple(a.shape) != tuple(b.shape):
        return float("inf")
    return float((a.float() - b.float()).abs().max().item())


def compare_heldout_fold_probabilities(
    *, runtime, cases: list[dict], selected: pd.DataFrame, oof_by_member: Mapping[str, pd.DataFrame], prob_atol: float
) -> pd.DataFrame:
    selected_idx = selected.set_index("uid")
    case_by_uid = {str(c["uid"]): c for c in cases}
    rows: list[dict] = []

    for member in FROZEN_MEMBERS:
        oof_idx = oof_by_member[member].set_index("uid")
        entries = runtime.manifest["members"][member]

        for fold in EXPECTED_FOLDS:
            fold_uids = selected.loc[selected["fold"] == fold, "uid"].astype(str).tolist()
            fold_cases = [case_by_uid[uid] for uid in fold_uids]
            fold_entry = entries[fold]
            if int(fold_entry["fold"]) != fold:
                raise RuntimeError(f"{member}: manifest fold order mismatch at fold {fold}")

            predicted, _ = runtime._predict_one_fold(
                member_id=member,
                fold_entry=fold_entry,
                cases=fold_cases,
            )

            for uid, observed in zip(fold_uids, predicted, strict=True):
                expected = float(oof_idx.loc[uid, "raw_probability"])
                recorded_fold = int(oof_idx.loc[uid, "fold"])
                if recorded_fold != fold:
                    raise RuntimeError(
                        f"{uid}/{member}: expected fold {recorded_fold}, selected fold {fold}"
                    )
                abs_diff = abs(float(observed) - expected)
                rows.append(
                    {
                        "uid": uid,
                        "fold": fold,
                        "is_pathologic": int(selected_idx.loc[uid, "is_pathologic"]),
                        "member": member,
                        "expected_raw_oof_probability": expected,
                        "runtime_heldout_fold_probability": float(observed),
                        "absolute_difference": abs_diff,
                        "probability_pass": bool(abs_diff <= prob_atol),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["fold", "uid", "member"], kind="stable"
    ).reset_index(drop=True)


def write_report(
    *, output_dir: Path, selected: pd.DataFrame, pre_df: pd.DataFrame, prob_df: pd.DataFrame,
    voxel_atol: float, affine_atol: float, prob_atol: float, workspace: Path,
) -> dict:
    preprocessing_pass = bool(pre_df["preprocessing_pass"].all())
    probability_pass = bool(prob_df["probability_pass"].all())
    overall_pass = bool(preprocessing_pass and probability_pass)

    summary = {
        "step": "25A",
        "name": "Quick held-out-fold runtime equivalence gate",
        "status": "PASS" if overall_pass else "REVIEW",
        "selected_subjects": int(len(selected)),
        "samples_per_fold": {
            str(k): int(v)
            for k, v in selected.groupby("fold").size().to_dict().items()
        },
        "class_counts": {
            str(k): int(v)
            for k, v in selected.groupby("is_pathologic").size().to_dict().items()
        },
        "members_checked": list(FROZEN_MEMBERS),
        "raw_probability_comparisons": int(len(prob_df)),
        "preprocessing_pass": preprocessing_pass,
        "heldout_fold_probability_pass": probability_pass,
        "thresholds": {
            "voxel_atol": voxel_atol,
            "affine_atol": affine_atol,
            "probability_atol": prob_atol,
        },
        "max_observed_differences": {
            "whole_voxel": float(pre_df["whole_voxel_max_abs_diff"].max()),
            "roi_voxel": float(pre_df["roi_voxel_max_abs_diff"].max()),
            "whole_affine": float(pre_df["whole_affine_max_abs_diff"].max()),
            "roi_affine": float(pre_df["roi_affine_max_abs_diff"].max()),
            "p2_input": float(pre_df["p2_input_max_abs_diff"].max()),
            "p7_input": float(pre_df["p7_input_max_abs_diff"].max()),
            "raw_probability": float(prob_df["absolute_difference"].max()),
        },
        "workspace": str(workspace),
        "interpretation": (
            "PASS means the sampled final runtime preprocessing and portable held-out-fold "
            "predictions reproduce the frozen training/OOF pipeline within configured "
            "numerical tolerances. Proceed directly to Step 25B."
            if overall_pass
            else
            "REVIEW means at least one sampled preprocessing or raw held-out-fold prediction "
            "does not reproduce the frozen training/OOF pipeline. Inspect the detailed CSVs "
            "before Step 25B training."
        ),
    }

    (output_dir / "step25a_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    worst_prob = prob_df.nlargest(min(10, len(prob_df)), "absolute_difference")
    report_lines = [
        "# Step 25A — Quick Held-Out-Fold Runtime Equivalence Gate",
        "",
        f"**STATUS: {summary['status']}**",
        "",
        f"Selected subjects: {len(selected)}",
        f"Raw member/fold probability comparisons: {len(prob_df)}",
        f"Preprocessing equivalence: {'PASS' if preprocessing_pass else 'REVIEW'}",
        f"Held-out-fold raw probability equivalence: {'PASS' if probability_pass else 'REVIEW'}",
        "",
        "## Maximum observed absolute differences",
        "",
        f"- Whole voxels: {summary['max_observed_differences']['whole_voxel']:.12g}",
        f"- ROI voxels: {summary['max_observed_differences']['roi_voxel']:.12g}",
        f"- Whole affine: {summary['max_observed_differences']['whole_affine']:.12g}",
        f"- ROI affine: {summary['max_observed_differences']['roi_affine']:.12g}",
        f"- P2 40-slice input tensor: {summary['max_observed_differences']['p2_input']:.12g}",
        f"- P7 ROI 2.5D input tensor: {summary['max_observed_differences']['p7_input']:.12g}",
        f"- Raw held-out-fold probability: {summary['max_observed_differences']['raw_probability']:.12g}",
        "",
        "## Gate thresholds",
        "",
        f"- Voxel atol: {voxel_atol}",
        f"- Affine atol: {affine_atol}",
        f"- Raw probability atol: {prob_atol}",
        "",
        "## Worst raw probability differences",
        "",
        "```text",
        worst_prob[[
            "uid", "fold", "member", "expected_raw_oof_probability",
            "runtime_heldout_fold_probability", "absolute_difference", "probability_pass"
        ]].to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
    ]
    (output_dir / "step25a_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    runtime_dir = resolve_under_root(project_root, args.runtime_dir)
    raw_dir = resolve_under_root(project_root, args.raw_dir)
    manifest_path = resolve_under_root(project_root, args.manifest)
    folds_path = resolve_under_root(project_root, args.folds)
    shortlist_path = resolve_under_root(project_root, args.shortlist)
    output_dir = resolve_under_root(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for required in (
        runtime_dir / "assets",
        runtime_dir / "pipeline_preprocessing",
        runtime_dir / "model_bundle",
        raw_dir,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    print("=" * 88)
    print("STEP 25A — QUICK HELD-OUT-FOLD RUNTIME EQUIVALENCE GATE")
    print("=" * 88)
    print(f"Project root : {project_root}")
    print(f"Runtime dir  : {runtime_dir}")
    print(f"Raw images   : {raw_dir}")
    print(f"Output       : {output_dir}")
    print(f"Subset       : {args.samples_per_fold} subjects/fold")
    print(f"Members      : {', '.join(FROZEN_MEMBERS)}")
    print("No training / no calibration / no ensemble selection")
    print()

    manifest, folds, shortlist, oof_by_member = load_authoritative_tables(
        project_root=project_root,
        manifest_path=manifest_path,
        folds_path=folds_path,
        shortlist_path=shortlist_path,
    )

    selected = choose_subset(
        oof_by_member["P1"],
        samples_per_fold=args.samples_per_fold,
        seed=args.seed,
    )
    selected.to_csv(output_dir / "selected_subjects.csv", index=False)

    print("Selected subset:")
    print(selected.groupby(["fold", "is_pathologic"]).size().unstack(fill_value=0))
    print()

    # Import the immutable portable runtime from the actual Step-23 bundle.
    if str(runtime_dir) not in sys.path:
        sys.path.insert(0, str(runtime_dir))
    from model_bundle.runtime_model import ENS328Runtime  # noqa: E402

    runtime = ENS328Runtime(
        bundle_root=runtime_dir / "model_bundle",
        device=args.device,
        amp=not args.no_amp,
        verify_hashes=True,
    )

    workspace, work_runtime = stage_preprocessing_workspace(
        runtime_dir=runtime_dir,
        raw_dir=raw_dir,
        selected_uids=selected["uid"].astype(str).tolist(),
        keep=args.keep_workspace,
    )

    success = False
    try:
        run_preprocessing(workspace, work_runtime, output_dir)

        cases, pre_df = build_cases_and_compare_preprocessing(
            project_root=project_root,
            workspace_root=workspace,
            selected=selected,
            manifest=manifest,
            runtime=runtime,
            voxel_atol=args.voxel_atol,
            affine_atol=args.affine_atol,
        )
        pre_df.to_csv(
            output_dir / "preprocessing_equivalence.csv",
            index=False,
            float_format="%.12g",
        )

        prob_df = compare_heldout_fold_probabilities(
            runtime=runtime,
            cases=cases,
            selected=selected,
            oof_by_member=oof_by_member,
            prob_atol=args.prob_atol,
        )
        prob_df.to_csv(
            output_dir / "heldout_fold_raw_probability_equivalence.csv",
            index=False,
            float_format="%.12g",
        )

        summary = write_report(
            output_dir=output_dir,
            selected=selected,
            pre_df=pre_df,
            prob_df=prob_df,
            voxel_atol=args.voxel_atol,
            affine_atol=args.affine_atol,
            prob_atol=args.prob_atol,
            workspace=workspace,
        )
        success = summary["status"] == "PASS"

        print()
        print("=" * 88)
        print("STEP 25A SUMMARY")
        print("=" * 88)
        print(f"Status                       : {summary['status']}")
        print(f"Subjects                     : {summary['selected_subjects']}")
        print(f"Preprocessing equivalence    : {summary['preprocessing_pass']}")
        print(f"Raw OOF probability replay   : {summary['heldout_fold_probability_pass']}")
        print(
            "Max raw probability diff     : "
            f"{summary['max_observed_differences']['raw_probability']:.12g}"
        )
        print(f"Saved                        : {output_dir}")
        print("=" * 88)
        print()

    finally:
        if not args.keep_workspace and workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)

    if not success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
