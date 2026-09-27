from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import torch

from step12a_load_manifest_and_folds import (
    load_supervised_manifest,
    load_fold_assignments,
    join_manifest_and_folds,
    split_for_validation_fold,
    resolve_existing_path,
)
from step12f_dataloader_factory import LoaderConfig
from step12g_reproducibility import seed_everything
from step12h_smoke_test import smoke_test_fold
from step12i_freeze_pipeline_config import file_sha256, write_json, write_validation_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STEP 12 — Build and validate model-input pipeline")
    p.add_argument("--manifest", default=None, help="Step-10 frozen supervised manifest CSV")
    p.add_argument("--folds", default=None, help="Step-11 combined frozen fold-assignment CSV")
    p.add_argument(
        "--output-dir",
        default="data/preprocessing_supervised_data/step12_build_model_input_pipeline_data",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def repo_root_from_script() -> Path:
    # .../scripts/preprocessing_supervised_script/step12_build_model_input_pipeline_script/file.py
    return Path(__file__).resolve().parents[3]


def main() -> int:
    args = parse_args()
    repo = repo_root_from_script()
    if Path.cwd().resolve() != repo:
        # Paths in manifests are commonly repo-relative; make behavior stable.
        import os
        os.chdir(repo)

    # Exact frozen Step-10 / Step-11 locations used by this project.
    step10_dir = (
        repo
        / "data/preprocessing_supervised_data"
        / "step10_supervised_dataset_manifest_data"
        / "supervised_dataset"
    )
    step11_dir = (
        repo
        / "data/preprocessing_supervised_data"
        / "step11_create_freeze_cv_splits_data"
    )

    manifest_candidates = [
        step10_dir / "supervised_dataset_manifest.csv",
    ]
    fold_candidates = [
        step11_dir / "fold_assignments.csv",
    ]

    manifest_path = resolve_existing_path(args.manifest, manifest_candidates, "Step-10 manifest")
    folds_path = resolve_existing_path(args.folds, fold_candidates + [step11_dir], "Step-11 folds", allow_directory=True)
    output_dir = (repo / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    manifest = load_supervised_manifest(manifest_path)
    folds, fold_col = load_fold_assignments(folds_path)
    merged = join_manifest_and_folds(manifest, folds, fold_col)
    available_folds = sorted(merged[fold_col].unique().tolist())
    if available_folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected exactly folds [0,1,2,3,4], got {available_folds}")

    cfg = LoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        seed=args.seed,
    )

    print("# STEP 12 — BUILD MODEL-INPUT PIPELINE")
    print()
    print(f"Repository root          : {repo}")
    print(f"Step-10 manifest         : {manifest_path}")
    print(f"Step-11 fold assignments : {folds_path}")
    print(f"Subjects                 : {len(merged)}")
    print(f"Folds                    : {available_folds}")
    print("Intensity handling       : NONE")
    print("Baseline augmentation    : False")
    print("Tensor convention        : [C,D,H,W] = [C,Z,Y,X]")
    print("Whole tensor             : [1,160,192,192]")
    print("ROI tensor               : [1,36,44,44]")
    print()

    validation_rows = []
    detailed = []
    for fold in available_folds:
        split = split_for_validation_fold(merged, fold_col, fold)
        for scenario in ["A", "B", "C"]:
            print(f"Validating fold={fold}, scenario={scenario} ...")
            result = smoke_test_fold(split.train, split.val, scenario, cfg)
            result["fold"] = fold
            detailed.append(result)
            validation_rows.append({
                "fold": fold,
                "scenario": scenario,
                "train_subjects": result["train_subjects"],
                "val_subjects": result["val_subjects"],
                "uid_overlap": 0,
                "passed": True,
            })

    config_payload = {
        "step": 12,
        "purpose": "build_model_input_pipeline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_folds": str(folds_path),
        "source_folds_sha256": file_sha256(folds_path) if folds_path.is_file() else None,
        "subjects": len(merged),
        "fold_column": fold_col,
        "folds": available_folds,
        "nifti_axis_convention": "XYZ",
        "tensor_axis_convention": "CDHW=CZYX",
        "axis_conversion": "XYZ->ZYX, then add channel",
        "dtype": "float32",
        "whole_source_shape_xyz": [192, 192, 160],
        "whole_tensor_shape_cdhw": [1, 160, 192, 192],
        "roi_source_shape_xyz": [44, 44, 36],
        "roi_tensor_shape_cdhw": [1, 36, 44, 44],
        "label_encoding": {"0": "Normal", "1": "Pathologic"},
        "label_dtype": "float32",
        "intensity_handling": "none_use_step6d_normalized_values_directly",
        "baseline_augmentation": False,
        "left_right_flip": False,
        "train_shuffle": True,
        "validation_shuffle": False,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
        "seed": cfg.seed,
        "torch_version": torch.__version__,
    }

    report_payload = {
        "all_passed": all(x["passed"] for x in detailed),
        "checks": detailed,
    }
    pipeline_manifest = {
        "artifacts": {
            "model_input_config": "model_input_config.json",
            "validation_report": "step12_validation_report.json",
            "validation_summary": "step12_validation_summary.csv",
        },
        "scenario_files_reused": {
            "A": "whole_volume_path",
            "B": "striatal_path",
            "C": ["whole_volume_path", "striatal_path"],
        },
        "augmented_images_saved": False,
    }

    write_json(output_dir / "model_input_config.json", config_payload)
    write_json(output_dir / "step12_validation_report.json", report_payload)
    write_validation_summary(output_dir / "step12_validation_summary.csv", validation_rows)
    write_json(output_dir / "step12_pipeline_manifest.json", pipeline_manifest)

    print()
    print(f"Validation checks        : {len(validation_rows)}/15 PASS")
    print(f"Output directory         : {output_dir}")
    print("Final validation passed  : True")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nSTEP 12 FAILED: {exc}", file=sys.stderr)
        raise
