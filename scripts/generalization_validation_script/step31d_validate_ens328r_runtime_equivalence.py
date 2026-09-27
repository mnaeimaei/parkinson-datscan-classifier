#!/usr/bin/env python3
"""
STEP 31D — ENS328R SOURCE-vs-PORTABLE RUNTIME EQUIVALENCE

Compare:
A) direct inference from the original 30 training checkpoints
B) inference from the Step31C tensor-only portable bundle

using the same frozen preprocessed Step-10 whole/ROI NIfTI inputs.

No preprocessing is rerun.
AMP is disabled for both paths.
The comparison therefore isolates checkpoint extraction/bundling/runtime logic.

Checks:
- same selected subjects and inputs
- raw five-fold mean probability for each of 6 members
- final ENS328R probability after member calibration, logit mean, temperature

Default sample: 10 subjects, 2 from each original Step-11 fold.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


MEMBERS = ("P1", "R_E6", "P2", "P3", "P7", "P8")
N_FOLDS = 5

BUILDERS = {
    "P1": ("src.models.model05_r3d18_scratch", "build_model"),
    "R_E6": ("src.models.model05_r3d18_scratch", "build_model"),
    "P2": ("src.models.model07_resnet18_2p5d_attention_scratch", "build_model"),
    "P3": ("src.models.model05_r3d18_scratch", "build_model"),
    "P7": ("src.models.model07_resnet18_2p5d_attention_scratch", "build_model"),
    "P8": ("src.models.model02_resnet18_3d_scratch", "build_model"),
}

DEFAULT_MANIFEST = Path(
    "data/preprocessing_supervised_data/"
    "step10_supervised_dataset_manifest_data/"
    "supervised_dataset/supervised_dataset_manifest.csv"
)
DEFAULT_FOLDS = Path(
    "data/preprocessing_supervised_data/"
    "step11_create_freeze_cv_splits_data/fold_assignments.csv"
)
DEFAULT_CHECKPOINT_MANIFEST = Path(
    "data/final_submission_preflight_data/ENS328R/"
    "final_30_checkpoint_manifest.csv"
)
DEFAULT_BUNDLE = Path(
    "data/final_runtime_bundle_data/ENS328R/model_bundle"
)
DEFAULT_CONFIG = Path(
    "data/final_predictor_data/ENS328R/final_predictor_config.json"
)
DEFAULT_OUTPUT = Path(
    "data/final_runtime_bundle_data/ENS328R/equivalence_validation"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate ENS328R portable equivalence.")
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--folds", type=Path, default=None)
    p.add_argument("--checkpoint-manifest", type=Path, default=None)
    p.add_argument("--bundle-root", type=Path, default=None)
    p.add_argument("--predictor-config", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--subjects-per-fold", type=int, default=2)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--member-atol", type=float, default=2e-6)
    p.add_argument("--final-atol", type=float, default=2e-6)
    return p.parse_args()


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def resolve_data_path(root: Path, value: Any) -> Path:
    p = Path(str(value)).expanduser()
    if p.is_file():
        return p.resolve()
    if not p.is_absolute():
        candidate = (root / p).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve data path: {value}")


def first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise RuntimeError(f"None of columns {names} found; available={list(df.columns)}")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model_state", "model"):
            value = checkpoint.get(key)
            if (
                isinstance(value, dict)
                and value
                and all(torch.is_tensor(v) for v in value.values())
            ):
                return value
        if checkpoint and all(
            isinstance(k, str) and torch.is_tensor(v)
            for k, v in checkpoint.items()
        ):
            return checkpoint
    raise RuntimeError("Could not extract source state_dict")


def build_source_model(member: str) -> torch.nn.Module:
    module_name, builder_name = BUILDERS[member]
    module = importlib.import_module(module_name)
    return getattr(module, builder_name)()


def choose_cases(
    *,
    root: Path,
    manifest_path: Path,
    folds_path: Path,
    subjects_per_fold: int,
) -> list[dict]:
    if subjects_per_fold < 1:
        raise ValueError("--subjects-per-fold must be >=1")

    manifest = pd.read_csv(manifest_path)
    folds = pd.read_csv(folds_path)

    uid_col = first_col(manifest, ("uid", "UID", "subject_uid", "subject_id"))
    whole_col = first_col(
        manifest,
        ("whole_path", "whole_volume_path", "whole_nifti_path"),
    )
    roi_col = first_col(
        manifest,
        ("roi_path", "striatal_path", "roi_nifti_path"),
    )

    fold_uid = first_col(folds, ("uid", "UID", "subject_uid", "subject_id"))
    fold_col = first_col(folds, ("fold", "Fold", "cv_fold"))

    manifest = manifest.copy()
    folds = folds.copy()
    manifest[uid_col] = manifest[uid_col].astype(str)
    folds[fold_uid] = folds[fold_uid].astype(str)
    folds[fold_col] = pd.to_numeric(folds[fold_col], errors="raise").astype(int)

    merged = manifest.merge(
        folds[[fold_uid, fold_col]],
        left_on=uid_col,
        right_on=fold_uid,
        validate="one_to_one",
        how="inner",
    )

    cases = []
    for fold in range(N_FOLDS):
        subset = merged.loc[merged[fold_col] == fold].copy()
        subset = subset.sort_values(uid_col, kind="stable").head(subjects_per_fold)
        if len(subset) != subjects_per_fold:
            raise RuntimeError(f"Fold {fold}: insufficient subjects")

        for _, row in subset.iterrows():
            cases.append(
                {
                    "uid": str(row[uid_col]),
                    "fold": fold,
                    "whole_path": str(resolve_data_path(root, row[whole_col])),
                    "roi_path": str(resolve_data_path(root, row[roi_col])),
                }
            )

    return cases


def direct_source_predictions(
    *,
    checkpoint_manifest: pd.DataFrame,
    cases: list[dict],
    runtime_module,
    final_predictor,
    device: torch.device,
):
    raw_member = {}
    fold_tables = {}

    for member in MEMBERS:
        rows = checkpoint_manifest.loc[
            checkpoint_manifest["Member ID"].astype(str) == member
        ].sort_values("Fold")
        if rows["Fold"].astype(int).tolist() != list(range(N_FOLDS)):
            raise RuntimeError(f"{member}: source manifest fold mismatch")

        matrix = np.empty((len(cases), N_FOLDS), dtype=np.float64)

        for _, row in rows.iterrows():
            fold = int(row["Fold"])
            checkpoint_path = Path(str(row["Checkpoint Path"])).expanduser().resolve()
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            state = extract_state_dict(checkpoint)

            model = build_source_model(member)
            incompatible = model.load_state_dict(state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"{member} fold {fold}: strict source load failed")

            model.eval()
            model.to(device)

            with torch.inference_mode():
                for i, case in enumerate(cases):
                    tensor, _ = runtime_module.ENS328RRuntime._prepare_case_input(
                        case,
                        member,
                    )
                    batch = tensor.unsqueeze(0).to(device)
                    logits = runtime_module._extract_logits(model(batch))
                    prob = torch.sigmoid(logits.float()).cpu().numpy().reshape(-1)
                    if prob.size != 1:
                        raise RuntimeError("Expected one probability")
                    matrix[i, fold] = float(prob[0])

            model.to("cpu")
            del model, state, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

        raw_member[member] = matrix.mean(axis=1)
        fold_tables[member] = matrix

    final = final_predictor.predict_with_intermediates(raw_member)
    final_probability = np.asarray(
        final["final_probability"], dtype=np.float64
    ).reshape(-1)

    return raw_member, fold_tables, final_probability


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else script_path.parents[2]
    )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.inference.final_ensemble import FinalEnsemblePredictor

    manifest_path = resolve(root, args.manifest or DEFAULT_MANIFEST)
    folds_path = resolve(root, args.folds or DEFAULT_FOLDS)
    checkpoint_manifest_path = resolve(
        root, args.checkpoint_manifest or DEFAULT_CHECKPOINT_MANIFEST
    )
    bundle_root = resolve(root, args.bundle_root or DEFAULT_BUNDLE)
    predictor_config = resolve(root, args.predictor_config or DEFAULT_CONFIG)
    output_dir = resolve(root, args.output_dir or DEFAULT_OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (
        manifest_path,
        folds_path,
        checkpoint_manifest_path,
        bundle_root / "runtime_model.py",
        bundle_root / "portable_checkpoint_manifest.json",
        predictor_config,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)

    cases = choose_cases(
        root=root,
        manifest_path=manifest_path,
        folds_path=folds_path,
        subjects_per_fold=args.subjects_per_fold,
    )

    checkpoint_manifest = pd.read_csv(checkpoint_manifest_path)
    checkpoint_manifest["Fold"] = pd.to_numeric(
        checkpoint_manifest["Fold"], errors="raise"
    ).astype(int)

    source_predictor = FinalEnsemblePredictor(config_path=predictor_config)

    # For source inference, import the runtime helper source directly from the
    # project-side Step31 runtime file. Its tensor/input geometry is identical
    # to the bundled copy.
    runtime_source_path = (
        root
        / "scripts/generalization_validation_script"
        / "step31_ens328r_runtime_model.py"
    )
    if not runtime_source_path.is_file():
        raise FileNotFoundError(runtime_source_path)

    import importlib.util
    source_spec = importlib.util.spec_from_file_location(
        "step31_source_runtime_helper",
        runtime_source_path,
    )
    source_runtime_module = importlib.util.module_from_spec(source_spec)
    assert source_spec.loader is not None
    source_spec.loader.exec_module(source_runtime_module)

    source_member, source_folds, source_final = direct_source_predictions(
        checkpoint_manifest=checkpoint_manifest,
        cases=cases,
        runtime_module=source_runtime_module,
        final_predictor=source_predictor,
        device=device,
    )

    # Remove project src modules from the module cache before importing the
    # portable runtime. Then put model_bundle itself first so `import src...`
    # resolves to model_bundle/src rather than the training repository.
    for module_name in list(sys.modules):
        if module_name == "src" or module_name.startswith("src."):
            del sys.modules[module_name]
        if module_name == "model_bundle" or module_name.startswith("model_bundle."):
            del sys.modules[module_name]

    bundle_parent = bundle_root.parent
    sys.path.insert(0, str(bundle_root))
    sys.path.insert(1, str(bundle_parent))

    import model_bundle.runtime_model as runtime_module

    portable_runtime = runtime_module.ENS328RRuntime(
        bundle_root=bundle_root,
        device=device_name,
        amp=False,
        verify_hashes=True,
    )
    portable_final, diagnostics = portable_runtime.predict_preprocessed_cases(
        cases,
        return_diagnostics=True,
    )

    rows = []
    member_max = {}
    fold_max = {}

    for member in MEMBERS:
        portable_member = np.asarray(
            diagnostics["raw_member_probabilities"][member],
            dtype=np.float64,
        )
        diff = np.abs(source_member[member] - portable_member)
        member_max[member] = float(diff.max())

        portable_fold = np.asarray(
            diagnostics["fold_probabilities"][member],
            dtype=np.float64,
        )
        fold_diff = np.abs(source_folds[member] - portable_fold)
        fold_max[member] = float(fold_diff.max())

        for i, case in enumerate(cases):
            rows.append(
                {
                    "uid": case["uid"],
                    "fold": case["fold"],
                    "member": member,
                    "source_member_mean_probability": source_member[member][i],
                    "portable_member_mean_probability": portable_member[i],
                    "absolute_difference": diff[i],
                }
            )

    final_diff = np.abs(
        source_final - np.asarray(portable_final, dtype=np.float64)
    )
    max_final = float(final_diff.max())

    member_pass = all(v <= args.member_atol for v in member_max.values())
    fold_pass = all(v <= args.member_atol for v in fold_max.values())
    final_pass = max_final <= args.final_atol
    passed = member_pass and fold_pass and final_pass

    pd.DataFrame(rows).to_csv(
        output_dir / "member_equivalence.csv",
        index=False,
        float_format="%.12f",
    )
    pd.DataFrame(
        {
            "uid": [c["uid"] for c in cases],
            "fold": [c["fold"] for c in cases],
            "source_final_probability": source_final,
            "portable_final_probability": portable_final,
            "absolute_difference": final_diff,
        }
    ).to_csv(
        output_dir / "final_equivalence.csv",
        index=False,
        float_format="%.12f",
    )

    summary = {
        "status": "PASS" if passed else "FAIL",
        "predictor": "ENS328R",
        "subjects": len(cases),
        "subjects_per_fold": args.subjects_per_fold,
        "device": str(device),
        "amp": False,
        "member_atol": args.member_atol,
        "final_atol": args.final_atol,
        "max_fold_probability_difference_by_member": fold_max,
        "max_member_mean_probability_difference_by_member": member_max,
        "max_final_probability_difference": max_final,
        "portable_hash_verification": True,
    }
    (output_dir / "step31d_equivalence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("STEP 31D — ENS328R SOURCE vs PORTABLE EQUIVALENCE")
    print("=" * 108)
    print(f"Subjects                 : {len(cases)}")
    print(f"Device                   : {device}")
    print("AMP                      : OFF")
    for member in MEMBERS:
        print(
            f"{member:<8s} fold max={fold_max[member]:.9g} "
            f"mean max={member_max[member]:.9g}"
        )
    print(f"Final max difference     : {max_final:.9g}")
    print(f"Member tolerance         : {args.member_atol}")
    print(f"Final tolerance          : {args.final_atol}")
    print(f"Status                   : {'PASS' if passed else 'FAIL'}")
    print("=" * 108)

    if not passed:
        raise RuntimeError("ENS328R portable runtime equivalence FAILED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
