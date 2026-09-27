#!/usr/bin/env python3
"""
Phase 13A — Final Submission / Runtime Preflight

Why this phase exists
---------------------
The final predictor is already frozen:

    ENS328 = P1 + P2 + P3 + P7 + P8
    equal-weight logit mean
    final ensemble temperature calibration

Before building the official competition submission runtime, verify that:

1. all 25 selected CV checkpoints exist and are unambiguous;
2. every checkpoint contains a usable model state dict;
3. the pretrained-trained checkpoints can be reconstructed OFFLINE from
   architecture-only constructors (no Kinetics/ImageNet download at inference);
4. P8 / E2's MONAI dependency is explicitly reported;
5. all frozen calibration artifacts exist;
6. the deterministic E4/E7 2.5D selector exists;
7. the validated preprocessing/runtime-gate source files and key assets exist;
8. a final checkpoint manifest with SHA256 hashes is frozen.

This phase does NOT run competition inference and does NOT modify model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


FROZEN_ENSEMBLE = "ENS328"
FROZEN_MEMBERS = ("P1", "P2", "P3", "P7", "P8")

# The exact trained experiments selected by Phase 12.
MEMBER_SPECS = {
    "P1": {
        "experiment": "E6",
        "model": "model06_r3d18_kinetics400_pretrained",
        "scenario_id": "S5",
        "input_type": "roi",
        "experiment_dir": (
            "data/experiments_data/"
            "model06_r3d18_kinetics400_pretrained_exp_data/"
            "scenario05_roi_aug_exp_data"
        ),
        # Offline inference does not need to initialize from Kinetics because
        # the full trained checkpoint will immediately overwrite all weights.
        "offline_builder_module": "src.models.model05_r3d18_scratch",
        "offline_builder_name": "build_model",
        "offline_builder_reason": (
            "E5 and E6 have the same frozen R3D-18 architecture; "
            "use E5 constructor to avoid Kinetics download, then strict-load E6 checkpoint."
        ),
    },
    "P2": {
        "experiment": "E7",
        "model": "model07_resnet18_2p5d_attention_scratch",
        "scenario_id": "S4",
        "input_type": "whole_2p5d",
        "experiment_dir": (
            "data/experiments_data/"
            "model07_resnet18_2p5d_attention_scratch_exp_data/"
            "scenario04_whole_aug_exp_data"
        ),
        "offline_builder_module": "src.models.model07_resnet18_2p5d_attention_scratch",
        "offline_builder_name": "build_model",
        "offline_builder_reason": "E7 is already scratch/offline-safe.",
    },
    "P3": {
        "experiment": "E6",
        "model": "model06_r3d18_kinetics400_pretrained",
        "scenario_id": "S2",
        "input_type": "roi",
        "experiment_dir": (
            "data/experiments_data/"
            "model06_r3d18_kinetics400_pretrained_exp_data/"
            "scenario02_roi_noaug_exp_data"
        ),
        "offline_builder_module": "src.models.model05_r3d18_scratch",
        "offline_builder_name": "build_model",
        "offline_builder_reason": (
            "E5 and E6 have the same frozen R3D-18 architecture; "
            "use E5 constructor to avoid Kinetics download, then strict-load E6 checkpoint."
        ),
    },
    "P7": {
        "experiment": "E4",
        "model": "model04_resnet18_2p5d_attention_imagenet_pretrained",
        "scenario_id": "S2",
        "input_type": "roi_2p5d",
        "experiment_dir": (
            "data/experiments_data/"
            "model04_resnet18_2p5d_attention_imagenet_pretrained_exp_data/"
            "scenario02_roi_noaug_exp_data"
        ),
        # E7 was intentionally built as the same architecture as E4, differing
        # only in initialization. Strict checkpoint loading proves portability.
        "offline_builder_module": "src.models.model07_resnet18_2p5d_attention_scratch",
        "offline_builder_name": "build_model",
        "offline_builder_reason": (
            "E4 and E7 share the same frozen 2.5D architecture; "
            "use E7 scratch constructor to avoid ImageNet download, "
            "then strict-load E4 checkpoint."
        ),
    },
    "P8": {
        "experiment": "E2",
        "model": "model02_resnet18_3d_scratch",
        "scenario_id": "S2",
        "input_type": "roi",
        "experiment_dir": (
            "data/experiments_data/"
            "model02_resnet18_3d_scratch_exp_data/"
            "scenario02_roi_noaug_exp_data"
        ),
        "offline_builder_module": "src.models.model02_resnet18_3d_scratch",
        "offline_builder_name": "build_model",
        "offline_builder_reason": (
            "E2 is scratch, but its current implementation may depend on MONAI."
        ),
    },
}

REQUIRED_PROJECT_FILES = (
    "src/inference/final_ensemble.py",
    "src/calibration/probability_calibration.py",
    "scripts/experiments_script/two_point_five_d_input.py",
    "data/final_predictor_data/ENS328/final_predictor_config.json",
    "data/final_predictor_data/ENS328/final_ensemble_temperature_calibrator.json",
    "data/final_predictor_data/ENS328/member_calibrators/P1.json",
    "data/final_predictor_data/ENS328/member_calibrators/P2.json",
    "data/final_predictor_data/ENS328/member_calibrators/P3.json",
    "data/final_predictor_data/ENS328/member_calibrators/P7.json",
    "data/final_predictor_data/ENS328/member_calibrators/P8.json",
)

# Source files from the already-developed deterministic preprocessing pipeline.
# We report these individually so the final runtime builder knows exactly what
# exists in the current training repo.
PREPROCESSING_SOURCE_CANDIDATES = (
    "scripts/preprocessing_image_script/step2_orientation_standardizer_script/step2_orientation_std.py",
    "scripts/preprocessing_image_script/step4_voxel_resampler_script/step4_resampling.py",
    "src/registration/registration_policy.py",
    "scripts/preprocessing_image_script/step7g_official_runtime_gate_script/step7i_registration_policy_replay.py",
    "scripts/preprocessing_image_script/step6a_initial_registration_script/step6a1_full_rigid_registration.py",
    "scripts/preprocessing_image_script/step6b_registration_rescue_script/step6b3_rescue_pnsm_similarity.py",
    "scripts/preprocessing_image_script/step6d_occipital_reference_script/step6d1_extract_occipital_reference.py",
    "scripts/preprocessing_image_script/step6e_intensity_normalization_script/step6e_normalize_occipital.py",
    "scripts/preprocessing_image_script/step8_bilateral_striatal_crop_script/step8c_striatal_crop_generator.py",
    "scripts/preprocessing_image_script/step9_classification_inputs_script/step9c_create_fixed_whole_volumes.py",
    "scripts/preprocessing_image_script/step9_classification_inputs_script/step9e_build_classification_scenarios.py",
)

RUNTIME_RELEVANT_DEPENDENCIES = (
    "numpy",
    "torch",
    "torchvision",
    "nibabel",
    "scipy",
    "SimpleITK",
    "monai",
    "pandas",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ENS328 checkpoints and offline runtime portability."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument(
        "--skip-strict-model-load",
        action="store_true",
        help=(
            "Only inventory checkpoint files/state dictionaries. "
            "Normally leave this OFF: strict model loading is the key portability test."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dependency_report() -> list[dict]:
    rows = []
    for package in RUNTIME_RELEVANT_DEPENDENCIES:
        available = importlib.util.find_spec(package) is not None
        version = None
        error = None

        if available:
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", None)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "package": package,
                "available": bool(available),
                "version": version,
                "import_error": error,
            }
        )
    return rows


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], str]:
    """
    Accept both the shared checkpoint format and older experiment formats.
    """
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model_state",
            "model",
        ):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value:
                if all(torch.is_tensor(v) for v in value.values()):
                    return value, key

        # A raw state dict is also valid.
        if checkpoint and all(
            isinstance(k, str) and torch.is_tensor(v)
            for k, v in checkpoint.items()
        ):
            return checkpoint, "<raw_state_dict>"

    raise RuntimeError(
        "Could not locate a model state dict in checkpoint. "
        f"Top-level type={type(checkpoint).__name__}"
    )


def resolve_fold_checkpoint(
    experiment_dir: Path,
    fold: int,
) -> tuple[Path, str]:
    """
    Prefer training_summary.json because it records the actual best checkpoint.
    Fall back to common best-checkpoint names only when needed.
    """
    fold_dir = experiment_dir / f"fold_{fold}"
    if not fold_dir.is_dir():
        raise FileNotFoundError(f"Missing fold directory: {fold_dir}")

    summary_path = fold_dir / "training_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        stored = summary.get("best_checkpoint")
        if stored:
            path = Path(str(stored)).expanduser()
            if not path.is_absolute():
                # First interpret relative to project cwd convention, then fold.
                project_candidate = (experiment_dir.parents[3] / path).resolve()
                fold_candidate = (fold_dir / path).resolve()
                if project_candidate.is_file():
                    path = project_candidate
                elif fold_candidate.is_file():
                    path = fold_candidate
            else:
                path = path.resolve()

            if path.is_file():
                return path, "training_summary.json"

    # Exact/familiar names.
    for name in (
        "best_model.pt",
        "best_checkpoint.pt",
        "best.pt",
        "model_best.pt",
    ):
        path = fold_dir / name
        if path.is_file():
            return path.resolve(), f"fallback:{name}"

    # Last resort: require exactly one *best*.pt/.pth candidate.
    candidates = sorted(
        p.resolve()
        for p in fold_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".pt", ".pth"}
        and "best" in p.name.lower()
    )

    if len(candidates) == 1:
        return candidates[0], "fallback:single_best_candidate"

    raise FileNotFoundError(
        f"Could not resolve one best checkpoint in {fold_dir}. "
        f"Best-like candidates: {[p.name for p in candidates]}"
    )


def build_offline_model(spec: dict) -> torch.nn.Module:
    module = importlib.import_module(spec["offline_builder_module"])
    builder = getattr(module, spec["offline_builder_name"])
    model = builder()
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"Offline builder returned {type(model).__name__}, not nn.Module."
        )
    return model


def parameter_count(state_dict: dict[str, torch.Tensor]) -> int:
    return int(sum(int(v.numel()) for v in state_dict.values()))


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = (
        args.project_root or script_path.parents[2]
    ).expanduser().resolve()

    output_dir = (
        args.output_dir
        or project_root
        / "data"
        / "final_submission_preflight_data"
        / "ENS328"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("\n" + "=" * 108)
    print("PHASE 13A — FINAL SUBMISSION / RUNTIME PREFLIGHT")
    print("=" * 108)
    print(f"Project root        : {project_root}")
    print(f"Output              : {output_dir}")
    print(f"Frozen ensemble     : {FROZEN_ENSEMBLE}")
    print(f"Members             : {'+'.join(FROZEN_MEMBERS)}")
    print(f"Expected checkpoints: {len(FROZEN_MEMBERS) * args.expected_folds}")
    print(f"Strict model load   : {not args.skip_strict_model_load}")
    print(f"Python              : {sys.version.split()[0]}")
    print(f"Platform            : {platform.platform()}")
    print("GPU required        : NO")
    print()

    failures: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # Frozen predictor contract.
    # ------------------------------------------------------------------
    config_path = (
        project_root
        / "data"
        / "final_predictor_data"
        / "ENS328"
        / "final_predictor_config.json"
    )

    if not config_path.is_file():
        failures.append(f"Missing final predictor config: {config_path}")
        config = {}
    else:
        config = read_json(config_path)
        observed_members = tuple(
            config.get("ensemble", {}).get("members", [])
        )
        observed_id = config.get("ensemble", {}).get("id")
        observed_rule = config.get("ensemble", {}).get(
            "aggregation_rule"
        )

        if observed_id != FROZEN_ENSEMBLE:
            failures.append(
                f"Frozen ensemble ID mismatch: {observed_id} != {FROZEN_ENSEMBLE}"
            )
        if observed_members != FROZEN_MEMBERS:
            failures.append(
                f"Frozen members mismatch: {observed_members} != {FROZEN_MEMBERS}"
            )
        if observed_rule != "logit_mean":
            failures.append(
                f"Frozen aggregation mismatch: {observed_rule} != logit_mean"
            )

    # ------------------------------------------------------------------
    # Required files.
    # ------------------------------------------------------------------
    project_file_rows = []
    for relative in REQUIRED_PROJECT_FILES:
        path = project_root / relative
        exists = path.is_file()
        project_file_rows.append(
            {
                "relative_path": relative,
                "exists": exists,
                "absolute_path": str(path.resolve()),
            }
        )
        if not exists:
            failures.append(f"Missing required project file: {relative}")

    pd.DataFrame(project_file_rows).to_csv(
        output_dir / "required_project_files.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Preprocessing source inventory.
    # ------------------------------------------------------------------
    preprocessing_rows = []
    for relative in PREPROCESSING_SOURCE_CANDIDATES:
        path = project_root / relative
        preprocessing_rows.append(
            {
                "relative_path": relative,
                "exists": path.is_file(),
                "absolute_path": str(path.resolve()),
            }
        )

    preprocessing_df = pd.DataFrame(preprocessing_rows)
    preprocessing_df.to_csv(
        output_dir / "preprocessing_source_inventory.csv",
        index=False,
    )

    missing_preprocess = preprocessing_df.loc[
        ~preprocessing_df["exists"], "relative_path"
    ].tolist()
    if missing_preprocess:
        warnings.append(
            "Some preprocessing-source candidates were not found. "
            "This is not automatically fatal because the final runtime may "
            "reuse the already-validated runtime implementation instead of "
            "these analysis scripts. Missing: "
            + ", ".join(missing_preprocess)
        )

    # ------------------------------------------------------------------
    # Dependency inventory.
    # ------------------------------------------------------------------
    dep_rows = dependency_report()
    dep_df = pd.DataFrame(dep_rows)
    dep_df.to_csv(
        output_dir / "idun_dependency_inventory.csv",
        index=False,
    )

    monai_row = next(
        row for row in dep_rows if row["package"] == "monai"
    )
    if not monai_row["available"]:
        failures.append(
            "MONAI is unavailable in this environment, so P8/E2 cannot be "
            "constructed from the current src implementation."
        )

    # ------------------------------------------------------------------
    # Checkpoint inventory and strict offline loading.
    # ------------------------------------------------------------------
    checkpoint_rows = []
    architecture_rows = []

    for member_id in FROZEN_MEMBERS:
        spec = MEMBER_SPECS[member_id]
        experiment_dir = project_root / spec["experiment_dir"]

        print()
        print("-" * 108)
        print(
            f"{member_id} — {spec['experiment']} {spec['scenario_id']} "
            f"{spec['model']}"
        )
        print(f"Input              : {spec['input_type']}")
        print(f"Experiment dir     : {experiment_dir}")
        print(
            f"Offline constructor: {spec['offline_builder_module']}."
            f"{spec['offline_builder_name']}"
        )
        print("-" * 108)

        if not experiment_dir.is_dir():
            failures.append(
                f"{member_id}: experiment directory missing: {experiment_dir}"
            )
            continue

        # Build only once per member. Every fold must load into the exact
        # same architecture.
        model = None
        model_build_error = None

        if not args.skip_strict_model_load:
            try:
                model = build_offline_model(spec)
                model.eval()
            except Exception as exc:
                model_build_error = f"{type(exc).__name__}: {exc}"
                failures.append(
                    f"{member_id}: offline architecture construction failed: "
                    f"{model_build_error}"
                )

        member_key_signature = None
        member_shape_signature = None

        for fold in range(args.expected_folds):
            try:
                checkpoint_path, resolution = resolve_fold_checkpoint(
                    experiment_dir,
                    fold,
                )

                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                state_dict, state_key = extract_state_dict(checkpoint)

                keys = tuple(state_dict.keys())
                shapes = tuple(
                    (key, tuple(tensor.shape))
                    for key, tensor in state_dict.items()
                )

                if member_key_signature is None:
                    member_key_signature = keys
                    member_shape_signature = shapes
                else:
                    if keys != member_key_signature:
                        raise RuntimeError(
                            "state-dict key sequence differs from fold 0"
                        )
                    if shapes != member_shape_signature:
                        raise RuntimeError(
                            "state-dict tensor shapes differ from fold 0"
                        )

                strict_ok = None
                strict_error = None

                if model is not None:
                    try:
                        incompatible = model.load_state_dict(
                            state_dict,
                            strict=True,
                        )
                        strict_ok = (
                            len(incompatible.missing_keys) == 0
                            and len(incompatible.unexpected_keys) == 0
                        )
                    except Exception as exc:
                        strict_ok = False
                        strict_error = f"{type(exc).__name__}: {exc}"
                        failures.append(
                            f"{member_id} fold {fold}: strict offline "
                            f"checkpoint load failed: {strict_error}"
                        )

                row = {
                    "Member ID": member_id,
                    "Experiment": spec["experiment"],
                    "Scenario ID": spec["scenario_id"],
                    "Model": spec["model"],
                    "Input Type": spec["input_type"],
                    "Fold": fold,
                    "Checkpoint Path": str(checkpoint_path),
                    "Checkpoint Resolution": resolution,
                    "Checkpoint Bytes": checkpoint_path.stat().st_size,
                    "Checkpoint SHA256": sha256_file(checkpoint_path),
                    "State Dict Container Key": state_key,
                    "State Dict Tensors": len(state_dict),
                    "State Dict Parameters": parameter_count(state_dict),
                    "Strict Offline Load": strict_ok,
                    "Strict Load Error": strict_error,
                    "Offline Builder Module":
                        spec["offline_builder_module"],
                    "Offline Builder Name":
                        spec["offline_builder_name"],
                    "Offline Builder Reason":
                        spec["offline_builder_reason"],
                }
                checkpoint_rows.append(row)

                print(
                    f"fold {fold}: {checkpoint_path.name} "
                    f"| tensors={len(state_dict)} "
                    f"| strict={strict_ok if strict_ok is not None else 'SKIP'}"
                )

            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                failures.append(
                    f"{member_id} fold {fold}: checkpoint preflight failed: {error}"
                )
                checkpoint_rows.append(
                    {
                        "Member ID": member_id,
                        "Experiment": spec["experiment"],
                        "Scenario ID": spec["scenario_id"],
                        "Model": spec["model"],
                        "Input Type": spec["input_type"],
                        "Fold": fold,
                        "Checkpoint Path": None,
                        "Checkpoint Resolution": None,
                        "Checkpoint Bytes": None,
                        "Checkpoint SHA256": None,
                        "State Dict Container Key": None,
                        "State Dict Tensors": None,
                        "State Dict Parameters": None,
                        "Strict Offline Load": False,
                        "Strict Load Error": error,
                        "Offline Builder Module":
                            spec["offline_builder_module"],
                        "Offline Builder Name":
                            spec["offline_builder_name"],
                        "Offline Builder Reason":
                            spec["offline_builder_reason"],
                    }
                )

        architecture_rows.append(
            {
                "Member ID": member_id,
                "Experiment": spec["experiment"],
                "Scenario ID": spec["scenario_id"],
                "Model": spec["model"],
                "Input Type": spec["input_type"],
                "Experiment Directory": str(experiment_dir.resolve()),
                "Offline Builder Module": spec["offline_builder_module"],
                "Offline Builder Name": spec["offline_builder_name"],
                "Offline Builder Reason": spec["offline_builder_reason"],
                "Model Build OK":
                    bool(model is not None)
                    if not args.skip_strict_model_load
                    else None,
                "Model Build Error": model_build_error,
            }
        )

        del model

    checkpoint_df = pd.DataFrame(checkpoint_rows)
    architecture_df = pd.DataFrame(architecture_rows)

    checkpoint_manifest_path = (
        output_dir / "final_25_checkpoint_manifest.csv"
    )
    checkpoint_df.to_csv(
        checkpoint_manifest_path,
        index=False,
    )
    architecture_df.to_csv(
        output_dir / "offline_architecture_plan.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Final-ensemble object load.
    # ------------------------------------------------------------------
    final_ensemble_load_ok = False
    final_ensemble_error = None
    if config_path.is_file():
        try:
            from src.inference.final_ensemble import FinalEnsemblePredictor
            predictor = FinalEnsemblePredictor(config_path=config_path)
            final_ensemble_load_ok = True
            del predictor
        except Exception as exc:
            final_ensemble_error = f"{type(exc).__name__}: {exc}"
            failures.append(
                "FinalEnsemblePredictor failed to load frozen config: "
                + final_ensemble_error
            )

    # ------------------------------------------------------------------
    # Conclusions.
    # ------------------------------------------------------------------
    expected_checkpoints = len(FROZEN_MEMBERS) * args.expected_folds
    resolved_checkpoints = int(
        checkpoint_df["Checkpoint Path"].notna().sum()
    ) if not checkpoint_df.empty else 0

    strict_values = (
        checkpoint_df["Strict Offline Load"].dropna().tolist()
        if (
            not checkpoint_df.empty
            and "Strict Offline Load" in checkpoint_df
        )
        else []
    )
    strict_pass = (
        all(bool(v) for v in strict_values)
        and len(strict_values) == expected_checkpoints
    ) if not args.skip_strict_model_load else None

    status = "PASS" if not failures else "FAIL"

    # Runtime packaging guidance is intentionally explicit.
    portability = {
        "frozen_ensemble": FROZEN_ENSEMBLE,
        "members": list(FROZEN_MEMBERS),
        "checkpoint_count_expected": expected_checkpoints,
        "checkpoint_count_resolved": resolved_checkpoints,
        "strict_offline_checkpoint_load_pass": strict_pass,
        "pretrained_runtime_policy": {
            "P1": (
                "Instantiate E5 scratch R3D-18 architecture offline, "
                "strict-load trained E6 checkpoint. Do NOT download Kinetics weights."
            ),
            "P3": (
                "Instantiate E5 scratch R3D-18 architecture offline, "
                "strict-load trained E6 checkpoint. Do NOT download Kinetics weights."
            ),
            "P7": (
                "Instantiate E7 scratch 2.5D architecture offline, "
                "strict-load trained E4 checkpoint. Do NOT download ImageNet weights."
            ),
        },
        "P8_MONAI": {
            "required_by_current_source": True,
            "idun_available": bool(monai_row["available"]),
            "idun_version": monai_row["version"],
            "competition_runtime_action": (
                "Verify MONAI inside the official submission container. "
                "If absent, vendor/replace E2 with an exactly state-dict-compatible "
                "pure-PyTorch implementation before final packaging."
            ),
        },
        "preprocessing": {
            "policy": (
                "Reuse the already-validated deterministic runtime preprocessing; "
                "do not rerun analysis/QC selection logic and do not apply training augmentation."
            ),
            "required_final_outputs": {
                "whole_nifti_geometry": "[X,Y,Z] corresponding to tensor [1,160,192,192]",
                "roi_nifti_geometry": "[X,Y,Z] corresponding to tensor [1,36,44,44]",
                "P2_whole_2p5d": (
                    "apply the frozen E4/E7 striatum-localized selector to obtain "
                    "[1,40,192,192]"
                ),
                "P7_roi_2p5d": "all 36 ROI slices; [1,36,44,44]",
            },
        },
    }

    portability_path = (
        output_dir / "runtime_portability_plan.json"
    )
    with portability_path.open("w", encoding="utf-8") as file:
        json.dump(portability, file, indent=2)

    summary = {
        "status": status,
        "project_root": str(project_root),
        "frozen_ensemble": FROZEN_ENSEMBLE,
        "members": list(FROZEN_MEMBERS),
        "expected_checkpoints": expected_checkpoints,
        "resolved_checkpoints": resolved_checkpoints,
        "strict_model_load_requested":
            not args.skip_strict_model_load,
        "strict_offline_checkpoint_load_pass": strict_pass,
        "final_ensemble_config_load_ok": final_ensemble_load_ok,
        "final_ensemble_config_load_error": final_ensemble_error,
        "dependency_inventory": dep_rows,
        "failures": failures,
        "warnings": warnings,
        "outputs": {
            "checkpoint_manifest": str(checkpoint_manifest_path),
            "offline_architecture_plan": str(
                output_dir / "offline_architecture_plan.csv"
            ),
            "dependency_inventory": str(
                output_dir / "idun_dependency_inventory.csv"
            ),
            "required_project_files": str(
                output_dir / "required_project_files.csv"
            ),
            "preprocessing_source_inventory": str(
                output_dir / "preprocessing_source_inventory.csv"
            ),
            "runtime_portability_plan": str(portability_path),
        },
    }

    summary_path = output_dir / "phase13a_preflight_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    report_lines = [
        "# Phase 13A — Final Submission Runtime Preflight",
        "",
        f"Status: **{status}**",
        "",
        f"- Frozen predictor: `{FROZEN_ENSEMBLE}`",
        f"- Members: `{'+'.join(FROZEN_MEMBERS)}`",
        f"- Checkpoints expected: `{expected_checkpoints}`",
        f"- Checkpoints resolved: `{resolved_checkpoints}`",
        f"- Strict offline checkpoint loading: `{strict_pass}`",
        f"- Final ensemble config load: `{final_ensemble_load_ok}`",
        f"- MONAI available on Idun: `{monai_row['available']}`",
        "",
        "## Failures",
        "",
    ]

    if failures:
        report_lines.extend(f"- {item}" for item in failures)
    else:
        report_lines.append("- None")

    report_lines.extend(["", "## Warnings", ""])
    if warnings:
        report_lines.extend(f"- {item}" for item in warnings)
    else:
        report_lines.append("- None")

    report_lines.extend(
        [
            "",
            "## Runtime policy",
            "",
            "- P1/P3: construct E5 scratch architecture, strict-load E6 weights.",
            "- P2: construct E7 scratch architecture, strict-load E7 weights.",
            "- P7: construct E7 scratch architecture, strict-load E4 weights.",
            "- P8: construct E2; MONAI runtime availability must be confirmed.",
            "- Never download pretrained weights in the competition runtime.",
            "- Reuse deterministic frozen preprocessing; no training augmentation.",
            "",
        ]
    )

    report_path = output_dir / "phase13a_preflight_report.md"
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print("PHASE 13A SUMMARY")
    print("=" * 108)
    print(f"Status                         : {status}")
    print(
        f"Checkpoints resolved           : "
        f"{resolved_checkpoints}/{expected_checkpoints}"
    )
    print(
        f"Strict offline checkpoint load : {strict_pass}"
    )
    print(
        f"Final ensemble config load     : {final_ensemble_load_ok}"
    )
    print(
        f"MONAI on Idun                  : "
        f"{monai_row['available']} ({monai_row['version']})"
    )
    print()

    if failures:
        print("FAILURES:")
        for item in failures:
            print(f"  - {item}")
        print()
    if warnings:
        print("WARNINGS:")
        for item in warnings:
            print(f"  - {item}")
        print()

    print(f"Saved: {checkpoint_manifest_path}")
    print(
        f"Saved: {output_dir / 'offline_architecture_plan.csv'}"
    )
    print(
        f"Saved: {output_dir / 'idun_dependency_inventory.csv'}"
    )
    print(
        f"Saved: {output_dir / 'preprocessing_source_inventory.csv'}"
    )
    print(f"Saved: {portability_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")
    print()
    print(f"STATUS: {status}")

    if status == "PASS":
        print(
            "Next step: build the official runtime submission bundle using "
            "this frozen 25-checkpoint manifest and the validated preprocessing."
        )
    else:
        print(
            "Resolve the reported runtime/checkpoint blocker(s) before "
            "building the final competition ZIP."
        )

    # Return non-zero for Slurm if there is a real blocker.
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
