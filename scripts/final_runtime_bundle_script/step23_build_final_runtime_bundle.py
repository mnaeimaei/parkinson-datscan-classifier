#!/usr/bin/env python3
"""
Phase 13B — Build Portable ENS328 Model Runtime Bundle

Consumes the Phase-13A manifest and creates a portable bundle that contains:

- exactly 25 selected MODEL state_dicts (training state removed)
- only the three runtime architecture sources actually needed
- calibration/final-ensemble sources
- frozen Phase-12 calibrators and config
- runtime inference module for already-preprocessed whole/ROI NIfTIs
- portable relative checkpoint manifest
- SHA256 provenance
- tar.gz transport artifact

No model parameters are changed.
No float conversion or quantization is performed.
Every extracted state_dict is strict-reloaded before being accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch


FROZEN_MEMBERS = ("P1", "P2", "P3", "P7", "P8")
EXPECTED_FOLDS = 5
EXPECTED_CHECKPOINTS = 25

BUILDER = {
    "P1": ("src.models.model05_r3d18_scratch", "build_model"),
    "P2": ("src.models.model07_resnet18_2p5d_attention_scratch", "build_model"),
    "P3": ("src.models.model05_r3d18_scratch", "build_model"),
    "P7": ("src.models.model07_resnet18_2p5d_attention_scratch", "build_model"),
    "P8": ("src.models.model02_resnet18_3d_scratch", "build_model"),
}

SOURCE_FILES = (
    "src/__init__.py",
    "src/models/__init__.py",
    "src/models/model02_resnet18_3d_scratch.py",
    "src/models/model05_r3d18_scratch.py",
    "src/models/model07_resnet18_2p5d_attention_scratch.py",
    "src/calibration/__init__.py",
    "src/calibration/probability_calibration.py",
    "src/inference/__init__.py",
    "src/inference/final_ensemble.py",
)

FINAL_PREDICTOR_FILES = (
    "final_predictor_config.json",
    "final_ensemble_temperature_calibrator.json",
    "member_calibrators/P1.json",
    "member_calibrators/P2.json",
    "member_calibrators/P3.json",
    "member_calibrators/P7.json",
    "member_calibrators/P8.json",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build portable ENS328 runtime weights/source bundle."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--checkpoint-manifest", type=Path, default=None)
    parser.add_argument("--runtime-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
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


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model_state",
            "model",
        ):
            value = checkpoint.get(key)
            if (
                isinstance(value, dict)
                and value
                and all(torch.is_tensor(v) for v in value.values())
            ):
                return value, key

        if (
            checkpoint
            and all(
                isinstance(k, str) and torch.is_tensor(v)
                for k, v in checkpoint.items()
            )
        ):
            return checkpoint, "<raw_state_dict>"

    raise RuntimeError(
        f"Could not extract model state_dict from {type(checkpoint).__name__}"
    )


def build_model(member_id: str):
    module_name, builder_name = BUILDER[member_id]
    module = importlib.import_module(module_name)
    builder = getattr(module, builder_name)
    model = builder()
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"{member_id}: builder returned {type(model).__name__}"
        )
    return model


def main():
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = (
        args.project_root or script_path.parents[2]
    ).expanduser().resolve()

    manifest_path = (
        args.checkpoint_manifest
        or project_root
        / "data"
        / "final_submission_preflight_data"
        / "ENS328"
        / "final_25_checkpoint_manifest.csv"
    ).expanduser().resolve()

    runtime_template = (
        args.runtime_template.expanduser().resolve()
    )

    output_dir = (
        args.output_dir
        or project_root
        / "data"
        / "final_runtime_bundle_data"
        / "ENS328"
    ).expanduser().resolve()

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("\n" + "=" * 108)
    print("PHASE 13B — BUILD PORTABLE ENS328 MODEL RUNTIME BUNDLE")
    print("=" * 108)
    print(f"Project root          : {project_root}")
    print(f"13A manifest          : {manifest_path}")
    print(f"Runtime template      : {runtime_template}")
    print(f"Output                : {output_dir}")
    print(f"Frozen members        : {'+'.join(FROZEN_MEMBERS)}")
    print(f"Expected weights      : {EXPECTED_CHECKPOINTS}")
    print("Precision             : unchanged")
    print("Quantization          : NONE")
    print("Training state        : REMOVED")
    print("GPU required          : NO")
    print()

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not runtime_template.is_dir():
        raise FileNotFoundError(runtime_template)

    manifest = pd.read_csv(manifest_path)

    required = {
        "Member ID",
        "Fold",
        "Checkpoint Path",
        "Checkpoint SHA256",
        "Strict Offline Load",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(
            f"Checkpoint manifest missing columns: {sorted(missing)}"
        )

    if len(manifest) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(
            f"Expected {EXPECTED_CHECKPOINTS} manifest rows, got {len(manifest)}"
        )

    manifest["Member ID"] = manifest["Member ID"].astype(str)
    manifest["Fold"] = pd.to_numeric(
        manifest["Fold"], errors="raise"
    ).astype(int)

    if set(manifest["Member ID"]) != set(FROZEN_MEMBERS):
        raise RuntimeError("Checkpoint member set does not match ENS328")

    for member_id in FROZEN_MEMBERS:
        folds = sorted(
            manifest.loc[
                manifest["Member ID"] == member_id,
                "Fold",
            ].tolist()
        )
        if folds != list(range(EXPECTED_FOLDS)):
            raise RuntimeError(
                f"{member_id}: folds {folds} != 0..4"
            )

    strict_values = manifest["Strict Offline Load"].astype(str).str.lower()
    if not strict_values.isin({"true", "1"}).all():
        raise RuntimeError(
            "Phase13A manifest contains a checkpoint that did not strict-load"
        )

    # Clean rebuild.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    bundle_dir = output_dir / "model_bundle"
    shutil.copytree(runtime_template, bundle_dir)

    # Copy minimal source package.
    for relative in SOURCE_FILES:
        source = project_root / relative
        target = bundle_dir / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    # Copy frozen predictor config/calibrators preserving relative layout.
    frozen_source_root = (
        project_root / "data" / "final_predictor_data" / "ENS328"
    )
    final_target_root = bundle_dir / "final_predictor"

    for relative in FINAL_PREDICTOR_FILES:
        source = frozen_source_root / relative
        target = final_target_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    portable_members = {}
    provenance_rows = []

    original_total = 0
    portable_total = 0

    for member_id in FROZEN_MEMBERS:
        print()
        print("-" * 108)
        print(f"{member_id}")
        print("-" * 108)

        member_rows = manifest.loc[
            manifest["Member ID"] == member_id
        ].sort_values("Fold")

        entries = []

        for _, row in member_rows.iterrows():
            fold = int(row["Fold"])
            source_path = Path(
                str(row["Checkpoint Path"])
            ).expanduser().resolve()

            if not source_path.is_file():
                raise FileNotFoundError(source_path)

            expected_source_sha = str(row["Checkpoint SHA256"])
            observed_source_sha = sha256_file(source_path)
            if observed_source_sha != expected_source_sha:
                raise RuntimeError(
                    f"{member_id} fold {fold}: original checkpoint SHA changed"
                )

            original_bytes = source_path.stat().st_size
            original_total += original_bytes

            checkpoint = torch.load(
                source_path,
                map_location="cpu",
                weights_only=False,
            )
            state_dict, container_key = extract_state_dict(checkpoint)

            # Exact tensors only. Do not cast/quantize/modify.
            portable_state = {
                str(key): tensor.detach().cpu().contiguous()
                for key, tensor in state_dict.items()
            }

            relative_path = (
                Path("weights")
                / member_id
                / f"fold_{fold}.pt"
            )
            target_path = bundle_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(portable_state, target_path)

            # Verify portable load with weights_only and exact strict architecture.
            reloaded = torch.load(
                target_path,
                map_location="cpu",
                weights_only=True,
            )

            model = build_model(member_id)
            incompatible = model.load_state_dict(
                reloaded,
                strict=True,
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"{member_id} fold {fold}: portable strict-load not exact"
                )

            # Exact tensor equality after round-trip.
            if tuple(reloaded.keys()) != tuple(portable_state.keys()):
                raise RuntimeError(
                    f"{member_id} fold {fold}: key order changed"
                )
            for key in portable_state:
                if not torch.equal(
                    portable_state[key],
                    reloaded[key],
                ):
                    raise RuntimeError(
                        f"{member_id} fold {fold}: tensor changed: {key}"
                    )

            portable_sha = sha256_file(target_path)
            portable_bytes = target_path.stat().st_size
            portable_total += portable_bytes

            entry = {
                "fold": fold,
                "relative_path": str(relative_path),
                "sha256": portable_sha,
                "bytes": portable_bytes,
                "tensor_count": len(portable_state),
                "parameter_count": int(
                    sum(v.numel() for v in portable_state.values())
                ),
            }
            entries.append(entry)

            provenance_rows.append(
                {
                    "Member ID": member_id,
                    "Fold": fold,
                    "Original Checkpoint Path": str(source_path),
                    "Original Checkpoint SHA256": observed_source_sha,
                    "Original Bytes": original_bytes,
                    "Original State Dict Container": container_key,
                    "Portable Relative Path": str(relative_path),
                    "Portable SHA256": portable_sha,
                    "Portable Bytes": portable_bytes,
                    "Tensor Count": len(portable_state),
                    "Parameter Count": entry["parameter_count"],
                    "Strict Reload": True,
                    "Exact Tensor Roundtrip": True,
                }
            )

            print(
                f"fold {fold}: "
                f"{original_bytes / 1024**2:.1f} MiB -> "
                f"{portable_bytes / 1024**2:.1f} MiB | strict=True"
            )

            del checkpoint, state_dict, portable_state, reloaded, model

        portable_members[member_id] = entries

    portable_manifest = {
        "schema_version": 1,
        "frozen_predictor": "ENS328",
        "member_order": list(FROZEN_MEMBERS),
        "folds_per_member": EXPECTED_FOLDS,
        "weights_policy": (
            "model state_dict only; exact tensor values; no quantization"
        ),
        "members": portable_members,
    }

    manifest_json_path = bundle_dir / "portable_checkpoint_manifest.json"
    with manifest_json_path.open("w", encoding="utf-8") as file:
        json.dump(portable_manifest, file, indent=2)

    provenance = pd.DataFrame(provenance_rows)
    provenance_path = output_dir / "portable_checkpoint_provenance.csv"
    provenance.to_csv(provenance_path, index=False)

    # Bundle SHA inventory.
    hashes = {}
    for path in sorted(
        p for p in bundle_dir.rglob("*") if p.is_file()
    ):
        hashes[str(path.relative_to(bundle_dir))] = sha256_file(path)

    hashes_path = bundle_dir / "bundle_sha256.json"
    with hashes_path.open("w", encoding="utf-8") as file:
        json.dump(hashes, file, indent=2, sort_keys=True)

    # Readable bundle summary.
    reduction = (
        100.0 * (1.0 - portable_total / original_total)
        if original_total
        else 0.0
    )

    summary = {
        "status": "PASS",
        "frozen_predictor": "ENS328",
        "members": list(FROZEN_MEMBERS),
        "portable_checkpoints": EXPECTED_CHECKPOINTS,
        "original_checkpoint_bytes": original_total,
        "portable_weight_bytes": portable_total,
        "training_state_bytes_removed": original_total - portable_total,
        "size_reduction_percent": reduction,
        "precision_changed": False,
        "quantized": False,
        "all_strict_reload": True,
        "all_exact_tensor_roundtrip": True,
    }

    summary_path = output_dir / "phase13b_bundle_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    # Create a transport tar.gz for copying from Idun to laptop.
    archive_path = output_dir / "ENS328_model_bundle.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(
            bundle_dir,
            arcname="model_bundle",
            recursive=True,
        )

    print()
    print("=" * 108)
    print("PHASE 13B SUMMARY")
    print("=" * 108)
    print("Status                    : PASS")
    print(f"Portable checkpoints      : {EXPECTED_CHECKPOINTS}/25")
    print(
        f"Original checkpoint total : "
        f"{original_total / 1024**3:.3f} GiB"
    )
    print(
        f"Portable weight total     : "
        f"{portable_total / 1024**3:.3f} GiB"
    )
    print(
        f"Size removed              : {reduction:.2f}%"
    )
    print("Precision changed          : NO")
    print("Quantization               : NO")
    print("Strict portable reload     : PASS")
    print("Exact tensor roundtrip     : PASS")
    print()
    print(f"Bundle directory: {bundle_dir}")
    print(f"Archive         : {archive_path}")
    print(f"Provenance      : {provenance_path}")
    print(f"Summary         : {summary_path}")
    print()
    print("STATUS: PASS")
    print(
        "Next: transfer ENS328_model_bundle.tar.gz to the local official "
        "runtime repository and connect it to the already-validated "
        "deterministic preprocessing main.py."
    )


if __name__ == "__main__":
    main()
