#!/usr/bin/env python3
"""
STEP 31C — BUILD PORTABLE ENS328R RUNTIME BUNDLE

Consumes:
- Step31B final_30_checkpoint_manifest.csv
- Step31A data/final_predictor_data/ENS328R/
- Step31 ENS328R runtime source

Produces:
data/final_runtime_bundle_data/ENS328R/
    model_bundle/
    ENS328R_model_bundle.tar.gz

Portable weights are tensor-only state_dict files.
Training optimizer/scheduler/scaler state is removed.
No quantization is applied.
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


MEMBERS = ("P1", "R_E6", "P2", "P3", "P7", "P8")
N_FOLDS = 5
EXPECTED = 30

BUILDERS = {
    "P1": ("src.models.model05_r3d18_scratch", "build_model"),
    "R_E6": ("src.models.model05_r3d18_scratch", "build_model"),
    "P2": ("src.models.model07_resnet18_2p5d_attention_scratch", "build_model"),
    "P3": ("src.models.model05_r3d18_scratch", "build_model"),
    "P7": ("src.models.model07_resnet18_2p5d_attention_scratch", "build_model"),
    "P8": ("src.models.model02_resnet18_3d_scratch", "build_model"),
}

DEFAULT_MANIFEST = Path(
    "data/final_submission_preflight_data/ENS328R/"
    "final_30_checkpoint_manifest.csv"
)
DEFAULT_PREDICTOR_DIR = Path("data/final_predictor_data/ENS328R")
DEFAULT_RUNTIME_SOURCE = Path(
    "scripts/generalization_validation_script/"
    "step31_ens328r_runtime_model.py"
)
DEFAULT_OUTPUT = Path("data/final_runtime_bundle_data/ENS328R")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build portable ENS328R runtime bundle.")
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--checkpoint-manifest", type=Path, default=None)
    p.add_argument("--predictor-dir", type=Path, default=None)
    p.add_argument("--runtime-source", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model_state", "model"):
            value = checkpoint.get(key)
            if (
                isinstance(value, dict)
                and value
                and all(torch.is_tensor(v) for v in value.values())
            ):
                return value, key

        if checkpoint and all(
            isinstance(k, str) and torch.is_tensor(v)
            for k, v in checkpoint.items()
        ):
            return checkpoint, "<raw_state_dict>"

    raise RuntimeError("Could not extract tensor model state_dict.")


def build_model(member: str) -> torch.nn.Module:
    module_name, builder_name = BUILDERS[member]
    module = importlib.import_module(module_name)
    model = getattr(module, builder_name)()
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{member}: invalid model builder output")
    return model


def copy_source_tree(project_root: Path, bundle_dir: Path) -> None:
    src_target = bundle_dir / "src"
    (src_target / "calibration").mkdir(parents=True, exist_ok=True)
    (src_target / "inference").mkdir(parents=True, exist_ok=True)
    (src_target / "models").mkdir(parents=True, exist_ok=True)

    files = {
        project_root / "src" / "__init__.py":
            src_target / "__init__.py",
        project_root / "src" / "calibration" / "__init__.py":
            src_target / "calibration" / "__init__.py",
        project_root / "src" / "calibration" / "probability_calibration.py":
            src_target / "calibration" / "probability_calibration.py",
        project_root / "src" / "inference" / "__init__.py":
            src_target / "inference" / "__init__.py",
        project_root / "src" / "inference" / "final_ensemble.py":
            src_target / "inference" / "final_ensemble.py",
        project_root / "src" / "models" / "model02_resnet18_3d_scratch.py":
            src_target / "models" / "model02_resnet18_3d_scratch.py",
        project_root / "src" / "models" / "model04_resnet18_2p5d_attention_imagenet_pretrained.py":
            src_target / "models" / "model04_resnet18_2p5d_attention_imagenet_pretrained.py",
        project_root / "src" / "models" / "model05_r3d18_scratch.py":
            src_target / "models" / "model05_r3d18_scratch.py",
        project_root / "src" / "models" / "model07_resnet18_2p5d_attention_scratch.py":
            src_target / "models" / "model07_resnet18_2p5d_attention_scratch.py",
    }

    for source, target in files.items():
        if source.name == "__init__.py" and not source.is_file():
            target.write_text("", encoding="utf-8")
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)

    # IMPORTANT:
    # Do NOT copy the training repository's src/models/__init__.py into the
    # portable bundle. The project initializer imports additional training-only
    # modules such as dual_input_fusion.py that the ENS328R runtime never uses.
    #
    # A minimal package initializer keeps the portable bundle self-contained
    # while runtime_model.py imports only the exact model modules required by
    # P1/R_E6/P2/P3/P7/P8.
    (src_target / "models" / "__init__.py").write_text(
        "# Minimal portable ENS328R models package.\n",
        encoding="utf-8",
    )


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

    manifest_path = resolve(root, args.checkpoint_manifest or DEFAULT_MANIFEST)
    predictor_dir = resolve(root, args.predictor_dir or DEFAULT_PREDICTOR_DIR)
    runtime_source = resolve(root, args.runtime_source or DEFAULT_RUNTIME_SOURCE)
    output_dir = resolve(root, args.output_dir or DEFAULT_OUTPUT)

    for path in (
        manifest_path,
        predictor_dir / "final_predictor_config.json",
        runtime_source,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"{output_dir} non-empty. Use --overwrite for intentional rebuild."
            )

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
        raise RuntimeError(f"Manifest missing {sorted(missing)}")

    manifest["Member ID"] = manifest["Member ID"].astype(str)
    manifest["Fold"] = pd.to_numeric(
        manifest["Fold"], errors="raise"
    ).astype(int)

    if len(manifest) != EXPECTED:
        raise RuntimeError(f"Expected {EXPECTED} manifest rows, got {len(manifest)}")
    if tuple(manifest["Member ID"].drop_duplicates()) != MEMBERS:
        raise RuntimeError("Manifest member order mismatch.")
    if not manifest["Strict Offline Load"].astype(bool).all():
        raise RuntimeError("Manifest includes failed strict offline load.")

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = output_dir / "model_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Package code and frozen predictor.
    shutil.copy2(runtime_source, bundle_dir / "runtime_model.py")
    (bundle_dir / "__init__.py").write_text(
        "from .runtime_model import ENS328RRuntime\n"
        "__all__ = ['ENS328RRuntime']\n",
        encoding="utf-8",
    )
    copy_source_tree(root, bundle_dir)

    predictor_target = bundle_dir / "final_predictor"
    shutil.copytree(predictor_dir, predictor_target)

    portable_members = {}
    provenance_rows = []
    original_total = 0
    portable_total = 0

    for member in MEMBERS:
        entries = []
        subset = manifest.loc[
            manifest["Member ID"] == member
        ].sort_values("Fold")

        if subset["Fold"].tolist() != list(range(N_FOLDS)):
            raise RuntimeError(f"{member}: fold list mismatch")

        target_member_dir = bundle_dir / "checkpoints" / member
        target_member_dir.mkdir(parents=True, exist_ok=True)

        for _, row in subset.iterrows():
            fold = int(row["Fold"])
            source = Path(str(row["Checkpoint Path"])).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)

            observed_sha = sha256_file(source)
            expected_sha = str(row["Checkpoint SHA256"])
            if observed_sha != expected_sha:
                raise RuntimeError(
                    f"{member} fold {fold}: source SHA changed before bundling"
                )

            original_total += source.stat().st_size

            checkpoint = torch.load(
                source,
                map_location="cpu",
                weights_only=False,
            )
            state, container_key = extract_state_dict(checkpoint)

            portable_state = {
                key: tensor.detach().cpu().contiguous()
                for key, tensor in state.items()
            }

            target_rel = Path("checkpoints") / member / f"fold_{fold}_state_dict.pt"
            target = bundle_dir / target_rel
            torch.save(portable_state, target)

            # Verify tensor-only reload and exact strict architecture.
            reloaded = torch.load(
                target,
                map_location="cpu",
                weights_only=True,
            )
            model = build_model(member)
            incompatible = model.load_state_dict(reloaded, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"{member} fold {fold}: portable strict reload failed"
                )

            for key in portable_state:
                if not torch.equal(portable_state[key], reloaded[key]):
                    raise RuntimeError(
                        f"{member} fold {fold}: tensor changed during portable save"
                    )

            portable_sha = sha256_file(target)
            portable_bytes = target.stat().st_size
            portable_total += portable_bytes

            entry = {
                "fold": fold,
                "relative_path": str(target_rel),
                "sha256": portable_sha,
                "bytes": portable_bytes,
                "tensor_count": len(portable_state),
                "parameter_count":
                    int(sum(v.numel() for v in portable_state.values())),
            }
            entries.append(entry)

            provenance_rows.append(
                {
                    "Member ID": member,
                    "Fold": fold,
                    "Original Checkpoint": str(source),
                    "Original SHA256": observed_sha,
                    "Original Bytes": source.stat().st_size,
                    "State Dict Container": container_key,
                    "Portable Path": str(target),
                    "Portable SHA256": portable_sha,
                    "Portable Bytes": portable_bytes,
                    "Tensor Count": len(portable_state),
                    "Parameter Count": entry["parameter_count"],
                    "Strict Portable Reload": True,
                }
            )

            del checkpoint, state, portable_state, reloaded, model

        portable_members[member] = entries

    portable_manifest = {
        "schema_version": 1,
        "predictor": "ENS328R",
        "member_order": list(MEMBERS),
        "n_folds": N_FOLDS,
        "expected_checkpoints": EXPECTED,
        "members": portable_members,
    }

    manifest_json = bundle_dir / "portable_checkpoint_manifest.json"
    manifest_json.write_text(
        json.dumps(portable_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    provenance_path = output_dir / "portable_checkpoint_provenance.csv"
    pd.DataFrame(provenance_rows).to_csv(provenance_path, index=False)

    summary = {
        "status": "PASS",
        "predictor": "ENS328R",
        "members": list(MEMBERS),
        "portable_checkpoints": EXPECTED,
        "original_checkpoint_bytes": original_total,
        "portable_weight_bytes": portable_total,
        "training_state_bytes_removed": original_total - portable_total,
        "precision_changed": False,
        "quantization": "NONE",
        "bundle_dir": str(bundle_dir),
    }
    (output_dir / "step31c_bundle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    archive = output_dir / "ENS328R_model_bundle.tar.gz"
    with tarfile.open(archive, mode="w:gz") as tar:
        tar.add(bundle_dir, arcname="model_bundle")

    # Archive must contain the runtime and manifest.
    if not archive.is_file() or archive.stat().st_size == 0:
        raise RuntimeError("ENS328R archive creation failed.")

    print("=" * 108)
    print("STEP 31C — PORTABLE ENS328R RUNTIME BUNDLE")
    print("=" * 108)
    print("Members                  : 6")
    print("Portable checkpoints     : 30/30")
    print("Strict portable reload   : PASS")
    print("Quantization             : NONE")
    print("Precision change         : NONE")
    print(f"Bundle                   : {bundle_dir}")
    print(f"Archive                  : {archive}")
    print("Status                   : PASS")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
