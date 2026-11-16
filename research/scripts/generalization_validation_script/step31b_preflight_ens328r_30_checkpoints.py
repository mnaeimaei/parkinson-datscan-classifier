#!/usr/bin/env python3
"""
STEP 31B — ENS328R 30-CHECKPOINT PREFLIGHT

Build and validate a new 30-checkpoint manifest:

    P1       5 folds
    R_E6     5 folds
    P2       5 folds
    P3       5 folds
    P7       5 folds
    P8       5 folds
    ----------------
             30 checkpoints

The existing ENS328 25-checkpoint manifest is used as the authoritative source
for P1/P2/P3/P7/P8. Five robust E6 checkpoints are discovered from Step 29.

All 30 checkpoints are SHA256-verified and strict-loaded into offline-safe
constructors. No pretrained weight download is required.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


MEMBERS = ("P1", "R_E6", "P2", "P3", "P7", "P8")
N_FOLDS = 5
EXPECTED = 30

DEFAULT_OLD_MANIFEST = Path(
    "data/final_submission_preflight_data/ENS328/final_25_checkpoint_manifest.csv"
)
DEFAULT_ROBUST_DIR = Path(
    "data/generalization_validation_data/"
    "step29_e6s5_robust_step11_confirmation"
)
DEFAULT_CONFIG = Path(
    "data/final_predictor_data/ENS328R/final_predictor_config.json"
)
DEFAULT_OUTPUT = Path(
    "data/final_submission_preflight_data/ENS328R"
)

BUILDERS = {
    "P1": ("src.models.model05_r3d18_scratch", "build_model"),
    "R_E6": ("src.models.model05_r3d18_scratch", "build_model"),
    "P2": ("src.models.model07_resnet18_2p5d_attention_scratch", "build_model"),
    "P3": ("src.models.model05_r3d18_scratch", "build_model"),
    "P7": ("src.models.model07_resnet18_2p5d_attention_scratch", "build_model"),
    "P8": ("src.models.model02_resnet18_3d_scratch", "build_model"),
}

INPUTS = {
    "P1": "roi",
    "R_E6": "roi",
    "P2": "whole_2p5d",
    "P3": "roi",
    "P7": "roi_2p5d",
    "P8": "roi",
}

MODELS = {
    "P1": "E6 / R3D-18 Kinetics-400 / S5 ROI+aug",
    "R_E6": "E6 / R3D-18 Kinetics-400 / S5 ROI+robust-aug",
    "P2": "E7 / ResNet18 2.5D attention scratch / S4 whole+aug",
    "P3": "E6 / R3D-18 Kinetics-400 / S2 ROI noaug",
    "P7": "E4 / ResNet18 2.5D attention ImageNet / S2 ROI noaug",
    "P8": "E2 / ResNet18 3D scratch / S2 ROI noaug",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate ENS328R 30 checkpoints.")
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--existing-manifest", type=Path, default=None)
    p.add_argument("--robust-dir", type=Path, default=None)
    p.add_argument("--predictor-config", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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

    raise RuntimeError(
        f"Could not extract tensor model state_dict from {type(checkpoint).__name__}"
    )


def build_model(member: str) -> torch.nn.Module:
    module_name, builder_name = BUILDERS[member]
    module = importlib.import_module(module_name)
    builder = getattr(module, builder_name)
    model = builder()
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{member}: builder returned {type(model).__name__}")
    return model


def resolve_checkpoint_from_training_summary(
    *,
    project_root: Path,
    fold_dir: Path,
) -> tuple[Path, str]:
    summary_path = fold_dir / "training_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        stored = summary.get("best_checkpoint")
        if stored:
            p = Path(str(stored)).expanduser()
            candidates = []
            if p.is_absolute():
                candidates.append(p.resolve())
            else:
                candidates.extend(
                    [
                        (project_root / p).resolve(),
                        (fold_dir / p).resolve(),
                    ]
                )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate, "training_summary.json"

    for name in ("best_model.pt", "best_checkpoint.pt", "best.pt", "model_best.pt"):
        p = fold_dir / name
        if p.is_file():
            return p.resolve(), f"fallback:{name}"

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
        f"Could not uniquely resolve best checkpoint in {fold_dir}: "
        f"{[p.name for p in candidates]}"
    )


def standardize_old_manifest(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Member ID",
        "Fold",
        "Checkpoint Path",
        "Checkpoint SHA256",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Existing ENS328 manifest missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["Member ID"] = df["Member ID"].astype(str)
    df["Fold"] = pd.to_numeric(df["Fold"], errors="raise").astype(int)
    return df


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

    old_manifest_path = resolve(
        root, args.existing_manifest or DEFAULT_OLD_MANIFEST
    )
    robust_dir = resolve(root, args.robust_dir or DEFAULT_ROBUST_DIR)
    config_path = resolve(root, args.predictor_config or DEFAULT_CONFIG)
    output_dir = resolve(root, args.output_dir or DEFAULT_OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (old_manifest_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not robust_dir.is_dir():
        raise FileNotFoundError(robust_dir)

    config = read_json(config_path)
    if config.get("ensemble", {}).get("id") != "ENS328R":
        raise RuntimeError("Predictor config is not ENS328R.")
    if tuple(config.get("ensemble", {}).get("members", [])) != MEMBERS:
        raise RuntimeError("ENS328R config member order mismatch.")

    old = standardize_old_manifest(pd.read_csv(old_manifest_path))
    old_members = set(old["Member ID"])
    expected_old = {"P1", "P2", "P3", "P7", "P8"}
    if old_members != expected_old:
        raise RuntimeError(
            f"Existing manifest members {old_members} != {expected_old}"
        )
    if len(old) != 25:
        raise RuntimeError(f"Expected 25 existing rows, got {len(old)}")

    rows = []
    failures = []

    # Revalidate all existing ENS328 checkpoints.
    for _, row in old.iterrows():
        member = str(row["Member ID"])
        fold = int(row["Fold"])
        path = Path(str(row["Checkpoint Path"])).expanduser().resolve()

        if not path.is_file():
            failures.append(f"{member} fold {fold}: missing {path}")
            continue

        observed_sha = sha256_file(path)
        expected_sha = str(row["Checkpoint SHA256"]).strip()
        if observed_sha != expected_sha:
            failures.append(
                f"{member} fold {fold}: SHA changed "
                f"{observed_sha} != {expected_sha}"
            )
            continue

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state, container_key = extract_state_dict(checkpoint)
        model = build_model(member)
        incompatible = model.load_state_dict(state, strict=True)
        strict_ok = (
            not incompatible.missing_keys
            and not incompatible.unexpected_keys
        )
        if not strict_ok:
            failures.append(
                f"{member} fold {fold}: strict offline model load failed"
            )

        rows.append(
            {
                "Member ID": member,
                "Fold": fold,
                "Model": MODELS[member],
                "Input Type": INPUTS[member],
                "Checkpoint Path": str(path),
                "Checkpoint SHA256": observed_sha,
                "Checkpoint Bytes": path.stat().st_size,
                "State Dict Container": container_key,
                "Tensor Count": len(state),
                "Parameter Count": int(sum(v.numel() for v in state.values())),
                "Offline Builder Module": BUILDERS[member][0],
                "Offline Builder Name": BUILDERS[member][1],
                "Strict Offline Load": bool(strict_ok),
                "Checkpoint Source": "existing ENS328 preflight manifest",
            }
        )

        del checkpoint, state, model

    # Discover and validate robust E6 checkpoints.
    for fold in range(N_FOLDS):
        fold_dir = robust_dir / f"fold_{fold}"
        if not fold_dir.is_dir():
            failures.append(f"R_E6 fold {fold}: missing directory {fold_dir}")
            continue

        path, source = resolve_checkpoint_from_training_summary(
            project_root=root,
            fold_dir=fold_dir,
        )

        observed_sha = sha256_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state, container_key = extract_state_dict(checkpoint)
        model = build_model("R_E6")
        incompatible = model.load_state_dict(state, strict=True)
        strict_ok = (
            not incompatible.missing_keys
            and not incompatible.unexpected_keys
        )
        if not strict_ok:
            failures.append(
                f"R_E6 fold {fold}: strict E5-architecture offline load failed"
            )

        rows.append(
            {
                "Member ID": "R_E6",
                "Fold": fold,
                "Model": MODELS["R_E6"],
                "Input Type": INPUTS["R_E6"],
                "Checkpoint Path": str(path),
                "Checkpoint SHA256": observed_sha,
                "Checkpoint Bytes": path.stat().st_size,
                "State Dict Container": container_key,
                "Tensor Count": len(state),
                "Parameter Count": int(sum(v.numel() for v in state.values())),
                "Offline Builder Module": BUILDERS["R_E6"][0],
                "Offline Builder Name": BUILDERS["R_E6"][1],
                "Strict Offline Load": bool(strict_ok),
                "Checkpoint Source": source,
            }
        )

        del checkpoint, state, model

    if failures:
        report = "\n".join(f"- {x}" for x in failures)
        raise RuntimeError("STEP31B PREFLIGHT FAILED:\n" + report)

    manifest = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MEMBERS)}
    manifest["_order"] = manifest["Member ID"].map(order)
    manifest = manifest.sort_values(
        ["_order", "Fold"], kind="stable"
    ).drop(columns="_order").reset_index(drop=True)

    if len(manifest) != EXPECTED:
        raise RuntimeError(f"Expected {EXPECTED} rows, found {len(manifest)}")

    if tuple(manifest["Member ID"].drop_duplicates()) != MEMBERS:
        raise RuntimeError("Manifest member order mismatch.")

    for member in MEMBERS:
        folds = manifest.loc[
            manifest["Member ID"] == member, "Fold"
        ].tolist()
        if folds != list(range(N_FOLDS)):
            raise RuntimeError(f"{member}: fold list {folds} != 0..4")
        if not manifest.loc[
            manifest["Member ID"] == member, "Strict Offline Load"
        ].all():
            raise RuntimeError(f"{member}: one or more strict loads failed.")

    manifest_path = output_dir / "final_30_checkpoint_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = {
        "status": "PASS",
        "predictor": "ENS328R",
        "members": list(MEMBERS),
        "checkpoints": EXPECTED,
        "strict_offline_loads": int(manifest["Strict Offline Load"].sum()),
        "all_sha256_verified": True,
        "robust_checkpoints": 5,
        "existing_ens328_checkpoints_reused": 25,
        "manifest": str(manifest_path),
    }
    (output_dir / "step31b_preflight_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("STEP 31B — ENS328R 30-CHECKPOINT PREFLIGHT")
    print("=" * 108)
    print("Checkpoints              : 30/30")
    print("SHA256                   : PASS")
    print("Strict offline load      : 30/30 PASS")
    print("Pretrained download      : NOT REQUIRED")
    print(f"Manifest                 : {manifest_path}")
    print("Status                   : PASS")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
