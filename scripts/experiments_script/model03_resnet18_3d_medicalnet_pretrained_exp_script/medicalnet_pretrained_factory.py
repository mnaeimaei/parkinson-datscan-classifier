#!/usr/bin/env python3
"""
medicalnet_pretrained_factory.py

Experiment-layer path/provenance helper for E3.

This module DOES NOT implement MedicalNet checkpoint loading.
The authoritative loading and coverage validation remain in:

    src.models.model03_resnet18_3d_medicalnet_pretrained

Responsibilities here are only:
    - resolve the frozen resnet_18_23dataset.pth path
    - require an unambiguous checkpoint
    - compute SHA256 for experiment provenance
    - call the src model factory
    - re-check the src load report's coverage fields

Checkpoint resolution
---------------------
Preferred:
    environment variable MEDICALNET_CHECKPOINT

If it is not set, the project root is searched recursively for exactly one
file named resnet_18_23dataset.pth.

If zero or multiple matches exist, execution fails loudly.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from src.models.model03_resnet18_3d_medicalnet_pretrained import (
    EXPECTED_CHECKPOINT_NAME,
    build_model as build_medicalnet_model,
)


ENV_NAME = "MEDICALNET_CHECKPOINT"


def resolve_medicalnet_checkpoint(project_root: Path) -> Path:
    project_root = Path(project_root).expanduser().resolve()

    explicit = os.environ.get(ENV_NAME, "").strip()

    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"{ENV_NAME} points to a missing file:\n    {path}"
            )

        if path.name != EXPECTED_CHECKPOINT_NAME:
            raise ValueError(
                "E3 requires the exact MedicalNet checkpoint filename.\n"
                f"Expected : {EXPECTED_CHECKPOINT_NAME}\n"
                f"Received : {path.name}"
            )

        return path

    matches = sorted(
        p.resolve()
        for p in project_root.rglob(EXPECTED_CHECKPOINT_NAME)
        if p.is_file()
    )

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            "Could not find the required MedicalNet checkpoint under the "
            f"project root:\n    {project_root}\n\n"
            f"Required filename: {EXPECTED_CHECKPOINT_NAME}\n\n"
            "Either place the checkpoint somewhere inside the project or "
            f"export {ENV_NAME}=/absolute/path/{EXPECTED_CHECKPOINT_NAME}."
        )

    raise RuntimeError(
        "Multiple MedicalNet checkpoints were found. The experiment must not "
        "guess which pretrained initialization to use.\n\n"
        + "\n".join(f"  - {p}" for p in matches)
        + f"\n\nSet {ENV_NAME} to the one frozen checkpoint intended for E3."
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def build_single_e3(checkpoint_path: Path):
    """
    Build one fresh E3 branch.

    Every call:
        1. constructs exactly the E2-compatible architecture,
        2. loads the MedicalNet encoder through src/,
        3. leaves the new binary classifier randomly initialized,
        4. verifies essentially complete intended encoder transfer.
    """
    model = build_medicalnet_model(
        checkpoint_path=Path(checkpoint_path),
    )

    report = model.pretrained_load_report

    if report.tensor_coverage < 0.999999:
        raise RuntimeError(
            "MedicalNet tensor coverage unexpectedly below requirement: "
            f"{report.tensor_coverage:.6%}"
        )

    if report.parameter_coverage < 0.999999:
        raise RuntimeError(
            "MedicalNet parameter coverage unexpectedly below requirement: "
            f"{report.parameter_coverage:.6%}"
        )

    return model
