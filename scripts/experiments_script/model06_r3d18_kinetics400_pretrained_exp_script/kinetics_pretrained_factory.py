#!/usr/bin/env python3
"""
kinetics_pretrained_factory.py

Thin experiment-layer factory for E6.

The architecture and actual Kinetics-400 transfer remain authoritative in:

    src.models.model06_r3d18_kinetics400_pretrained
    src.models.model05_r3d18_scratch

This helper does NOT modify weights or preprocessing.

It performs:
    - construction through the src factory,
    - cheap per-model contract checks,
    - one deep E5/E6 architecture + Kinetics-transfer QC per Python process.

The deep QC is intentionally performed only once because
verify_kinetics_transfer() constructs an untouched official Torchvision
Kinetics model for exact tensor-by-tensor comparison. Repeating that for
every fold/branch would add unnecessary overhead without adding information.
"""

from __future__ import annotations

from src.models.model05_r3d18_scratch import (
    BLOCK_STRUCTURE,
    FEATURE_DIM,
    STAGE_CHANNELS,
    STEM_IN_CHANNELS,
    STEM_KERNEL_SIZE,
    STEM_OUT_CHANNELS,
    STEM_PADDING,
    STEM_STRIDE,
)
from src.models.model06_r3d18_kinetics400_pretrained import (
    KINETICS400_WEIGHTS,
    build_model as build_e6_model,
    verify_architecture_matches_e5,
    verify_kinetics_transfer,
)


_FULL_QC_DONE = False


def _check_model_contract(model) -> None:
    if not bool(model.pretrained):
        raise RuntimeError(
            "E6 must be Kinetics-pretrained, but model.pretrained is False."
        )

    if model.grayscale_mode != "rgb_sum":
        raise RuntimeError(
            "E6 grayscale_mode must be 'rgb_sum', received "
            f"{model.grayscale_mode!r}."
        )

    if getattr(model, "pretrained_source", None) != (
        "R3D_18_Weights.KINETICS400_V1"
    ):
        raise RuntimeError(
            "Unexpected E6 pretrained source: "
            f"{getattr(model, 'pretrained_source', None)!r}."
        )

    if int(model.feature_dim) != int(FEATURE_DIM) or int(FEATURE_DIM) != 512:
        raise RuntimeError(
            f"Unexpected E6 feature dimension: {model.feature_dim}."
        )

    stem = model.backbone.stem[0]

    if int(stem.in_channels) != STEM_IN_CHANNELS:
        raise RuntimeError(
            f"E6 stem input channels must be {STEM_IN_CHANNELS}."
        )

    if int(stem.out_channels) != STEM_OUT_CHANNELS:
        raise RuntimeError(
            f"E6 stem output channels must be {STEM_OUT_CHANNELS}."
        )

    if tuple(stem.kernel_size) != tuple(STEM_KERNEL_SIZE):
        raise RuntimeError(
            "Unexpected E6 stem kernel: "
            f"{tuple(stem.kernel_size)}."
        )

    if tuple(stem.stride) != tuple(STEM_STRIDE):
        raise RuntimeError(
            "Unexpected E6 stem stride: "
            f"{tuple(stem.stride)}."
        )

    if tuple(stem.padding) != tuple(STEM_PADDING):
        raise RuntimeError(
            "Unexpected E6 stem padding: "
            f"{tuple(stem.padding)}."
        )

    if model.classifier.in_features != FEATURE_DIM:
        raise RuntimeError(
            "E6 binary classifier must receive 512 features."
        )

    if model.classifier.out_features != 1:
        raise RuntimeError(
            "E6 binary classifier must produce one raw logit."
        )


def build_single_e6():
    """
    Build one fresh E6 branch.

    Every call constructs a new E6 model initialized from the official
    Torchvision Kinetics-400 V1 weights.

    The first successful call in the process additionally verifies:
        1. exact E5/E6 state-dict name/shape identity,
        2. exact RGB-sum grayscale stem transfer,
        3. exact preservation of every other pretrained backbone tensor,
        4. removal of the 400-class Kinetics head,
        5. presence of the new Linear(512 -> 1) binary head.
    """
    global _FULL_QC_DONE

    try:
        model = build_e6_model()
    except Exception as exc:
        raise RuntimeError(
            "E6 could not construct the Kinetics-400 pretrained R3D-18. "
            "Torchvision must be able to load "
            "R3D_18_Weights.KINETICS400_V1, normally from the local Torch "
            "cache or by downloading the official weights if the HPC node "
            "has network access. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    _check_model_contract(model)

    if not _FULL_QC_DONE:
        verify_architecture_matches_e5(model)
        verify_kinetics_transfer(model)
        _FULL_QC_DONE = True

    return model
