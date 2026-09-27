#!/usr/bin/env python3
"""
swin3d_kinetics_pretrained_factory.py

Thin experiment-layer factory for E10.

The authoritative model architecture and Kinetics-400 transfer remain in:

    src.models.model09_swin3d_t_scratch
    src.models.model10_swin3d_t_kinetics400_pretrained

This helper does NOT modify model weights and does NOT apply any Torchvision
Kinetics preprocessing.

It performs:
    - E10 construction through the src factory,
    - cheap per-model contract checks,
    - one deep E9/E10 architecture + pretrained-transfer QC per Python process.

The deep QC is intentionally performed only once because
verify_kinetics_transfer() constructs an untouched official pretrained
Swin3D-Tiny for exact tensor comparison. Repeating that for every fold/branch
would add overhead without adding scientific information.
"""

from __future__ import annotations

import torch.nn as nn

from src.models.model09_swin3d_t_scratch import (
    DEPTHS,
    EMBED_DIM,
    FEATURE_DIM,
    INPUT_CHANNELS,
    NUM_HEADS,
    PATCH_SIZE,
    STAGE_CHANNELS,
    STOCHASTIC_DEPTH_PROB,
    WINDOW_SIZE,
)
from src.models.model10_swin3d_t_kinetics400_pretrained import (
    KINETICS400_WEIGHTS,
    build_model as build_e10_model,
    count_all_parameters,
    verify_architecture_matches_e9,
    verify_kinetics_transfer,
)


EXPECTED_PARAMETER_COUNT = 27_845_095

WHOLE_DHW = (160, 192, 192)
ROI_DHW = (36, 44, 44)

EXPECTED_WHOLE_GRIDS = (
    (80, 48, 48),
    (80, 24, 24),
    (80, 12, 12),
    (80, 6, 6),
)

EXPECTED_ROI_GRIDS = (
    (18, 11, 11),
    (18, 6, 6),
    (18, 3, 3),
    (18, 2, 2),
)

_FULL_QC_DONE = False


def _check_shape_report(report, expected_grids, name: str) -> None:
    if not report.compatible:
        raise RuntimeError(
            f"E10 {name} input is incompatible with Swin3D hierarchy: "
            f"{report}"
        )

    observed = (
        tuple(report.stage1_grid),
        tuple(report.stage2_grid),
        tuple(report.stage3_grid),
        tuple(report.stage4_grid),
    )

    if observed != expected_grids:
        raise RuntimeError(
            f"Unexpected E10 {name} Swin grids: "
            f"{observed} != {expected_grids}"
        )


def _check_model_contract(model) -> None:
    if not bool(model.pretrained):
        raise RuntimeError(
            "E10 must be Kinetics-pretrained; model.pretrained is False."
        )

    if getattr(model, "pretrained_source", None) != (
        "Swin3D_T_Weights.KINETICS400_V1"
    ):
        raise RuntimeError(
            "Unexpected E10 pretrained source: "
            f"{getattr(model, 'pretrained_source', None)!r}."
        )

    if int(model.feature_dim) != int(FEATURE_DIM) or int(FEATURE_DIM) != 768:
        raise RuntimeError(
            f"Unexpected E10 feature dimension: {model.feature_dim}."
        )

    report = model.initialization_report

    if report.pretrained_source != "Swin3D_T_Weights.KINETICS400_V1":
        raise RuntimeError(
            "Unexpected E10 initialization-report source: "
            f"{report.pretrained_source!r}."
        )

    if tuple(report.original_patch_shape) != (96, 3, 2, 4, 4):
        raise RuntimeError(
            "Unexpected pretrained RGB patch-embedding shape: "
            f"{report.original_patch_shape}."
        )

    if tuple(report.grayscale_patch_shape) != (96, 1, 2, 4, 4):
        raise RuntimeError(
            "Unexpected E10 grayscale patch-embedding shape: "
            f"{report.grayscale_patch_shape}."
        )

    if report.grayscale_method != "RGB_SUM":
        raise RuntimeError(
            "E10 grayscale conversion must be RGB_SUM, received "
            f"{report.grayscale_method!r}."
        )

    if int(report.feature_dim) != 768:
        raise RuntimeError(
            f"Unexpected E10 initialization feature_dim: {report.feature_dim}."
        )

    if int(report.original_classes) != 400:
        raise RuntimeError(
            "E10 pretrained classifier must originate from 400 classes."
        )

    if int(report.target_classes) != 1:
        raise RuntimeError(
            "E10 target classifier must have one output."
        )

    patch_conv = model.backbone.patch_embed.proj

    if not isinstance(patch_conv, nn.Conv3d):
        raise RuntimeError(
            "E10 patch embedding must be nn.Conv3d."
        )

    if patch_conv.in_channels != INPUT_CHANNELS or INPUT_CHANNELS != 1:
        raise RuntimeError(
            "E10 patch embedding must have one input channel."
        )

    if patch_conv.out_channels != EMBED_DIM or EMBED_DIM != 96:
        raise RuntimeError(
            f"Unexpected E10 embedding dimension: {patch_conv.out_channels}."
        )

    if tuple(patch_conv.kernel_size) != tuple(PATCH_SIZE):
        raise RuntimeError(
            f"Unexpected E10 patch kernel: {patch_conv.kernel_size}."
        )

    if tuple(patch_conv.stride) != tuple(PATCH_SIZE):
        raise RuntimeError(
            f"Unexpected E10 patch stride: {patch_conv.stride}."
        )

    if tuple(DEPTHS) != (2, 2, 6, 2):
        raise RuntimeError(
            f"Unexpected E10 stage depths: {DEPTHS}."
        )

    if tuple(NUM_HEADS) != (3, 6, 12, 24):
        raise RuntimeError(
            f"Unexpected E10 attention heads: {NUM_HEADS}."
        )

    if tuple(WINDOW_SIZE) != (8, 7, 7):
        raise RuntimeError(
            f"Unexpected E10 window size: {WINDOW_SIZE}."
        )

    if tuple(STAGE_CHANNELS) != (96, 192, 384, 768):
        raise RuntimeError(
            f"Unexpected E10 stage channels: {STAGE_CHANNELS}."
        )

    if abs(float(STOCHASTIC_DEPTH_PROB) - 0.1) > 1e-12:
        raise RuntimeError(
            "Unexpected E10 stochastic-depth probability: "
            f"{STOCHASTIC_DEPTH_PROB}."
        )

    if model.backbone.head.in_features != 768:
        raise RuntimeError(
            "E10 binary classifier must receive 768 features."
        )

    if model.backbone.head.out_features != 1:
        raise RuntimeError(
            "E10 binary classifier must output one raw logit."
        )

    parameter_count = count_all_parameters(model)

    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            "Unexpected E10 parameter count: "
            f"{parameter_count:,} != {EXPECTED_PARAMETER_COUNT:,}."
        )

    whole_report = model.analyze_input_shape(*WHOLE_DHW)
    roi_report = model.analyze_input_shape(*ROI_DHW)

    _check_shape_report(
        whole_report,
        EXPECTED_WHOLE_GRIDS,
        "whole-volume",
    )

    _check_shape_report(
        roi_report,
        EXPECTED_ROI_GRIDS,
        "ROI",
    )


def build_single_e10():
    """
    Build one fresh E10 branch.

    Every call constructs a new E10 model from the official
    Swin3D_T_Weights.KINETICS400_V1 weights.

    The first successful call in a Python job additionally verifies:
        1. E9/E10 state-dict names and shapes match exactly,
        2. grayscale patch weight exactly equals RGB-channel sum,
        3. pretrained patch bias is preserved,
        4. every other pretrained encoder tensor is unchanged,
        5. the 400-class Kinetics head is gone,
        6. the new binary head is Linear(768 -> 1).
    """
    global _FULL_QC_DONE

    try:
        model = build_e10_model()
    except Exception as exc:
        raise RuntimeError(
            "E10 could not construct the Kinetics-400 pretrained "
            "Swin3D-Tiny. Torchvision must be able to load "
            "Swin3D_T_Weights.KINETICS400_V1, normally from the local Torch "
            "cache or by downloading the official weights if the HPC node "
            "has network access. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    _check_model_contract(model)

    if not _FULL_QC_DONE:
        verify_architecture_matches_e9(model)
        verify_kinetics_transfer(model)
        _FULL_QC_DONE = True

    return model
