#!/usr/bin/env python3
"""
swin3d_scratch_factory.py

Thin experiment-layer factory for E9.

The authoritative architecture and initialization remain in:
    src.models.model09_swin3d_t_scratch

This helper does not alter model weights or preprocessing. It simply checks
the frozen E9 contract before a fold begins.
"""

from __future__ import annotations

import torch.nn as nn

from src.models.model09_swin3d_t_scratch import (
    INPUT_CHANNELS,
    PATCH_SIZE,
    EMBED_DIM,
    DEPTHS,
    NUM_HEADS,
    WINDOW_SIZE,
    STAGE_CHANNELS,
    FEATURE_DIM,
    STOCHASTIC_DEPTH_PROB,
    build_model as build_e9_model,
    count_all_parameters,
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


def _check_shape_report(report, expected_grids, name: str) -> None:
    if not report.compatible:
        raise RuntimeError(
            f"E9 {name} input is incompatible with Swin3D hierarchy: {report}"
        )

    observed = (
        tuple(report.stage1_grid),
        tuple(report.stage2_grid),
        tuple(report.stage3_grid),
        tuple(report.stage4_grid),
    )

    if observed != expected_grids:
        raise RuntimeError(
            f"Unexpected E9 {name} Swin grids: {observed} != {expected_grids}"
        )


def build_single_e9():
    model = build_e9_model()

    if bool(model.pretrained):
        raise RuntimeError("E9 must be scratch; model.pretrained must be False.")

    if int(model.feature_dim) != int(FEATURE_DIM) or int(FEATURE_DIM) != 768:
        raise RuntimeError(
            f"Unexpected E9 feature dimension: {model.feature_dim}."
        )

    patch_conv = model.backbone.patch_embed.proj

    if not isinstance(patch_conv, nn.Conv3d):
        raise RuntimeError("E9 patch embedding must be Conv3d.")

    if patch_conv.in_channels != INPUT_CHANNELS or INPUT_CHANNELS != 1:
        raise RuntimeError(
            f"E9 patch embedding must have one input channel; "
            f"received {patch_conv.in_channels}."
        )

    if patch_conv.out_channels != EMBED_DIM or EMBED_DIM != 96:
        raise RuntimeError(
            f"Unexpected E9 embedding dimension: {patch_conv.out_channels}."
        )

    if tuple(patch_conv.kernel_size) != tuple(PATCH_SIZE):
        raise RuntimeError(
            f"Unexpected E9 patch kernel: {patch_conv.kernel_size}."
        )

    if tuple(patch_conv.stride) != tuple(PATCH_SIZE):
        raise RuntimeError(
            f"Unexpected E9 patch stride: {patch_conv.stride}."
        )

    if tuple(DEPTHS) != (2, 2, 6, 2):
        raise RuntimeError(f"Unexpected E9 stage depths: {DEPTHS}.")

    if tuple(NUM_HEADS) != (3, 6, 12, 24):
        raise RuntimeError(f"Unexpected E9 attention heads: {NUM_HEADS}.")

    if tuple(WINDOW_SIZE) != (8, 7, 7):
        raise RuntimeError(f"Unexpected E9 window size: {WINDOW_SIZE}.")

    if tuple(STAGE_CHANNELS) != (96, 192, 384, 768):
        raise RuntimeError(
            f"Unexpected E9 stage channels: {STAGE_CHANNELS}."
        )

    if abs(float(STOCHASTIC_DEPTH_PROB) - 0.1) > 1e-12:
        raise RuntimeError(
            f"Unexpected E9 stochastic-depth probability: "
            f"{STOCHASTIC_DEPTH_PROB}."
        )

    if model.backbone.head.in_features != 768:
        raise RuntimeError("E9 classifier must receive 768 features.")

    if model.backbone.head.out_features != 1:
        raise RuntimeError("E9 classifier must output one raw logit.")

    parameter_count = count_all_parameters(model)
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Unexpected E9 parameter count: "
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

    return model
