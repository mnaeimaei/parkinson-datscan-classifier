#!/usr/bin/env python3
"""
scratch_2p5d_factory.py

Thin experiment-layer factory for E7.

The authoritative architecture and initialization remain in:
    src.models.model07_resnet18_2p5d_attention_scratch

E7 directly inherits the same ResNet18_2p5D_Attention_Base used by E4.
Therefore the architecture, attention module, and binary head are shared by
construction; only backbone initialization differs.

This helper:
    - constructs E7 through src,
    - calls the src scratch-initialization verifier,
    - checks the expected 512-D feature / attention / classifier contract,
    - does NOT construct E4 and therefore does NOT require ImageNet weights.
"""

from __future__ import annotations

import torch.nn as nn

from src.models.model04_resnet18_2p5d_attention_imagenet_pretrained import (
    DROPOUT,
    FEATURE_DIM,
)
from src.models.model07_resnet18_2p5d_attention_scratch import (
    build_model as build_e7_model,
    count_all_parameters,
    verify_scratch_initialization,
)


EXPECTED_PARAMETER_COUNT = 11_171_266


def build_single_e7():
    model = build_e7_model()

    verify_scratch_initialization(model)

    if bool(model.imagenet_pretrained):
        raise RuntimeError(
            "E7 must not use ImageNet pretrained weights."
        )

    if int(model.feature_dim) != int(FEATURE_DIM) or int(FEATURE_DIM) != 512:
        raise RuntimeError(
            f"Unexpected E7 feature dimension: {model.feature_dim}."
        )

    if abs(float(model.dropout_probability) - float(DROPOUT)) > 1e-12:
        raise RuntimeError(
            "Unexpected E7 dropout probability: "
            f"{model.dropout_probability}."
        )

    if not isinstance(model.backbone.conv1, nn.Conv2d):
        raise RuntimeError("E7 backbone.conv1 must be Conv2d.")

    if model.backbone.conv1.in_channels != 1:
        raise RuntimeError(
            "E7 must have a native single-channel Conv2d stem."
        )

    if tuple(model.backbone.conv1.kernel_size) != (7, 7):
        raise RuntimeError(
            f"Unexpected E7 conv1 kernel: {model.backbone.conv1.kernel_size}."
        )

    if tuple(model.backbone.conv1.stride) != (2, 2):
        raise RuntimeError(
            f"Unexpected E7 conv1 stride: {model.backbone.conv1.stride}."
        )

    if model.slice_attention.in_features != 512:
        raise RuntimeError(
            "E7 slice-attention scorer must receive 512 features."
        )

    if model.slice_attention.out_features != 1:
        raise RuntimeError(
            "E7 slice-attention scorer must emit one score per slice."
        )

    if model.classifier.in_features != 512:
        raise RuntimeError(
            "E7 classifier must receive 512 subject features."
        )

    if model.classifier.out_features != 1:
        raise RuntimeError(
            "E7 classifier must output one raw logit."
        )

    parameter_count = count_all_parameters(model)

    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            "Unexpected E7 parameter count: "
            f"{parameter_count:,} != {EXPECTED_PARAMETER_COUNT:,}."
        )

    return model
