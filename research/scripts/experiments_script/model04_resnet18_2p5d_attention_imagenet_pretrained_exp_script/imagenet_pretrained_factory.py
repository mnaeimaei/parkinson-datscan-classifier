#!/usr/bin/env python3
"""
imagenet_pretrained_factory.py

Thin E4 experiment-layer factory.

The architecture and actual ImageNet initialization remain authoritative in:
    src.models.model04_resnet18_2p5d_attention_imagenet_pretrained

This helper only verifies the initialization report before a fold starts.
It does NOT apply torchvision/ImageNet preprocessing transforms.
"""

from __future__ import annotations

from src.models.model04_resnet18_2p5d_attention_imagenet_pretrained import (
    FEATURE_DIM,
    build_model as build_e4_model,
)


def build_single_e4():
    try:
        model = build_e4_model()
    except Exception as exc:
        raise RuntimeError(
            "E4 could not construct the ImageNet-pretrained ResNet-18. "
            "Torchvision must be able to load "
            "ResNet18_Weights.IMAGENET1K_V1 (normally from the local Torch "
            "cache or by downloading it if the HPC node has network access). "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    report = model.initialization_report

    if not report.pretrained:
        raise RuntimeError(
            "E4 initialization report says pretrained=False."
        )

    if report.weight_name != "ResNet18_Weights.IMAGENET1K_V1":
        raise RuntimeError(
            "Unexpected E4 ImageNet weight identifier: "
            f"{report.weight_name}"
        )

    if tuple(report.original_conv1_shape) != (64, 3, 7, 7):
        raise RuntimeError(
            "Unexpected original ImageNet conv1 shape: "
            f"{report.original_conv1_shape}"
        )

    if tuple(report.grayscale_conv1_shape) != (64, 1, 7, 7):
        raise RuntimeError(
            "Unexpected grayscale conv1 shape: "
            f"{report.grayscale_conv1_shape}"
        )

    if report.grayscale_method != "RGB_SUM":
        raise RuntimeError(
            "E4 grayscale conversion must use RGB_SUM, received "
            f"{report.grayscale_method!r}."
        )

    if int(model.feature_dim) != int(FEATURE_DIM) or int(FEATURE_DIM) != 512:
        raise RuntimeError(
            f"Unexpected E4 feature dimension: {model.feature_dim}"
        )

    return model
