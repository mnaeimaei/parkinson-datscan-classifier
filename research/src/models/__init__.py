"""Model package for the DaT-SPECT classification project."""

from src.models.dual_input_fusion import (
    DualInputFeatureFusionClassifier,
    build_dual_input_fusion,
)

__all__ = [
    "DualInputFeatureFusionClassifier",
    "build_dual_input_fusion",
]
