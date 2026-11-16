"""
model05_r3d18_scratch.py

E5 — Torchvision R3D-18 trained from scratch.

Architecture
------------
Input:
    [B, 1, D, H, W]

Interpretation:
    D is treated as the temporal dimension expected by Torchvision R3D-18.

Stem:
    Conv3d(
        1 -> 64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(1, 3, 3),
        bias=False,
    )
    BatchNorm3d(64)
    ReLU

Residual architecture:
    BasicBlock [2, 2, 2, 2]

Channels:
    64 -> 128 -> 256 -> 512

Residual convolutions:
    full 3x3x3 Conv3d

Head:
    AdaptiveAvgPool3d((1, 1, 1))
    Flatten
    Linear(512 -> 1)

Output:
    [B, 1] raw binary-classification logit

Initialization:
    Scratch / random initialization only.
    Convolution layers use Kaiming initialization.
    No Kinetics pretrained weights are loaded.

Important
---------
No sigmoid is applied inside the model.

Training should use:
    BCEWithLogitsLoss

Probability calculation:
    probability = torch.sigmoid(logit)

E5 vs E6
--------
E5:
    exact R3D-18 architecture
    scratch initialization

E6:
    exact same architecture
    Kinetics-400 pretrained initialization

Therefore E6 should reuse the base architecture defined here.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models.video import (
    R3D_18_Weights,
    r3d_18,
)


# =====================================================================
# FROZEN ARCHITECTURE CONSTANTS
# =====================================================================

FEATURE_DIM = 512

STEM_IN_CHANNELS = 1
STEM_OUT_CHANNELS = 64

STEM_KERNEL_SIZE = (3, 7, 7)
STEM_STRIDE = (1, 2, 2)
STEM_PADDING = (1, 3, 3)

BLOCK_STRUCTURE = [2, 2, 2, 2]

STAGE_CHANNELS = [64, 128, 256, 512]


# =====================================================================
# SHARED E5 / E6 ARCHITECTURE
# =====================================================================


class R3D18_3D_Base(nn.Module):
    """
    Shared architecture for E5 and E6.

    E5:
        weights=None
        scratch=True

    E6:
        Kinetics-400 initialization

    This base class ensures that E5 and E6 use the same model topology.
    """

    feature_dim: int = FEATURE_DIM

    def __init__(
        self,
        *,
        weights: Optional[R3D_18_Weights] = None,
        grayscale_mode: str = "scratch",
    ) -> None:
        super().__init__()

        if grayscale_mode not in {
            "scratch",
            "rgb_sum",
        }:
            raise ValueError(
                "grayscale_mode must be one of "
                "{'scratch', 'rgb_sum'}, "
                f"received {grayscale_mode!r}."
            )

        if weights is None and grayscale_mode == "rgb_sum":
            raise ValueError(
                "grayscale_mode='rgb_sum' requires pretrained RGB weights."
            )

        # -------------------------------------------------------------
        # 1. Build native Torchvision R3D-18
        # -------------------------------------------------------------

        backbone = r3d_18(
            weights=weights,
        )

        # -------------------------------------------------------------
        # 2. Verify expected native R3D stem
        # -------------------------------------------------------------
        #
        # Torchvision R3D-18 originally expects:
        #
        #     Conv3d(
        #         3 -> 64,
        #         kernel=(3,7,7),
        #         stride=(1,2,2),
        #         padding=(1,3,3)
        #     )
        #
        # -------------------------------------------------------------

        original_conv = backbone.stem[0]

        if not isinstance(original_conv, nn.Conv3d):
            raise RuntimeError(
                "Unexpected R3D-18 stem. "
                "Expected stem[0] to be nn.Conv3d."
            )

        if original_conv.in_channels != 3:
            raise RuntimeError(
                "Unexpected R3D-18 input channels. "
                f"Expected 3, received {original_conv.in_channels}."
            )

        if original_conv.out_channels != STEM_OUT_CHANNELS:
            raise RuntimeError(
                "Unexpected R3D-18 stem output channels. "
                f"Expected {STEM_OUT_CHANNELS}, "
                f"received {original_conv.out_channels}."
            )

        if tuple(original_conv.kernel_size) != STEM_KERNEL_SIZE:
            raise RuntimeError(
                "Unexpected R3D-18 stem kernel. "
                f"Expected {STEM_KERNEL_SIZE}, "
                f"received {tuple(original_conv.kernel_size)}."
            )

        if tuple(original_conv.stride) != STEM_STRIDE:
            raise RuntimeError(
                "Unexpected R3D-18 stem stride. "
                f"Expected {STEM_STRIDE}, "
                f"received {tuple(original_conv.stride)}."
            )

        if tuple(original_conv.padding) != STEM_PADDING:
            raise RuntimeError(
                "Unexpected R3D-18 stem padding. "
                f"Expected {STEM_PADDING}, "
                f"received {tuple(original_conv.padding)}."
            )

        # Keep RGB weights temporarily when E6 uses pretrained weights.
        original_rgb_weight = (
            original_conv.weight.detach().clone()
            if weights is not None
            else None
        )

        # -------------------------------------------------------------
        # 3. Replace RGB stem with native single-channel stem
        # -------------------------------------------------------------

        grayscale_conv = nn.Conv3d(
            in_channels=STEM_IN_CHANNELS,
            out_channels=STEM_OUT_CHANNELS,
            kernel_size=STEM_KERNEL_SIZE,
            stride=STEM_STRIDE,
            padding=STEM_PADDING,
            bias=False,
        )

        if grayscale_mode == "scratch":

            # ---------------------------------------------------------
            # E5:
            #
            # Completely new single-channel stem.
            # ---------------------------------------------------------

            nn.init.kaiming_normal_(
                grayscale_conv.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

        elif grayscale_mode == "rgb_sum":

            # ---------------------------------------------------------
            # Reserved for E6.
            #
            # Convert pretrained RGB weights:
            #
            # W_gray = W_R + W_G + W_B
            # ---------------------------------------------------------

            if original_rgb_weight is None:
                raise RuntimeError(
                    "RGB pretrained weights are unavailable."
                )

            grayscale_weight = original_rgb_weight.sum(
                dim=1,
                keepdim=True,
            )

            if (
                grayscale_weight.shape
                != grayscale_conv.weight.shape
            ):
                raise RuntimeError(
                    "RGB-to-grayscale R3D stem conversion produced "
                    "an unexpected tensor shape."
                )

            with torch.no_grad():
                grayscale_conv.weight.copy_(
                    grayscale_weight
                )

        backbone.stem[0] = grayscale_conv

        # -------------------------------------------------------------
        # 4. Verify feature dimension
        # -------------------------------------------------------------

        if not isinstance(backbone.fc, nn.Linear):
            raise RuntimeError(
                "Unexpected R3D-18 classifier type."
            )

        if backbone.fc.in_features != FEATURE_DIM:
            raise RuntimeError(
                "Unexpected R3D-18 feature dimension. "
                f"Expected {FEATURE_DIM}, "
                f"received {backbone.fc.in_features}."
            )

        # -------------------------------------------------------------
        # 5. Remove original Kinetics classifier
        #
        # Native backbone:
        #
        #     AdaptiveAvgPool3d
        #          ↓
        #       Flatten
        #          ↓
        #      fc(512 -> classes)
        #
        # Replace fc with Identity so:
        #
        #     backbone(x) -> [B,512]
        # -------------------------------------------------------------

        backbone.fc = nn.Identity()

        self.backbone = backbone

        # -------------------------------------------------------------
        # 6. Binary classification head
        # -------------------------------------------------------------

        self.classifier = nn.Linear(
            in_features=FEATURE_DIM,
            out_features=1,
            bias=True,
        )

        nn.init.kaiming_normal_(
            self.classifier.weight,
            mode="fan_in",
            nonlinearity="linear",
        )

        nn.init.zeros_(
            self.classifier.bias
        )

        self.pretrained = weights is not None
        self.grayscale_mode = grayscale_mode

    # =================================================================
    # INPUT CONTRACT
    # =================================================================

    @staticmethod
    def _validate_input(
        x: Tensor,
    ) -> None:
        """
        Validate model input.

        Expected:
            [B, 1, D, H, W]
        """

        if not isinstance(x, Tensor):
            raise TypeError(
                "Input must be torch.Tensor, "
                f"received {type(x).__name__}."
            )

        if x.ndim != 5:
            raise ValueError(
                "R3D-18 expects a 5D tensor with shape "
                "[B, C, D, H, W], "
                f"received {tuple(x.shape)}."
            )

        if x.shape[0] < 1:
            raise ValueError(
                "Batch size must be >= 1."
            )

        if x.shape[1] != 1:
            raise ValueError(
                "R3D-18 DaT model expects exactly one input channel, "
                f"received C={x.shape[1]}."
            )

        if min(x.shape[2:]) < 1:
            raise ValueError(
                "All spatial/depth dimensions must be positive, "
                f"received {tuple(x.shape[2:])}."
            )

        if not torch.is_floating_point(x):
            raise TypeError(
                "R3D-18 input must be floating point, "
                f"received dtype={x.dtype}."
            )

    # =================================================================
    # FEATURE EXTRACTION
    # =================================================================

    def forward_features(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Extract one 512-D representation per subject.

        Parameters
        ----------
        x:
            [B, 1, D, H, W]

        Returns
        -------
        Tensor:
            [B, 512]
        """

        self._validate_input(x)

        features = self.backbone(x)

        if features.ndim != 2:
            raise RuntimeError(
                "Unexpected R3D-18 feature tensor rank. "
                f"Expected [B,{FEATURE_DIM}], "
                f"received {tuple(features.shape)}."
            )

        if features.shape[1] != FEATURE_DIM:
            raise RuntimeError(
                "Unexpected R3D-18 feature dimension. "
                f"Expected {FEATURE_DIM}, "
                f"received {features.shape[1]}."
            )

        return features

    # =================================================================
    # CLASSIFICATION
    # =================================================================

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Forward classification pass.

        Parameters
        ----------
        x:
            [B, 1, D, H, W]

        Returns
        -------
        Tensor:
            Raw binary logit:

                [B, 1]

        No sigmoid is applied here.
        """

        features = self.forward_features(x)

        logits = self.classifier(
            features
        )

        return logits


# =====================================================================
# E5 — SCRATCH
# =====================================================================


class R3D18_3D_Scratch(
    R3D18_3D_Base
):
    """
    E5 — Torchvision R3D-18 trained completely from scratch.

    No Kinetics weights are loaded.

    The first convolution is natively single-channel.
    """

    def __init__(self) -> None:
        super().__init__(
            weights=None,
            grayscale_mode="scratch",
        )


# =====================================================================
# FACTORY
# =====================================================================


def build_model() -> R3D18_3D_Scratch:
    """
    Build a fresh E5 model.

    Every call creates a newly randomized network.
    """

    return R3D18_3D_Scratch()


# =====================================================================
# PARAMETER UTILITIES
# =====================================================================


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_all_parameters(
    model: nn.Module,
) -> int:
    """
    Count all parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# =====================================================================
# ARCHITECTURE VALIDATION
# =====================================================================


if __name__ == "__main__":

    # -------------------------------------------------------------
    # Architecture sanity check only.
    #
    # NO:
    #     optimizer
    #     loss
    #     backward()
    #     training
    #     parameter update
    # -------------------------------------------------------------

    model = build_model()

    model.eval()

    print("=" * 76)
    print(
        "MODEL 05 — TORCHVISION R3D-18 — SCRATCH"
    )
    print("=" * 76)

    print()
    print("Architecture:")
    print(
        f"  Blocks           : {BLOCK_STRUCTURE}"
    )
    print(
        f"  Channels         : {STAGE_CHANNELS}"
    )
    print(
        f"  Stem channels    : "
        f"{STEM_IN_CHANNELS} -> {STEM_OUT_CHANNELS}"
    )
    print(
        f"  Stem kernel      : {STEM_KERNEL_SIZE}"
    )
    print(
        f"  Stem stride      : {STEM_STRIDE}"
    )
    print(
        f"  Stem padding     : {STEM_PADDING}"
    )
    print(
        f"  Feature dim      : {FEATURE_DIM}"
    )
    print(
        f"  Pretrained       : {model.pretrained}"
    )

    total_parameters = count_all_parameters(
        model
    )

    trainable_parameters = (
        count_trainable_parameters(
            model
        )
    )

    print()
    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )
    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # Expected approximately:
    #
    #     33,147,969
    #
    # because the original Torchvision R3D-18 has:
    #
    #     RGB input + 400-class output
    #
    # whereas this model has:
    #
    #     1-channel input + 1-class output

    # -------------------------------------------------------------
    # Small project-compatible synthetic test.
    # -------------------------------------------------------------

    dummy_input = torch.zeros(
        2,
        1,
        36,
        44,
        44,
        dtype=torch.float32,
    )

    with torch.no_grad():

        features = model.forward_features(
            dummy_input
        )

        logits = model(
            dummy_input
        )

    print()
    print(
        f"Input shape        : "
        f"{tuple(dummy_input.shape)}"
    )
    print(
        f"Feature shape      : "
        f"{tuple(features.shape)}"
    )
    print(
        f"Output/logit shape : "
        f"{tuple(logits.shape)}"
    )

    # -------------------------------------------------------------
    # Contract validation
    # -------------------------------------------------------------

    assert features.shape == (
        2,
        FEATURE_DIM,
    )

    assert logits.shape == (
        2,
        1,
    )

    assert torch.isfinite(
        features
    ).all()

    assert torch.isfinite(
        logits
    ).all()

    # Check final binary classifier.
    assert model.classifier.in_features == FEATURE_DIM
    assert model.classifier.out_features == 1

    # Check single-channel stem.
    assert model.backbone.stem[0].in_channels == 1
    assert model.backbone.stem[0].out_channels == 64

    assert tuple(
        model.backbone.stem[0].kernel_size
    ) == STEM_KERNEL_SIZE

    assert tuple(
        model.backbone.stem[0].stride
    ) == STEM_STRIDE

    print()
    print("Scratch initialization : PASS")
    print("Single-channel stem     : PASS")
    print("512-D feature output    : PASS")
    print("Binary classifier       : PASS")
    print("Forward-pass check      : PASS")