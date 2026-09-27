"""
model08_densenet121_3d_scratch.py

E8 — MONAI 3D DenseNet-121 trained from scratch.

Architecture
------------
Input:
    [B, 1, D, H, W]

MONAI DenseNet-121:
    spatial_dims = 3
    in_channels  = 1
    out_channels = 1

Initial stem:
    Conv3d(
        1 -> 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    BatchNorm3d
    ReLU
    MaxPool3d(
        kernel_size=3,
        stride=2,
        padding=1,
    )

DenseNet configuration:
    growth_rate = 32
    init_features = 64
    bn_size = 4
    block_config = (6, 12, 24, 16)
    dropout_prob = 0.0

Dense layer:
    BatchNorm
    ReLU
    Conv3d(1x1x1)
    BatchNorm
    ReLU
    Conv3d(3x3x3)

Transition:
    BatchNorm
    ReLU
    Conv3d(1x1x1)
    AvgPool3d(kernel_size=2, stride=2)

Feature progression:
    Stem                 :   64

    Dense block 1:
        64 + 6*32        =  256
    Transition 1:
        256 / 2          =  128

    Dense block 2:
        128 + 12*32      =  512
    Transition 2:
        512 / 2          =  256

    Dense block 3:
        256 + 24*32      = 1024
    Transition 3:
        1024 / 2         =  512

    Dense block 4:
        512 + 16*32      = 1024

Final representation:
    BatchNorm3d
    ReLU
    AdaptiveAvgPool3d(1)
    Flatten
    -> [B, 1024]

Binary head:
    Linear(1024 -> 1)

Output:
    [B, 1] raw binary-classification logit

Initialization:
    Scratch only.
    No pretrained weights.

Important
---------
No sigmoid is applied inside the model.

Training:
    BCEWithLogitsLoss

Probability:
    torch.sigmoid(logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from monai.networks.nets import DenseNet121


# =====================================================================
# FROZEN ARCHITECTURE CONSTANTS
# =====================================================================

SPATIAL_DIMS = 3

INPUT_CHANNELS = 1

INIT_FEATURES = 64

GROWTH_RATE = 32

BLOCK_CONFIG = (6, 12, 24, 16)

BN_SIZE = 4

DROPOUT_PROB = 0.0

FEATURE_DIM = 1024


# =====================================================================
# MODEL
# =====================================================================


class DenseNet121_3D_Scratch(nn.Module):
    """
    E8 — MONAI 3D DenseNet-121 trained from scratch.

    Supported project inputs
    ------------------------
    Scenario A — whole volume:

        [B, 1, 160, 192, 192]

    Scenario B — striatal ROI:

        [B, 1, 36, 44, 44]

    Adaptive global average pooling allows the same network to process
    both input sizes.

    The network returns one RAW logit per subject.
    """

    feature_dim: int = FEATURE_DIM

    def __init__(self) -> None:
        super().__init__()

        # -------------------------------------------------------------
        # Native MONAI DenseNet-121
        #
        # pretrained=False is explicit because E8 is a scratch model.
        # -------------------------------------------------------------

        self.backbone = DenseNet121(
            spatial_dims=SPATIAL_DIMS,
            in_channels=INPUT_CHANNELS,
            out_channels=1,

            init_features=INIT_FEATURES,
            growth_rate=GROWTH_RATE,
            block_config=BLOCK_CONFIG,

            bn_size=BN_SIZE,

            act=("relu", {"inplace": True}),
            norm="batch",

            dropout_prob=DROPOUT_PROB,

            pretrained=False,
        )

        # -------------------------------------------------------------
        # MONAI already creates:
        #
        # self.backbone.class_layers:
        #
        #     ReLU
        #     AdaptiveAvgPool3d(1)
        #     Flatten
        #     Linear(1024 -> 1)
        #
        # We retain that exact classifier.
        # -------------------------------------------------------------

        self._validate_architecture()

    # =================================================================
    # ARCHITECTURE VALIDATION
    # =================================================================

    def _validate_architecture(self) -> None:
        """
        Verify that the instantiated MONAI model matches the frozen
        E8 experiment definition.

        Fail loudly if a future MONAI version changes an important
        architectural assumption.
        """

        # -------------------------------------------------------------
        # Stem convolution
        # -------------------------------------------------------------

        conv0 = self.backbone.features.conv0

        if not isinstance(conv0, nn.Conv3d):
            raise RuntimeError(
                "DenseNet121 stem must be Conv3d."
            )

        if conv0.in_channels != INPUT_CHANNELS:
            raise RuntimeError(
                "Unexpected DenseNet121 input channels. "
                f"Expected {INPUT_CHANNELS}, "
                f"received {conv0.in_channels}."
            )

        if conv0.out_channels != INIT_FEATURES:
            raise RuntimeError(
                "Unexpected DenseNet121 initial features. "
                f"Expected {INIT_FEATURES}, "
                f"received {conv0.out_channels}."
            )

        if tuple(conv0.kernel_size) != (7, 7, 7):
            raise RuntimeError(
                "Unexpected DenseNet121 stem kernel. "
                f"Received {conv0.kernel_size}."
            )

        if tuple(conv0.stride) != (2, 2, 2):
            raise RuntimeError(
                "Unexpected DenseNet121 stem stride. "
                f"Received {conv0.stride}."
            )

        if tuple(conv0.padding) != (3, 3, 3):
            raise RuntimeError(
                "Unexpected DenseNet121 stem padding. "
                f"Received {conv0.padding}."
            )

        # -------------------------------------------------------------
        # Dense-block depths
        # -------------------------------------------------------------

        actual_block_config = (
            len(self.backbone.features.denseblock1),
            len(self.backbone.features.denseblock2),
            len(self.backbone.features.denseblock3),
            len(self.backbone.features.denseblock4),
        )

        if actual_block_config != BLOCK_CONFIG:
            raise RuntimeError(
                "Unexpected DenseNet121 block configuration.\n"
                f"Expected: {BLOCK_CONFIG}\n"
                f"Actual  : {actual_block_config}"
            )

        # -------------------------------------------------------------
        # Binary classifier
        # -------------------------------------------------------------

        classifier = self.backbone.class_layers.out

        if not isinstance(classifier, nn.Linear):
            raise RuntimeError(
                "DenseNet121 final classifier must be nn.Linear."
            )

        if classifier.in_features != FEATURE_DIM:
            raise RuntimeError(
                "Unexpected DenseNet121 feature dimension. "
                f"Expected {FEATURE_DIM}, "
                f"received {classifier.in_features}."
            )

        if classifier.out_features != 1:
            raise RuntimeError(
                "E8 must produce exactly one raw logit."
            )

    # =================================================================
    # INPUT VALIDATION
    # =================================================================

    @staticmethod
    def _validate_input(
        x: Tensor,
    ) -> None:
        """
        Validate E8 model input.

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
                "DenseNet121_3D_Scratch expects input shape "
                "[B, C, D, H, W], "
                f"received {tuple(x.shape)}."
            )

        if x.shape[0] < 1:
            raise ValueError(
                "Batch size must be at least 1."
            )

        if x.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                "DenseNet121_3D_Scratch expects one input channel, "
                f"received C={x.shape[1]}."
            )

        if min(x.shape[2:]) < 1:
            raise ValueError(
                "All spatial dimensions must be positive. "
                f"Received {tuple(x.shape[2:])}."
            )

        if not torch.is_floating_point(x):
            raise TypeError(
                "DenseNet input must be floating point, "
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
        Produce the 1024-D DenseNet subject representation.

        Parameters
        ----------
        x:
            [B, 1, D, H, W]

        Returns
        -------
        Tensor:
            [B, 1024]

        This method is exposed so Scenario C can later use the same
        encoder for whole-volume + ROI feature fusion.
        """

        self._validate_input(x)

        # -------------------------------------------------------------
        # DenseNet convolutional feature extractor
        # -------------------------------------------------------------

        x = self.backbone.features(x)

        # -------------------------------------------------------------
        # MONAI classifier pre-head:
        #
        # final ReLU
        # adaptive global average pooling
        # flatten
        #
        # We stop BEFORE class_layers.out.
        # -------------------------------------------------------------

        x = self.backbone.class_layers.relu(x)

        x = self.backbone.class_layers.pool(x)

        x = self.backbone.class_layers.flatten(x)

        # -------------------------------------------------------------
        # Expected:
        #     [B,1024]
        # -------------------------------------------------------------

        if x.ndim != 2:
            raise RuntimeError(
                "Unexpected DenseNet feature shape. "
                f"Expected [B,{FEATURE_DIM}], "
                f"received {tuple(x.shape)}."
            )

        if x.shape[1] != FEATURE_DIM:
            raise RuntimeError(
                "Unexpected DenseNet feature dimension. "
                f"Expected {FEATURE_DIM}, "
                f"received {x.shape[1]}."
            )

        return x

    # =================================================================
    # CLASSIFICATION
    # =================================================================

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Binary classification forward pass.

        Returns
        -------
        Tensor:
            [B,1] raw logits.

        Important
        ---------
        Do NOT apply sigmoid here.

        Training:

            loss = BCEWithLogitsLoss(logits, labels)

        Probability:

            probabilities = torch.sigmoid(logits)
        """

        features = self.forward_features(x)

        logits = self.backbone.class_layers.out(
            features
        )

        return logits


# =====================================================================
# FACTORY
# =====================================================================


def build_model(
) -> DenseNet121_3D_Scratch:
    """
    Create a newly initialized E8 model.

    No pretrained weights are loaded.
    """

    return DenseNet121_3D_Scratch()


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
# ARCHITECTURE SANITY CHECK
# =====================================================================


if __name__ == "__main__":

    # =============================================================
    # QC ONLY
    #
    # NO:
    #     optimizer
    #     loss
    #     backward()
    #     training
    #     parameter update
    # =============================================================

    model = build_model()

    model.eval()

    print("=" * 78)
    print(
        "MODEL 08 — MONAI 3D DENSENET-121 — SCRATCH"
    )
    print("=" * 78)

    print()
    print("Architecture:")

    print(
        f"  Spatial dims      : {SPATIAL_DIMS}"
    )

    print(
        f"  Input channels    : {INPUT_CHANNELS}"
    )

    print(
        f"  Initial features  : {INIT_FEATURES}"
    )

    print(
        f"  Growth rate       : {GROWTH_RATE}"
    )

    print(
        f"  Block config      : {BLOCK_CONFIG}"
    )

    print(
        f"  Bottleneck size   : {BN_SIZE}"
    )

    print(
        f"  Dense dropout     : {DROPOUT_PROB}"
    )

    print(
        f"  Feature dim       : {FEATURE_DIM}"
    )

    print(
        "  Classifier        : Linear(1024 -> 1)"
    )

    print(
        "  Pretrained        : NO"
    )

    # -------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Project ROI-sized synthetic test.
    #
    # Batch size 2 is deliberately used.
    # -------------------------------------------------------------

    batch_size = 2

    dummy_input = torch.zeros(
        batch_size,
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
    # Contract checks
    # -------------------------------------------------------------

    assert features.shape == (
        batch_size,
        FEATURE_DIM,
    )

    assert logits.shape == (
        batch_size,
        1,
    )

    assert torch.isfinite(
        features
    ).all()

    assert torch.isfinite(
        logits
    ).all()

    assert (
        model.backbone.features.conv0.in_channels
        == 1
    )

    assert (
        model.backbone.class_layers.out.in_features
        == FEATURE_DIM
    )

    assert (
        model.backbone.class_layers.out.out_features
        == 1
    )

    print()
    print("Scratch initialization : PASS")
    print("Single-channel input    : PASS")
    print("Dense blocks            : PASS")
    print("1024-D feature output   : PASS")
    print("Binary classifier       : PASS")
    print("Forward-pass check      : PASS")