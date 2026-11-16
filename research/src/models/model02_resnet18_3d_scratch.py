"""
model02_resnet18_3d_scratch.py

MedicalNet-compatible MONAI 3D ResNet-18 trained from scratch.

Purpose
-------
E2 provides the scratch baseline for E3:

    E2 = identical architecture + random initialization
    E3 = identical architecture + MedicalNet pretrained initialization

Therefore, the architecture used here MUST also be used by
model03_resnet18_3d_medicalnet_pretrained.py.

Architecture
------------
Input:
    [B, 1, D, H, W]

Backbone:
    MONAI 3D ResNet-18

    BasicBlock structure:
        [2, 2, 2, 2]

    Channels:
        64 -> 128 -> 256 -> 512

Stem:
    Conv3d(
        1 -> 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    BatchNorm3d(64)
    ReLU
    MaxPool3d(
        kernel_size=3,
        stride=2,
        padding=1
    )

Residual stages:
    layer1:
        64 channels
        2 BasicBlocks
        stride=1

    layer2:
        128 channels
        2 BasicBlocks
        stride=2

    layer3:
        256 channels
        2 BasicBlocks
        stride=2

    layer4:
        512 channels
        2 BasicBlocks
        stride=2

MedicalNet compatibility settings:
    spatial_dims=3
    n_input_channels=1
    shortcut_type="A"
    bias_downsample=True
    feed_forward=False

Head:
    AdaptiveAvgPool3d(1)
    -> 512-D feature vector
    -> Linear(512 -> 1)

Output:
    [B, 1] raw binary-classification logit

Initialization:
    Random / Kaiming initialization only.
    NO pretrained weights are loaded in E2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from monai.networks.nets.resnet import ResNet, ResNetBlock


# ---------------------------------------------------------------------
# Frozen architecture constants
# ---------------------------------------------------------------------

FEATURE_DIM = 512

BLOCK_STRUCTURE = [2, 2, 2, 2]

BLOCK_INPLANES = [64, 128, 256, 512]

SHORTCUT_TYPE = "A"

BIAS_DOWNSAMPLE = True


class ResNet18_3D_Scratch(nn.Module):
    """
    MedicalNet-compatible MONAI 3D ResNet-18 trained from scratch.

    Expected project inputs
    -----------------------
    Scenario A — whole volume:
        [B, 1, 160, 192, 192]

    Scenario B — striatal ROI:
        [B, 1, 36, 44, 44]

    The AdaptiveAvgPool3d head allows both spatial input sizes.

    The model returns ONE raw logit per subject.

    No sigmoid is applied inside the model.
    Use BCEWithLogitsLoss during training.
    """

    feature_dim: int = FEATURE_DIM

    def __init__(self) -> None:
        super().__init__()

        # -------------------------------------------------------------
        # MONAI ResNet-18 encoder
        # -------------------------------------------------------------
        #
        # feed_forward=False removes MONAI's own final FC classifier.
        #
        # The encoder therefore returns the 512-dimensional representation
        # after global average pooling.
        #
        # These parameters MUST remain identical in E3.
        # -------------------------------------------------------------

        self.encoder = ResNet(
            block=ResNetBlock,
            layers=BLOCK_STRUCTURE,
            block_inplanes=BLOCK_INPLANES,
            spatial_dims=3,
            n_input_channels=1,

            # MedicalNet / ResNet stem
            conv1_t_size=7,
            conv1_t_stride=2,
            no_max_pool=False,

            # MedicalNet ResNet-18 compatibility
            shortcut_type=SHORTCUT_TYPE,
            bias_downsample=BIAS_DOWNSAMPLE,

            widen_factor=1.0,

            # We provide our own binary classifier below.
            feed_forward=False,

            # Irrelevant when feed_forward=False, but set explicitly.
            num_classes=1,

            # Match MedicalNet / MONAI ResNet conventions.
            act=("relu", {"inplace": True}),
            norm="batch",
        )

        # -------------------------------------------------------------
        # Binary classification head
        # -------------------------------------------------------------

        self.classifier = nn.Linear(
            in_features=self.feature_dim,
            out_features=1,
            bias=True,
        )

        # Only the new classification head needs explicit initialization.
        #
        # MONAI already initializes the ResNet convolutional layers using
        # Kaiming initialization and BatchNorm weights/biases appropriately.
        self._initialize_classifier()

    def _initialize_classifier(self) -> None:
        """
        Randomly initialize the binary classification head.

        E2 must contain no pretrained parameters.
        """

        nn.init.kaiming_normal_(
            self.classifier.weight,
            mode="fan_in",
            nonlinearity="linear",
        )

        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Extract the subject-level 512-D representation.

        Parameters
        ----------
        x:
            Tensor of shape:

                [B, 1, D, H, W]

        Returns
        -------
        Tensor:
            Feature tensor:

                [B, 512]

        This method is exposed intentionally so the same encoder can later
        be used for Scenario C whole-volume + ROI fusion.
        """

        self._validate_input(x)

        features = self.encoder(x)

        if features.ndim != 2:
            raise RuntimeError(
                "Unexpected ResNet feature shape. "
                f"Expected [B, {self.feature_dim}], "
                f"received {tuple(features.shape)}."
            )

        if features.shape[1] != self.feature_dim:
            raise RuntimeError(
                "Unexpected ResNet feature dimension. "
                f"Expected {self.feature_dim}, "
                f"received {features.shape[1]}."
            )

        return features

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform binary classification.

        Parameters
        ----------
        x:
            [B, 1, D, H, W]

        Returns
        -------
        Tensor:
            Raw logits:

                [B, 1]

        Important
        ---------
        Do NOT apply sigmoid here.

        Training:
            BCEWithLogitsLoss(logits, labels)

        Probability calculation:
            probabilities = torch.sigmoid(logits)
        """

        features = self.forward_features(x)

        logits = self.classifier(features)

        return logits

    @staticmethod
    def _validate_input(x: Tensor) -> None:
        """
        Validate the model-input contract.
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "Input must be a torch.Tensor, "
                f"received {type(x).__name__}."
            )

        if x.ndim != 5:
            raise ValueError(
                "ResNet18_3D_Scratch expects input shape "
                "[B, C, D, H, W], "
                f"but received {tuple(x.shape)}."
            )

        if x.shape[1] != 1:
            raise ValueError(
                "ResNet18_3D_Scratch requires one input channel, "
                f"but received C={x.shape[1]}."
            )


def build_model() -> ResNet18_3D_Scratch:
    """
    Build a fresh E2 model.

    Every call creates a newly randomized model.

    No MedicalNet checkpoint is loaded.
    No pretrained weights are used.
    """

    return ResNet18_3D_Scratch()


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_all_parameters(model: nn.Module) -> int:
    """
    Count all model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


if __name__ == "__main__":
    # =============================================================
    # ARCHITECTURE SANITY CHECK ONLY
    #
    # This section:
    #
    #   - does NOT train
    #   - does NOT create an optimizer
    #   - does NOT calculate a loss
    #   - does NOT call backward()
    #   - does NOT update parameters
    #
    # It only verifies the model contract.
    # =============================================================

    model = build_model()

    model.eval()

    print("=" * 72)
    print("MODEL 02 — MEDICALNET-COMPATIBLE MONAI 3D RESNET-18 — SCRATCH")
    print("=" * 72)

    print()
    print("Architecture:")
    print(f"  Blocks          : {BLOCK_STRUCTURE}")
    print(f"  Channels        : {BLOCK_INPLANES}")
    print(f"  Shortcut type   : {SHORTCUT_TYPE}")
    print(f"  Bias downsample : {BIAS_DOWNSAMPLE}")
    print(f"  Feature dim     : {FEATURE_DIM}")
    print("  Pretrained      : NO")
    print()

    total_parameters = count_all_parameters(model)
    trainable_parameters = count_trainable_parameters(model)

    print(f"Total parameters     : {total_parameters:,}")
    print(f"Trainable parameters : {trainable_parameters:,}")

    # -------------------------------------------------------------
    # Use ROI-sized dummy input for the sanity test.
    #
    # Avoid allocating the full:
    #
    #     [B, 1, 160, 192, 192]
    #
    # whole-volume tensor unnecessarily during a simple architecture
    # verification.
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
        features = model.forward_features(dummy_input)
        logits = model(dummy_input)

    print()
    print(f"Dummy input shape : {tuple(dummy_input.shape)}")
    print(f"Feature shape     : {tuple(features.shape)}")
    print(f"Logit shape       : {tuple(logits.shape)}")

    assert features.shape == (2, FEATURE_DIM)
    assert logits.shape == (2, 1)

    assert torch.isfinite(features).all()
    assert torch.isfinite(logits).all()

    print()
    print("Forward-pass check : PASS")