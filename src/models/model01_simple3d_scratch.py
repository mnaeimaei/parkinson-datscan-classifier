"""
model01_simple3d_scratch.py

Simple3D custom compact 3D residual CNN trained from scratch.

Architecture
------------
Input:
    [B, 1, D, H, W]

Stem:
    Conv3d(1 -> 32, kernel_size=3, padding=1)
    BatchNorm3d
    ReLU
    MaxPool3d(kernel_size=2)

Residual stages:
    Stage 1: 32  -> 32,  stride=1
    Stage 2: 32  -> 64,  stride=2
    Stage 3: 64  -> 128, stride=2
    Stage 4: 128 -> 256, stride=2

Each residual block:
    Conv3d(3x3x3)
    BatchNorm3d
    ReLU
    Conv3d(3x3x3)
    BatchNorm3d
    Residual addition
    ReLU

If spatial/channel dimensions change:
    Conv3d(1x1x1) projection shortcut
    BatchNorm3d

Head:
    AdaptiveAvgPool3d(1)
    Flatten
    Dropout(p=0.3)
    Linear(256 -> 1)

Output:
    [B, 1] raw binary-classification logits

Expected trainable parameters:
    ~3.59 million
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ResidualBlock3D(nn.Module):
    """
    Basic 3D residual block with two 3x3x3 convolutions.

    A 1x1x1 projection shortcut is used when:
        - the number of channels changes, or
        - spatial downsampling is requested.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm3d(out_channels)

        # Projection shortcut when dimensions change.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class Simple3DScratch(nn.Module):
    """
    Compact 3D residual CNN for binary DaT-SPECT classification.

    Supports variable spatial input sizes, including the frozen project inputs:

        Whole volume:
            [B, 1, 160, 192, 192]

        Striatal ROI:
            [B, 1, 36, 44, 44]

    The network returns a single raw logit per subject.
    """

    feature_dim: int = 256

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must satisfy 0 <= dropout < 1, got {dropout}"
            )

        # -------------------------------------------------------------
        # Stem
        # -------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=2,
                stride=2,
            ),
        )

        # -------------------------------------------------------------
        # Residual stages
        # -------------------------------------------------------------

        # Stage 1:
        # 32 -> 32
        # no spatial downsampling
        self.stage1 = ResidualBlock3D(
            in_channels=32,
            out_channels=32,
            stride=1,
        )

        # Stage 2:
        # 32 -> 64
        # spatial downsampling by 2
        self.stage2 = ResidualBlock3D(
            in_channels=32,
            out_channels=64,
            stride=2,
        )

        # Stage 3:
        # 64 -> 128
        # spatial downsampling by 2
        self.stage3 = ResidualBlock3D(
            in_channels=64,
            out_channels=128,
            stride=2,
        )

        # Stage 4:
        # 128 -> 256
        # spatial downsampling by 2
        self.stage4 = ResidualBlock3D(
            in_channels=128,
            out_channels=256,
            stride=2,
        )

        # -------------------------------------------------------------
        # Classification head
        # -------------------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=1)

        self.dropout = nn.Dropout(p=dropout)

        self.classifier = nn.Linear(
            in_features=self.feature_dim,
            out_features=1,
        )

        # Scratch initialization.
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize all model parameters from scratch.

        Conv3d:
            Kaiming/He normal initialization.

        BatchNorm3d:
            scale = 1
            bias  = 0

        Linear:
            Kaiming/He normal initialization.
        """

        for module in self.modules():

            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm3d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="linear",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Run the feature extractor.

        Parameters
        ----------
        x:
            Tensor with shape:

                [B, 1, D, H, W]

        Returns
        -------
        Tensor:
            Subject-level feature representation with shape:

                [B, 256]

        This method is intentionally exposed so that the same encoder can
        later be reused for Scenario C whole-volume + ROI shared-encoder
        fusion.
        """

        self._validate_input(x)

        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.global_pool(x)

        # [B, 256, 1, 1, 1] -> [B, 256]
        x = torch.flatten(x, start_dim=1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            [B, 1, D, H, W]

        Returns
        -------
        Tensor:
            Raw binary-classification logits:

                [B, 1]

        Do NOT apply sigmoid here when training with
        BCEWithLogitsLoss.
        """

        features = self.forward_features(x)

        features = self.dropout(features)

        logits = self.classifier(features)

        return logits

    @staticmethod
    def _validate_input(x: Tensor) -> None:
        """
        Basic model-input contract validation.
        """

        if x.ndim != 5:
            raise ValueError(
                "Simple3DScratch expects a 5D tensor with shape "
                "[B, C, D, H, W], "
                f"but received shape {tuple(x.shape)}."
            )

        if x.shape[1] != 1:
            raise ValueError(
                "Simple3DScratch expects exactly one input channel, "
                f"but received C={x.shape[1]}."
            )


def build_model(
    dropout: float = 0.3,
) -> Simple3DScratch:
    """
    Factory function used by experiment scripts.

    Returns
    -------
    Simple3DScratch
        Newly initialized model trained from scratch.
    """

    return Simple3DScratch(dropout=dropout)


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    # -------------------------------------------------------------
    # Lightweight architecture sanity check only.
    #
    # This is NOT training.
    # No optimizer.
    # No loss.
    # No backward pass.
    # -------------------------------------------------------------

    model = build_model()

    model.eval()

    total_params = count_trainable_parameters(model)

    print("=" * 70)
    print("MODEL 01 — SIMPLE3D SCRATCH")
    print("=" * 70)

    print(f"Trainable parameters: {total_params:,}")

    # Small synthetic volume for architecture verification.
    # We deliberately avoid allocating a full 160x192x192 tensor here.
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

    print(f"Input shape        : {tuple(dummy_input.shape)}")
    print(f"Feature shape      : {tuple(features.shape)}")
    print(f"Output/logit shape : {tuple(logits.shape)}")

    assert features.shape == (2, 256)
    assert logits.shape == (2, 1)

    print("Forward-pass check : PASS")