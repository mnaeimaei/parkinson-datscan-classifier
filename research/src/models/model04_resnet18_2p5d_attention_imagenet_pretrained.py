"""
model04_resnet18_2p5d_attention_imagenet_pretrained.py

E4 — 2.5D ResNet-18 + learned slice attention
     ImageNet-1K pretrained.

Architecture
------------
Input:
    [B, 1, S, H, W]

where:
    B = subjects
    1 = grayscale DaT-SPECT channel
    S = deterministically selected axial slices
    H, W = slice spatial dimensions

Slice encoder:
    Torchvision 2D ResNet-18
    BasicBlock structure [2, 2, 2, 2]

    ImageNet-1K pretrained:
        ResNet18_Weights.IMAGENET1K_V1

RGB -> grayscale adaptation:
    Original first convolution:
        Conv2d(3 -> 64, 7x7, stride=2, padding=3)

    New first convolution:
        Conv2d(1 -> 64, 7x7, stride=2, padding=3)

    Pretrained grayscale kernel:
        W_gray = W_R + W_G + W_B

Slice representation:
    Each axial slice -> 512-D feature vector

Learned slice attention:
    512-D slice feature -> Linear(512 -> 1)
    Softmax across slices
    Weighted sum of slice embeddings

Subject representation:
    [B, 512]

Classification head:
    Dropout(0.3)
    Linear(512 -> 1)

Output:
    [B, 1] raw binary-classification logit

Important preprocessing rule
----------------------------
DO NOT use the torchvision ImageNet transforms.

In particular:
    - no ImageNet RGB mean/std normalization
    - no automatic RGB conversion
    - no ImageNet resize/crop preprocessing

The model expects images that already went through the frozen
DaT-SPECT preprocessing pipeline.

E4 vs E7
--------
E4:
    Same architecture + ImageNet-1K initialization

E7:
    Same architecture + scratch initialization

Therefore E7 should reuse the base class defined in this file or reproduce
these settings exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


# =====================================================================
# FROZEN ARCHITECTURE CONSTANTS
# =====================================================================

FEATURE_DIM = 512

DROPOUT = 0.30

IMAGENET_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


@dataclass(frozen=True)
class BackboneInitializationReport:
    """
    Information about the ResNet-18 initialization.
    """

    pretrained: bool
    weight_name: str
    original_conv1_shape: tuple[int, ...]
    grayscale_conv1_shape: tuple[int, ...]
    grayscale_method: str


class ResNet18_2p5D_Attention_Base(nn.Module):
    """
    Shared E4/E7 architecture.

    E4:
        imagenet_pretrained=True

    E7:
        imagenet_pretrained=False

    Keeping the common architecture in one class guarantees that the
    E4-vs-E7 comparison isolates initialization rather than architecture.
    """

    feature_dim: int = FEATURE_DIM

    def __init__(
        self,
        *,
        imagenet_pretrained: bool,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1, "
                f"received {dropout}."
            )

        self.imagenet_pretrained = imagenet_pretrained
        self.dropout_probability = dropout

        # -------------------------------------------------------------
        # Build 2D ResNet-18 slice encoder
        # -------------------------------------------------------------

        if imagenet_pretrained:
            weights = IMAGENET_WEIGHTS
        else:
            weights = None

        backbone = resnet18(
            weights=weights,
        )

        # -------------------------------------------------------------
        # Verify expected torchvision ResNet-18 contract.
        # -------------------------------------------------------------

        if backbone.conv1.in_channels != 3:
            raise RuntimeError(
                "Unexpected torchvision ResNet-18 conv1 input channels. "
                f"Expected 3, received {backbone.conv1.in_channels}."
            )

        if backbone.fc.in_features != FEATURE_DIM:
            raise RuntimeError(
                "Unexpected torchvision ResNet-18 feature dimension. "
                f"Expected {FEATURE_DIM}, "
                f"received {backbone.fc.in_features}."
            )

        original_conv1_weight = backbone.conv1.weight.detach().clone()

        original_conv1_shape = tuple(
            original_conv1_weight.shape
        )

        # Expected:
        # [64, 3, 7, 7]
        if original_conv1_weight.ndim != 4:
            raise RuntimeError(
                "Unexpected ResNet-18 conv1 tensor rank: "
                f"{original_conv1_weight.ndim}"
            )

        if original_conv1_weight.shape[1] != 3:
            raise RuntimeError(
                "Expected RGB pretrained conv1 with three input channels, "
                f"received shape {tuple(original_conv1_weight.shape)}."
            )

        # -------------------------------------------------------------
        # Replace RGB conv1 with single-channel conv1.
        # -------------------------------------------------------------

        grayscale_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=backbone.conv1.out_channels,
            kernel_size=backbone.conv1.kernel_size,
            stride=backbone.conv1.stride,
            padding=backbone.conv1.padding,
            dilation=backbone.conv1.dilation,
            groups=backbone.conv1.groups,
            bias=False,
            padding_mode=backbone.conv1.padding_mode,
        )

        if imagenet_pretrained:

            # ---------------------------------------------------------
            # User-defined transfer-learning policy:
            #
            # W_gray = W_R + W_G + W_B
            #
            # Shape:
            # [64, 3, 7, 7]
            #       ↓ sum over RGB dimension
            # [64, 1, 7, 7]
            # ---------------------------------------------------------

            grayscale_weight = original_conv1_weight.sum(
                dim=1,
                keepdim=True,
            )

            with torch.no_grad():
                grayscale_conv1.weight.copy_(grayscale_weight)

        else:

            # ---------------------------------------------------------
            # E7 scratch initialization.
            #
            # This ensures the grayscale stem has random Kaiming
            # initialization while preserving the exact same architecture.
            # ---------------------------------------------------------

            nn.init.kaiming_normal_(
                grayscale_conv1.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

        backbone.conv1 = grayscale_conv1

        # -------------------------------------------------------------
        # Remove original ImageNet 1000-class classifier.
        #
        # After this replacement:
        #
        #     backbone(slice) -> [N, 512]
        # -------------------------------------------------------------

        backbone.fc = nn.Identity()

        self.backbone = backbone

        # -------------------------------------------------------------
        # Learned slice-attention scorer
        #
        # One scalar attention score for every 512-D slice embedding.
        #
        # Initializing scores at zero makes the initial attention uniform
        # across slices while still allowing attention to learn.
        # -------------------------------------------------------------

        self.slice_attention = nn.Linear(
            FEATURE_DIM,
            1,
            bias=True,
        )

        nn.init.zeros_(self.slice_attention.weight)
        nn.init.zeros_(self.slice_attention.bias)

        # -------------------------------------------------------------
        # Binary classification head
        # -------------------------------------------------------------

        self.dropout = nn.Dropout(
            p=dropout,
        )

        self.classifier = nn.Linear(
            FEATURE_DIM,
            1,
            bias=True,
        )

        nn.init.kaiming_normal_(
            self.classifier.weight,
            mode="fan_in",
            nonlinearity="linear",
        )

        nn.init.zeros_(self.classifier.bias)

        self.initialization_report = BackboneInitializationReport(
            pretrained=imagenet_pretrained,
            weight_name=(
                "ResNet18_Weights.IMAGENET1K_V1"
                if imagenet_pretrained
                else "NONE"
            ),
            original_conv1_shape=original_conv1_shape,
            grayscale_conv1_shape=tuple(
                self.backbone.conv1.weight.shape
            ),
            grayscale_method=(
                "RGB_SUM"
                if imagenet_pretrained
                else "SCRATCH_KAIMING"
            ),
        )

    # =================================================================
    # INPUT VALIDATION
    # =================================================================

    @staticmethod
    def _validate_input(x: Tensor) -> None:
        """
        Validate 2.5D input.

        Expected:
            [B, 1, S, H, W]
        """

        if not isinstance(x, Tensor):
            raise TypeError(
                "Input must be a torch.Tensor, "
                f"received {type(x).__name__}."
            )

        if x.ndim != 5:
            raise ValueError(
                "2.5D model expects input shape "
                "[B, C, S, H, W], "
                f"received {tuple(x.shape)}."
            )

        batch_size, channels, slices, height, width = x.shape

        if batch_size < 1:
            raise ValueError(
                "Batch size must be >= 1."
            )

        if channels != 1:
            raise ValueError(
                "2.5D ResNet-18 expects exactly one grayscale channel, "
                f"received C={channels}."
            )

        if slices < 1:
            raise ValueError(
                "At least one axial slice is required."
            )

        if height < 1 or width < 1:
            raise ValueError(
                "Slice dimensions must be positive, "
                f"received H={height}, W={width}."
            )

        if not torch.is_floating_point(x):
            raise TypeError(
                "Input must be floating point, "
                f"received dtype={x.dtype}."
            )

    # =================================================================
    # SLICE ENCODING
    # =================================================================

    def encode_slices(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Encode every axial slice independently using the shared
        2D ResNet-18 backbone.

        Parameters
        ----------
        x:
            [B, 1, S, H, W]

        Returns
        -------
        Tensor:
            [B, S, 512]
        """

        self._validate_input(x)

        batch_size, channels, num_slices, height, width = x.shape

        # -------------------------------------------------------------
        # Convert:
        #
        # [B, 1, S, H, W]
        #
        # to:
        #
        # [B, S, 1, H, W]
        #
        # then flatten subject/slice dimensions:
        #
        # [B*S, 1, H, W]
        # -------------------------------------------------------------

        slices = x.permute(
            0,
            2,
            1,
            3,
            4,
        ).contiguous()

        slices = slices.view(
            batch_size * num_slices,
            channels,
            height,
            width,
        )

        # -------------------------------------------------------------
        # Shared ResNet-18 encoder
        #
        # [B*S,1,H,W] -> [B*S,512]
        # -------------------------------------------------------------

        slice_features = self.backbone(slices)

        if slice_features.ndim != 2:
            raise RuntimeError(
                "Unexpected ResNet-18 slice-feature rank. "
                f"Expected 2D [B*S,512], "
                f"received {tuple(slice_features.shape)}."
            )

        if slice_features.shape[1] != FEATURE_DIM:
            raise RuntimeError(
                "Unexpected ResNet-18 feature dimension. "
                f"Expected {FEATURE_DIM}, "
                f"received {slice_features.shape[1]}."
            )

        # -------------------------------------------------------------
        # Restore subject grouping:
        #
        # [B*S,512] -> [B,S,512]
        # -------------------------------------------------------------

        slice_features = slice_features.view(
            batch_size,
            num_slices,
            FEATURE_DIM,
        )

        return slice_features

    # =================================================================
    # LEARNED SLICE ATTENTION
    # =================================================================

    def aggregate_slices(
        self,
        slice_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply learned attention across slice embeddings.

        Parameters
        ----------
        slice_features:
            [B, S, 512]

        Returns
        -------
        subject_features:
            [B, 512]

        attention_weights:
            [B, S]
        """

        if slice_features.ndim != 3:
            raise ValueError(
                "slice_features must have shape [B,S,F], "
                f"received {tuple(slice_features.shape)}."
            )

        if slice_features.shape[-1] != FEATURE_DIM:
            raise ValueError(
                "Unexpected feature dimension. "
                f"Expected {FEATURE_DIM}, "
                f"received {slice_features.shape[-1]}."
            )

        # -------------------------------------------------------------
        # Score every slice independently:
        #
        # [B,S,512]
        #     ↓
        # Linear(512 -> 1)
        #     ↓
        # [B,S,1]
        #     ↓
        # [B,S]
        # -------------------------------------------------------------

        attention_logits = self.slice_attention(
            slice_features
        ).squeeze(-1)

        # -------------------------------------------------------------
        # Convert scores into weights that sum to one per subject.
        # -------------------------------------------------------------

        attention_weights = torch.softmax(
            attention_logits,
            dim=1,
        )

        # -------------------------------------------------------------
        # Weighted subject representation:
        #
        # subject =
        #     Σ attention_weight_s * feature_s
        #
        # [B,S,512] -> [B,512]
        # -------------------------------------------------------------

        subject_features = torch.sum(
            slice_features
            * attention_weights.unsqueeze(-1),
            dim=1,
        )

        return (
            subject_features,
            attention_weights,
        )

    # =================================================================
    # SUBJECT FEATURE EXTRACTION
    # =================================================================

    def forward_features(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Produce one 512-D feature vector per subject.

        Parameters
        ----------
        x:
            [B,1,S,H,W]

        Returns
        -------
        Tensor:
            [B,512]
        """

        slice_features = self.encode_slices(x)

        subject_features, _ = self.aggregate_slices(
            slice_features
        )

        return subject_features

    # =================================================================
    # STANDARD FORWARD
    # =================================================================

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Standard classification forward pass.

        Returns
        -------
        Tensor:
            [B,1] raw logits

        No sigmoid is applied here.
        """

        subject_features = self.forward_features(x)

        subject_features = self.dropout(
            subject_features
        )

        logits = self.classifier(
            subject_features
        )

        return logits

    # =================================================================
    # INTERPRETABLE FORWARD
    # =================================================================

    def forward_with_attention(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass that also exposes learned slice weights.

        Useful for:
            - visual QC
            - interpretation
            - verifying which slices contribute most

        Returns
        -------
        logits:
            [B,1]

        attention_weights:
            [B,S]
        """

        slice_features = self.encode_slices(x)

        subject_features, attention_weights = (
            self.aggregate_slices(slice_features)
        )

        logits = self.classifier(
            self.dropout(subject_features)
        )

        return (
            logits,
            attention_weights,
        )


# =====================================================================
# E4 — IMAGENET-PRETRAINED MODEL
# =====================================================================


class ResNet18_2p5D_Attention_ImageNet_Pretrained(
    ResNet18_2p5D_Attention_Base
):
    """
    E4.

    2D ResNet-18 + learned slice attention.

    Backbone:
        ImageNet-1K pretrained.

    First RGB convolution:
        converted to one channel using:

            W_gray = W_R + W_G + W_B
    """

    def __init__(
        self,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__(
            imagenet_pretrained=True,
            dropout=dropout,
        )


# =====================================================================
# FACTORY
# =====================================================================


def build_model(
    dropout: float = DROPOUT,
) -> ResNet18_2p5D_Attention_ImageNet_Pretrained:
    """
    Build E4.
    """

    return ResNet18_2p5D_Attention_ImageNet_Pretrained(
        dropout=dropout,
    )


# =====================================================================
# UTILITIES
# =====================================================================


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable model parameters.
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
    Count all model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# =====================================================================
# ARCHITECTURE SANITY CHECK
# =====================================================================


if __name__ == "__main__":

    # -------------------------------------------------------------
    # This block performs architecture validation only.
    #
    # NO:
    #     optimizer
    #     loss
    #     backward
    #     training
    #     parameter update
    # -------------------------------------------------------------

    model = build_model()

    model.eval()

    print("=" * 78)
    print(
        "MODEL 04 — RESNET18 2.5D + SLICE ATTENTION "
        "— IMAGENET PRETRAINED"
    )
    print("=" * 78)

    report = model.initialization_report

    print()
    print("Backbone initialization:")
    print(
        f"  Pretrained          : {report.pretrained}"
    )
    print(
        f"  Weights             : {report.weight_name}"
    )
    print(
        f"  Original conv1      : {report.original_conv1_shape}"
    )
    print(
        f"  Grayscale conv1     : {report.grayscale_conv1_shape}"
    )
    print(
        f"  Grayscale method    : {report.grayscale_method}"
    )

    print()
    print("Architecture:")
    print(f"  Slice feature dim   : {FEATURE_DIM}")
    print("  Attention scorer    : Linear(512 -> 1)")
    print(f"  Dropout             : {DROPOUT}")
    print("  Classifier          : Linear(512 -> 1)")

    print()
    print(
        f"Total parameters     : "
        f"{count_all_parameters(model):,}"
    )
    print(
        f"Trainable parameters : "
        f"{count_trainable_parameters(model):,}"
    )

    # -------------------------------------------------------------
    # ROI example:
    #
    # Step-8 ROI:
    #     [1,36,44,44]
    #
    # Batched:
    #     [B,1,36,44,44]
    #
    # Using batch=2 here.
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

        slice_features = model.encode_slices(
            dummy_input
        )

        subject_features = model.forward_features(
            dummy_input
        )

        logits, attention_weights = (
            model.forward_with_attention(
                dummy_input
            )
        )

    print()
    print(
        f"Input shape             : "
        f"{tuple(dummy_input.shape)}"
    )

    print(
        f"Slice-feature shape     : "
        f"{tuple(slice_features.shape)}"
    )

    print(
        f"Subject-feature shape   : "
        f"{tuple(subject_features.shape)}"
    )

    print(
        f"Attention-weight shape  : "
        f"{tuple(attention_weights.shape)}"
    )

    print(
        f"Logit shape             : "
        f"{tuple(logits.shape)}"
    )

    # -------------------------------------------------------------
    # Contract checks
    # -------------------------------------------------------------

    assert slice_features.shape == (
        2,
        36,
        FEATURE_DIM,
    )

    assert subject_features.shape == (
        2,
        FEATURE_DIM,
    )

    assert attention_weights.shape == (
        2,
        36,
    )

    assert logits.shape == (
        2,
        1,
    )

    assert torch.isfinite(
        slice_features
    ).all()

    assert torch.isfinite(
        subject_features
    ).all()

    assert torch.isfinite(
        attention_weights
    ).all()

    assert torch.isfinite(
        logits
    ).all()

    # Attention must sum to one for every subject.
    attention_sum = attention_weights.sum(
        dim=1
    )

    assert torch.allclose(
        attention_sum,
        torch.ones_like(attention_sum),
        atol=1e-6,
    )

    print()
    print("ImageNet backbone load      : PASS")
    print("RGB -> grayscale conversion : PASS")
    print("Slice encoding              : PASS")
    print("Slice attention             : PASS")
    print("Attention sum = 1           : PASS")
    print("Forward-pass check          : PASS")