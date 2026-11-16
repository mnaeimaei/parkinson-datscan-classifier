"""
model07_resnet18_2p5d_attention_scratch.py

E7 — 2D ResNet-18 + learned slice attention, trained from scratch.

Architecture
------------
Input:
    [B, 1, S, H, W]

where:
    B = batch size
    1 = single-channel DaT-SPECT input
    S = deterministically selected axial slices
    H = slice height
    W = slice width

IMPORTANT:
    Slice selection is NOT performed inside this model.

    The experiment/input pipeline must deterministically select the slices
    before passing them to this network.

    Examples:

    Whole-volume input:
        use the frozen deterministic whole-volume slice-selection rule.

    ROI input:
        use all 36 ROI slices, or a separately frozen deterministic subset.

Backbone:
    Torchvision 2D ResNet-18
    BasicBlock structure:
        [2, 2, 2, 2]

    Channels:
        64 -> 128 -> 256 -> 512

Input stem:
    Native single-channel Conv2d:
        Conv2d(
            1 -> 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

Slice representation:
    Every selected axial slice is encoded independently by the SAME
    shared ResNet-18 backbone.

    Each slice:
        [1, H, W]
            ->
        ResNet-18
            ->
        512-D feature vector

Learned slice attention:
    Each 512-D slice embedding receives one learned scalar score:

        Linear(512 -> 1)

    Softmax across slices gives attention weights:

        alpha_1 ... alpha_S

    with:

        sum(alpha) = 1

Subject representation:
    Weighted sum of slice embeddings:

        subject_feature =
            sum(alpha_s * slice_feature_s)

    Result:
        [B, 512]

Classification head:
    Dropout(0.3)
    Linear(512 -> 1)

Output:
    [B, 1] raw binary-classification logit

Initialization:
    SCRATCH ONLY.

    No ImageNet weights are loaded.

E4 vs E7
--------
E4:
    same architecture
    ImageNet-1K pretrained ResNet-18 backbone

E7:
    same architecture
    randomly initialized ResNet-18 backbone

The attention module and binary classification head are initialized
identically in E4 and E7.

No sigmoid is applied inside the model.

Training:
    BCEWithLogitsLoss

Probability:
    torch.sigmoid(logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.model04_resnet18_2p5d_attention_imagenet_pretrained import (
    DROPOUT,
    FEATURE_DIM,
    ResNet18_2p5D_Attention_Base,
    ResNet18_2p5D_Attention_ImageNet_Pretrained,
)


# =====================================================================
# E7 — SCRATCH
# =====================================================================


class ResNet18_2p5D_Attention_Scratch(
    ResNet18_2p5D_Attention_Base
):
    """
    E7 — 2D ResNet-18 + learned slice attention trained from scratch.

    Architecture is exactly identical to E4.

    The only intended difference is backbone initialization:

        E4:
            ImageNet-1K pretrained

        E7:
            random/Kaiming scratch initialization
    """

    def __init__(
        self,
        dropout: float = DROPOUT,
    ) -> None:

        super().__init__(
            imagenet_pretrained=False,
            dropout=dropout,
        )


# =====================================================================
# MODEL FACTORY
# =====================================================================


def build_model(
    dropout: float = DROPOUT,
) -> ResNet18_2p5D_Attention_Scratch:
    """
    Build a fresh E7 model.

    Every call returns a newly initialized scratch model.

    No pretrained weights are loaded.
    """

    return ResNet18_2p5D_Attention_Scratch(
        dropout=dropout,
    )


# =====================================================================
# PARAMETER UTILITIES
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
# E4 / E7 ARCHITECTURE VERIFICATION
# =====================================================================


def verify_architecture_matches_e4(
    e7_model: ResNet18_2p5D_Attention_Scratch,
) -> None:
    """
    Verify that E4 and E7 have exactly the same parameter names
    and tensor shapes.

    Parameter VALUES are intentionally not compared.

    E4 values:
        ImageNet initialization

    E7 values:
        scratch initialization

    But architecture must be identical.
    """

    # -------------------------------------------------------------
    # Creating E4 here may download ImageNet weights if they are not
    # already present in the Torch cache.
    #
    # This function is intended for one-time architecture QC.
    # -------------------------------------------------------------

    e4_model = (
        ResNet18_2p5D_Attention_ImageNet_Pretrained(
            dropout=e7_model.dropout_probability,
        )
    )

    e4_state = e4_model.state_dict()
    e7_state = e7_model.state_dict()

    # -------------------------------------------------------------
    # Parameter/buffer names must be identical.
    # -------------------------------------------------------------

    if e4_state.keys() != e7_state.keys():

        e4_only = sorted(
            set(e4_state.keys())
            - set(e7_state.keys())
        )

        e7_only = sorted(
            set(e7_state.keys())
            - set(e4_state.keys())
        )

        lines = [
            "E4/E7 architecture mismatch.",
            "",
            "State-dict keys only in E4:",
        ]

        lines.extend(
            f"  - {key}"
            for key in e4_only
        )

        lines.append("")
        lines.append(
            "State-dict keys only in E7:"
        )

        lines.extend(
            f"  - {key}"
            for key in e7_only
        )

        raise RuntimeError(
            "\n".join(lines)
        )

    # -------------------------------------------------------------
    # Tensor shapes must also be identical.
    # -------------------------------------------------------------

    shape_mismatches = []

    for key in e4_state:

        e4_shape = tuple(
            e4_state[key].shape
        )

        e7_shape = tuple(
            e7_state[key].shape
        )

        if e4_shape != e7_shape:

            shape_mismatches.append(
                (
                    key,
                    e4_shape,
                    e7_shape,
                )
            )

    if shape_mismatches:

        lines = [
            "E4/E7 architecture tensor-shape mismatch:",
            "",
        ]

        for (
            key,
            e4_shape,
            e7_shape,
        ) in shape_mismatches:

            lines.append(
                f"  {key}: "
                f"E4={e4_shape}, "
                f"E7={e7_shape}"
            )

        raise RuntimeError(
            "\n".join(lines)
        )


# =====================================================================
# SCRATCH INITIALIZATION CHECK
# =====================================================================


def verify_scratch_initialization(
    model: ResNet18_2p5D_Attention_Scratch,
) -> None:
    """
    Verify the important E7 initialization contract.

    Checks:
        - no ImageNet-pretrained flag
        - one-channel stem
        - correct 512-D feature dimension
        - attention architecture
        - binary classifier architecture
    """

    if model.imagenet_pretrained:
        raise RuntimeError(
            "E7 must NOT use ImageNet pretrained weights."
        )

    # -------------------------------------------------------------
    # Single-channel stem
    # -------------------------------------------------------------

    conv1 = model.backbone.conv1

    if not isinstance(
        conv1,
        nn.Conv2d,
    ):
        raise RuntimeError(
            "E7 backbone.conv1 must be nn.Conv2d."
        )

    if conv1.in_channels != 1:
        raise RuntimeError(
            "E7 must use a single-channel stem. "
            f"Received in_channels={conv1.in_channels}."
        )

    if conv1.out_channels != 64:
        raise RuntimeError(
            "Unexpected E7 stem output channels. "
            f"Expected 64, "
            f"received {conv1.out_channels}."
        )

    if tuple(conv1.kernel_size) != (
        7,
        7,
    ):
        raise RuntimeError(
            "Unexpected E7 stem kernel size. "
            f"Received {conv1.kernel_size}."
        )

    if tuple(conv1.stride) != (
        2,
        2,
    ):
        raise RuntimeError(
            "Unexpected E7 stem stride. "
            f"Received {conv1.stride}."
        )

    # -------------------------------------------------------------
    # Original ResNet classifier must have been removed.
    # -------------------------------------------------------------

    if not isinstance(
        model.backbone.fc,
        nn.Identity,
    ):
        raise RuntimeError(
            "Original ResNet-18 classifier must be replaced "
            "with nn.Identity."
        )

    # -------------------------------------------------------------
    # Attention scorer
    # -------------------------------------------------------------

    if (
        model.slice_attention.in_features
        != FEATURE_DIM
    ):
        raise RuntimeError(
            "Slice-attention scorer must receive "
            f"{FEATURE_DIM} features."
        )

    if (
        model.slice_attention.out_features
        != 1
    ):
        raise RuntimeError(
            "Slice-attention scorer must produce "
            "one scalar per slice."
        )

    # -------------------------------------------------------------
    # Binary classifier
    # -------------------------------------------------------------

    if (
        model.classifier.in_features
        != FEATURE_DIM
    ):
        raise RuntimeError(
            "Binary classifier must receive "
            f"{FEATURE_DIM} features."
        )

    if (
        model.classifier.out_features
        != 1
    ):
        raise RuntimeError(
            "Binary classifier must produce "
            "one raw logit."
        )


# =====================================================================
# ARCHITECTURE SANITY CHECK
# =====================================================================


if __name__ == "__main__":

    # =============================================================
    # QC ONLY
    #
    # This block does NOT:
    #
    #     train
    #     create optimizer
    #     calculate loss
    #     call backward()
    #     update parameters
    #
    # It only verifies model architecture and forward behavior.
    # =============================================================

    model = build_model()

    model.eval()

    print("=" * 78)
    print(
        "MODEL 07 — RESNET18 2.5D + "
        "SLICE ATTENTION — SCRATCH"
    )
    print("=" * 78)

    print()
    print("Architecture:")

    print(
        "  Backbone           : "
        "Torchvision ResNet-18"
    )

    print(
        "  Blocks             : "
        "[2, 2, 2, 2]"
    )

    print(
        "  Channels           : "
        "64 -> 128 -> 256 -> 512"
    )

    print(
        "  Input channels     : "
        "1"
    )

    print(
        f"  Slice feature dim  : "
        f"{FEATURE_DIM}"
    )

    print(
        "  Attention scorer   : "
        "Linear(512 -> 1)"
    )

    print(
        f"  Dropout            : "
        f"{DROPOUT}"
    )

    print(
        "  Classifier         : "
        "Linear(512 -> 1)"
    )

    print()
    print("Initialization:")

    print(
        "  Pretrained         : "
        "NO"
    )

    print(
        "  Backbone           : "
        "scratch / Kaiming"
    )

    print(
        "  Grayscale stem     : "
        "scratch / Kaiming"
    )

    print(
        "  Attention          : "
        "zero-score initialization"
    )

    print(
        "  Binary head        : "
        "scratch / Kaiming"
    )

    # -------------------------------------------------------------
    # Parameter counts
    # -------------------------------------------------------------

    total_parameters = (
        count_all_parameters(
            model
        )
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
    # Verify initialization contract.
    # -------------------------------------------------------------

    verify_scratch_initialization(
        model
    )

    # -------------------------------------------------------------
    # ROI input:
    #
    # Frozen ROI:
    #     [1,36,44,44]
    #
    # Batched:
    #     [B,1,36,44,44]
    #
    # All 36 ROI slices are represented here.
    # -------------------------------------------------------------

    batch_size = 2
    number_of_slices = 36

    dummy_input = torch.zeros(
        batch_size,
        1,
        number_of_slices,
        44,
        44,
        dtype=torch.float32,
    )

    with torch.no_grad():

        slice_features = (
            model.encode_slices(
                dummy_input
            )
        )

        (
            subject_features,
            attention_weights,
        ) = model.aggregate_slices(
            slice_features
        )

        logits = model(
            dummy_input
        )

    print()
    print("Forward shapes:")

    print(
        f"  Input             : "
        f"{tuple(dummy_input.shape)}"
    )

    print(
        f"  Slice features    : "
        f"{tuple(slice_features.shape)}"
    )

    print(
        f"  Attention weights : "
        f"{tuple(attention_weights.shape)}"
    )

    print(
        f"  Subject features  : "
        f"{tuple(subject_features.shape)}"
    )

    print(
        f"  Logits            : "
        f"{tuple(logits.shape)}"
    )

    # -------------------------------------------------------------
    # Shape checks
    # -------------------------------------------------------------

    assert slice_features.shape == (
        batch_size,
        number_of_slices,
        FEATURE_DIM,
    )

    assert attention_weights.shape == (
        batch_size,
        number_of_slices,
    )

    assert subject_features.shape == (
        batch_size,
        FEATURE_DIM,
    )

    assert logits.shape == (
        batch_size,
        1,
    )

    # -------------------------------------------------------------
    # Numerical checks
    # -------------------------------------------------------------

    assert torch.isfinite(
        slice_features
    ).all()

    assert torch.isfinite(
        attention_weights
    ).all()

    assert torch.isfinite(
        subject_features
    ).all()

    assert torch.isfinite(
        logits
    ).all()

    # -------------------------------------------------------------
    # Attention weights must sum to one.
    # -------------------------------------------------------------

    attention_sums = (
        attention_weights.sum(
            dim=1
        )
    )

    assert torch.allclose(
        attention_sums,
        torch.ones_like(
            attention_sums
        ),
        atol=1e-6,
    )

    print()
    print(
        "Scratch initialization     : PASS"
    )

    print(
        "Single-channel backbone    : PASS"
    )

    print(
        "Slice encoding             : PASS"
    )

    print(
        "Learned slice attention    : PASS"
    )

    print(
        "Attention sum = 1          : PASS"
    )

    print(
        "512-D subject feature      : PASS"
    )

    print(
        "Binary classifier          : PASS"
    )

    print(
        "Forward-pass check         : PASS"
    )