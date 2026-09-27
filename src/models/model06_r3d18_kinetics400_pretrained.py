"""
model06_r3d18_kinetics400_pretrained.py

E6 — Torchvision R3D-18 pretrained on Kinetics-400.

E5 vs E6
--------
E5:
    Exact same R3D-18 architecture
    Random/Kaiming initialization

E6:
    Exact same R3D-18 architecture
    Kinetics-400 pretrained initialization

The ONLY intended difference is initialization.

Architecture
------------
Input:
    [B, 1, D, H, W]

Depth D is treated as the temporal dimension expected by R3D-18.

Stem:
    Conv3d(
        1 -> 64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(1, 3, 3),
        bias=False,
    )
    BatchNorm3d
    ReLU

Residual architecture:
    BasicBlock [2, 2, 2, 2]

Channels:
    64 -> 128 -> 256 -> 512

Convolutions:
    full 3x3x3 Conv3d

Head:
    AdaptiveAvgPool3d((1, 1, 1))
    Flatten
    Linear(512 -> 1)

Pretrained initialization
-------------------------
Torchvision:
    R3D_18_Weights.KINETICS400_V1

Original RGB stem:
    [64, 3, 3, 7, 7]

Converted grayscale stem:
    [64, 1, 3, 7, 7]

Conversion:
    W_gray = W_R + W_G + W_B

All pretrained encoder tensors other than the RGB-input adaptation are
preserved exactly.

The original Kinetics-400 classifier:
    Linear(512 -> 400)

is discarded and replaced by:
    Linear(512 -> 1)

Important preprocessing rule
----------------------------
DO NOT use the torchvision Kinetics RGB normalization/transforms.

Input must use the project's already-frozen DaT-SPECT preprocessing and
normalization.

Output
------
[B, 1] raw binary-classification logits.

Do NOT apply sigmoid inside the model.
Use BCEWithLogitsLoss during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models.video import (
    R3D_18_Weights,
    r3d_18,
)

from src.models.model05_r3d18_scratch import (
    BLOCK_STRUCTURE,
    FEATURE_DIM,
    STAGE_CHANNELS,
    STEM_IN_CHANNELS,
    STEM_KERNEL_SIZE,
    STEM_OUT_CHANNELS,
    STEM_PADDING,
    STEM_STRIDE,
    R3D18_3D_Base,
    R3D18_3D_Scratch,
)


# =====================================================================
# FROZEN PRETRAINED WEIGHTS
# =====================================================================

KINETICS400_WEIGHTS = R3D_18_Weights.KINETICS400_V1


# =====================================================================
# E6
# =====================================================================


class R3D18_3D_Kinetics400_Pretrained(
    R3D18_3D_Base
):
    """
    E6 — Torchvision R3D-18 with Kinetics-400 pretrained initialization.

    Architecture is exactly identical to E5.

    Differences from E5:
        1. Encoder initialized from Kinetics-400.
        2. RGB stem converted to one channel using RGB-weight summation.

    All other pretrained encoder tensors are retained.
    """

    def __init__(self) -> None:

        super().__init__(
            weights=KINETICS400_WEIGHTS,
            grayscale_mode="rgb_sum",
        )

        self.pretrained_source = (
            "R3D_18_Weights.KINETICS400_V1"
        )


# =====================================================================
# FACTORY
# =====================================================================


def build_model(
) -> R3D18_3D_Kinetics400_Pretrained:
    """
    Build E6.

    Returns
    -------
    R3D18_3D_Kinetics400_Pretrained
        Kinetics-400 pretrained R3D-18 with:

            RGB -> grayscale stem conversion
            Linear(512 -> 1) classification head
    """

    return R3D18_3D_Kinetics400_Pretrained()


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
    Count all model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# =====================================================================
# E5 / E6 ARCHITECTURE VERIFICATION
# =====================================================================


def verify_architecture_matches_e5(
    e6_model: R3D18_3D_Kinetics400_Pretrained,
) -> None:
    """
    Verify E5 and E6 have exactly the same parameter names and shapes.

    Parameter VALUES are intentionally different because:

        E5 = scratch
        E6 = pretrained

    But names and dimensions must be identical.
    """

    e5_model = R3D18_3D_Scratch()

    e5_state = e5_model.state_dict()
    e6_state = e6_model.state_dict()

    if e5_state.keys() != e6_state.keys():

        e5_only = sorted(
            set(e5_state) - set(e6_state)
        )

        e6_only = sorted(
            set(e6_state) - set(e5_state)
        )

        lines = [
            "E5/E6 architecture mismatch.",
            "",
            "Keys only in E5:",
        ]

        lines.extend(
            f"  {key}"
            for key in e5_only
        )

        lines.append("")
        lines.append("Keys only in E6:")

        lines.extend(
            f"  {key}"
            for key in e6_only
        )

        raise RuntimeError(
            "\n".join(lines)
        )

    shape_mismatches = []

    for key in e5_state:

        if (
            e5_state[key].shape
            != e6_state[key].shape
        ):
            shape_mismatches.append(
                (
                    key,
                    tuple(e5_state[key].shape),
                    tuple(e6_state[key].shape),
                )
            )

    if shape_mismatches:

        lines = [
            "E5/E6 architecture shape mismatch:",
            "",
        ]

        for (
            key,
            e5_shape,
            e6_shape,
        ) in shape_mismatches:

            lines.append(
                f"  {key}: "
                f"E5={e5_shape}, "
                f"E6={e6_shape}"
            )

        raise RuntimeError(
            "\n".join(lines)
        )


# =====================================================================
# PRETRAINED TRANSFER VERIFICATION
# =====================================================================


def verify_kinetics_transfer(
    e6_model: R3D18_3D_Kinetics400_Pretrained,
) -> None:
    """
    Verify E6 against an untouched official Torchvision Kinetics model.

    This is intended as a one-time sanity/QC check, not something that
    should run during every training epoch.

    Checks
    ------
    1. Official Kinetics R3D-18 loads successfully.
    2. E6 grayscale stem equals the sum of the RGB pretrained kernels.
    3. Every remaining backbone tensor is exactly preserved.
    4. Original Kinetics fc layer is excluded.
    5. E6 binary classifier has shape Linear(512 -> 1).
    """

    # -------------------------------------------------------------
    # Official untouched pretrained reference.
    # -------------------------------------------------------------

    reference = r3d_18(
        weights=KINETICS400_WEIGHTS,
    )

    reference.eval()

    reference_state = reference.state_dict()

    e6_backbone_state = (
        e6_model.backbone.state_dict()
    )

    # -------------------------------------------------------------
    # 1. Verify grayscale stem.
    #
    # Official:
    #     [64,3,3,7,7]
    #
    # Expected E6:
    #     sum RGB -> [64,1,3,7,7]
    # -------------------------------------------------------------

    rgb_stem_weight = reference_state[
        "stem.0.weight"
    ]

    expected_gray_weight = (
        rgb_stem_weight.sum(
            dim=1,
            keepdim=True,
        )
    )

    actual_gray_weight = (
        e6_backbone_state[
            "stem.0.weight"
        ]
    )

    if (
        expected_gray_weight.shape
        != actual_gray_weight.shape
    ):
        raise RuntimeError(
            "R3D RGB-to-grayscale stem shape mismatch.\n"
            f"Expected: {tuple(expected_gray_weight.shape)}\n"
            f"Actual  : {tuple(actual_gray_weight.shape)}"
        )

    if not torch.equal(
        expected_gray_weight,
        actual_gray_weight,
    ):
        raise RuntimeError(
            "R3D grayscale stem does not exactly equal "
            "W_R + W_G + W_B."
        )

    # -------------------------------------------------------------
    # 2. Verify every remaining pretrained backbone tensor.
    #
    # The following are intentionally excluded:
    #
    #     stem.0.weight
    #         because RGB was converted to grayscale
    #
    #     fc.weight
    #     fc.bias
    #         because the Kinetics classifier was removed
    # -------------------------------------------------------------

    excluded_reference_keys = {
        "stem.0.weight",
        "fc.weight",
        "fc.bias",
    }

    mismatched_keys = []
    missing_keys = []

    for (
        reference_key,
        reference_tensor,
    ) in reference_state.items():

        if (
            reference_key
            in excluded_reference_keys
        ):
            continue

        if (
            reference_key
            not in e6_backbone_state
        ):
            missing_keys.append(
                reference_key
            )
            continue

        e6_tensor = (
            e6_backbone_state[
                reference_key
            ]
        )

        if (
            reference_tensor.shape
            != e6_tensor.shape
        ):
            mismatched_keys.append(
                (
                    reference_key,
                    "shape",
                    tuple(reference_tensor.shape),
                    tuple(e6_tensor.shape),
                )
            )

            continue

        if not torch.equal(
            reference_tensor,
            e6_tensor,
        ):
            mismatched_keys.append(
                (
                    reference_key,
                    "value",
                    None,
                    None,
                )
            )

    if missing_keys:

        raise RuntimeError(
            "Pretrained R3D encoder tensors are missing:\n"
            + "\n".join(
                f"  - {key}"
                for key in missing_keys
            )
        )

    if mismatched_keys:

        lines = [
            "Pretrained R3D encoder tensors were modified unexpectedly:",
            "",
        ]

        for item in mismatched_keys:

            key = item[0]
            problem = item[1]

            if problem == "shape":

                lines.append(
                    f"  {key}: "
                    f"reference={item[2]}, "
                    f"E6={item[3]}"
                )

            else:

                lines.append(
                    f"  {key}: values differ"
                )

        raise RuntimeError(
            "\n".join(lines)
        )

    # -------------------------------------------------------------
    # 3. Confirm Kinetics head is gone.
    # -------------------------------------------------------------

    if "fc.weight" in e6_backbone_state:
        raise RuntimeError(
            "Original Kinetics fc.weight is still present "
            "inside the E6 backbone."
        )

    if "fc.bias" in e6_backbone_state:
        raise RuntimeError(
            "Original Kinetics fc.bias is still present "
            "inside the E6 backbone."
        )

    # -------------------------------------------------------------
    # 4. Verify new binary classifier.
    # -------------------------------------------------------------

    if (
        e6_model.classifier.in_features
        != FEATURE_DIM
    ):
        raise RuntimeError(
            "E6 classifier must receive 512 features."
        )

    if (
        e6_model.classifier.out_features
        != 1
    ):
        raise RuntimeError(
            "E6 classifier must produce one raw logit."
        )


# =====================================================================
# ARCHITECTURE / PRETRAINING SANITY CHECK
# =====================================================================


if __name__ == "__main__":

    # -------------------------------------------------------------
    # QC ONLY.
    #
    # NO:
    #     optimizer
    #     loss
    #     backward()
    #     training
    #     parameter updates
    #
    # Running this file may download the official Kinetics-400 weights
    # if they are not already in the Torch cache.
    # -------------------------------------------------------------

    model = build_model()

    model.eval()

    print("=" * 78)
    print(
        "MODEL 06 — TORCHVISION R3D-18 "
        "— KINETICS-400 PRETRAINED"
    )
    print("=" * 78)

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
        f"{STEM_IN_CHANNELS} -> "
        f"{STEM_OUT_CHANNELS}"
    )
    print(
        f"  Stem kernel      : "
        f"{STEM_KERNEL_SIZE}"
    )
    print(
        f"  Stem stride      : "
        f"{STEM_STRIDE}"
    )
    print(
        f"  Stem padding     : "
        f"{STEM_PADDING}"
    )
    print(
        f"  Feature dim      : "
        f"{FEATURE_DIM}"
    )

    print()
    print("Pretraining:")
    print(
        "  Source           : "
        "R3D_18_Weights.KINETICS400_V1"
    )
    print(
        "  Grayscale method : "
        "W_gray = W_R + W_G + W_B"
    )
    print(
        "  Kinetics head    : discarded"
    )
    print(
        "  New head         : Linear(512 -> 1)"
    )

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
    # Verify E5/E6 architecture identity.
    # -------------------------------------------------------------

    verify_architecture_matches_e5(
        model
    )

    # -------------------------------------------------------------
    # Verify exact Kinetics transfer.
    # -------------------------------------------------------------

    verify_kinetics_transfer(
        model
    )

    # -------------------------------------------------------------
    # Project-compatible ROI-sized synthetic input.
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
    # Contract checks.
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

    assert (
        model.backbone.stem[0].in_channels
        == 1
    )

    assert (
        model.backbone.stem[0].out_channels
        == 64
    )

    assert tuple(
        model.backbone.stem[0].kernel_size
    ) == STEM_KERNEL_SIZE

    assert tuple(
        model.backbone.stem[0].stride
    ) == STEM_STRIDE

    assert (
        model.classifier.in_features
        == FEATURE_DIM
    )

    assert (
        model.classifier.out_features
        == 1
    )

    print()
    print("Kinetics-400 weight load      : PASS")
    print("RGB -> grayscale conversion   : PASS")
    print("Remaining pretrained weights  : PASS")
    print("Original Kinetics head removed: PASS")
    print("E5/E6 architecture match      : PASS")
    print("512-D feature output          : PASS")
    print("Binary classifier             : PASS")
    print("Forward-pass check            : PASS")