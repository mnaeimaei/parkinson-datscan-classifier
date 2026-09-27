"""
model10_swin3d_t_kinetics400_pretrained.py

E10 — Torchvision Swin3D-Tiny pretrained on Kinetics-400.

E9 vs E10
----------
E9:
    Exact same Swin3D-Tiny architecture
    Scratch initialization

E10:
    Exact same Swin3D-Tiny architecture
    Kinetics-400 pretrained initialization

The ONLY intended difference is initialization.

Architecture
------------
Input:
    [B, 1, D, H, W]

Patch embedding:
    Conv3d(
        1 -> 96,
        kernel_size=(2, 4, 4),
        stride=(2, 4, 4),
    )
    LayerNorm(96)

Swin stages:
    depths:
        [2, 2, 6, 2]

    attention heads:
        [3, 6, 12, 24]

    window size:
        (8, 7, 7)

    feature dimensions:
        96 -> 192 -> 384 -> 768

Final representation:
    LayerNorm(768)
    AdaptiveAvgPool3d(1)
    Flatten
    -> [B, 768]

Binary classifier:
    Linear(768 -> 1)

Pretrained initialization
-------------------------
Torchvision:
    Swin3D_T_Weights.KINETICS400_V1

Original RGB patch embedding:
    [96, 3, 2, 4, 4]

Converted grayscale patch embedding:
    [96, 1, 2, 4, 4]

Conversion:
    W_gray = W_R + W_G + W_B

The pretrained patch-embedding bias is preserved.

All remaining pretrained Swin3D encoder tensors are retained unchanged.

Original Kinetics classifier:
    Linear(768 -> 400)

is discarded and replaced by:
    Linear(768 -> 1)

Important preprocessing rule
----------------------------
DO NOT use the torchvision Kinetics transforms.

Specifically:
    - no Kinetics RGB mean/std normalization
    - no RGB conversion
    - no Kinetics resize/crop preprocessing

Input must use the project's frozen DaT-SPECT preprocessing and
normalization.

Output
------
[B, 1] raw binary-classification logit.

Do NOT apply sigmoid inside the model.

Training:
    BCEWithLogitsLoss

Probability:
    torch.sigmoid(logits)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models.video import (
    Swin3D_T_Weights,
    swin3d_t,
)

from src.models.model09_swin3d_t_scratch import (
    DEPTHS,
    EMBED_DIM,
    FEATURE_DIM,
    INPUT_CHANNELS,
    NUM_HEADS,
    PATCH_SIZE,
    STAGE_CHANNELS,
    STOCHASTIC_DEPTH_PROB,
    WINDOW_SIZE,
    Swin3D_T_Base,
    Swin3D_T_Scratch,
)


# =====================================================================
# FROZEN PRETRAINED WEIGHT SET
# =====================================================================

KINETICS400_WEIGHTS = (
    Swin3D_T_Weights.KINETICS400_V1
)


# =====================================================================
# INITIALIZATION REPORT
# =====================================================================


@dataclass(frozen=True)
class PretrainedInitializationReport:
    """
    Summary of E10 pretrained initialization.
    """

    pretrained_source: str

    original_patch_shape: tuple[int, ...]
    grayscale_patch_shape: tuple[int, ...]

    grayscale_method: str

    feature_dim: int

    original_classes: int
    target_classes: int


# =====================================================================
# E10
# =====================================================================


class Swin3D_T_Kinetics400_Pretrained(
    Swin3D_T_Base
):
    """
    E10 — Torchvision Swin3D-Tiny with Kinetics-400 pretraining.

    Architecture is exactly identical to E9.

    The only intended E9/E10 difference is initialization.
    """

    feature_dim: int = FEATURE_DIM

    def __init__(self) -> None:

        # -------------------------------------------------------------
        # IMPORTANT
        #
        # We intentionally do NOT call:
        #
        #     Swin3D_T_Base.__init__()
        #
        # because E9's base constructor creates a scratch backbone.
        #
        # Instead, initialize nn.Module itself and then construct the
        # exact same Torchvision architecture using pretrained weights.
        #
        # All forward/input/shape-analysis methods are inherited from
        # Swin3D_T_Base.
        # -------------------------------------------------------------

        nn.Module.__init__(self)

        # -------------------------------------------------------------
        # 1. Load official Kinetics-400 Swin3D-Tiny.
        #
        # Do NOT pass num_classes=1 here.
        #
        # When pretrained weights are supplied, Torchvision constructs
        # the original 400-class Kinetics model first.
        # -------------------------------------------------------------

        backbone = swin3d_t(
            weights=KINETICS400_WEIGHTS,
        )

        # -------------------------------------------------------------
        # 2. Verify native Torchvision architecture.
        #
        # This inherited E9 check validates:
        #
        #     RGB input
        #     patch size
        #     embed dim
        #     final feature dim
        # -------------------------------------------------------------

        self._validate_native_backbone(
            backbone
        )

        # -------------------------------------------------------------
        # 3. Verify original classifier contract.
        # -------------------------------------------------------------

        if not isinstance(
            backbone.head,
            nn.Linear,
        ):
            raise RuntimeError(
                "Unexpected pretrained Swin3D classifier type."
            )

        if (
            backbone.head.in_features
            != FEATURE_DIM
        ):
            raise RuntimeError(
                "Unexpected pretrained Swin3D feature dimension. "
                f"Expected {FEATURE_DIM}, "
                f"received {backbone.head.in_features}."
            )

        original_number_of_classes = (
            backbone.head.out_features
        )

        if original_number_of_classes != 400:
            raise RuntimeError(
                "Unexpected Kinetics classifier output dimension. "
                f"Expected 400, "
                f"received {original_number_of_classes}."
            )

        # -------------------------------------------------------------
        # 4. Get pretrained RGB patch embedding.
        #
        # Original:
        #
        #     [96, 3, 2, 4, 4]
        # -------------------------------------------------------------

        original_patch_conv = (
            backbone.patch_embed.proj
        )

        if not isinstance(
            original_patch_conv,
            nn.Conv3d,
        ):
            raise RuntimeError(
                "Expected Swin3D patch_embed.proj "
                "to be nn.Conv3d."
            )

        original_rgb_weight = (
            original_patch_conv
            .weight
            .detach()
            .clone()
        )

        original_patch_shape = tuple(
            original_rgb_weight.shape
        )

        expected_rgb_shape = (
            EMBED_DIM,
            3,
            *PATCH_SIZE,
        )

        if (
            original_patch_shape
            != expected_rgb_shape
        ):
            raise RuntimeError(
                "Unexpected pretrained Swin3D RGB patch embedding.\n"
                f"Expected: {expected_rgb_shape}\n"
                f"Actual  : {original_patch_shape}"
            )

        # Preserve pretrained bias as well.
        original_patch_bias = None

        if original_patch_conv.bias is not None:

            original_patch_bias = (
                original_patch_conv
                .bias
                .detach()
                .clone()
            )

        # -------------------------------------------------------------
        # 5. Create EXACT same patch embedding except input channels = 1.
        # -------------------------------------------------------------

        grayscale_patch_conv = nn.Conv3d(
            in_channels=INPUT_CHANNELS,
            out_channels=original_patch_conv.out_channels,
            kernel_size=original_patch_conv.kernel_size,
            stride=original_patch_conv.stride,
            padding=original_patch_conv.padding,
            dilation=original_patch_conv.dilation,
            groups=original_patch_conv.groups,
            bias=(
                original_patch_conv.bias
                is not None
            ),
            padding_mode=(
                original_patch_conv.padding_mode
            ),
        )

        # -------------------------------------------------------------
        # 6. RGB -> grayscale adaptation.
        #
        # Frozen project rule:
        #
        #     W_gray = W_R + W_G + W_B
        #
        # [96,3,2,4,4]
        #        ↓
        # [96,1,2,4,4]
        # -------------------------------------------------------------

        grayscale_weight = (
            original_rgb_weight.sum(
                dim=1,
                keepdim=True,
            )
        )

        if (
            grayscale_weight.shape
            != grayscale_patch_conv.weight.shape
        ):
            raise RuntimeError(
                "RGB-to-grayscale patch-embedding "
                "conversion produced an unexpected shape.\n"
                f"Expected: "
                f"{tuple(grayscale_patch_conv.weight.shape)}\n"
                f"Actual  : "
                f"{tuple(grayscale_weight.shape)}"
            )

        with torch.no_grad():

            grayscale_patch_conv.weight.copy_(
                grayscale_weight
            )

            # ---------------------------------------------------------
            # Bias is independent of input-channel count, therefore it
            # can be preserved exactly from the pretrained model.
            # ---------------------------------------------------------

            if (
                original_patch_bias is not None
                and grayscale_patch_conv.bias
                is not None
            ):
                grayscale_patch_conv.bias.copy_(
                    original_patch_bias
                )

        backbone.patch_embed.proj = (
            grayscale_patch_conv
        )

        # -------------------------------------------------------------
        # 7. Remove original Kinetics-400 classification head.
        #
        # Original:
        #
        #     Linear(768 -> 400)
        #
        # New:
        #
        #     Linear(768 -> 1)
        # -------------------------------------------------------------

        backbone.head = nn.Linear(
            in_features=FEATURE_DIM,
            out_features=1,
            bias=True,
        )

        # Match Torchvision Swin linear-layer initialization convention.
        nn.init.trunc_normal_(
            backbone.head.weight,
            std=0.02,
        )

        if backbone.head.bias is not None:
            nn.init.zeros_(
                backbone.head.bias
            )

        # -------------------------------------------------------------
        # 8. Save backbone.
        # -------------------------------------------------------------

        self.backbone = backbone

        self.pretrained = True

        self.pretrained_source = (
            "Swin3D_T_Weights.KINETICS400_V1"
        )

        # -------------------------------------------------------------
        # 9. Final architecture validation inherited from E9.
        # -------------------------------------------------------------

        self._validate_final_architecture()

        # -------------------------------------------------------------
        # 10. Initialization report.
        # -------------------------------------------------------------

        self.initialization_report = (
            PretrainedInitializationReport(
                pretrained_source=(
                    self.pretrained_source
                ),
                original_patch_shape=(
                    original_patch_shape
                ),
                grayscale_patch_shape=tuple(
                    self.backbone
                    .patch_embed
                    .proj
                    .weight
                    .shape
                ),
                grayscale_method=(
                    "RGB_SUM"
                ),
                feature_dim=FEATURE_DIM,
                original_classes=(
                    original_number_of_classes
                ),
                target_classes=1,
            )
        )


# =====================================================================
# FACTORY
# =====================================================================


def build_model(
) -> Swin3D_T_Kinetics400_Pretrained:
    """
    Build E10.

    Returns
    -------
    Swin3D_T_Kinetics400_Pretrained
        Kinetics-400 pretrained Swin3D-Tiny with:

            RGB patch embedding -> grayscale
            W_gray = W_R + W_G + W_B

            Kinetics head:
                Linear(768 -> 400)

            replaced by:
                Linear(768 -> 1)
    """

    return Swin3D_T_Kinetics400_Pretrained()


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
# E9 / E10 ARCHITECTURE VERIFICATION
# =====================================================================


def verify_architecture_matches_e9(
    e10_model: Swin3D_T_Kinetics400_Pretrained,
) -> None:
    """
    Verify E9 and E10 have identical parameter/buffer names and shapes.

    Values are intentionally different:

        E9  = scratch
        E10 = Kinetics-400 pretrained

    Architecture, however, must be identical.
    """

    e9_model = Swin3D_T_Scratch()

    e9_state = e9_model.state_dict()

    e10_state = e10_model.state_dict()

    # -------------------------------------------------------------
    # Keys must match exactly.
    # -------------------------------------------------------------

    if (
        e9_state.keys()
        != e10_state.keys()
    ):

        e9_only = sorted(
            set(e9_state.keys())
            - set(e10_state.keys())
        )

        e10_only = sorted(
            set(e10_state.keys())
            - set(e9_state.keys())
        )

        lines = [
            "E9/E10 architecture mismatch.",
            "",
            "State-dict keys only in E9:",
        ]

        lines.extend(
            f"  - {key}"
            for key in e9_only
        )

        lines.append("")
        lines.append(
            "State-dict keys only in E10:"
        )

        lines.extend(
            f"  - {key}"
            for key in e10_only
        )

        raise RuntimeError(
            "\n".join(lines)
        )

    # -------------------------------------------------------------
    # Shapes must match exactly.
    # -------------------------------------------------------------

    shape_mismatches = []

    for key in e9_state:

        e9_shape = tuple(
            e9_state[key].shape
        )

        e10_shape = tuple(
            e10_state[key].shape
        )

        if e9_shape != e10_shape:

            shape_mismatches.append(
                (
                    key,
                    e9_shape,
                    e10_shape,
                )
            )

    if shape_mismatches:

        lines = [
            "E9/E10 tensor-shape mismatch:",
            "",
        ]

        for (
            key,
            e9_shape,
            e10_shape,
        ) in shape_mismatches:

            lines.append(
                f"  {key}: "
                f"E9={e9_shape}, "
                f"E10={e10_shape}"
            )

        raise RuntimeError(
            "\n".join(lines)
        )


# =====================================================================
# PRETRAINED TRANSFER VERIFICATION
# =====================================================================


def verify_kinetics_transfer(
    e10_model: Swin3D_T_Kinetics400_Pretrained,
) -> None:
    """
    Verify the exact Kinetics-400 -> E10 transfer.

    This is intended as one-time model QC.

    Checks
    ------
    1. Official Kinetics Swin3D-Tiny loads.
    2. Grayscale patch kernel exactly equals RGB-channel sum.
    3. Patch-embedding bias is preserved.
    4. Every other encoder tensor is exactly unchanged.
    5. Kinetics 400-class head is excluded.
    6. New binary head is Linear(768 -> 1).
    """

    # -------------------------------------------------------------
    # Build untouched official pretrained reference.
    #
    # If already downloaded, Torchvision uses its local cache.
    # -------------------------------------------------------------

    reference = swin3d_t(
        weights=KINETICS400_WEIGHTS,
    )

    reference.eval()

    reference_state = (
        reference.state_dict()
    )

    e10_backbone_state = (
        e10_model.backbone.state_dict()
    )

    # -------------------------------------------------------------
    # 1. Verify RGB -> grayscale patch weight.
    # -------------------------------------------------------------

    rgb_patch_weight = (
        reference_state[
            "patch_embed.proj.weight"
        ]
    )

    expected_gray_weight = (
        rgb_patch_weight.sum(
            dim=1,
            keepdim=True,
        )
    )

    actual_gray_weight = (
        e10_backbone_state[
            "patch_embed.proj.weight"
        ]
    )

    if (
        expected_gray_weight.shape
        != actual_gray_weight.shape
    ):
        raise RuntimeError(
            "Swin3D RGB-to-grayscale patch shape mismatch.\n"
            f"Expected: "
            f"{tuple(expected_gray_weight.shape)}\n"
            f"Actual  : "
            f"{tuple(actual_gray_weight.shape)}"
        )

    if not torch.equal(
        expected_gray_weight,
        actual_gray_weight,
    ):
        raise RuntimeError(
            "E10 grayscale patch embedding does not exactly equal "
            "W_R + W_G + W_B."
        )

    # -------------------------------------------------------------
    # 2. Verify patch-embedding bias was preserved.
    # -------------------------------------------------------------

    patch_bias_key = (
        "patch_embed.proj.bias"
    )

    if (
        patch_bias_key
        in reference_state
    ):

        if (
            patch_bias_key
            not in e10_backbone_state
        ):
            raise RuntimeError(
                "Pretrained patch-embedding bias "
                "is missing from E10."
            )

        if not torch.equal(
            reference_state[
                patch_bias_key
            ],
            e10_backbone_state[
                patch_bias_key
            ],
        ):
            raise RuntimeError(
                "Pretrained patch-embedding bias "
                "was not preserved exactly."
            )

    # -------------------------------------------------------------
    # 3. Verify every other pretrained encoder tensor.
    #
    # Intentionally excluded:
    #
    #     patch_embed.proj.weight
    #         because RGB -> grayscale conversion is required
    #
    #     head.weight
    #     head.bias
    #         because the 400-class Kinetics head is replaced
    # -------------------------------------------------------------

    excluded_reference_keys = {
        "patch_embed.proj.weight",
        "head.weight",
        "head.bias",
    }

    missing_keys = []

    shape_mismatches = []

    value_mismatches = []

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
            not in e10_backbone_state
        ):
            missing_keys.append(
                reference_key
            )

            continue

        e10_tensor = (
            e10_backbone_state[
                reference_key
            ]
        )

        if (
            reference_tensor.shape
            != e10_tensor.shape
        ):

            shape_mismatches.append(
                (
                    reference_key,
                    tuple(
                        reference_tensor.shape
                    ),
                    tuple(
                        e10_tensor.shape
                    ),
                )
            )

            continue

        if not torch.equal(
            reference_tensor,
            e10_tensor,
        ):

            value_mismatches.append(
                reference_key
            )

    if missing_keys:

        raise RuntimeError(
            "Pretrained Swin3D encoder tensors "
            "are missing from E10:\n"
            + "\n".join(
                f"  - {key}"
                for key in missing_keys
            )
        )

    if shape_mismatches:

        lines = [
            "Pretrained Swin3D tensor-shape mismatches:",
            "",
        ]

        for (
            key,
            reference_shape,
            e10_shape,
        ) in shape_mismatches:

            lines.append(
                f"  {key}: "
                f"reference={reference_shape}, "
                f"E10={e10_shape}"
            )

        raise RuntimeError(
            "\n".join(lines)
        )

    if value_mismatches:

        raise RuntimeError(
            "Pretrained Swin3D encoder tensors "
            "were modified unexpectedly:\n"
            + "\n".join(
                f"  - {key}"
                for key in value_mismatches
            )
        )

    # -------------------------------------------------------------
    # 4. Verify Kinetics classifier is gone.
    # -------------------------------------------------------------

    if (
        e10_model.backbone.head.in_features
        != FEATURE_DIM
    ):
        raise RuntimeError(
            "E10 classifier must receive "
            f"{FEATURE_DIM} features."
        )

    if (
        e10_model.backbone.head.out_features
        != 1
    ):
        raise RuntimeError(
            "E10 classifier must produce one raw logit."
        )


# =====================================================================
# ARCHITECTURE / PRETRAINING SANITY CHECK
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
    #
    # Running this file may download the official Kinetics-400
    # checkpoint if it is not already available in the Torch cache.
    # =============================================================

    model = build_model()

    model.eval()

    report = (
        model.initialization_report
    )

    print("=" * 80)

    print(
        "MODEL 10 — TORCHVISION SWIN3D-TINY "
        "— KINETICS-400 PRETRAINED"
    )

    print("=" * 80)

    print()
    print("Architecture:")

    print(
        f"  Input channels       : "
        f"{INPUT_CHANNELS}"
    )

    print(
        f"  Patch size           : "
        f"{PATCH_SIZE}"
    )

    print(
        f"  Embedding dim        : "
        f"{EMBED_DIM}"
    )

    print(
        f"  Stage depths         : "
        f"{DEPTHS}"
    )

    print(
        f"  Attention heads      : "
        f"{NUM_HEADS}"
    )

    print(
        f"  Window size          : "
        f"{WINDOW_SIZE}"
    )

    print(
        f"  Stage channels       : "
        f"{STAGE_CHANNELS}"
    )

    print(
        f"  Feature dim          : "
        f"{FEATURE_DIM}"
    )

    print(
        f"  Stochastic depth     : "
        f"{STOCHASTIC_DEPTH_PROB}"
    )

    print()
    print("Pretraining:")

    print(
        f"  Source               : "
        f"{report.pretrained_source}"
    )

    print(
        f"  Original patch       : "
        f"{report.original_patch_shape}"
    )

    print(
        f"  Grayscale patch      : "
        f"{report.grayscale_patch_shape}"
    )

    print(
        f"  Grayscale method     : "
        f"{report.grayscale_method}"
    )

    print(
        f"  Original classes     : "
        f"{report.original_classes}"
    )

    print(
        f"  Target classes       : "
        f"{report.target_classes}"
    )

    # -------------------------------------------------------------
    # Parameter count
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
        f"Total parameters       : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters   : "
        f"{trainable_parameters:,}"
    )

    # Expected approximately:
    #
    #     27,845,095
    #
    # This should match E9 exactly.

    # -------------------------------------------------------------
    # E9/E10 architecture identity.
    # -------------------------------------------------------------

    verify_architecture_matches_e9(
        model
    )

    # -------------------------------------------------------------
    # Exact Kinetics transfer QC.
    # -------------------------------------------------------------

    verify_kinetics_transfer(
        model
    )

    # -------------------------------------------------------------
    # Verify the two frozen project input dimensions.
    # -------------------------------------------------------------

    roi_report = (
        model.analyze_input_shape(
            36,
            44,
            44,
        )
    )

    whole_report = (
        model.analyze_input_shape(
            160,
            192,
            192,
        )
    )

    print()
    print("ROI shape analysis:")

    print(
        f"  Input               : "
        f"{roi_report.input_shape}"
    )

    print(
        f"  Stage 1             : "
        f"{roi_report.stage1_grid}"
    )

    print(
        f"  Stage 2             : "
        f"{roi_report.stage2_grid}"
    )

    print(
        f"  Stage 3             : "
        f"{roi_report.stage3_grid}"
    )

    print(
        f"  Stage 4             : "
        f"{roi_report.stage4_grid}"
    )

    print(
        f"  Compatible          : "
        f"{roi_report.compatible}"
    )

    print()
    print("Whole-volume shape analysis:")

    print(
        f"  Input               : "
        f"{whole_report.input_shape}"
    )

    print(
        f"  Stage 1             : "
        f"{whole_report.stage1_grid}"
    )

    print(
        f"  Stage 2             : "
        f"{whole_report.stage2_grid}"
    )

    print(
        f"  Stage 3             : "
        f"{whole_report.stage3_grid}"
    )

    print(
        f"  Stage 4             : "
        f"{whole_report.stage4_grid}"
    )

    print(
        f"  Compatible          : "
        f"{whole_report.compatible}"
    )

    assert roi_report.compatible
    assert whole_report.compatible

    # -------------------------------------------------------------
    # Actual forward-pass QC using ROI-sized input.
    #
    # Batch 1 is acceptable because Swin uses LayerNorm rather than
    # BatchNorm.
    # -------------------------------------------------------------

    dummy_input = torch.zeros(
        1,
        1,
        36,
        44,
        44,
        dtype=torch.float32,
    )

    with torch.no_grad():

        features = (
            model.forward_features(
                dummy_input
            )
        )

        logits = model(
            dummy_input
        )

    print()

    print(
        f"Input shape            : "
        f"{tuple(dummy_input.shape)}"
    )

    print(
        f"Feature shape          : "
        f"{tuple(features.shape)}"
    )

    print(
        f"Output/logit shape     : "
        f"{tuple(logits.shape)}"
    )

    # -------------------------------------------------------------
    # Final contract checks.
    # -------------------------------------------------------------

    assert features.shape == (
        1,
        FEATURE_DIM,
    )

    assert logits.shape == (
        1,
        1,
    )

    assert torch.isfinite(
        features
    ).all()

    assert torch.isfinite(
        logits
    ).all()

    patch_conv = (
        model.backbone
        .patch_embed
        .proj
    )

    assert (
        patch_conv.in_channels
        == 1
    )

    assert (
        patch_conv.out_channels
        == EMBED_DIM
    )

    assert tuple(
        patch_conv.kernel_size
    ) == PATCH_SIZE

    assert tuple(
        patch_conv.stride
    ) == PATCH_SIZE

    assert (
        model.backbone.head.in_features
        == FEATURE_DIM
    )

    assert (
        model.backbone.head.out_features
        == 1
    )

    print()
    print(
        "Kinetics-400 weight load      : PASS"
    )

    print(
        "RGB -> grayscale conversion   : PASS"
    )

    print(
        "Patch bias preserved          : PASS"
    )

    print(
        "Remaining pretrained weights  : PASS"
    )

    print(
        "Original Kinetics head removed: PASS"
    )

    print(
        "E9/E10 architecture match     : PASS"
    )

    print(
        "ROI shape compatibility       : PASS"
    )

    print(
        "Whole shape compatibility     : PASS"
    )

    print(
        "768-D feature output          : PASS"
    )

    print(
        "Binary classifier             : PASS"
    )

    print(
        "Forward-pass check            : PASS"
    )