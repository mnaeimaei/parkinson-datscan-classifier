"""
model09_swin3d_t_scratch.py

E9 — Torchvision Swin3D-Tiny trained from scratch.

Architecture
------------
Input:
    [B, 1, D, H, W]

Depth D is treated as the temporal/depth dimension.

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

    channel dimensions:
        96 -> 192 -> 384 -> 768

Attention:
    3D shifted-window self-attention.

Patch merging:
    Between stages 1-2, 2-3, and 3-4.

Final representation:
    LayerNorm(768)
    AdaptiveAvgPool3d(1)
    Flatten
    -> [B, 768]

Binary classifier:
    Linear(768 -> 1)

Output:
    [B, 1] raw binary-classification logit

Initialization:
    Scratch / random only.
    No Kinetics-400 pretrained weights are loaded.

Important
---------
No sigmoid is applied inside the model.

Training:
    BCEWithLogitsLoss

Probability:
    torch.sigmoid(logits)

E9 vs E10
----------
E9:
    exact Swin3D-Tiny architecture
    scratch initialization

E10:
    exact same architecture
    Kinetics-400 pretrained initialization

Therefore E10 should reuse the shared architecture defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models.video import swin3d_t


# =====================================================================
# FROZEN ARCHITECTURE CONSTANTS
# =====================================================================

INPUT_CHANNELS = 1

PATCH_SIZE = (2, 4, 4)

EMBED_DIM = 96

DEPTHS = (2, 2, 6, 2)

NUM_HEADS = (3, 6, 12, 24)

WINDOW_SIZE = (8, 7, 7)

STAGE_CHANNELS = (96, 192, 384, 768)

FEATURE_DIM = 768

STOCHASTIC_DEPTH_PROB = 0.1


# =====================================================================
# INPUT-SHAPE REPORT
# =====================================================================


@dataclass(frozen=True)
class Swin3DShapeReport:
    """
    Deterministic description of token-grid dimensions through Swin3D-Tiny.

    Torchvision's Video Swin implementation performs temporal/down-depth
    reduction in the initial patch embedding.

    Subsequent PatchMerging layers reduce H and W while preserving the
    temporal/depth token dimension.
    """

    input_shape: tuple[int, int, int]

    patch_grid: tuple[int, int, int]

    stage1_grid: tuple[int, int, int]
    stage2_grid: tuple[int, int, int]
    stage3_grid: tuple[int, int, int]
    stage4_grid: tuple[int, int, int]

    compatible: bool


# =====================================================================
# SHARED E9 / E10 BASE
# =====================================================================


class Swin3D_T_Base(nn.Module):
    """
    Shared Swin3D-Tiny architecture for E9 and E10.

    Parameters
    ----------
    pretrained_backbone:
        False for E9.

        E10 will later use the same architecture with Kinetics-400
        pretrained initialization.

    Notes
    -----
    This class currently implements the scratch construction path.

    E10 can reuse the same feature/head interface while changing only
    initialization.
    """

    feature_dim: int = FEATURE_DIM

    def __init__(self) -> None:
        super().__init__()

        # -------------------------------------------------------------
        # Build native Torchvision Swin3D-Tiny.
        #
        # weights=None:
        #     no pretrained parameters
        #
        # num_classes=1:
        #     binary raw-logit classifier
        # -------------------------------------------------------------

        backbone = swin3d_t(
            weights=None,
            num_classes=1,
        )

        # -------------------------------------------------------------
        # Validate native Torchvision architecture before changing stem.
        # -------------------------------------------------------------

        self._validate_native_backbone(backbone)

        # -------------------------------------------------------------
        # Replace native RGB patch embedding:
        #
        #     Conv3d(3 -> 96)
        #
        # with:
        #
        #     Conv3d(1 -> 96)
        #
        # Architecture otherwise stays unchanged.
        # -------------------------------------------------------------

        original_patch_conv = backbone.patch_embed.proj

        grayscale_patch_conv = nn.Conv3d(
            in_channels=INPUT_CHANNELS,
            out_channels=EMBED_DIM,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )

        # -------------------------------------------------------------
        # Scratch initialization for new patch embedding.
        # -------------------------------------------------------------

        nn.init.kaiming_normal_(
            grayscale_patch_conv.weight,
            mode="fan_out",
            nonlinearity="linear",
        )

        if grayscale_patch_conv.bias is not None:
            nn.init.zeros_(
                grayscale_patch_conv.bias
            )

        backbone.patch_embed.proj = (
            grayscale_patch_conv
        )

        # -------------------------------------------------------------
        # Binary classifier is already Linear(768 -> 1) because
        # swin3d_t(..., num_classes=1) was requested.
        #
        # Reinitialize it explicitly as a new task head.
        # -------------------------------------------------------------

        if not isinstance(
            backbone.head,
            nn.Linear,
        ):
            raise RuntimeError(
                "Unexpected Swin3D-Tiny classifier type."
            )

        if (
            backbone.head.in_features
            != FEATURE_DIM
        ):
            raise RuntimeError(
                "Unexpected Swin3D feature dimension. "
                f"Expected {FEATURE_DIM}, "
                f"received {backbone.head.in_features}."
            )

        if backbone.head.out_features != 1:
            raise RuntimeError(
                "E9 classifier must produce exactly one logit."
            )

        nn.init.trunc_normal_(
            backbone.head.weight,
            std=0.02,
        )

        if backbone.head.bias is not None:
            nn.init.zeros_(
                backbone.head.bias
            )

        self.backbone = backbone

        self.pretrained = False

        # -------------------------------------------------------------
        # Final architecture validation after grayscale conversion.
        # -------------------------------------------------------------

        self._validate_final_architecture()

    # =================================================================
    # NATIVE BACKBONE VALIDATION
    # =================================================================

    @staticmethod
    def _validate_native_backbone(
        backbone: nn.Module,
    ) -> None:
        """
        Verify critical Torchvision Swin3D-Tiny assumptions.

        This protects the experiment from silent architecture changes
        after future torchvision upgrades.
        """

        if not hasattr(
            backbone,
            "patch_embed",
        ):
            raise RuntimeError(
                "Unexpected Swin3D implementation: "
                "missing patch_embed."
            )

        if not hasattr(
            backbone.patch_embed,
            "proj",
        ):
            raise RuntimeError(
                "Unexpected Swin3D patch embedding: "
                "missing patch_embed.proj."
            )

        patch_conv = (
            backbone.patch_embed.proj
        )

        if not isinstance(
            patch_conv,
            nn.Conv3d,
        ):
            raise RuntimeError(
                "Swin3D patch embedding must use Conv3d."
            )

        if patch_conv.in_channels != 3:
            raise RuntimeError(
                "Expected native RGB Swin3D stem with "
                "3 input channels, "
                f"received {patch_conv.in_channels}."
            )

        if patch_conv.out_channels != EMBED_DIM:
            raise RuntimeError(
                "Unexpected Swin3D embedding dimension. "
                f"Expected {EMBED_DIM}, "
                f"received {patch_conv.out_channels}."
            )

        if tuple(
            patch_conv.kernel_size
        ) != PATCH_SIZE:

            raise RuntimeError(
                "Unexpected Swin3D patch size. "
                f"Expected {PATCH_SIZE}, "
                f"received {patch_conv.kernel_size}."
            )

        if tuple(
            patch_conv.stride
        ) != PATCH_SIZE:

            raise RuntimeError(
                "Unexpected Swin3D patch stride. "
                f"Expected {PATCH_SIZE}, "
                f"received {patch_conv.stride}."
            )

        if (
            backbone.num_features
            != FEATURE_DIM
        ):
            raise RuntimeError(
                "Unexpected Swin3D final feature dimension. "
                f"Expected {FEATURE_DIM}, "
                f"received {backbone.num_features}."
            )

    # =================================================================
    # FINAL ARCHITECTURE VALIDATION
    # =================================================================

    def _validate_final_architecture(
        self,
    ) -> None:
        """
        Verify the frozen E9 architecture.
        """

        patch_conv = (
            self.backbone.patch_embed.proj
        )

        if patch_conv.in_channels != 1:
            raise RuntimeError(
                "E9 patch embedding must have one input channel."
            )

        if patch_conv.out_channels != EMBED_DIM:
            raise RuntimeError(
                "E9 patch embedding must produce "
                f"{EMBED_DIM} features."
            )

        if tuple(
            patch_conv.kernel_size
        ) != PATCH_SIZE:

            raise RuntimeError(
                "Incorrect E9 patch size."
            )

        if tuple(
            patch_conv.stride
        ) != PATCH_SIZE:

            raise RuntimeError(
                "Incorrect E9 patch stride."
            )

        if (
            self.backbone.head.in_features
            != FEATURE_DIM
        ):
            raise RuntimeError(
                "E9 classifier must receive "
                f"{FEATURE_DIM} features."
            )

        if (
            self.backbone.head.out_features
            != 1
        ):
            raise RuntimeError(
                "E9 classifier must produce one raw logit."
            )

    # =================================================================
    # INPUT CONTRACT
    # =================================================================

    @staticmethod
    def _validate_input(
        x: Tensor,
    ) -> None:
        """
        Validate project model input.

        Expected:
            [B, 1, D, H, W]
        """

        if not isinstance(
            x,
            Tensor,
        ):
            raise TypeError(
                "Input must be a torch.Tensor, "
                f"received {type(x).__name__}."
            )

        if x.ndim != 5:
            raise ValueError(
                "Swin3D-Tiny expects input shape "
                "[B, C, D, H, W], "
                f"received {tuple(x.shape)}."
            )

        if x.shape[0] < 1:
            raise ValueError(
                "Batch size must be >= 1."
            )

        if x.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                "Swin3D-Tiny expects one input channel, "
                f"received C={x.shape[1]}."
            )

        depth, height, width = (
            x.shape[2],
            x.shape[3],
            x.shape[4],
        )

        if min(
            depth,
            height,
            width,
        ) < 1:

            raise ValueError(
                "All input dimensions must be positive, "
                f"received D/H/W="
                f"{(depth, height, width)}."
            )

        if not torch.is_floating_point(x):
            raise TypeError(
                "Swin3D-Tiny input must be floating point, "
                f"received dtype={x.dtype}."
            )

        # -------------------------------------------------------------
        # Explicitly validate that the resulting hierarchical token
        # grids never collapse.
        # -------------------------------------------------------------

        report = (
            Swin3D_T_Base.analyze_input_shape(
                depth,
                height,
                width,
            )
        )

        if not report.compatible:
            raise ValueError(
                "Input dimensions are incompatible with "
                "the frozen Swin3D-Tiny hierarchy.\n"
                f"Input: {(depth, height, width)}\n"
                f"Report: {report}"
            )

    # =================================================================
    # INPUT-SHAPE ANALYSIS
    # =================================================================

    @staticmethod
    def analyze_input_shape(
        depth: int,
        height: int,
        width: int,
    ) -> Swin3DShapeReport:
        """
        Analyze token-grid dimensions produced by Swin3D-Tiny.

        Torchvision pads the input when necessary during patch embedding.

        Initial patch embedding:

            D -> ceil(D / 2)
            H -> ceil(H / 4)
            W -> ceil(W / 4)

        Video Swin's PatchMerging then reduces H and W between stages
        while retaining the temporal/depth token dimension.

        Patch merging uses padding for odd spatial dimensions.
        """

        if min(
            depth,
            height,
            width,
        ) < 1:

            return Swin3DShapeReport(
                input_shape=(
                    depth,
                    height,
                    width,
                ),
                patch_grid=(0, 0, 0),
                stage1_grid=(0, 0, 0),
                stage2_grid=(0, 0, 0),
                stage3_grid=(0, 0, 0),
                stage4_grid=(0, 0, 0),
                compatible=False,
            )

        # -------------------------------------------------------------
        # Patch embedding:
        #     kernel = stride = (2,4,4)
        #
        # Torchvision pads dimensions to patch multiples.
        # -------------------------------------------------------------

        d1 = ceil(
            depth
            / PATCH_SIZE[0]
        )

        h1 = ceil(
            height
            / PATCH_SIZE[1]
        )

        w1 = ceil(
            width
            / PATCH_SIZE[2]
        )

        # Stage 1.
        stage1 = (
            d1,
            h1,
            w1,
        )

        # -------------------------------------------------------------
        # PatchMerging 1:
        # H/W /2, temporal dimension unchanged.
        # -------------------------------------------------------------

        h2 = ceil(h1 / 2)
        w2 = ceil(w1 / 2)

        stage2 = (
            d1,
            h2,
            w2,
        )

        # PatchMerging 2.
        h3 = ceil(h2 / 2)
        w3 = ceil(w2 / 2)

        stage3 = (
            d1,
            h3,
            w3,
        )

        # PatchMerging 3.
        h4 = ceil(h3 / 2)
        w4 = ceil(w3 / 2)

        stage4 = (
            d1,
            h4,
            w4,
        )

        compatible = all(
            value >= 1
            for grid in (
                stage1,
                stage2,
                stage3,
                stage4,
            )
            for value in grid
        )

        return Swin3DShapeReport(
            input_shape=(
                depth,
                height,
                width,
            ),
            patch_grid=stage1,
            stage1_grid=stage1,
            stage2_grid=stage2,
            stage3_grid=stage3,
            stage4_grid=stage4,
            compatible=compatible,
        )

    # =================================================================
    # FEATURE EXTRACTION
    # =================================================================

    def forward_features(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Extract one 768-D representation per subject.

        Parameters
        ----------
        x:
            [B, 1, D, H, W]

        Returns
        -------
        Tensor:
            [B, 768]

        This interface is intentionally exposed for later Scenario C
        whole-volume + ROI feature fusion.
        """

        self._validate_input(x)

        # -------------------------------------------------------------
        # Follow Torchvision SwinTransformer3d.forward exactly,
        # stopping immediately before self.backbone.head.
        # -------------------------------------------------------------

        # [B,C,D,H,W]
        #       ↓
        # [B,D',H',W',96]
        x = self.backbone.patch_embed(x)

        x = self.backbone.pos_drop(x)

        # Hierarchical Swin stages and patch merging.
        x = self.backbone.features(x)

        # Final LayerNorm.
        x = self.backbone.norm(x)

        # Torchvision Swin3D representation:
        #
        # [B,D',H',W',768]
        #          ↓
        # [B,768,D',H',W']
        x = x.permute(
            0,
            4,
            1,
            2,
            3,
        )

        # Global average pooling.
        x = self.backbone.avgpool(x)

        # [B,768,1,1,1]
        #       ↓
        # [B,768]
        x = torch.flatten(
            x,
            1,
        )

        if x.ndim != 2:
            raise RuntimeError(
                "Unexpected Swin3D feature tensor rank. "
                f"Expected [B,{FEATURE_DIM}], "
                f"received {tuple(x.shape)}."
            )

        if x.shape[1] != FEATURE_DIM:
            raise RuntimeError(
                "Unexpected Swin3D feature dimension. "
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

        Do NOT apply sigmoid here.
        """

        features = self.forward_features(
            x
        )

        logits = self.backbone.head(
            features
        )

        return logits


# =====================================================================
# E9 — SCRATCH
# =====================================================================


class Swin3D_T_Scratch(
    Swin3D_T_Base
):
    """
    E9 — single-channel Torchvision Swin3D-Tiny trained from scratch.
    """

    def __init__(self) -> None:
        super().__init__()


# =====================================================================
# FACTORY
# =====================================================================


def build_model(
) -> Swin3D_T_Scratch:
    """
    Build a fresh E9 model.

    No pretrained weights are loaded.
    """

    return Swin3D_T_Scratch()


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
        "MODEL 09 — TORCHVISION SWIN3D-TINY — SCRATCH"
    )
    print("=" * 78)

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

    print(
        "  Classifier           : "
        "Linear(768 -> 1)"
    )

    print(
        "  Pretrained           : NO"
    )

    # -------------------------------------------------------------
    # Parameter counts.
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
    # after:
    #
    #     RGB patch embedding -> 1 channel
    #     Kinetics 400-class head -> 1 logit
    #
    # i.e. approximately 27.85M parameters.

    # -------------------------------------------------------------
    # Check both frozen project input shapes.
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
        f"  Input        : "
        f"{roi_report.input_shape}"
    )
    print(
        f"  Stage 1      : "
        f"{roi_report.stage1_grid}"
    )
    print(
        f"  Stage 2      : "
        f"{roi_report.stage2_grid}"
    )
    print(
        f"  Stage 3      : "
        f"{roi_report.stage3_grid}"
    )
    print(
        f"  Stage 4      : "
        f"{roi_report.stage4_grid}"
    )
    print(
        f"  Compatible   : "
        f"{roi_report.compatible}"
    )

    print()
    print("Whole-volume shape analysis:")
    print(
        f"  Input        : "
        f"{whole_report.input_shape}"
    )
    print(
        f"  Stage 1      : "
        f"{whole_report.stage1_grid}"
    )
    print(
        f"  Stage 2      : "
        f"{whole_report.stage2_grid}"
    )
    print(
        f"  Stage 3      : "
        f"{whole_report.stage3_grid}"
    )
    print(
        f"  Stage 4      : "
        f"{whole_report.stage4_grid}"
    )
    print(
        f"  Compatible   : "
        f"{whole_report.compatible}"
    )

    assert roi_report.compatible
    assert whole_report.compatible

    # -------------------------------------------------------------
    # Use ROI input for actual forward-pass QC because it is much
    # lighter than the full whole-volume tensor.
    #
    # Batch size 1 is sufficient because Swin uses LayerNorm rather
    # than BatchNorm.
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
    # Contract checks.
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
        model.backbone.patch_embed.proj
    )

    assert (
        patch_conv.in_channels
        == INPUT_CHANNELS
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
        "Scratch initialization      : PASS"
    )
    print(
        "Single-channel patch embed  : PASS"
    )
    print(
        "ROI shape compatibility     : PASS"
    )
    print(
        "Whole shape compatibility   : PASS"
    )
    print(
        "768-D feature output        : PASS"
    )
    print(
        "Binary classifier           : PASS"
    )
    print(
        "Forward-pass check          : PASS"
    )