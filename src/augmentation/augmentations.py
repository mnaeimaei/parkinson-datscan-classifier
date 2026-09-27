"""
augmentations.py

Training-only augmentation for DaT-SPECT classification.

Supported inputs
----------------
Single subject:
    [C, D, H, W]

Batched:
    [B, C, D, H, W]

Project convention:
    C = 1

Supported spatial modes
-----------------------
"3d":
    For true volumetric models:
        E1, E2, E3, E5, E6, E8, E9, E10

    Applies one small 3-D affine transformation.

"inplane_2d":
    For 2.5-D models:
        E4, E7

    Applies the SAME 2-D affine transformation to every selected
    axial slice.

    This intentionally avoids interpolating/mixing across selected
    slices because whole-volume 2.5-D slices may be a deterministic
    sparse subset rather than contiguous slices.

Scenario C
----------
augment_pair(whole, roi)

uses the same sampled augmentation parameters for both streams.

Important
---------
Augmentation must be enabled ONLY for training.

Validation:
    OFF

OOF prediction:
    OFF

Calibration prediction:
    OFF

Normal inference:
    OFF

No random left-right flipping is included in the baseline.

Spatial interpolation:
    trilinear/bilinear via torch.grid_sample

Padding:
    zero

Intensity policy
----------------
The project already has frozen intensity normalization.

Therefore augmentation performs only mild:
    - multiplicative intensity scaling
    - Gaussian noise relative to foreground standard deviation

No ImageNet/Kinetics normalization is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, pi, sin
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


SpatialMode = Literal[
    "3d",
    "inplane_2d",
]


# =====================================================================
# CONFIGURATION
# =====================================================================


@dataclass(frozen=True)
class AugmentationConfig:
    """
    Conservative DaT-SPECT augmentation configuration.

    These are baseline values and should be frozen before the final
    experiment sweep.
    """

    enabled: bool = True

    # Probability of applying spatial augmentation.
    spatial_probability: float = 0.80

    # Maximum absolute rotation in degrees.
    rotation_degrees: float = 5.0

    # Translation as a fraction of field of view.
    translation_fraction: float = 0.03

    # Uniform scale range:
    #
    #   [1-scale_fraction, 1+scale_fraction]
    scale_fraction: float = 0.05

    # Probability of multiplicative intensity scaling.
    intensity_probability: float = 0.50

    # Multiplicative intensity range:
    #
    #   [1-intensity_scale_fraction,
    #    1+intensity_scale_fraction]
    intensity_scale_fraction: float = 0.05

    # Gaussian-noise probability.
    noise_probability: float = 0.30

    # Noise SD:
    #
    # foreground_std * noise_std_fraction
    noise_std_fraction: float = 0.01

    # Keep scans nonnegative when the original scan is nonnegative.
    preserve_nonnegative: bool = True

    def validate(self) -> None:

        probabilities = {
            "spatial_probability":
                self.spatial_probability,

            "intensity_probability":
                self.intensity_probability,

            "noise_probability":
                self.noise_probability,
        }

        for name, value in probabilities.items():

            if not 0.0 <= value <= 1.0:

                raise ValueError(
                    f"{name} must be within [0,1], "
                    f"received {value}."
                )

        if self.rotation_degrees < 0:

            raise ValueError(
                "rotation_degrees must be >= 0."
            )

        if not 0.0 <= self.translation_fraction < 0.5:

            raise ValueError(
                "translation_fraction must satisfy "
                "0 <= value < 0.5."
            )

        if not 0.0 <= self.scale_fraction < 1.0:

            raise ValueError(
                "scale_fraction must satisfy "
                "0 <= value < 1."
            )

        if self.intensity_scale_fraction < 0:

            raise ValueError(
                "intensity_scale_fraction must be >= 0."
            )

        if self.noise_std_fraction < 0:

            raise ValueError(
                "noise_std_fraction must be >= 0."
            )


# =====================================================================
# RANDOM PARAMETERS
# =====================================================================


@dataclass(frozen=True)
class AugmentationParameters:
    """
    One subject's sampled augmentation parameters.
    """

    apply_spatial: bool

    rotation_x_deg: float
    rotation_y_deg: float
    rotation_z_deg: float

    translation_x: float
    translation_y: float
    translation_z: float

    scale: float

    apply_intensity: bool
    intensity_scale: float

    apply_noise: bool


# =====================================================================
# RANDOM HELPERS
# =====================================================================


def _random_uniform(
    low: float,
    high: float,
    *,
    device: torch.device,
) -> float:
    """
    Draw scalar uniformly using Torch RNG.

    Using Torch RNG allows DataLoader-worker seeding to control
    augmentation randomness.
    """

    value = torch.empty(
        (),
        device=device,
        dtype=torch.float32,
    ).uniform_(
        low,
        high,
    )

    return float(
        value.item()
    )


def _random_bool(
    probability: float,
    *,
    device: torch.device,
) -> bool:

    if probability <= 0:
        return False

    if probability >= 1:
        return True

    return bool(
        torch.rand(
            (),
            device=device,
        ).item()
        < probability
    )


# =====================================================================
# AUGMENTER
# =====================================================================


class DaTSPECTAugmentation:
    """
    Conservative stochastic augmentation.

    This object is intentionally NOT an nn.Module because it does not
    contain trainable parameters.
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        *,
        spatial_mode: SpatialMode = "3d",
    ) -> None:

        self.config = (
            config
            if config is not None
            else AugmentationConfig()
        )

        self.config.validate()

        if spatial_mode not in {
            "3d",
            "inplane_2d",
        }:

            raise ValueError(
                "spatial_mode must be "
                "'3d' or 'inplane_2d'."
            )

        self.spatial_mode = spatial_mode

    # =================================================================
    # PARAMETER SAMPLING
    # =================================================================

    def sample_parameters(
        self,
        *,
        device: torch.device,
    ) -> AugmentationParameters:
        """
        Sample one augmentation configuration.
        """

        c = self.config

        apply_spatial = _random_bool(
            c.spatial_probability,
            device=device,
        )

        if apply_spatial:

            rotation_x = _random_uniform(
                -c.rotation_degrees,
                c.rotation_degrees,
                device=device,
            )

            rotation_y = _random_uniform(
                -c.rotation_degrees,
                c.rotation_degrees,
                device=device,
            )

            rotation_z = _random_uniform(
                -c.rotation_degrees,
                c.rotation_degrees,
                device=device,
            )

            translation_x = _random_uniform(
                -c.translation_fraction,
                c.translation_fraction,
                device=device,
            )

            translation_y = _random_uniform(
                -c.translation_fraction,
                c.translation_fraction,
                device=device,
            )

            translation_z = _random_uniform(
                -c.translation_fraction,
                c.translation_fraction,
                device=device,
            )

            scale = _random_uniform(
                1.0 - c.scale_fraction,
                1.0 + c.scale_fraction,
                device=device,
            )

        else:

            rotation_x = 0.0
            rotation_y = 0.0
            rotation_z = 0.0

            translation_x = 0.0
            translation_y = 0.0
            translation_z = 0.0

            scale = 1.0

        apply_intensity = _random_bool(
            c.intensity_probability,
            device=device,
        )

        if apply_intensity:

            intensity_scale = _random_uniform(
                1.0 - c.intensity_scale_fraction,
                1.0 + c.intensity_scale_fraction,
                device=device,
            )

        else:

            intensity_scale = 1.0

        apply_noise = _random_bool(
            c.noise_probability,
            device=device,
        )

        return AugmentationParameters(
            apply_spatial=apply_spatial,

            rotation_x_deg=rotation_x,
            rotation_y_deg=rotation_y,
            rotation_z_deg=rotation_z,

            translation_x=translation_x,
            translation_y=translation_y,
            translation_z=translation_z,

            scale=scale,

            apply_intensity=apply_intensity,
            intensity_scale=intensity_scale,

            apply_noise=apply_noise,
        )

    # =================================================================
    # VALIDATION
    # =================================================================

    @staticmethod
    def _validate_single_volume(
        x: Tensor,
    ) -> None:
        """
        Expected:
            [C,D,H,W]
        """

        if not isinstance(
            x,
            Tensor,
        ):

            raise TypeError(
                "Augmentation input must be torch.Tensor."
            )

        if x.ndim != 4:

            raise ValueError(
                "Single-volume augmentation expects "
                "[C,D,H,W], "
                f"received {tuple(x.shape)}."
            )

        if x.shape[0] != 1:

            raise ValueError(
                "Project augmentation expects one channel, "
                f"received C={x.shape[0]}."
            )

        if not torch.is_floating_point(
            x
        ):

            raise TypeError(
                "Augmentation input must be floating point."
            )

        if not torch.isfinite(
            x
        ).all():

            raise ValueError(
                "Augmentation input contains NaN/Inf."
            )

    # =================================================================
    # MATRICES
    # =================================================================

    @staticmethod
    def _rotation_matrix_3d(
        params: AugmentationParameters,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Build 3-D rotation matrix.

        Coordinate convention used by affine_grid:
            x = width
            y = height
            z = depth
        """

        rx = (
            params.rotation_x_deg
            * pi
            / 180.0
        )

        ry = (
            params.rotation_y_deg
            * pi
            / 180.0
        )

        rz = (
            params.rotation_z_deg
            * pi
            / 180.0
        )

        rx_matrix = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos(rx), -sin(rx)],
                [0.0, sin(rx), cos(rx)],
            ],
            device=device,
            dtype=dtype,
        )

        ry_matrix = torch.tensor(
            [
                [cos(ry), 0.0, sin(ry)],
                [0.0, 1.0, 0.0],
                [-sin(ry), 0.0, cos(ry)],
            ],
            device=device,
            dtype=dtype,
        )

        rz_matrix = torch.tensor(
            [
                [cos(rz), -sin(rz), 0.0],
                [sin(rz), cos(rz), 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )

        return (
            rz_matrix
            @ ry_matrix
            @ rx_matrix
        )

    # =================================================================
    # 3-D SPATIAL AUGMENTATION
    # =================================================================

    def _apply_3d_affine(
        self,
        x: Tensor,
        params: AugmentationParameters,
    ) -> Tensor:

        if not params.apply_spatial:
            return x

        device = x.device
        dtype = x.dtype

        rotation = (
            self._rotation_matrix_3d(
                params,
                device=device,
                dtype=dtype,
            )
        )

        # affine_grid uses normalized coordinates [-1,1].
        #
        # A fraction f of field of view corresponds approximately
        # to 2*f in normalized coordinates.
        translation = torch.tensor(
            [
                2.0 * params.translation_x,
                2.0 * params.translation_y,
                2.0 * params.translation_z,
            ],
            device=device,
            dtype=dtype,
        )

        linear = (
            rotation
            * params.scale
        )

        theta = torch.zeros(
            (1, 3, 4),
            device=device,
            dtype=dtype,
        )

        theta[
            0,
            :3,
            :3,
        ] = linear

        theta[
            0,
            :3,
            3,
        ] = translation

        batched = x.unsqueeze(
            0
        )

        grid = F.affine_grid(
            theta,
            size=batched.shape,
            align_corners=False,
        )

        transformed = F.grid_sample(
            batched,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        return transformed[
            0
        ]

    # =================================================================
    # 2-D IN-PLANE SPATIAL AUGMENTATION
    # =================================================================

    def _apply_inplane_affine(
        self,
        x: Tensor,
        params: AugmentationParameters,
    ) -> Tensor:
        """
        Apply identical H/W transform to every slice.

        Input:
            [C,D,H,W]

        No interpolation is performed across D.
        """

        if not params.apply_spatial:
            return x

        _, depth, height, width = (
            x.shape
        )

        angle = (
            params.rotation_z_deg
            * pi
            / 180.0
        )

        c = cos(
            angle
        )

        s = sin(
            angle
        )

        theta = torch.tensor(
            [
                [
                    params.scale * c,
                    -params.scale * s,
                    2.0 * params.translation_x,
                ],
                [
                    params.scale * s,
                    params.scale * c,
                    2.0 * params.translation_y,
                ],
            ],
            dtype=x.dtype,
            device=x.device,
        )

        theta = theta.unsqueeze(
            0
        ).expand(
            depth,
            -1,
            -1,
        ).contiguous()

        # [C,D,H,W]
        #      ↓
        # [D,C,H,W]
        slices = x.permute(
            1,
            0,
            2,
            3,
        ).contiguous()

        grid = F.affine_grid(
            theta,
            size=slices.shape,
            align_corners=False,
        )

        slices = F.grid_sample(
            slices,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        return slices.permute(
            1,
            0,
            2,
            3,
        ).contiguous()

    # =================================================================
    # INTENSITY AUGMENTATION
    # =================================================================

    def _apply_intensity(
        self,
        x: Tensor,
        params: AugmentationParameters,
    ) -> Tensor:

        output = x

        original_nonnegative = bool(
            output.min().item()
            >= 0.0
        )

        # -------------------------------------------------------------
        # Preserve zero-valued background.
        # -------------------------------------------------------------

        foreground = (
            output != 0
        )

        if params.apply_intensity:

            output = torch.where(
                foreground,
                output
                * params.intensity_scale,
                output,
            )

        if (
            params.apply_noise
            and self.config.noise_std_fraction > 0
        ):

            foreground_values = (
                output[
                    foreground
                ]
            )

            if foreground_values.numel() > 1:

                foreground_std = (
                    foreground_values.std(
                        unbiased=False
                    )
                )

                noise_std = (
                    foreground_std
                    * self.config.noise_std_fraction
                )

                if noise_std.item() > 0:

                    noise = torch.randn_like(
                        output
                    ) * noise_std

                    output = torch.where(
                        foreground,
                        output + noise,
                        output,
                    )

        if (
            self.config.preserve_nonnegative
            and original_nonnegative
        ):

            output = output.clamp_min(
                0.0
            )

        return output

    # =================================================================
    # APPLY ONE PARAMETER SET
    # =================================================================

    def apply_parameters(
        self,
        x: Tensor,
        params: AugmentationParameters,
    ) -> Tensor:
        """
        Apply already-sampled parameters to one [C,D,H,W] tensor.
        """

        self._validate_single_volume(
            x
        )

        if not self.config.enabled:
            return x

        if self.spatial_mode == "3d":

            output = (
                self._apply_3d_affine(
                    x,
                    params,
                )
            )

        else:

            output = (
                self._apply_inplane_affine(
                    x,
                    params,
                )
            )

        output = self._apply_intensity(
            output,
            params,
        )

        if not torch.isfinite(
            output
        ).all():

            raise FloatingPointError(
                "Augmentation produced NaN/Inf."
            )

        return output

    # =================================================================
    # SINGLE OR BATCH CALL
    # =================================================================

    def __call__(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Augment:

            [C,D,H,W]

        or:

            [B,C,D,H,W]

        Each subject receives independently sampled parameters.
        """

        if not self.config.enabled:
            return x

        if x.ndim == 4:

            params = self.sample_parameters(
                device=x.device,
            )

            return self.apply_parameters(
                x,
                params,
            )

        if x.ndim != 5:

            raise ValueError(
                "Expected [C,D,H,W] or [B,C,D,H,W], "
                f"received {tuple(x.shape)}."
            )

        augmented = []

        for sample in x:

            params = (
                self.sample_parameters(
                    device=sample.device,
                )
            )

            augmented.append(
                self.apply_parameters(
                    sample,
                    params,
                )
            )

        return torch.stack(
            augmented,
            dim=0,
        )

    # =================================================================
    # SCENARIO C TRANSLATION SYNCHRONIZATION
    # =================================================================

    @staticmethod
    def _translation_for_target_shape(
        params: AugmentationParameters,
        *,
        reference_shape: tuple[int, int, int, int],
        target_shape: tuple[int, int, int, int],
    ) -> AugmentationParameters:
        """
        Convert normalized translation fractions so two differently sized
        tensors receive the same voxel displacement.

        The project freezes whole and ROI to the same voxel spacing.
        Therefore equal voxel displacement also means equal physical-mm
        displacement.  Without this conversion, a 3% translation would be
        much larger in the whole volume than in the smaller ROI.

        Shapes are [C,D,H,W]. Rotation and scale are unchanged.
        """

        if len(reference_shape) != 4 or len(target_shape) != 4:
            raise ValueError(
                "reference_shape and target_shape must be [C,D,H,W]."
            )

        _, ref_d, ref_h, ref_w = reference_shape
        _, tgt_d, tgt_h, tgt_w = target_shape

        if min(ref_d, ref_h, ref_w, tgt_d, tgt_h, tgt_w) <= 0:
            raise ValueError("Spatial dimensions must be positive.")

        return replace(
            params,
            translation_x=(
                params.translation_x
                * float(ref_w)
                / float(tgt_w)
            ),
            translation_y=(
                params.translation_y
                * float(ref_h)
                / float(tgt_h)
            ),
            translation_z=(
                params.translation_z
                * float(ref_d)
                / float(tgt_d)
            ),
        )

    # =================================================================
    # SCENARIO C
    # =================================================================

    def augment_pair(
        self,
        whole: Tensor,
        roi: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        """
        Apply synchronized random augmentation parameters to:

            whole
            ROI

        Supports either:

            [C,D,H,W]

        or batched:

            [B,C,D,H,W]

        Rotation/scaling/intensity factor are identical between the two
        streams.

        Translation is synchronized as the same voxel displacement in
        both streams. Because the frozen whole and ROI tensors use the same
        voxel spacing, this also preserves the same physical-mm shift.
        """

        if not self.config.enabled:

            return (
                whole,
                roi,
            )

        if whole.ndim != roi.ndim:

            raise ValueError(
                "Whole and ROI tensor ranks must match."
            )

        if whole.ndim == 4:

            params = self.sample_parameters(
                device=whole.device,
            )

            roi_params = self._translation_for_target_shape(
                params,
                reference_shape=tuple(whole.shape),
                target_shape=tuple(roi.shape),
            )

            return (
                self.apply_parameters(
                    whole,
                    params,
                ),
                self.apply_parameters(
                    roi,
                    roi_params,
                ),
            )

        if whole.ndim != 5:

            raise ValueError(
                "Scenario C tensors must be 4-D or 5-D."
            )

        if (
            whole.shape[0]
            != roi.shape[0]
        ):

            raise ValueError(
                "Whole/ROI batch sizes differ."
            )

        augmented_whole = []
        augmented_roi = []

        for index in range(
            whole.shape[0]
        ):

            params = self.sample_parameters(
                device=whole.device,
            )

            roi_params = self._translation_for_target_shape(
                params,
                reference_shape=tuple(whole[index].shape),
                target_shape=tuple(roi[index].shape),
            )

            augmented_whole.append(
                self.apply_parameters(
                    whole[index],
                    params,
                )
            )

            augmented_roi.append(
                self.apply_parameters(
                    roi[index],
                    roi_params,
                )
            )

        return (
            torch.stack(
                augmented_whole,
                dim=0,
            ),
            torch.stack(
                augmented_roi,
                dim=0,
            ),
        )


# =====================================================================
# FACTORY
# =====================================================================


def build_augmentation(
    *,
    enabled: bool,
    spatial_mode: SpatialMode,
    config: AugmentationConfig | None = None,
) -> DaTSPECTAugmentation:
    """
    Build project augmentation from the shared configuration.

    Passing ``config`` is recommended for experiment scripts so the values
    frozen in ``src.configs.training_config`` remain the single source of
    truth. Only the scenario-specific ``enabled`` flag is overridden here.
    """

    base_config = (
        config
        if config is not None
        else AugmentationConfig()
    )

    resolved_config = replace(
        base_config,
        enabled=bool(enabled),
    )

    return DaTSPECTAugmentation(
        config=resolved_config,
        spatial_mode=spatial_mode,
    )

