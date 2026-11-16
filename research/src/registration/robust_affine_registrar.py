from __future__ import annotations

from typing import Any

import numpy as np
import SimpleITK as sitk

from src.registration.affine_registrar import (
    create_head_mask,
    mask_dice,
    robust_registration_image,
)


def _initialize_rigid(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    mode: str,
) -> sitk.Euler3DTransform:
    """
    Create deterministic rigid initialization.

    MOMENTS:
        aligns intensity/foreground centers.

    GEOMETRY:
        aligns image geometric centers.

    We deliberately test both because heterogeneous FOV can
    make either initialization preferable for a given scan.
    """

    if mode == "moments":
        initializer_mode = (
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )

    elif mode == "geometry":
        initializer_mode = (
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

    else:
        raise ValueError(
            f"Unsupported initialization mode: {mode}"
        )

    transform = sitk.CenteredTransformInitializer(
        sitk.Cast(
            fixed_mask,
            sitk.sitkFloat32,
        ),
        sitk.Cast(
            moving_mask,
            sitk.sitkFloat32,
        ),
        sitk.Euler3DTransform(),
        initializer_mode,
    )

    return sitk.Euler3DTransform(
        transform
    )


def _configure_registration(
    transform: sitk.Transform,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    learning_rate: float,
    iterations: int,
) -> sitk.ImageRegistrationMethod:

    registration = (
        sitk.ImageRegistrationMethod()
    )

    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50,
    )

    registration.SetMetricSamplingStrategy(
        registration.RANDOM
    )

    registration.SetMetricSamplingPercentage(
        0.20,
        seed=42,
    )

    registration.SetMetricFixedMask(
        fixed_mask
    )

    registration.SetMetricMovingMask(
        moving_mask
    )

    registration.SetInterpolator(
        sitk.sitkLinear
    )

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=1e-4,
        numberOfIterations=iterations,
        gradientMagnitudeTolerance=1e-6,
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetShrinkFactorsPerLevel(
        [4, 2, 1]
    )

    registration.SetSmoothingSigmasPerLevel(
        [2, 1, 0]
    )

    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetInitialTransform(
        transform,
        inPlace=True,
    )

    return registration


def _affine_determinant(
    transform: sitk.AffineTransform,
) -> float:

    matrix = np.asarray(
        transform.GetMatrix(),
        dtype=float,
    ).reshape(3, 3)

    return float(
        np.linalg.det(matrix)
    )


def _affine_condition_number(
    transform: sitk.AffineTransform,
) -> float:

    matrix = np.asarray(
        transform.GetMatrix(),
        dtype=float,
    ).reshape(3, 3)

    return float(
        np.linalg.cond(matrix)
    )


def register_single_start(
    fixed_template: sitk.Image,
    moving_subject: sitk.Image,
    initialization_mode: str,
) -> tuple[sitk.AffineTransform, dict[str, Any]]:
    """
    Run rigid + affine registration from one initialization.
    """

    fixed_reg = robust_registration_image(
        fixed_template
    )

    moving_reg = robust_registration_image(
        moving_subject
    )

    fixed_mask = create_head_mask(
        fixed_reg
    )

    moving_mask = create_head_mask(
        moving_reg
    )

    # --------------------------------------------------
    # Rigid
    # --------------------------------------------------

    rigid = _initialize_rigid(
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        mode=initialization_mode,
    )

    rigid_registration = _configure_registration(
        transform=rigid,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        learning_rate=2.0,
        iterations=200,
    )

    rigid_registration.Execute(
        fixed_reg,
        moving_reg,
    )

    rigid_metric = float(
        rigid_registration.GetMetricValue()
    )

    # --------------------------------------------------
    # Affine initialized from rigid result
    # --------------------------------------------------

    affine = sitk.AffineTransform(3)

    affine.SetCenter(
        rigid.GetCenter()
    )

    affine.SetMatrix(
        rigid.GetMatrix()
    )

    affine.SetTranslation(
        rigid.GetTranslation()
    )

    affine_registration = _configure_registration(
        transform=affine,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        learning_rate=0.5,
        iterations=300,
    )

    affine_registration.Execute(
        fixed_reg,
        moving_reg,
    )

    affine_metric = float(
        affine_registration.GetMetricValue()
    )

    dice = mask_dice(
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        transform=affine,
    )

    metadata = {
        "initialization_mode": (
            initialization_mode
        ),

        "rigid_metric": (
            rigid_metric
        ),

        "affine_metric": (
            affine_metric
        ),

        "head_mask_dice": (
            float(dice)
        ),

        "affine_determinant": (
            _affine_determinant(
                affine
            )
        ),

        "affine_condition_number": (
            _affine_condition_number(
                affine
            )
        ),

        "rigid_stop_condition": (
            rigid_registration
            .GetOptimizerStopConditionDescription()
        ),

        "affine_stop_condition": (
            affine_registration
            .GetOptimizerStopConditionDescription()
        ),
    }

    return affine, metadata


def register_affine_dual_start(
    fixed_template: sitk.Image,
    moving_subject: sitk.Image,
) -> tuple[
    sitk.AffineTransform,
    dict[str, Any],
]:
    """
    Run two deterministic registrations:

        1. MOMENTS initialization
        2. GEOMETRY initialization

    The transform with the larger head-mask Dice is selected.

    Head-mask Dice does not use class labels or striatal
    pathology information.
    """

    candidates = []

    for mode in (
        "moments",
        "geometry",
    ):

        try:
            (
                transform,
                metadata,
            ) = register_single_start(
                fixed_template=(
                    fixed_template
                ),
                moving_subject=(
                    moving_subject
                ),
                initialization_mode=mode,
            )

            candidates.append(
                {
                    "transform": (
                        transform
                    ),
                    "metadata": (
                        metadata
                    ),
                }
            )

        except Exception as exc:

            candidates.append(
                {
                    "transform": None,

                    "metadata": {
                        "initialization_mode": mode,
                        "failed": True,
                        "error": str(exc),
                        "head_mask_dice": None,
                    },
                }
            )

    successful = [
        candidate
        for candidate in candidates
        if (
            candidate["transform"]
            is not None
        )
    ]

    if not successful:
        raise RuntimeError(
            "Both registration initializations failed."
        )

    best = max(
        successful,
        key=lambda candidate: (
            candidate[
                "metadata"
            ][
                "head_mask_dice"
            ]
        ),
    )

    selected_transform = (
        best["transform"]
    )

    selected_metadata = dict(
        best["metadata"]
    )

    selected_metadata[
        "selected_initialization"
    ] = (
        selected_metadata[
            "initialization_mode"
        ]
    )

    selected_metadata[
        "candidate_results"
    ] = {
        candidate[
            "metadata"
        ][
            "initialization_mode"
        ]: candidate[
            "metadata"
        ]
        for candidate in candidates
    }

    return (
        selected_transform,
        selected_metadata,
    )
