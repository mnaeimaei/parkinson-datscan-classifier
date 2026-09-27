from __future__ import annotations

from typing import Any

import numpy as np
import SimpleITK as sitk

from src.registration.affine_registrar import (
    create_head_mask,
    mask_dice,
    robust_registration_image,
)


def _create_rigid_initialization(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    initialization_mode: str,
) -> sitk.Euler3DTransform:
    """
    Create a rigid initialization using either:

        moments
        geometry
    """

    if initialization_mode == "moments":
        mode = (
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )

    elif initialization_mode == "geometry":
        mode = (
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

    else:
        raise ValueError(
            "initialization_mode must be "
            "'moments' or 'geometry'."
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
        mode,
    )

    return sitk.Euler3DTransform(
        transform
    )


def _configure_registration(
    transform: sitk.Transform,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    learning_rate: float,
    number_of_iterations: int,
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
        numberOfIterations=number_of_iterations,
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


def _matrix_statistics(
    transform: sitk.Transform,
) -> dict[str, float]:
    """
    Calculate determinant and condition number from the
    3x3 linear transform matrix.
    """

    matrix = np.asarray(
        transform.GetMatrix(),
        dtype=float,
    ).reshape(3, 3)

    return {
        "determinant": float(
            np.linalg.det(matrix)
        ),

        "condition_number": float(
            np.linalg.cond(matrix)
        ),
    }


def _run_rigid_stage(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    initialization_mode: str,
) -> tuple[
    sitk.Euler3DTransform,
    dict[str, Any],
]:

    rigid = _create_rigid_initialization(
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        initialization_mode=(
            initialization_mode
        ),
    )

    registration = _configure_registration(
        transform=rigid,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        learning_rate=2.0,
        number_of_iterations=200,
    )

    registration.Execute(
        fixed_image,
        moving_image,
    )

    dice = mask_dice(
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        transform=rigid,
    )

    matrix_stats = _matrix_statistics(
        rigid
    )

    metadata = {
        "transform_type": "rigid",

        "initialization_mode": (
            initialization_mode
        ),

        "metric": float(
            registration.GetMetricValue()
        ),

        "head_mask_dice": float(
            dice
        ),

        "determinant": (
            matrix_stats["determinant"]
        ),

        "condition_number": (
            matrix_stats[
                "condition_number"
            ]
        ),

        "stop_condition": (
            registration
            .GetOptimizerStopConditionDescription()
        ),
    }

    return rigid, metadata


def register_rigid_candidate(
    fixed_template: sitk.Image,
    moving_subject: sitk.Image,
    initialization_mode: str,
) -> tuple[
    sitk.Euler3DTransform,
    dict[str, Any],
]:
    """
    Rigid registration:

        translation
        rotation

    No scale and no shear.
    """

    fixed_image = robust_registration_image(
        fixed_template
    )

    moving_image = robust_registration_image(
        moving_subject
    )

    fixed_mask = create_head_mask(
        fixed_image
    )

    moving_mask = create_head_mask(
        moving_image
    )

    return _run_rigid_stage(
        fixed_image=fixed_image,
        moving_image=moving_image,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        initialization_mode=(
            initialization_mode
        ),
    )


def _rigid_to_similarity(
    rigid: sitk.Euler3DTransform,
) -> sitk.Similarity3DTransform:
    """
    Initialize a Similarity3DTransform from a rigid result.

    Similarity transform permits:

        translation
        rotation
        one isotropic scale

    but no anisotropic scaling and no shear.
    """

    similarity = (
        sitk.Similarity3DTransform()
    )

    similarity.SetCenter(
        rigid.GetCenter()
    )

    similarity.SetMatrix(
        rigid.GetMatrix()
    )

    similarity.SetTranslation(
        rigid.GetTranslation()
    )

    similarity.SetScale(
        1.0
    )

    return similarity


def register_similarity_candidate(
    fixed_template: sitk.Image,
    moving_subject: sitk.Image,
    initialization_mode: str,
) -> tuple[
    sitk.Similarity3DTransform,
    dict[str, Any],
]:
    """
    Registration sequence:

        initialization
            ↓
        rigid
            ↓
        similarity

    Similarity allows one global scale only.
    """

    fixed_image = robust_registration_image(
        fixed_template
    )

    moving_image = robust_registration_image(
        moving_subject
    )

    fixed_mask = create_head_mask(
        fixed_image
    )

    moving_mask = create_head_mask(
        moving_image
    )

    # --------------------------------------------------
    # First obtain rigid alignment
    # --------------------------------------------------

    (
        rigid,
        rigid_metadata,
    ) = _run_rigid_stage(
        fixed_image=fixed_image,
        moving_image=moving_image,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        initialization_mode=(
            initialization_mode
        ),
    )

    # --------------------------------------------------
    # Convert rigid -> similarity
    # --------------------------------------------------

    similarity = _rigid_to_similarity(
        rigid
    )

    registration = _configure_registration(
        transform=similarity,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        learning_rate=0.5,
        number_of_iterations=300,
    )

    registration.Execute(
        fixed_image,
        moving_image,
    )

    dice = mask_dice(
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        transform=similarity,
    )

    matrix_stats = _matrix_statistics(
        similarity
    )

    scale = float(
        similarity.GetScale()
    )

    metadata = {
        "transform_type": (
            "similarity"
        ),

        "initialization_mode": (
            initialization_mode
        ),

        "rigid_metric": (
            rigid_metadata["metric"]
        ),

        "similarity_metric": float(
            registration.GetMetricValue()
        ),

        "head_mask_dice": float(
            dice
        ),

        "scale": (
            scale
        ),

        "determinant": (
            matrix_stats["determinant"]
        ),

        "condition_number": (
            matrix_stats[
                "condition_number"
            ]
        ),

        "rigid_stop_condition": (
            rigid_metadata[
                "stop_condition"
            ]
        ),

        "similarity_stop_condition": (
            registration
            .GetOptimizerStopConditionDescription()
        ),
    }

    return similarity, metadata
