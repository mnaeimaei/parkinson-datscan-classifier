from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def robust_registration_image(
    image: sitk.Image,
) -> sitk.Image:
    """
    Create an intensity-scaled image ONLY for registration.

    This does not modify the image later used to measure
    reference-region uptake.
    """
    array = sitk.GetArrayFromImage(
        image
    ).astype(np.float32)

    finite = np.isfinite(array)
    positive = array[finite & (array > 0)]

    if positive.size == 0:
        raise ValueError(
            "Image has no positive finite voxels."
        )

    low = float(
        np.percentile(positive, 1)
    )

    high = float(
        np.percentile(positive, 99)
    )

    if high <= low:
        high = float(positive.max())

    clipped = np.clip(
        array,
        low,
        high,
    )

    clipped[~finite] = 0

    if high > low:
        clipped = (
            clipped - low
        ) / (
            high - low
        )

    clipped = np.clip(
        clipped,
        0,
        1,
    )

    output = sitk.GetImageFromArray(
        clipped.astype(np.float32)
    )

    output.CopyInformation(image)

    return output


def create_head_mask(
    image: sitk.Image,
) -> sitk.Image:

    mask = sitk.OtsuThreshold(
        image,
        0,
        1,
        128,
    )

    mask = sitk.BinaryMorphologicalClosing(
        mask,
        [2, 2, 2],
    )

    return sitk.Cast(
        mask,
        sitk.sitkUInt8,
    )


def _configure_registration(
    initial_transform: sitk.Transform,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    learning_rate: float,
    iterations: int,
) -> sitk.ImageRegistrationMethod:

    registration = (
        sitk.ImageRegistrationMethod()
    )

    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50
    )

    registration.SetMetricSamplingStrategy(
        registration.RANDOM
    )

    # Fixed nonzero seed = deterministic sampling.
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
        initial_transform,
        inPlace=True,
    )

    return registration


def mask_dice(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    transform: sitk.Transform,
) -> float:

    warped_moving = sitk.Resample(
        moving_mask,
        fixed_mask,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    fixed_array = (
        sitk.GetArrayFromImage(fixed_mask) > 0
    )

    moving_array = (
        sitk.GetArrayFromImage(warped_moving) > 0
    )

    denominator = (
        fixed_array.sum()
        + moving_array.sum()
    )

    if denominator == 0:
        return 0.0

    intersection = np.logical_and(
        fixed_array,
        moving_array,
    ).sum()

    return float(
        2.0 * intersection / denominator
    )


def register_affine(
    fixed_template: sitk.Image,
    moving_subject: sitk.Image,
) -> tuple[sitk.AffineTransform, dict]:

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

    # Initial rigid alignment based on the geometry/moments
    # of the broader foreground masks.
    rigid = (
        sitk.CenteredTransformInitializer(
            sitk.Cast(
                fixed_mask,
                sitk.sitkFloat32,
            ),
            sitk.Cast(
                moving_mask,
                sitk.sitkFloat32,
            ),
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )
    )

    rigid_registration = _configure_registration(
        initial_transform=rigid,
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

    # Initialize affine from rigid solution.
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
        initial_transform=affine,
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
        fixed_mask,
        moving_mask,
        affine,
    )

    metadata = {
        "rigid_metric": rigid_metric,
        "affine_metric": affine_metric,
        "head_mask_dice": dice,
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


def save_transform(
    transform: sitk.Transform,
    output_path: str | Path,
) -> None:

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sitk.WriteTransform(
        transform,
        str(output_path),
    )