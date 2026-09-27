from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def map_template_mask_to_subject(
    template_mask: sitk.Image,
    subject_image: sitk.Image,
    subject_to_template_transform: sitk.Transform,
) -> sitk.Image:
    """
    Registration transform is used to resample subject -> template.

    To bring a template-space mask back into subject space,
    use the inverse transformation.
    """
    inverse = (
        subject_to_template_transform
        .GetInverse()
    )

    mapped = sitk.Resample(
        template_mask,
        subject_image,
        inverse,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    return mapped


def mask_touches_border(
    mask_array: np.ndarray,
) -> bool:

    if not mask_array.any():
        return False

    return bool(
        mask_array[0, :, :].any()
        or mask_array[-1, :, :].any()
        or mask_array[:, 0, :].any()
        or mask_array[:, -1, :].any()
        or mask_array[:, :, 0].any()
        or mask_array[:, :, -1].any()
    )


def analyze_reference_region(
    subject_image: sitk.Image,
    subject_mask: sitk.Image,
    template_mask: sitk.Image,
) -> dict:

    data = sitk.GetArrayFromImage(
        subject_image
    ).astype(np.float32)

    mask = (
        sitk.GetArrayFromImage(
            subject_mask
        ) > 0
    )

    template_mask_array = (
        sitk.GetArrayFromImage(
            template_mask
        ) > 0
    )

    mapped_voxels = int(mask.sum())

    template_voxels = int(
        template_mask_array.sum()
    )

    if mapped_voxels == 0:
        return {
            "mapped_voxels": 0,
            "valid": False,
            "touches_border": False,
        }

    values = data[mask]

    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.size == 0:
        return {
            "mapped_voxels": mapped_voxels,
            "valid": False,
            "touches_border": (
                mask_touches_border(mask)
            ),
        }

    positive_values = finite_values[
        finite_values > 0
    ]

    positive_fraction = float(
        positive_values.size
        / finite_values.size
    )

    spacing_subject = np.asarray(
        subject_image.GetSpacing(),
        dtype=float,
    )

    spacing_template = np.asarray(
        template_mask.GetSpacing(),
        dtype=float,
    )

    mapped_volume_mm3 = float(
        mapped_voxels
        * np.prod(spacing_subject)
    )

    template_volume_mm3 = float(
        template_voxels
        * np.prod(spacing_template)
    )

    if positive_values.size == 0:
        stats_values = finite_values
    else:
        stats_values = positive_values

    return {
        "valid": True,

        "mapped_voxels": mapped_voxels,

        "mapped_volume_mm3": (
            mapped_volume_mm3
        ),

        "template_volume_mm3": (
            template_volume_mm3
        ),

        "volume_ratio": (
            mapped_volume_mm3
            / template_volume_mm3
            if template_volume_mm3 > 0
            else None
        ),

        "positive_fraction": (
            positive_fraction
        ),

        "touches_border": (
            mask_touches_border(mask)
        ),

        "mean": float(
            np.mean(stats_values)
        ),

        "median": float(
            np.median(stats_values)
        ),

        "p75": float(
            np.percentile(
                stats_values,
                75,
            )
        ),

        "p90": float(
            np.percentile(
                stats_values,
                90,
            )
        ),

        "p95": float(
            np.percentile(
                stats_values,
                95,
            )
        ),
    }