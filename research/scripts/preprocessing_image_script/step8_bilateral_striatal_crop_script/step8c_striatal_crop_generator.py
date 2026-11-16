#!/usr/bin/env python3

"""
Step 8C — Generate fixed bilateral striatal crops.

Frozen decisions
----------------
Localization:
    Step-7 final center
    L1 primary → L0 fallback

Crop size:
    Read ONLY from Step-8B selected_crop_size.json.

Intensity:
    Use Step-6E normalized whole-volume image.
    No additional normalization.
    No interpolation.
    No resizing.

Padding:
    Zero padding when the requested crop extends beyond
    the subject image boundary.

Template:
    The frozen bilateral DaT striatal template mask is loaded
    and validated against the Step-8B configuration for
    provenance/consistency.

Important
---------
The template ROI is NOT transformed again in this step.

Step 8B already used transformed template geometry to freeze the
global crop dimensions. Step 8C therefore uses only:

    frozen Step-7 center
            +
    frozen Step-8B crop size
            ↓
    deterministic fixed crop

Outputs
-------
output_dir/
    crops/
        <uid>.nii.gz

    striatal_crop_manifest.csv
    striatal_crop_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 8C: generate fixed striatal crops."
    )

    parser.add_argument(
        "--localization-manifest",
        type=Path,
        required=True,
        help="Frozen Step-7 localization manifest.",
    )

    parser.add_argument(
        "--selected-crop-config",
        type=Path,
        required=True,
        help="Step-8B selected_crop_size.json.",
    )

    parser.add_argument(
        "--normalized-dir",
        type=Path,
        required=True,
        help="Directory containing Step-6E normalized NIfTI scans.",
    )

    parser.add_argument(
        "--template-striatal-mask",
        type=Path,
        required=True,
        help="Frozen bilateral DaT striatal template mask.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Step-8C output directory.",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# General helpers
# -------------------------------------------------------------------------


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:

    lookup = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise ValueError(
            "Required column not found.\n"
            f"Tried: {candidates}\n"
            f"Available: {list(df.columns)}"
        )

    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def resolve_image(
    image_dir: Path,
    uid: str,
) -> Path:

    candidates = [
        image_dir / f"{uid}.nii.gz",
        image_dir / f"{uid}.nii",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No normalized NIfTI found for UID={uid} "
        f"in {image_dir}"
    )


def calculate_crop_start(
    center: float,
    size: int,
) -> int:
    """
    EXACTLY the same crop convention frozen in Step 8B.

    start = floor(center - (size - 1) / 2)
    """

    return int(
        math.floor(
            float(center)
            - (float(size) - 1.0) / 2.0
        )
    )


# -------------------------------------------------------------------------
# Crop implementation
# -------------------------------------------------------------------------


def crop_with_zero_padding(
    data: np.ndarray,
    center: np.ndarray,
    crop_shape: tuple[int, int, int],
) -> tuple[
    np.ndarray,
    tuple[int, int, int],
    dict[str, int],
]:

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D image, got shape {data.shape}"
        )

    image_shape = np.asarray(
        data.shape,
        dtype=int,
    )

    crop_shape_array = np.asarray(
        crop_shape,
        dtype=int,
    )

    starts = np.array(
        [
            calculate_crop_start(
                center[axis],
                crop_shape[axis],
            )
            for axis in range(3)
        ],
        dtype=int,
    )

    ends = starts + crop_shape_array

    source_starts = np.maximum(
        starts,
        0,
    )

    source_ends = np.minimum(
        ends,
        image_shape,
    )

    if np.any(source_ends <= source_starts):
        raise ValueError(
            "Requested crop has no overlap with source image. "
            f"Image shape={tuple(image_shape)}, "
            f"center={tuple(center)}, "
            f"crop={crop_shape}"
        )

    destination_starts = (
        source_starts - starts
    )

    destination_ends = (
        destination_starts
        + (
            source_ends
            - source_starts
        )
    )

    crop = np.zeros(
        crop_shape,
        dtype=data.dtype,
    )

    crop[
        destination_starts[0]:destination_ends[0],
        destination_starts[1]:destination_ends[1],
        destination_starts[2]:destination_ends[2],
    ] = data[
        source_starts[0]:source_ends[0],
        source_starts[1]:source_ends[1],
        source_starts[2]:source_ends[2],
    ]

    padding = {
        "pad_x_before": int(
            max(0, -starts[0])
        ),
        "pad_x_after": int(
            max(0, ends[0] - image_shape[0])
        ),

        "pad_y_before": int(
            max(0, -starts[1])
        ),
        "pad_y_after": int(
            max(0, ends[1] - image_shape[1])
        ),

        "pad_z_before": int(
            max(0, -starts[2])
        ),
        "pad_z_after": int(
            max(0, ends[2] - image_shape[2])
        ),
    }

    return (
        crop,
        tuple(int(v) for v in starts),
        padding,
    )


def cropped_affine(
    original_affine: np.ndarray,
    starts: tuple[int, int, int],
) -> np.ndarray:
    """
    Preserve world coordinates after cropping.

    Output voxel (0,0,0) corresponds to the requested crop-start
    coordinate in the original subject voxel system.

    This remains geometrically consistent even when the requested
    start is negative and zero padding is introduced.
    """

    translation = np.eye(
        4,
        dtype=float,
    )

    translation[:3, 3] = np.asarray(
        starts,
        dtype=float,
    )

    return original_affine @ translation


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------

    required_files = [
        args.localization_manifest,
        args.selected_crop_config,
        args.template_striatal_mask,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    if not args.normalized_dir.is_dir():
        raise NotADirectoryError(
            f"Normalized directory not found: "
            f"{args.normalized_dir}"
        )

    crop_dir = (
        args.output_dir
        / "crops"
    )

    crop_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Read frozen Step-8B configuration
    # ------------------------------------------------------------------

    with open(
        args.selected_crop_config,
        "r",
        encoding="utf-8",
    ) as f:
        crop_config = json.load(f)

    if not crop_config.get(
        "selection_successful",
        False,
    ):
        raise RuntimeError(
            "Step-8B crop selection was not successful."
        )

    crop_size_config = crop_config.get(
        "crop_size_vox"
    )

    if crop_size_config is None:
        raise ValueError(
            "crop_size_vox missing from Step-8B configuration."
        )

    crop_shape = (
        int(crop_size_config["x"]),
        int(crop_size_config["y"]),
        int(crop_size_config["z"]),
    )

    if any(
        dimension <= 0
        for dimension in crop_shape
    ):
        raise ValueError(
            f"Invalid crop shape: {crop_shape}"
        )

    # ------------------------------------------------------------------
    # Verify frozen template
    # ------------------------------------------------------------------

    template_img = nib.load(
        str(
            args.template_striatal_mask
        )
    )

    template_data = np.asarray(
        template_img.dataobj
    )

    template_roi_voxels = int(
        np.count_nonzero(
            template_data > 0
        )
    )

    if template_roi_voxels == 0:
        raise ValueError(
            "Template striatal mask is empty."
        )

    expected_template_voxels = (
        crop_config.get(
            "template_roi_voxels"
        )
    )

    if (
        expected_template_voxels is not None
        and int(expected_template_voxels)
        != template_roi_voxels
    ):
        raise ValueError(
            "Template mask does not match the mask "
            "used when Step 8B was frozen.\n"
            f"Step-8B ROI voxels: "
            f"{expected_template_voxels}\n"
            f"Current ROI voxels: "
            f"{template_roi_voxels}"
        )

    template_hash = sha256_file(
        args.template_striatal_mask
    )

    # ------------------------------------------------------------------
    # Read Step-7 localization
    # ------------------------------------------------------------------

    localization_df = pd.read_csv(
        args.localization_manifest
    )

    uid_col = first_existing_column(
        localization_df,
        [
            "uid",
            "subject_uid",
            "subject_id",
            "id",
        ],
    )

    center_x_col = first_existing_column(
        localization_df,
        [
            "final_center_x_vox",
            "selected_center_x_vox",
            "center_x_vox",
            "final_center_x",
            "center_x",
        ],
    )

    center_y_col = first_existing_column(
        localization_df,
        [
            "final_center_y_vox",
            "selected_center_y_vox",
            "center_y_vox",
            "final_center_y",
            "center_y",
        ],
    )

    center_z_col = first_existing_column(
        localization_df,
        [
            "final_center_z_vox",
            "selected_center_z_vox",
            "center_z_vox",
            "final_center_z",
            "center_z",
        ],
    )

    center_source_col = first_existing_column(
        localization_df,
        [
            "center_source",
            "selected_center_source",
            "final_center_source",
            "localization_source",
            "selected_method",
        ],
        required=False,
    )

    localization_df[uid_col] = (
        localization_df[
            uid_col
        ].astype(str)
    )

    if localization_df[
        uid_col
    ].duplicated().any():
        raise ValueError(
            "Duplicate UID found in Step-7 localization manifest."
        )

    # ------------------------------------------------------------------
    # Print configuration
    # ------------------------------------------------------------------

    print("=" * 72)
    print(
        "STEP 8C — FIXED STRIATAL CROP GENERATION"
    )
    print("=" * 72)

    print(
        f"Localization rows:       "
        f"{len(localization_df)}"
    )

    print(
        f"Normalized directory:    "
        f"{args.normalized_dir}"
    )

    print(
        f"Selected crop:           "
        f"{crop_shape[0]} × "
        f"{crop_shape[1]} × "
        f"{crop_shape[2]} vox"
    )

    print(
        f"Physical crop:           "
        f"{crop_shape[0] * 2.46:.2f} × "
        f"{crop_shape[1] * 2.46:.2f} × "
        f"{crop_shape[2] * 2.46:.2f} mm"
    )

    print(
        f"Template:                "
        f"{args.template_striatal_mask}"
    )

    print(
        f"Template ROI voxels:     "
        f"{template_roi_voxels}"
    )

    print(
        f"Template SHA256:         "
        f"{template_hash}"
    )

    print()
    print(
        "Intensity policy:        "
        "preserve Step-6E values"
    )

    print(
        "Interpolation:           NONE"
    )

    print(
        "Resize:                  NONE"
    )

    print(
        "Padding:                 ZERO"
    )

    print()

    # ------------------------------------------------------------------
    # Generate crops
    # ------------------------------------------------------------------

    records: list[
        dict[str, Any]
    ] = []

    for index, row in localization_df.iterrows():

        uid = str(
            row[uid_col]
        )

        record: dict[str, Any] = {
            "uid": uid,
            "successful": False,
            "error": "",
        }

        try:

            center = np.asarray(
                [
                    float(
                        row[
                            center_x_col
                        ]
                    ),
                    float(
                        row[
                            center_y_col
                        ]
                    ),
                    float(
                        row[
                            center_z_col
                        ]
                    ),
                ],
                dtype=float,
            )

            if not np.all(
                np.isfinite(center)
            ):
                raise ValueError(
                    "Non-finite localization center."
                )

            center_source = (
                str(
                    row[
                        center_source_col
                    ]
                )
                if center_source_col
                is not None
                else "unknown"
            )

            source_path = resolve_image(
                args.normalized_dir,
                uid,
            )

            source_img = nib.load(
                str(source_path)
            )

            source_data = np.asanyarray(
                source_img.dataobj
            )

            if source_data.ndim != 3:
                raise ValueError(
                    f"Expected 3D NIfTI, "
                    f"got {source_data.shape}"
                )

            source_shape = tuple(
                int(v)
                for v in source_data.shape
            )

            # ----------------------------------------------------------
            # Orientation must remain RAS
            # ----------------------------------------------------------

            orientation = tuple(
                nib.aff2axcodes(
                    source_img.affine
                )
            )

            if orientation != (
                "R",
                "A",
                "S",
            ):
                raise ValueError(
                    f"Expected RAS orientation, "
                    f"got {orientation}"
                )

            spacing = np.asarray(
                source_img.header.get_zooms()[:3],
                dtype=float,
            )

            if not np.allclose(
                spacing,
                np.asarray(
                    [2.46, 2.46, 2.46]
                ),
                atol=0.01,
                rtol=0.0,
            ):
                raise ValueError(
                    "Unexpected voxel spacing: "
                    f"{spacing}"
                )

            # ----------------------------------------------------------
            # Frozen center must be inside subject image
            # ----------------------------------------------------------

            if np.any(
                center < 0
            ) or np.any(
                center
                > (
                    np.asarray(
                        source_shape
                    )
                    - 1
                )
            ):
                raise ValueError(
                    "Localization center outside image. "
                    f"center={center}, "
                    f"shape={source_shape}"
                )

            # ----------------------------------------------------------
            # Deterministic fixed crop
            # ----------------------------------------------------------

            (
                crop_data,
                starts,
                padding,
            ) = crop_with_zero_padding(
                data=source_data,
                center=center,
                crop_shape=crop_shape,
            )

            if crop_data.shape != crop_shape:
                raise RuntimeError(
                    "Generated crop shape mismatch. "
                    f"Expected {crop_shape}, "
                    f"got {crop_data.shape}"
                )

            if not np.all(
                np.isfinite(
                    crop_data
                )
            ):
                raise ValueError(
                    "Generated crop contains NaN or Inf."
                )

            new_affine = cropped_affine(
                original_affine=np.asarray(
                    source_img.affine,
                    dtype=float,
                ),
                starts=starts,
            )

            header = (
                source_img.header.copy()
            )

            output_img = nib.Nifti1Image(
                crop_data,
                new_affine,
                header=header,
            )

            qform_code = int(
                source_img.header[
                    "qform_code"
                ]
            )

            sform_code = int(
                source_img.header[
                    "sform_code"
                ]
            )

            output_img.set_qform(
                new_affine,
                code=(
                    qform_code
                    if qform_code > 0
                    else 1
                ),
            )

            output_img.set_sform(
                new_affine,
                code=(
                    sform_code
                    if sform_code > 0
                    else 1
                ),
            )

            output_path = (
                crop_dir
                / f"{uid}.nii.gz"
            )

            nib.save(
                output_img,
                str(output_path),
            )

            # ----------------------------------------------------------
            # Validate saved file
            # ----------------------------------------------------------

            check_img = nib.load(
                str(output_path)
            )

            if tuple(
                check_img.shape[:3]
            ) != crop_shape:
                raise RuntimeError(
                    "Saved crop shape validation failed."
                )

            check_orientation = tuple(
                nib.aff2axcodes(
                    check_img.affine
                )
            )

            if check_orientation != (
                "R",
                "A",
                "S",
            ):
                raise RuntimeError(
                    "Saved crop orientation "
                    "is not RAS."
                )

            total_padding = int(
                sum(
                    padding.values()
                )
            )

            padding_required = bool(
                total_padding > 0
            )

            record.update(
                {
                    "successful": True,

                    "source_path": str(
                        source_path
                    ),

                    "output_path": str(
                        output_path
                    ),

                    "center_source": (
                        center_source
                    ),

                    "center_x_vox": float(
                        center[0]
                    ),
                    "center_y_vox": float(
                        center[1]
                    ),
                    "center_z_vox": float(
                        center[2]
                    ),

                    "source_shape_x": (
                        source_shape[0]
                    ),
                    "source_shape_y": (
                        source_shape[1]
                    ),
                    "source_shape_z": (
                        source_shape[2]
                    ),

                    "crop_x_vox": (
                        crop_shape[0]
                    ),
                    "crop_y_vox": (
                        crop_shape[1]
                    ),
                    "crop_z_vox": (
                        crop_shape[2]
                    ),

                    "crop_start_x": (
                        starts[0]
                    ),
                    "crop_start_y": (
                        starts[1]
                    ),
                    "crop_start_z": (
                        starts[2]
                    ),

                    **padding,

                    "padding_required": (
                        padding_required
                    ),

                    "total_padding_vox": (
                        total_padding
                    ),

                    "spacing_x_mm": float(
                        spacing[0]
                    ),
                    "spacing_y_mm": float(
                        spacing[1]
                    ),
                    "spacing_z_mm": float(
                        spacing[2]
                    ),

                    "orientation": (
                        "".join(
                            orientation
                        )
                    ),

                    "source_min": float(
                        np.min(
                            source_data
                        )
                    ),

                    "source_max": float(
                        np.max(
                            source_data
                        )
                    ),

                    "crop_min": float(
                        np.min(
                            crop_data
                        )
                    ),

                    "crop_max": float(
                        np.max(
                            crop_data
                        )
                    ),

                    "crop_mean": float(
                        np.mean(
                            crop_data
                        )
                    ),

                    "crop_nonzero_fraction": float(
                        np.count_nonzero(
                            crop_data
                        )
                        / crop_data.size
                    ),
                }
            )

        except Exception as exc:

            record[
                "error"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        records.append(
            record
        )

        if (
            index + 1
        ) % 100 == 0:

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(localization_df)}"
            )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    manifest_df = pd.DataFrame(
        records
    )

    manifest_path = (
        args.output_dir
        / "striatal_crop_manifest.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False,
    )

    successful_df = manifest_df[
        manifest_df[
            "successful"
        ]
        == True  # noqa: E712
    ].copy()

    failed_df = manifest_df[
        manifest_df[
            "successful"
        ]
        != True  # noqa: E712
    ].copy()

    # ------------------------------------------------------------------
    # Global validation
    # ------------------------------------------------------------------

    generated_files = list(
        crop_dir.glob(
            "*.nii.gz"
        )
    )

    unique_uids = int(
        manifest_df[
            "uid"
        ].nunique()
    )

    padding_subjects = int(
        successful_df[
            "padding_required"
        ].sum()
    )

    max_total_padding = (
        int(
            successful_df[
                "total_padding_vox"
            ].max()
        )
        if not successful_df.empty
        else 0
    )

    center_source_counts = (
        successful_df[
            "center_source"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    validation_passed = bool(
        len(successful_df)
        == len(localization_df)
        and len(failed_df) == 0
        and unique_uids
        == len(localization_df)
        and len(generated_files)
        == len(localization_df)
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary = {
        "step": "8C",

        "description": (
            "Generate fixed bilateral "
            "striatal crops"
        ),

        "number_expected": int(
            len(localization_df)
        ),

        "number_successful": int(
            len(successful_df)
        ),

        "number_failed": int(
            len(failed_df)
        ),

        "number_unique_uids": (
            unique_uids
        ),

        "number_output_files": int(
            len(generated_files)
        ),

        "crop_size_vox": {
            "x": crop_shape[0],
            "y": crop_shape[1],
            "z": crop_shape[2],
        },

        "crop_size_mm": {
            "x": (
                crop_shape[0]
                * 2.46
            ),
            "y": (
                crop_shape[1]
                * 2.46
            ),
            "z": (
                crop_shape[2]
                * 2.46
            ),
        },

        "template": {
            "path": str(
                args.template_striatal_mask
            ),
            "roi_voxels": (
                template_roi_voxels
            ),
            "sha256": (
                template_hash
            ),
        },

        "localization_manifest": str(
            args.localization_manifest
        ),

        "selected_crop_config": str(
            args.selected_crop_config
        ),

        "normalized_source_dir": str(
            args.normalized_dir
        ),

        "center_source_counts": (
            center_source_counts
        ),

        "padding": {
            "policy": "zero",
            "subjects_requiring_padding": (
                padding_subjects
            ),
            "fraction_requiring_padding": (
                padding_subjects
                / len(successful_df)
                if len(successful_df)
                else 0.0
            ),
            "maximum_total_padding_vox": (
                max_total_padding
            ),
        },

        "intensity_policy": (
            "Step-6E normalized values "
            "preserved; no renormalization"
        ),

        "interpolation": "none",

        "resize": "none",

        "orientation": "RAS",

        "validation_passed": (
            validation_passed
        ),
    }

    summary_path = (
        args.output_dir
        / "striatal_crop_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Terminal report
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "STEP 8C — RESULTS"
    )
    print("=" * 72)

    print(
        f"Expected:                   "
        f"{len(localization_df)}"
    )

    print(
        f"Successful:                 "
        f"{len(successful_df)}"
    )

    print(
        f"Failed:                     "
        f"{len(failed_df)}"
    )

    print(
        f"Unique UIDs:                "
        f"{unique_uids}"
    )

    print(
        f"Output NIfTI files:         "
        f"{len(generated_files)}"
    )

    print()
    print(
        f"Crop shape:                 "
        f"{crop_shape}"
    )

    print(
        f"Padding subjects:           "
        f"{padding_subjects}"
    )

    print(
        f"Maximum total padding:      "
        f"{max_total_padding} vox"
    )

    print(
        f"Validation passed:          "
        f"{validation_passed}"
    )

    print()
    print(
        f"Crops:    {crop_dir}"
    )

    print(
        f"Manifest: {manifest_path}"
    )

    print(
        f"Summary:  {summary_path}"
    )

    print()

    if not validation_passed:

        print(
            "STEP 8C FAILED VALIDATION."
        )

        print(
            "Do NOT continue to Step 8D."
        )

        raise RuntimeError(
            "Step 8C global validation failed."
        )

    print(
        "STEP 8C COMPLETE"
    )

    print(
        "All fixed striatal crops were generated successfully."
    )


if __name__ == "__main__":
    main()
