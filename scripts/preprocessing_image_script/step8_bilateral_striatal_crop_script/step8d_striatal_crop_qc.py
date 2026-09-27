#!/usr/bin/env python3

"""
Step 8D — Automatic QC of fixed bilateral striatal crops.

Frozen inputs
-------------
Step 7:
    Frozen localization center
    L1 primary → L0 fallback

Step 8A:
    Transformed bilateral striatal ROI extent

Step 8B:
    Fixed crop size = frozen selected_crop_size.json

Step 8C:
    Generated fixed NIfTI crops and crop manifest

Template:
    Frozen bilateral DaT striatal mask

QC checks
---------
For every subject:

1. Crop exists
2. Crop shape is exactly the frozen Step-8B shape
3. Crop orientation is RAS
4. Crop spacing remains 2.46 × 2.46 × 2.46 mm
5. No NaN / Inf
6. Frozen localization center lies inside the crop
7. Crop affine/world geometry is consistent with source image
8. Non-padded voxels exactly preserve Step-6D intensity values
9. Padded regions contain zero only
10. Padding recorded in Step-8C manifest matches recomputed padding
11. Bilateral transformed template ROI is fully inside crop
    for Step-8A extent-valid subjects
12. Template provenance is unchanged

Important
---------
Padding itself is NOT a QC failure.

Subjects requiring zero padding are valid if:
    - the padding matches the expected geometry
    - padded voxels are zero
    - the copied source region is unchanged

Output
------
step8_bilateral_striatal_crop_data/step8d_striatal_crop_qc/
    crop_qc_manifest.csv
    crop_qc_summary.json
    review_cases.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
        description="Step 8D: automatic QC of striatal crops."
    )

    parser.add_argument(
        "--crop-manifest",
        type=Path,
        required=True,
        help="Step-8C crop manifest CSV.",
    )

    parser.add_argument(
        "--extent-statistics",
        type=Path,
        required=True,
        help="Step-8A striatal extent statistics CSV.",
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
        help="Step-6D normalized whole-volume directory.",
    )

    parser.add_argument(
        "--crop-dir",
        type=Path,
        required=True,
        help="Step-8C generated crop directory.",
    )

    parser.add_argument(
        "--template-striatal-mask",
        type=Path,
        required=True,
        help="Frozen bilateral DaT striatal mask.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(2.46, 2.46, 2.46),
        metavar=("X", "Y", "Z"),
    )

    parser.add_argument(
        "--spacing-tolerance",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--intensity-atol",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--intensity-rtol",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--world-tolerance-mm",
        type=float,
        default=1e-4,
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def to_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "pass",
        "passed",
    }


def resolve_image(
    directory: Path,
    uid: str,
) -> Path:

    candidates = [
        directory / f"{uid}.nii.gz",
        directory / f"{uid}.nii",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"NIfTI not found for UID={uid} in {directory}"
    )


def world_coordinate(
    affine: np.ndarray,
    voxel_coordinate: np.ndarray,
) -> np.ndarray:

    homogeneous = np.append(
        np.asarray(voxel_coordinate, dtype=float),
        1.0,
    )

    return (
        np.asarray(affine, dtype=float)
        @ homogeneous
    )[:3]


def reconstruct_expected_crop(
    source_data: np.ndarray,
    starts: np.ndarray,
    crop_shape: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, int]]:

    source_shape = np.asarray(
        source_data.shape,
        dtype=int,
    )

    crop_shape_array = np.asarray(
        crop_shape,
        dtype=int,
    )

    ends = (
        starts
        + crop_shape_array
    )

    source_starts = np.maximum(
        starts,
        0,
    )

    source_ends = np.minimum(
        ends,
        source_shape,
    )

    destination_starts = (
        source_starts
        - starts
    )

    destination_ends = (
        destination_starts
        + (
            source_ends
            - source_starts
        )
    )

    expected = np.zeros(
        crop_shape,
        dtype=source_data.dtype,
    )

    expected[
        destination_starts[0]:destination_ends[0],
        destination_starts[1]:destination_ends[1],
        destination_starts[2]:destination_ends[2],
    ] = source_data[
        source_starts[0]:source_ends[0],
        source_starts[1]:source_ends[1],
        source_starts[2]:source_ends[2],
    ]

    padding = {
        "pad_x_before": int(
            max(0, -starts[0])
        ),
        "pad_x_after": int(
            max(
                0,
                ends[0] - source_shape[0],
            )
        ),

        "pad_y_before": int(
            max(0, -starts[1])
        ),
        "pad_y_after": int(
            max(
                0,
                ends[1] - source_shape[1],
            )
        ),

        "pad_z_before": int(
            max(0, -starts[2])
        ),
        "pad_z_after": int(
            max(
                0,
                ends[2] - source_shape[2],
            )
        ),
    }

    return expected, padding


def transformed_roi_inside_crop(
    extent_row: pd.Series,
    starts: np.ndarray,
    crop_shape: tuple[int, int, int],
) -> tuple[bool, float]:

    crop_shape_array = np.asarray(
        crop_shape,
        dtype=float,
    )

    # Same support convention used in Step 8B.
    crop_min = (
        starts.astype(float)
        - 0.5
    )

    crop_max = (
        starts.astype(float)
        + crop_shape_array
        - 0.5
    )

    roi_min = np.asarray(
        [
            float(
                extent_row[
                    "roi_min_x_vox"
                ]
            ),
            float(
                extent_row[
                    "roi_min_y_vox"
                ]
            ),
            float(
                extent_row[
                    "roi_min_z_vox"
                ]
            ),
        ]
    )

    roi_max = np.asarray(
        [
            float(
                extent_row[
                    "roi_max_x_vox"
                ]
            ),
            float(
                extent_row[
                    "roi_max_y_vox"
                ]
            ),
            float(
                extent_row[
                    "roi_max_z_vox"
                ]
            ),
        ]
    )

    lower_margin = (
        roi_min
        - crop_min
    )

    upper_margin = (
        crop_max
        - roi_max
    )

    all_margins = np.concatenate(
        [
            lower_margin,
            upper_margin,
        ]
    )

    minimum_margin = float(
        np.min(
            all_margins
        )
    )

    covered = bool(
        np.all(
            lower_margin >= 0.0
        )
        and np.all(
            upper_margin >= 0.0
        )
    )

    return (
        covered,
        minimum_margin,
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Validate input paths
    # ------------------------------------------------------------------

    for path in [
        args.crop_manifest,
        args.extent_statistics,
        args.selected_crop_config,
        args.template_striatal_mask,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    for directory in [
        args.normalized_dir,
        args.crop_dir,
    ]:
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Directory not found: {directory}"
            )

    # ------------------------------------------------------------------
    # Frozen Step-8B configuration
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

    crop_shape = (
        int(
            crop_config[
                "crop_size_vox"
            ]["x"]
        ),
        int(
            crop_config[
                "crop_size_vox"
            ]["y"]
        ),
        int(
            crop_config[
                "crop_size_vox"
            ]["z"]
        ),
    )

    # ------------------------------------------------------------------
    # Template validation
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

    template_hash = sha256_file(
        args.template_striatal_mask
    )

    frozen_roi_voxels = (
        crop_config.get(
            "template_roi_voxels"
        )
    )

    template_roi_match = bool(
        frozen_roi_voxels is None
        or int(
            frozen_roi_voxels
        )
        == template_roi_voxels
    )

    # ------------------------------------------------------------------
    # Read Step-8A and Step-8C
    # ------------------------------------------------------------------

    crop_df = pd.read_csv(
        args.crop_manifest
    )

    extent_df = pd.read_csv(
        args.extent_statistics
    )

    required_crop_columns = [
        "uid",
        "successful",

        "center_x_vox",
        "center_y_vox",
        "center_z_vox",

        "crop_start_x",
        "crop_start_y",
        "crop_start_z",

        "pad_x_before",
        "pad_x_after",
        "pad_y_before",
        "pad_y_after",
        "pad_z_before",
        "pad_z_after",
    ]

    missing = [
        col
        for col in required_crop_columns
        if col not in crop_df.columns
    ]

    if missing:
        raise ValueError(
            "Step-8C manifest missing columns:\n"
            + "\n".join(
                missing
            )
        )

    extent_required_columns = [
        "uid",
        "extent_valid_for_size_selection",

        "roi_min_x_vox",
        "roi_min_y_vox",
        "roi_min_z_vox",

        "roi_max_x_vox",
        "roi_max_y_vox",
        "roi_max_z_vox",
    ]

    missing = [
        col
        for col in extent_required_columns
        if col not in extent_df.columns
    ]

    if missing:
        raise ValueError(
            "Step-8A statistics missing columns:\n"
            + "\n".join(
                missing
            )
        )

    crop_df["uid"] = (
        crop_df["uid"]
        .astype(str)
    )

    extent_df["uid"] = (
        extent_df["uid"]
        .astype(str)
    )

    if crop_df[
        "uid"
    ].duplicated().any():
        raise ValueError(
            "Duplicate UID in Step-8C manifest."
        )

    if extent_df[
        "uid"
    ].duplicated().any():
        raise ValueError(
            "Duplicate UID in Step-8A statistics."
        )

    extent_lookup = (
        extent_df
        .set_index(
            "uid"
        )
    )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    print("=" * 72)
    print(
        "STEP 8D — AUTOMATIC STRIATAL CROP QC"
    )
    print("=" * 72)

    print(
        f"Crop manifest rows:       "
        f"{len(crop_df)}"
    )

    print(
        f"Expected crop shape:      "
        f"{crop_shape}"
    )

    print(
        f"Expected spacing:         "
        f"{tuple(args.spacing)}"
    )

    print(
        f"Template ROI voxels:      "
        f"{template_roi_voxels}"
    )

    print(
        f"Template ROI match:       "
        f"{template_roi_match}"
    )

    print(
        f"Template SHA256:          "
        f"{template_hash}"
    )

    print()

    # ------------------------------------------------------------------
    # Subject QC
    # ------------------------------------------------------------------

    records: list[
        dict[str, Any]
    ] = []

    for index, row in crop_df.iterrows():

        uid = str(
            row["uid"]
        )

        record: dict[str, Any] = {
            "uid": uid,
            "qc_pass": False,
            "review_required": True,
            "error": "",
        }

        try:

            if not to_bool(
                row["successful"]
            ):
                raise RuntimeError(
                    "Step-8C marked crop unsuccessful."
                )

            if uid not in extent_lookup.index:
                raise KeyError(
                    "UID missing from Step-8A statistics."
                )

            extent_row = (
                extent_lookup.loc[
                    uid
                ]
            )

            source_path = resolve_image(
                args.normalized_dir,
                uid,
            )

            crop_path = resolve_image(
                args.crop_dir,
                uid,
            )

            source_img = nib.load(
                str(
                    source_path
                )
            )

            crop_img = nib.load(
                str(
                    crop_path
                )
            )

            source_data = np.asanyarray(
                source_img.dataobj
            )

            crop_data = np.asanyarray(
                crop_img.dataobj
            )

            # ----------------------------------------------------------
            # Shape
            # ----------------------------------------------------------

            shape_pass = bool(
                tuple(
                    crop_data.shape
                )
                == crop_shape
            )

            # ----------------------------------------------------------
            # Finite values
            # ----------------------------------------------------------

            finite_pass = bool(
                np.all(
                    np.isfinite(
                        crop_data
                    )
                )
            )

            # ----------------------------------------------------------
            # Orientation
            # ----------------------------------------------------------

            crop_orientation = tuple(
                nib.aff2axcodes(
                    crop_img.affine
                )
            )

            orientation_pass = bool(
                crop_orientation
                == (
                    "R",
                    "A",
                    "S",
                )
            )

            # ----------------------------------------------------------
            # Spacing
            # ----------------------------------------------------------

            crop_spacing = np.asarray(
                crop_img.header.get_zooms()[:3],
                dtype=float,
            )

            spacing_pass = bool(
                np.allclose(
                    crop_spacing,
                    np.asarray(
                        args.spacing,
                        dtype=float,
                    ),
                    atol=(
                        args.spacing_tolerance
                    ),
                    rtol=0.0,
                )
            )

            # ----------------------------------------------------------
            # Center
            # ----------------------------------------------------------

            center_source_vox = np.asarray(
                [
                    float(
                        row[
                            "center_x_vox"
                        ]
                    ),
                    float(
                        row[
                            "center_y_vox"
                        ]
                    ),
                    float(
                        row[
                            "center_z_vox"
                        ]
                    ),
                ],
                dtype=float,
            )

            starts = np.asarray(
                [
                    int(
                        row[
                            "crop_start_x"
                        ]
                    ),
                    int(
                        row[
                            "crop_start_y"
                        ]
                    ),
                    int(
                        row[
                            "crop_start_z"
                        ]
                    ),
                ],
                dtype=int,
            )

            center_crop_vox = (
                center_source_vox
                - starts
            )

            center_inside_pass = bool(
                np.all(
                    center_crop_vox
                    >= 0.0
                )
                and np.all(
                    center_crop_vox
                    <= (
                        np.asarray(
                            crop_shape,
                            dtype=float,
                        )
                        - 1.0
                    )
                )
            )

            # ----------------------------------------------------------
            # World-coordinate consistency
            # ----------------------------------------------------------

            source_center_world = world_coordinate(
                source_img.affine,
                center_source_vox,
            )

            crop_center_world = world_coordinate(
                crop_img.affine,
                center_crop_vox,
            )

            world_error_mm = float(
                np.linalg.norm(
                    source_center_world
                    - crop_center_world
                )
            )

            affine_pass = bool(
                world_error_mm
                <= args.world_tolerance_mm
            )

            # ----------------------------------------------------------
            # Reconstruct exact expected crop from source
            # ----------------------------------------------------------

            (
                expected_crop,
                expected_padding,
            ) = reconstruct_expected_crop(
                source_data=source_data,
                starts=starts,
                crop_shape=crop_shape,
            )

            intensity_pass = bool(
                np.allclose(
                    crop_data,
                    expected_crop,
                    atol=args.intensity_atol,
                    rtol=args.intensity_rtol,
                    equal_nan=False,
                )
            )

            max_absolute_intensity_error = float(
                np.max(
                    np.abs(
                        crop_data.astype(
                            np.float64
                        )
                        - expected_crop.astype(
                            np.float64
                        )
                    )
                )
            )

            # ----------------------------------------------------------
            # Padding manifest consistency
            # ----------------------------------------------------------

            manifest_padding = {
                "pad_x_before": int(
                    row[
                        "pad_x_before"
                    ]
                ),
                "pad_x_after": int(
                    row[
                        "pad_x_after"
                    ]
                ),
                "pad_y_before": int(
                    row[
                        "pad_y_before"
                    ]
                ),
                "pad_y_after": int(
                    row[
                        "pad_y_after"
                    ]
                ),
                "pad_z_before": int(
                    row[
                        "pad_z_before"
                    ]
                ),
                "pad_z_after": int(
                    row[
                        "pad_z_after"
                    ]
                ),
            }

            padding_manifest_pass = bool(
                manifest_padding
                == expected_padding
            )

            padding_required = bool(
                sum(
                    expected_padding.values()
                )
                > 0
            )

            total_padding_vox = int(
                sum(
                    expected_padding.values()
                )
            )

            # ----------------------------------------------------------
            # Check actual padded regions are zero
            # ----------------------------------------------------------

            padded_zero_pass = True

            px0 = expected_padding[
                "pad_x_before"
            ]
            px1 = expected_padding[
                "pad_x_after"
            ]

            py0 = expected_padding[
                "pad_y_before"
            ]
            py1 = expected_padding[
                "pad_y_after"
            ]

            pz0 = expected_padding[
                "pad_z_before"
            ]
            pz1 = expected_padding[
                "pad_z_after"
            ]

            padded_regions = []

            if px0 > 0:
                padded_regions.append(
                    crop_data[
                        :px0,
                        :,
                        :,
                    ]
                )

            if px1 > 0:
                padded_regions.append(
                    crop_data[
                        crop_shape[0]
                        - px1:,
                        :,
                        :,
                    ]
                )

            if py0 > 0:
                padded_regions.append(
                    crop_data[
                        :,
                        :py0,
                        :,
                    ]
                )

            if py1 > 0:
                padded_regions.append(
                    crop_data[
                        :,
                        crop_shape[1]
                        - py1:,
                        :,
                    ]
                )

            if pz0 > 0:
                padded_regions.append(
                    crop_data[
                        :,
                        :,
                        :pz0,
                    ]
                )

            if pz1 > 0:
                padded_regions.append(
                    crop_data[
                        :,
                        :,
                        crop_shape[2]
                        - pz1:,
                    ]
                )

            for region in padded_regions:
                if not np.all(
                    region == 0
                ):
                    padded_zero_pass = False
                    break

            # ----------------------------------------------------------
            # Step-8A transformed ROI coverage
            # ----------------------------------------------------------

            extent_valid = to_bool(
                extent_row[
                    "extent_valid_for_size_selection"
                ]
            )

            if extent_valid:

                (
                    roi_coverage_pass,
                    minimum_roi_margin_vox,
                ) = transformed_roi_inside_crop(
                    extent_row=extent_row,
                    starts=starts,
                    crop_shape=crop_shape,
                )

            else:

                # Center remains valid but transformed extent was
                # rejected in Step 8A (similarity-rescue or
                # scale/anisotropy exclusion).
                roi_coverage_pass = None
                minimum_roi_margin_vox = None

            # ----------------------------------------------------------
            # Final subject QC
            # ----------------------------------------------------------

            mandatory_checks = [
                shape_pass,
                finite_pass,
                orientation_pass,
                spacing_pass,
                center_inside_pass,
                affine_pass,
                intensity_pass,
                padding_manifest_pass,
                padded_zero_pass,
            ]

            if extent_valid:
                mandatory_checks.append(
                    bool(
                        roi_coverage_pass
                    )
                )

            qc_pass = bool(
                all(
                    mandatory_checks
                )
            )

            record.update(
                {
                    "qc_pass": qc_pass,
                    "review_required": (
                        not qc_pass
                    ),

                    "shape_pass": (
                        shape_pass
                    ),
                    "finite_pass": (
                        finite_pass
                    ),
                    "orientation_pass": (
                        orientation_pass
                    ),
                    "spacing_pass": (
                        spacing_pass
                    ),
                    "center_inside_pass": (
                        center_inside_pass
                    ),
                    "affine_pass": (
                        affine_pass
                    ),
                    "intensity_pass": (
                        intensity_pass
                    ),
                    "padding_manifest_pass": (
                        padding_manifest_pass
                    ),
                    "padded_zero_pass": (
                        padded_zero_pass
                    ),

                    "extent_valid": (
                        extent_valid
                    ),

                    "roi_coverage_pass": (
                        roi_coverage_pass
                    ),

                    "minimum_roi_margin_vox": (
                        minimum_roi_margin_vox
                    ),

                    "world_coordinate_error_mm": (
                        world_error_mm
                    ),

                    "max_absolute_intensity_error": (
                        max_absolute_intensity_error
                    ),

                    "padding_required": (
                        padding_required
                    ),

                    "total_padding_vox": (
                        total_padding_vox
                    ),

                    "crop_orientation": (
                        "".join(
                            crop_orientation
                        )
                    ),

                    "crop_spacing_x": float(
                        crop_spacing[0]
                    ),
                    "crop_spacing_y": float(
                        crop_spacing[1]
                    ),
                    "crop_spacing_z": float(
                        crop_spacing[2]
                    ),

                    "center_crop_x_vox": float(
                        center_crop_vox[0]
                    ),
                    "center_crop_y_vox": float(
                        center_crop_vox[1]
                    ),
                    "center_crop_z_vox": float(
                        center_crop_vox[2]
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
                f"{len(crop_df)}"
            )

    # ------------------------------------------------------------------
    # Save subject QC
    # ------------------------------------------------------------------

    qc_df = pd.DataFrame(
        records
    )

    qc_manifest_path = (
        args.output_dir
        / "crop_qc_manifest.csv"
    )

    qc_df.to_csv(
        qc_manifest_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Review cases
    # ------------------------------------------------------------------

    review_df = qc_df[
        qc_df[
            "review_required"
        ]
        == True  # noqa: E712
    ].copy()

    review_path = (
        args.output_dir
        / "review_cases.csv"
    )

    review_df.to_csv(
        review_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Aggregate QC
    # ------------------------------------------------------------------

    number_pass = int(
        qc_df[
            "qc_pass"
        ].sum()
    )

    number_fail = int(
        len(qc_df)
        - number_pass
    )

    padding_subjects = int(
        qc_df[
            "padding_required"
        ].sum()
    )

    eligible_extent_df = qc_df[
        qc_df[
            "extent_valid"
        ]
        == True  # noqa: E712
    ]

    roi_coverage_count = int(
        eligible_extent_df[
            "roi_coverage_pass"
        ].sum()
    )

    minimum_margin = (
        float(
            pd.to_numeric(
                eligible_extent_df[
                    "minimum_roi_margin_vox"
                ],
                errors="coerce",
            )
            .dropna()
            .min()
        )
        if not eligible_extent_df.empty
        else None
    )

    check_columns = [
        "shape_pass",
        "finite_pass",
        "orientation_pass",
        "spacing_pass",
        "center_inside_pass",
        "affine_pass",
        "intensity_pass",
        "padding_manifest_pass",
        "padded_zero_pass",
    ]

    check_failures = {}

    for column in check_columns:

        check_failures[
            column
        ] = int(
            (
                qc_df[column]
                != True  # noqa: E712
            ).sum()
        )

    global_pass = bool(
        number_fail == 0
        and template_roi_match
        and roi_coverage_count
        == len(
            eligible_extent_df
        )
    )

    summary = {
        "step": "8D",

        "description": (
            "Automatic QC of fixed bilateral "
            "striatal crops"
        ),

        "number_subjects": int(
            len(qc_df)
        ),

        "number_qc_pass": (
            number_pass
        ),

        "number_qc_fail": (
            number_fail
        ),

        "number_review_required": int(
            len(review_df)
        ),

        "crop_shape_expected": {
            "x": crop_shape[0],
            "y": crop_shape[1],
            "z": crop_shape[2],
        },

        "padding_subjects": (
            padding_subjects
        ),

        "maximum_total_padding_vox": int(
            qc_df[
                "total_padding_vox"
            ].max()
        ),

        "template": {
            "path": str(
                args.template_striatal_mask
            ),
            "roi_voxels": (
                template_roi_voxels
            ),
            "roi_voxel_count_matches_step8b": (
                template_roi_match
            ),
            "sha256": (
                template_hash
            ),
        },

        "extent_coverage": {
            "number_extent_valid": int(
                len(
                    eligible_extent_df
                )
            ),
            "number_fully_covered": (
                roi_coverage_count
            ),
            "coverage_fraction": (
                roi_coverage_count
                / len(
                    eligible_extent_df
                )
                if len(
                    eligible_extent_df
                )
                else None
            ),
            "minimum_observed_margin_vox": (
                minimum_margin
            ),
        },

        "check_failure_counts": (
            check_failures
        ),

        "maximum_world_coordinate_error_mm": float(
            qc_df[
                "world_coordinate_error_mm"
            ].max()
        ),

        "maximum_absolute_intensity_error": float(
            qc_df[
                "max_absolute_intensity_error"
            ].max()
        ),

        "global_qc_pass": (
            global_pass
        ),
    }

    summary_path = (
        args.output_dir
        / "crop_qc_summary.json"
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
        "STEP 8D — RESULTS"
    )
    print("=" * 72)

    print(
        f"Subjects:                   "
        f"{len(qc_df)}"
    )

    print(
        f"QC passed:                  "
        f"{number_pass}"
    )

    print(
        f"QC failed:                  "
        f"{number_fail}"
    )

    print(
        f"Review required:            "
        f"{len(review_df)}"
    )

    print()

    print(
        f"Shape failures:             "
        f"{check_failures['shape_pass']}"
    )

    print(
        f"Finite-value failures:      "
        f"{check_failures['finite_pass']}"
    )

    print(
        f"Orientation failures:       "
        f"{check_failures['orientation_pass']}"
    )

    print(
        f"Spacing failures:           "
        f"{check_failures['spacing_pass']}"
    )

    print(
        f"Center failures:            "
        f"{check_failures['center_inside_pass']}"
    )

    print(
        f"Affine failures:            "
        f"{check_failures['affine_pass']}"
    )

    print(
        f"Intensity-copy failures:    "
        f"{check_failures['intensity_pass']}"
    )

    print(
        f"Padding-manifest failures:  "
        f"{check_failures['padding_manifest_pass']}"
    )

    print(
        f"Padding-zero failures:      "
        f"{check_failures['padded_zero_pass']}"
    )

    print()

    print(
        f"Padding subjects:           "
        f"{padding_subjects}"
    )

    print(
        f"Extent-valid subjects:      "
        f"{len(eligible_extent_df)}"
    )

    print(
        f"Template ROI fully covered: "
        f"{roi_coverage_count}/"
        f"{len(eligible_extent_df)}"
    )

    print(
        f"Minimum ROI margin:         "
        f"{minimum_margin:.3f} vox"
        if minimum_margin is not None
        else "Minimum ROI margin:         N/A"
    )

    print()

    print(
        f"Global QC pass:             "
        f"{global_pass}"
    )

    print()
    print(
        f"QC manifest:  {qc_manifest_path}"
    )

    print(
        f"Review cases: {review_path}"
    )

    print(
        f"Summary:      {summary_path}"
    )

    print()

    if not global_pass:

        print(
            "STEP 8D FAILED."
        )

        print(
            "Do NOT proceed to Step 8E until "
            "the automatic QC failures are reviewed."
        )

        raise RuntimeError(
            "Step 8D automatic QC failed."
        )

    print(
        "STEP 8D COMPLETE"
    )

    print(
        "Automatic QC passed for all generated crops."
    )


if __name__ == "__main__":
    main()
