#!/usr/bin/env python3

"""
Step 8F — Freeze final bilateral striatal crop pipeline.

This step DOES NOT modify, crop, normalize, interpolate, or resize images.

It verifies and freezes the outputs of:

    Step 8A — crop extent analysis
    Step 8B — fixed crop-size selection
    Step 8C — crop generation
    Step 8D — automatic QC
    Step 8E — manual visual QC

Frozen decisions
----------------
Localization:
    Step-7 frozen center
    L1 primary → L0 fallback

Crop dimensions:
    Read from Step-8B selected_crop_size.json

Intensity:
    Step-6D normalized whole-volume values preserved

Interpolation:
    None during cropping

Padding:
    Zero padding

Template:
    Frozen bilateral DaT striatum mask

Outputs
-------
output_dir/
    final_striatal_crop_manifest.csv
    final_striatal_crop_config.json
    freeze_summary.json

The final manifest contains one row for every final crop and includes a
SHA256 digest so the frozen crop set can later be checked for accidental
modification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
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
        description="Step 8F: freeze final striatal crop pipeline."
    )

    parser.add_argument(
        "--extent-statistics",
        type=Path,
        required=True,
        help="Step-8A striatal_extent_statistics.csv",
    )

    parser.add_argument(
        "--crop-config",
        type=Path,
        required=True,
        help="Step-8B selected_crop_size.json",
    )

    parser.add_argument(
        "--crop-manifest",
        type=Path,
        required=True,
        help="Step-8C crop manifest CSV",
    )

    parser.add_argument(
        "--crop-summary",
        type=Path,
        required=True,
        help="Step-8C crop summary JSON",
    )

    parser.add_argument(
        "--automatic-qc-manifest",
        type=Path,
        required=True,
        help="Step-8D automatic-QC manifest",
    )

    parser.add_argument(
        "--automatic-qc-summary",
        type=Path,
        required=True,
        help="Step-8D automatic-QC summary JSON",
    )

    parser.add_argument(
        "--visual-qc-selection",
        type=Path,
        required=True,
        help="Step-8E selected-case CSV",
    )

    parser.add_argument(
        "--visual-qc-summary",
        type=Path,
        required=True,
        help="Step-8E visual-QC summary JSON",
    )

    parser.add_argument(
        "--visual-qc-dir",
        type=Path,
        required=True,
        help="Directory containing Step-8E QC PNGs",
    )

    parser.add_argument(
        "--crop-dir",
        type=Path,
        required=True,
        help="Step-8C final crop directory",
    )

    parser.add_argument(
        "--template-striatal-mask",
        type=Path,
        required=True,
        help="Frozen bilateral DaT striatum mask",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    # Manual visual-QC approval.
    parser.add_argument(
        "--visual-qc-approved",
        action="store_true",
        help=(
            "Explicit confirmation that the selected Step-8E figures "
            "were manually reviewed and approved."
        ),
    )

    parser.add_argument(
        "--visual-qc-reviewed-count",
        type=int,
        default=None,
        help=(
            "Number of Step-8E cases that were reviewed. "
            "Defaults to the Step-8E selection size."
        ),
    )

    parser.add_argument(
        "--visual-qc-pass-count",
        type=int,
        default=None,
        help=(
            "Number of Step-8E cases that passed review. "
            "Defaults to the Step-8E selection size when approved."
        ),
    )

    parser.add_argument(
        "--visual-qc-review-count",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--visual-qc-fail-count",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--visual-qc-reviewer",
        type=str,
        default="manual_review",
    )

    parser.add_argument(
        "--expected-subjects",
        type=int,
        default=None,
        help=(
            "Expected subject count. Defaults to the Step-8A row "
            "count; Step 8C and 8D must match."
        ),
    )

    parser.add_argument(
        "--expected-spacing",
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

    return parser.parse_args()


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def to_bool(value: Any) -> bool:
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
            "pass",
            "passed",
        }
    )


def resolve_crop(
    crop_dir: Path,
    uid: str,
) -> Path:

    candidates = [
        crop_dir / f"{uid}.nii.gz",
        crop_dir / f"{uid}.nii",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Final crop not found for UID={uid}"
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

    required_files = [
        args.extent_statistics,
        args.crop_config,
        args.crop_manifest,
        args.crop_summary,
        args.automatic_qc_manifest,
        args.automatic_qc_summary,
        args.visual_qc_selection,
        args.visual_qc_summary,
        args.template_striatal_mask,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required frozen input not found: {path}"
            )

    for directory in [
        args.crop_dir,
        args.visual_qc_dir,
    ]:

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Required directory not found: {directory}"
            )

    # ------------------------------------------------------------------
    # Read frozen summaries/configuration
    # ------------------------------------------------------------------

    crop_config = read_json(
        args.crop_config
    )

    crop_summary = read_json(
        args.crop_summary
    )

    automatic_qc_summary = read_json(
        args.automatic_qc_summary
    )

    visual_qc_summary = read_json(
        args.visual_qc_summary
    )

    # ------------------------------------------------------------------
    # STEP 8B validation
    # ------------------------------------------------------------------

    if not crop_config.get(
        "selection_successful",
        False,
    ):
        raise RuntimeError(
            "Step 8B is not marked successful."
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
    # Template freeze
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

    frozen_template_voxels = (
        crop_config.get(
            "template_roi_voxels"
        )
    )

    if (
        frozen_template_voxels is not None
        and int(
            frozen_template_voxels
        )
        != template_roi_voxels
    ):
        raise RuntimeError(
            "Template ROI voxel count differs from Step 8B."
        )

    # Check against Step 8C.
    step8c_template = crop_summary.get(
        "template",
        {},
    )

    step8c_hash = step8c_template.get(
        "sha256"
    )

    if (
        step8c_hash is not None
        and step8c_hash
        != template_hash
    ):
        raise RuntimeError(
            "Template SHA256 differs from Step 8C."
        )

    # Check against Step 8D.
    step8d_template = automatic_qc_summary.get(
        "template",
        {},
    )

    step8d_hash = step8d_template.get(
        "sha256"
    )

    if (
        step8d_hash is not None
        and step8d_hash
        != template_hash
    ):
        raise RuntimeError(
            "Template SHA256 differs from Step 8D."
        )

    # Check against Step 8E.
    step8e_template = visual_qc_summary.get(
        "template",
        {},
    )

    step8e_hash = step8e_template.get(
        "sha256"
    )

    if (
        step8e_hash is not None
        and step8e_hash
        != template_hash
    ):
        raise RuntimeError(
            "Template SHA256 differs from Step 8E."
        )

    # ------------------------------------------------------------------
    # STEP 8A
    # ------------------------------------------------------------------

    extent_df = pd.read_csv(
        args.extent_statistics
    )

    extent_df[
        "uid"
    ] = (
        extent_df[
            "uid"
        ].astype(str)
    )

    if args.expected_subjects is None:
        args.expected_subjects = int(
            len(extent_df)
        )

    if len(
        extent_df
    ) != args.expected_subjects:
        raise RuntimeError(
            "Unexpected number of Step-8A subjects: "
            f"{len(extent_df)}"
        )

    if extent_df[
        "uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate UID in Step-8A statistics."
        )

    # ------------------------------------------------------------------
    # STEP 8C
    # ------------------------------------------------------------------

    if not crop_summary.get(
        "validation_passed",
        False,
    ):
        raise RuntimeError(
            "Step 8C global validation did not pass."
        )

    if int(
        crop_summary.get(
            "number_successful",
            -1,
        )
    ) != args.expected_subjects:
        raise RuntimeError(
            "Step 8C did not generate all subjects."
        )

    crop_df = pd.read_csv(
        args.crop_manifest
    )

    crop_df[
        "uid"
    ] = (
        crop_df[
            "uid"
        ].astype(str)
    )

    if len(
        crop_df
    ) != args.expected_subjects:
        raise RuntimeError(
            "Unexpected Step-8C manifest size."
        )

    if crop_df[
        "uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate UID in Step-8C manifest."
        )

    # ------------------------------------------------------------------
    # STEP 8D
    # ------------------------------------------------------------------

    if not automatic_qc_summary.get(
        "global_qc_pass",
        False,
    ):
        raise RuntimeError(
            "Step 8D global automatic QC failed."
        )

    if int(
        automatic_qc_summary.get(
            "number_qc_pass",
            -1,
        )
    ) != args.expected_subjects:
        raise RuntimeError(
            "Step 8D did not pass all subjects."
        )

    if int(
        automatic_qc_summary.get(
            "number_qc_fail",
            -1,
        )
    ) != 0:
        raise RuntimeError(
            "Step 8D contains QC failures."
        )

    automatic_qc_df = pd.read_csv(
        args.automatic_qc_manifest
    )

    automatic_qc_df[
        "uid"
    ] = (
        automatic_qc_df[
            "uid"
        ].astype(str)
    )

    if len(
        automatic_qc_df
    ) != args.expected_subjects:
        raise RuntimeError(
            "Unexpected Step-8D manifest size."
        )

    if automatic_qc_df[
        "uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate UID in Step-8D manifest."
        )

    if not automatic_qc_df[
        "qc_pass"
    ].map(
        to_bool
    ).all():
        raise RuntimeError(
            "At least one Step-8D subject did not pass."
        )

    # ------------------------------------------------------------------
    # STEP 8E — generation + manual approval
    # ------------------------------------------------------------------

    visual_selection_df = pd.read_csv(
        args.visual_qc_selection
    )

    visual_selection_df[
        "uid"
    ] = (
        visual_selection_df[
            "uid"
        ].astype(str)
    )

    selected_count = int(
        len(
            visual_selection_df
        )
    )

    if args.visual_qc_reviewed_count is None:
        args.visual_qc_reviewed_count = selected_count

    if args.visual_qc_pass_count is None:
        args.visual_qc_pass_count = selected_count

    generated_count = int(
        visual_qc_summary.get(
            "number_images_generated",
            -1,
        )
    )

    generation_failures = int(
        visual_qc_summary.get(
            "number_generation_failures",
            -1,
        )
    )

    if generated_count != selected_count:
        raise RuntimeError(
            "Step 8E generated-image count does not "
            "match selected-case count."
        )

    if generation_failures != 0:
        raise RuntimeError(
            "Step 8E contains image-generation failures."
        )

    png_files = list(
        args.visual_qc_dir.glob(
            "*_visual_qc.png"
        )
    )

    if len(
        png_files
    ) < selected_count:
        raise RuntimeError(
            "Not all Step-8E visual-QC PNGs are present."
        )

    if not args.visual_qc_approved:
        raise RuntimeError(
            "Manual Step-8E visual-QC approval was not provided."
        )

    if (
        args.visual_qc_reviewed_count
        != selected_count
    ):
        raise RuntimeError(
            "Manual visual-QC reviewed count "
            "does not match Step-8E selection count."
        )

    if (
        args.visual_qc_pass_count
        != selected_count
    ):
        raise RuntimeError(
            "Not every Step-8E selected case was manually passed."
        )

    if args.visual_qc_review_count != 0:
        raise RuntimeError(
            "One or more visual-QC subjects remain under review."
        )

    if args.visual_qc_fail_count != 0:
        raise RuntimeError(
            "One or more visual-QC subjects failed."
        )

    # ------------------------------------------------------------------
    # Cross-manifest UID consistency
    # ------------------------------------------------------------------

    extent_uids = set(
        extent_df[
            "uid"
        ]
    )

    crop_uids = set(
        crop_df[
            "uid"
        ]
    )

    qc_uids = set(
        automatic_qc_df[
            "uid"
        ]
    )

    if not (
        extent_uids
        == crop_uids
        == qc_uids
    ):
        raise RuntimeError(
            "UID sets differ between Steps 8A, 8C and 8D."
        )

    # ------------------------------------------------------------------
    # Prepare final merged manifest
    # ------------------------------------------------------------------

    useful_crop_columns = [
        column
        for column in [
            "uid",
            "source_path",
            "output_path",
            "center_source",

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

            "padding_required",
            "total_padding_vox",

            "spacing_x_mm",
            "spacing_y_mm",
            "spacing_z_mm",

            "orientation",
        ]
        if column in crop_df.columns
    ]

    useful_qc_columns = [
        column
        for column in [
            "uid",
            "qc_pass",
            "extent_valid",
            "roi_coverage_pass",
            "minimum_roi_margin_vox",
            "world_coordinate_error_mm",
            "max_absolute_intensity_error",
        ]
        if column in automatic_qc_df.columns
    ]

    final_df = crop_df[
        useful_crop_columns
    ].merge(
        automatic_qc_df[
            useful_qc_columns
        ],
        on="uid",
        how="inner",
    )

    if len(
        final_df
    ) != args.expected_subjects:
        raise RuntimeError(
            "Final manifest merge produced unexpected row count."
        )

    # ------------------------------------------------------------------
    # Final on-disk validation + SHA256
    # ------------------------------------------------------------------

    print("=" * 72)
    print(
        "STEP 8F — FREEZE FINAL STRIATAL CROP PIPELINE"
    )
    print("=" * 72)

    print(
        f"Subjects expected:       "
        f"{args.expected_subjects}"
    )

    print(
        f"Frozen crop shape:       "
        f"{crop_shape}"
    )

    print(
        f"Expected spacing:        "
        f"{tuple(args.expected_spacing)}"
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

    crop_hashes = []
    actual_crop_paths = []

    final_shape_failures = 0
    final_orientation_failures = 0
    final_spacing_failures = 0

    for index, row in final_df.iterrows():

        uid = str(
            row["uid"]
        )

        crop_path = resolve_crop(
            args.crop_dir,
            uid,
        )

        image = nib.load(
            str(
                crop_path
            )
        )

        shape = tuple(
            int(v)
            for v in image.shape[:3]
        )

        if shape != crop_shape:
            final_shape_failures += 1

        orientation = tuple(
            nib.aff2axcodes(
                image.affine
            )
        )

        if orientation != (
            "R",
            "A",
            "S",
        ):
            final_orientation_failures += 1

        spacing = np.asarray(
            image.header.get_zooms()[:3],
            dtype=float,
        )

        if not np.allclose(
            spacing,
            np.asarray(
                args.expected_spacing,
                dtype=float,
            ),
            atol=args.spacing_tolerance,
            rtol=0.0,
        ):
            final_spacing_failures += 1

        crop_hashes.append(
            sha256_file(
                crop_path
            )
        )

        actual_crop_paths.append(
            str(
                crop_path
            )
        )

        if (
            index + 1
        ) % 100 == 0:

            print(
                f"Verified and hashed "
                f"{index + 1}/"
                f"{len(final_df)}"
            )

    final_df[
        "final_crop_path"
    ] = actual_crop_paths

    final_df[
        "final_crop_sha256"
    ] = crop_hashes

    final_df[
        "frozen_crop_x_vox"
    ] = crop_shape[0]

    final_df[
        "frozen_crop_y_vox"
    ] = crop_shape[1]

    final_df[
        "frozen_crop_z_vox"
    ] = crop_shape[2]

    final_df[
        "pipeline_status"
    ] = "FROZEN"

    # ------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------

    duplicate_hash_count = int(
        final_df[
            "final_crop_sha256"
        ].duplicated().sum()
    )

    final_validation_passed = bool(
        len(
            final_df
        )
        == args.expected_subjects
        and final_shape_failures == 0
        and final_orientation_failures == 0
        and final_spacing_failures == 0
        and final_df[
            "qc_pass"
        ].map(
            to_bool
        ).all()
    )

    if not final_validation_passed:
        raise RuntimeError(
            "Final Step-8F validation failed."
        )

    # ------------------------------------------------------------------
    # Save final manifest
    # ------------------------------------------------------------------

    final_manifest_path = (
        args.output_dir
        / "final_striatal_crop_manifest.csv"
    )

    final_df.to_csv(
        final_manifest_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Freeze configuration
    # ------------------------------------------------------------------

    freeze_timestamp = (
        datetime.now(
            timezone.utc
        )
        .replace(
            microsecond=0
        )
        .isoformat()
    )

    final_config = {
        "pipeline_step": "8",

        "pipeline_status": "FROZEN",

        "freeze_timestamp_utc": (
            freeze_timestamp
        ),

        "number_subjects": (
            args.expected_subjects
        ),

        "localization": {
            "policy": (
                "Frozen Step-7 localization center"
            ),
            "primary": "L1 template-transform center",
            "fallback": "L0 array center",
            "l2_status": "rejected",
        },

        "crop": {
            "shape_vox": {
                "x": crop_shape[0],
                "y": crop_shape[1],
                "z": crop_shape[2],
            },

            "shape_mm": {
                "x": (
                    crop_shape[0]
                    * args.expected_spacing[0]
                ),
                "y": (
                    crop_shape[1]
                    * args.expected_spacing[1]
                ),
                "z": (
                    crop_shape[2]
                    * args.expected_spacing[2]
                ),
            },

            "spacing_mm": {
                "x": args.expected_spacing[0],
                "y": args.expected_spacing[1],
                "z": args.expected_spacing[2],
            },

            "crop_index_policy": (
                "start = floor(center - "
                "(crop_size - 1) / 2)"
            ),

            "padding_policy": "zero",

            "interpolation": "none",

            "resize": "none",

            "orientation": "RAS",
        },

        "intensity": {
            "source": (
                "Step-6D normalized whole scan"
            ),
            "additional_normalization_after_crop": False,
            "intensity_values_preserved": True,
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

        "step8a": {
            "number_subjects": int(
                len(
                    extent_df
                )
            ),

            "extent_valid_subjects": int(
                extent_df[
                    "extent_valid_for_size_selection"
                ].map(
                    to_bool
                ).sum()
            ),
        },

        "step8b": {
            "selection_successful": True,
            "selected_crop_size_vox": {
                "x": crop_shape[0],
                "y": crop_shape[1],
                "z": crop_shape[2],
            },
        },

        "step8c": {
            "number_successful": int(
                crop_summary[
                    "number_successful"
                ]
            ),

            "validation_passed": bool(
                crop_summary[
                    "validation_passed"
                ]
            ),

            "padding_subjects": int(
                crop_summary[
                    "padding"
                ][
                    "subjects_requiring_padding"
                ]
            ),

            "maximum_total_padding_vox": int(
                crop_summary[
                    "padding"
                ][
                    "maximum_total_padding_vox"
                ]
            ),
        },

        "step8d": {
            "number_qc_pass": int(
                automatic_qc_summary[
                    "number_qc_pass"
                ]
            ),

            "number_qc_fail": int(
                automatic_qc_summary[
                    "number_qc_fail"
                ]
            ),

            "global_qc_pass": bool(
                automatic_qc_summary[
                    "global_qc_pass"
                ]
            ),

            "extent_coverage_fraction": float(
                automatic_qc_summary[
                    "extent_coverage"
                ][
                    "coverage_fraction"
                ]
            ),

            "minimum_roi_margin_vox": float(
                automatic_qc_summary[
                    "extent_coverage"
                ][
                    "minimum_observed_margin_vox"
                ]
            ),
        },

        "step8e": {
            "number_selected": (
                selected_count
            ),

            "number_reviewed": (
                args.visual_qc_reviewed_count
            ),

            "number_pass": (
                args.visual_qc_pass_count
            ),

            "number_review": (
                args.visual_qc_review_count
            ),

            "number_fail": (
                args.visual_qc_fail_count
            ),

            "manual_visual_qc_passed": True,

            "reviewer": (
                args.visual_qc_reviewer
            ),
        },

        "final_manifest": str(
            final_manifest_path
        ),

        "crop_files_hashed": True,

        "note": (
            "This configuration freezes Step 8. "
            "Any future change to localization, crop size, template, "
            "padding, normalization or crop-generation policy requires "
            "a new preprocessing version rather than silently modifying "
            "this frozen configuration."
        ),
    }

    config_path = (
        args.output_dir
        / "final_striatal_crop_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_config,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Freeze summary
    # ------------------------------------------------------------------

    summary = {
        "step": "8F",

        "description": (
            "Freeze final bilateral striatal crop pipeline"
        ),

        "number_subjects": int(
            len(
                final_df
            )
        ),

        "crop_shape_vox": {
            "x": crop_shape[0],
            "y": crop_shape[1],
            "z": crop_shape[2],
        },

        "template_sha256": (
            template_hash
        ),

        "step8a_pass": True,
        "step8b_pass": True,
        "step8c_pass": True,
        "step8d_pass": True,
        "step8e_manual_pass": True,

        "final_shape_failures": (
            final_shape_failures
        ),

        "final_orientation_failures": (
            final_orientation_failures
        ),

        "final_spacing_failures": (
            final_spacing_failures
        ),

        "duplicate_crop_hash_count": (
            duplicate_hash_count
        ),

        "final_validation_passed": (
            final_validation_passed
        ),

        "step8_complete": (
            final_validation_passed
        ),
    }

    summary_path = (
        args.output_dir
        / "freeze_summary.json"
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
    # Terminal output
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "STEP 8F — FINAL RESULT"
    )
    print("=" * 72)

    print(
        f"Subjects frozen:            "
        f"{len(final_df)}"
    )

    print(
        f"Crop shape:                 "
        f"{crop_shape}"
    )

    print(
        f"Shape failures:             "
        f"{final_shape_failures}"
    )

    print(
        f"Orientation failures:       "
        f"{final_orientation_failures}"
    )

    print(
        f"Spacing failures:           "
        f"{final_spacing_failures}"
    )

    print(
        f"Automatic QC failures:      0"
    )

    print(
        f"Visual QC:                  "
        f"{args.visual_qc_pass_count}/"
        f"{args.visual_qc_reviewed_count} PASS"
    )

    print(
        f"Template consistency:       PASS"
    )

    print(
        f"Crop SHA256 generated:      "
        f"{len(crop_hashes)}"
    )

    print()

    print(
        f"Final validation passed:    "
        f"{final_validation_passed}"
    )

    print()

    print(
        f"Final manifest: "
        f"{final_manifest_path}"
    )

    print(
        f"Final config:   "
        f"{config_path}"
    )

    print(
        f"Summary:        "
        f"{summary_path}"
    )

    print()

    print("=" * 72)
    print(
        "STEP 8 COMPLETE — STRIATAL CROP PIPELINE FROZEN"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
