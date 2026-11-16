from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESAMPLED_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step4_voxel_resampler_data"
)

TEMPLATE_DIR = (
    PROJECT_ROOT
    / "data/template/dat_spect"
)

REFERENCE_MASK_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data"
    / "step5c_reference_region_analysis_data"
    / "build_reference_masks"
)

TEMPLATE_PATH = (
    TEMPLATE_DIR
    / "fpcit_template_mni.nii"
)

FINAL_REGISTRATION_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6c_finalize_registration_data/step6c_finalize_all_registration_transforms"
)

FINAL_MANIFEST = (
    FINAL_REGISTRATION_DIR
    / "final_registration_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6d_occipital_reference_data/step6d1_extract_occipital_reference"
)

MAPPED_MASK_DIR = (
    OUTPUT_DIR
    / "mapped_masks"
)

REFERENCE_CSV = (
    OUTPUT_DIR
    / "occipital_reference_statistics.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "occipital_reference_summary.json"
)


# ============================================================
# QC SCREENING
#
# These are SCREENING flags, not new frozen clinical rules.
# ============================================================

MIN_POSITIVE_FRACTION = 0.90


# ============================================================
# MASK DISCOVERY
# ============================================================

def is_nifti(path: Path) -> bool:

    name = path.name.lower()

    return (
        name.endswith(".nii")
        or
        name.endswith(".nii.gz")
    )


def find_occipital_mask(
    explicit_path: str | None,
) -> Path:
    """
    Use --occipital-mask if supplied.

    Otherwise search step5c reference-mask output recursively
    for exactly one NIfTI file containing 'occip' in its name.

    This avoids hard-coding a mask filename that may differ
    from the one created during the pilot work.
    """

    if explicit_path:

        path = Path(explicit_path)

        if not path.is_absolute():

            path = (
                PROJECT_ROOT
                / path
            )

        if not path.exists():

            raise FileNotFoundError(
                f"Occipital mask not found: {path}"
            )

        return path

    candidates = sorted(
        path
        for path in REFERENCE_MASK_DIR.rglob("*")
        if (
            path.is_file()
            and
            is_nifti(path)
            and
            "occip" in path.name.lower()
        )
    )

    if len(candidates) == 0:

        raise FileNotFoundError(
            "\nNo occipital NIfTI mask was found under:\n"
            f"{REFERENCE_MASK_DIR}\n\n"
            "Run again with the exact path, for example:\n"
            "--occipital-mask "
            "data/preprocessing_image_data/"
            "step5c_reference_region_analysis_data/"
            "build_reference_masks/"
            "<your_occipital_mask>.nii.gz"
        )

    if len(candidates) > 1:

        message = "\n".join(
            f"  - {path}"
            for path in candidates
        )

        raise RuntimeError(
            "\nMore than one occipital mask was found.\n"
            "Specify the correct one with --occipital-mask.\n"
            f"{message}"
        )

    return candidates[0]


# ============================================================
# IMAGE / MASK HELPERS
# ============================================================

def binary_mask(
    image: sitk.Image,
) -> sitk.Image:

    return sitk.Cast(
        image > 0,
        sitk.sitkUInt8,
    )


def put_mask_on_template_grid(
    mask: sitk.Image,
    template: sitk.Image,
) -> sitk.Image:
    """
    Ensure the occipital mask is represented exactly
    on the DaT template grid.

    Physical coordinates are preserved.
    """

    identity = sitk.Transform(
        3,
        sitk.sitkIdentity,
    )

    result = sitk.Resample(
        binary_mask(mask),
        template,
        identity,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    return result


def map_template_mask_to_subject(
    template_mask: sitk.Image,
    subject: sitk.Image,
    final_transform: sitk.Transform,
) -> sitk.Image:
    """
    IMPORTANT TRANSFORM DIRECTION

    Registration transform produced earlier is used as:

        Resample(
            moving_subject,
            fixed_template,
            final_transform
        )

    Therefore final_transform maps:

        template/output coordinates
            ->
        subject/input coordinates

    To resample a TEMPLATE MASK into SUBJECT SPACE,
    Resample needs:

        subject/output coordinates
            ->
        template/input coordinates

    Therefore we must use the inverse transform.
    """

    inverse_transform = (
        final_transform.GetInverse()
    )

    subject_mask = sitk.Resample(
        template_mask,
        subject,
        inverse_transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    return binary_mask(
        subject_mask
    )


# ============================================================
# QC HELPERS
# ============================================================

def mask_touches_border(
    mask_array: np.ndarray,
) -> bool:

    mask = mask_array > 0

    if not np.any(mask):
        return False

    return bool(
        np.any(mask[0, :, :])
        or
        np.any(mask[-1, :, :])
        or
        np.any(mask[:, 0, :])
        or
        np.any(mask[:, -1, :])
        or
        np.any(mask[:, :, 0])
        or
        np.any(mask[:, :, -1])
    )


def physical_voxel_volume(
    image: sitk.Image,
) -> float:

    spacing = np.asarray(
        image.GetSpacing(),
        dtype=np.float64,
    )

    return float(
        np.prod(spacing)
    )


def calculate_statistics(
    subject: sitk.Image,
    mapped_mask: sitk.Image,
    template_mask_volume_mm3: float,
) -> dict:

    image_array = (
        sitk.GetArrayFromImage(
            subject
        ).astype(
            np.float64,
            copy=False,
        )
    )

    mask_array = (
        sitk.GetArrayFromImage(
            mapped_mask
        ) > 0
    )

    mask_voxel_count = int(
        np.count_nonzero(
            mask_array
        )
    )

    if mask_voxel_count == 0:

        raise RuntimeError(
            "Mapped occipital mask contains zero voxels."
        )

    values = image_array[
        mask_array
    ]

    finite_mask = np.isfinite(
        values
    )

    finite_count = int(
        np.count_nonzero(
            finite_mask
        )
    )

    if finite_count == 0:

        raise RuntimeError(
            "Mapped occipital region contains "
            "no finite intensity values."
        )

    finite_values = values[
        finite_mask
    ]

    positive_values = finite_values[
        finite_values > 0
    ]

    positive_voxel_count = int(
        positive_values.size
    )

    positive_fraction = float(
        positive_voxel_count
        /
        finite_count
    )

    zero_or_negative_fraction = float(
        1.0
        -
        positive_fraction
    )

    # --------------------------------------------------------
    # Primary reference statistic
    #
    # R1 from the previous analysis:
    # mean intensity of the mapped occipital ROI
    # in ORIGINAL RESAMPLED SUBJECT SPACE.
    # --------------------------------------------------------

    occipital_mean = float(
        np.mean(
            finite_values
        )
    )

    occipital_median = float(
        np.median(
            finite_values
        )
    )

    occipital_std = float(
        np.std(
            finite_values
        )
    )

    occipital_p05 = float(
        np.percentile(
            finite_values,
            5,
        )
    )

    occipital_p25 = float(
        np.percentile(
            finite_values,
            25,
        )
    )

    occipital_p75 = float(
        np.percentile(
            finite_values,
            75,
        )
    )

    occipital_p95 = float(
        np.percentile(
            finite_values,
            95,
        )
    )

    occipital_min = float(
        np.min(
            finite_values
        )
    )

    occipital_max = float(
        np.max(
            finite_values
        )
    )

    # Also save positive-only values as diagnostics.
    # These are NOT automatically used as the normalization
    # reference scalar.
    if positive_voxel_count > 0:

        positive_mean = float(
            np.mean(
                positive_values
            )
        )

        positive_median = float(
            np.median(
                positive_values
            )
        )

    else:

        positive_mean = float("nan")
        positive_median = float("nan")

    subject_voxel_volume = (
        physical_voxel_volume(
            subject
        )
    )

    mapped_mask_volume_mm3 = float(
        mask_voxel_count
        *
        subject_voxel_volume
    )

    if template_mask_volume_mm3 > 0:

        physical_volume_ratio = float(
            mapped_mask_volume_mm3
            /
            template_mask_volume_mm3
        )

    else:

        physical_volume_ratio = float(
            "nan"
        )

    border_touch = (
        mask_touches_border(
            mask_array
        )
    )

    # --------------------------------------------------------
    # Screening flags
    #
    # Do not silently reject scans here.
    # Keep the full input distribution first.
    # --------------------------------------------------------

    screening_reasons = []

    if (
        positive_fraction
        <
        MIN_POSITIVE_FRACTION
    ):

        screening_reasons.append(
            "positive_fraction_lt_0.90"
        )

    if border_touch:

        screening_reasons.append(
            "mask_touches_image_border"
        )

    if (
        not np.isfinite(
            occipital_mean
        )
        or
        occipital_mean <= 0
    ):

        screening_reasons.append(
            "invalid_occipital_mean"
        )

    screening_status = (
        "pass"
        if len(
            screening_reasons
        ) == 0
        else "review"
    )

    return {

        "occipital_voxel_count":
            mask_voxel_count,

        "finite_voxel_count":
            finite_count,

        "positive_voxel_count":
            positive_voxel_count,

        "positive_fraction":
            positive_fraction,

        "zero_or_negative_fraction":
            zero_or_negative_fraction,

        # PRIMARY R1 scalar
        "occipital_mean":
            occipital_mean,

        # Diagnostics
        "occipital_median":
            occipital_median,

        "occipital_std":
            occipital_std,

        "occipital_p05":
            occipital_p05,

        "occipital_p25":
            occipital_p25,

        "occipital_p75":
            occipital_p75,

        "occipital_p95":
            occipital_p95,

        "occipital_min":
            occipital_min,

        "occipital_max":
            occipital_max,

        "positive_only_mean":
            positive_mean,

        "positive_only_median":
            positive_median,

        "mapped_mask_volume_mm3":
            mapped_mask_volume_mm3,

        "template_mask_volume_mm3":
            template_mask_volume_mm3,

        "physical_volume_ratio":
            physical_volume_ratio,

        "border_touch":
            border_touch,

        "screening_status":
            screening_status,

        "screening_reason":
            ";".join(
                screening_reasons
            ),
    }


# ============================================================
# SUMMARY HELPERS
# ============================================================

def numeric_summary(
    values: pd.Series,
) -> dict:

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values[
        np.isfinite(
            values
        )
    ].to_numpy(
        dtype=np.float64
    )

    if len(values) == 0:

        return {
            "count": 0
        }

    return {

        "count":
            int(
                len(values)
            ),

        "min":
            float(
                np.min(values)
            ),

        "p01":
            float(
                np.percentile(
                    values,
                    1,
                )
            ),

        "p05":
            float(
                np.percentile(
                    values,
                    5,
                )
            ),

        "p25":
            float(
                np.percentile(
                    values,
                    25,
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "p75":
            float(
                np.percentile(
                    values,
                    75,
                )
            ),

        "p95":
            float(
                np.percentile(
                    values,
                    95,
                )
            ),

        "p99":
            float(
                np.percentile(
                    values,
                    99,
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),

        "std":
            float(
                np.std(
                    values
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 6C: map the template-space "
            "occipital reference ROI into each "
            "subject's original resampled space "
            "and extract reference statistics."
        )
    )

    parser.add_argument(
        "--occipital-mask",
        type=str,
        default=None,
        help=(
            "Path to the template-space occipital "
            "mask. If omitted, the script searches "
            "step5c_reference_region_analysis_data/build_reference_masks/"
            "dat_spect for exactly one NIfTI filename "
            "containing 'occip'."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Technical test only. "
            "Process first N scans."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete previous Step-6C output."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    for required in [
        RESAMPLED_DIR,
        TEMPLATE_PATH,
        FINAL_MANIFEST,
    ]:

        if not required.exists():

            raise FileNotFoundError(
                f"Required path missing: "
                f"{required}"
            )

    occipital_mask_path = (
        find_occipital_mask(
            args.occipital_mask
        )
    )

    print(
        f"Occipital mask: "
        f"{occipital_mask_path}"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    if (
        args.overwrite
        and
        OUTPUT_DIR.exists()
    ):

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MAPPED_MASK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # FINAL REGISTRATION MANIFEST
    # ========================================================

    manifest = pd.read_csv(
        FINAL_MANIFEST,
        dtype={
            "uid": str,
            "file_name": str,
        },
    )

    if manifest[
        "uid"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate UIDs found in final "
            "registration manifest."
        )

    if args.limit is None:

        if (
            manifest["status"]
            !=
            "success"
        ).any():

            raise RuntimeError(
                "Final registration manifest "
                "contains unsuccessful rows."
            )

    if args.limit is not None:

        manifest = manifest.head(
            args.limit
        ).copy()

    manifest = manifest.reset_index(
        drop=True
    )

    n_scans = len(manifest)

    # ========================================================
    # LOAD TEMPLATE + TEMPLATE MASK
    # ========================================================

    template = sitk.ReadImage(
        str(TEMPLATE_PATH),
        sitk.sitkFloat32,
    )

    raw_occipital_mask = (
        sitk.ReadImage(
            str(occipital_mask_path)
        )
    )

    template_occipital_mask = (
        put_mask_on_template_grid(
            raw_occipital_mask,
            template,
        )
    )

    template_mask_array = (
        sitk.GetArrayViewFromImage(
            template_occipital_mask
        ) > 0
    )

    template_mask_voxels = int(
        np.count_nonzero(
            template_mask_array
        )
    )

    if template_mask_voxels == 0:

        raise RuntimeError(
            "Template occipital mask is empty "
            "after resampling to template grid."
        )

    template_mask_volume_mm3 = float(
        template_mask_voxels
        *
        physical_voxel_volume(
            template_occipital_mask
        )
    )

    # ========================================================
    # PROCESS
    # ========================================================

    print("=" * 90)

    print(
        "STEP 6C — OCCIPITAL REFERENCE EXTRACTION"
    )

    print("=" * 90)

    print(
        f"Scans: "
        f"{len(manifest)}"
    )

    print(
        f"Template mask voxels: "
        f"{template_mask_voxels}"
    )

    print(
        f"Template mask volume: "
        f"{template_mask_volume_mm3:.2f} mm3"
    )

    print("=" * 90)

    rows = []

    errors = []

    total = len(manifest)

    for index, registration_row in (
        manifest.iterrows()
    ):

        uid = str(
            registration_row[
                "uid"
            ]
        )

        file_name = str(
            registration_row[
                "file_name"
            ]
        )

        print(
            f"[{index + 1}/{total}] "
            f"{uid}"
        )

        subject_path = (
            RESAMPLED_DIR
            /
            file_name
        )

        transform_relative = str(
            registration_row[
                "final_transform"
            ]
        )

        transform_path = Path(
            transform_relative
        )

        if not transform_path.is_absolute():

            transform_path = (
                PROJECT_ROOT
                /
                transform_path
            )

        output_mask_path = (
            MAPPED_MASK_DIR
            /
            f"{uid}_occipital_mask.nii.gz"
        )

        try:

            if not subject_path.exists():

                raise FileNotFoundError(
                    f"Subject image missing: "
                    f"{subject_path}"
                )

            if not transform_path.exists():

                raise FileNotFoundError(
                    f"Final transform missing: "
                    f"{transform_path}"
                )

            # -----------------------------------------------
            # ORIGINAL RESAMPLED SUBJECT
            # -----------------------------------------------

            subject = sitk.ReadImage(
                str(subject_path),
                sitk.sitkFloat32,
            )

            # -----------------------------------------------
            # FINAL TRANSFORM
            # -----------------------------------------------

            final_transform = (
                sitk.ReadTransform(
                    str(transform_path)
                )
            )

            # -----------------------------------------------
            # MAP OCCIPITAL MASK TO SUBJECT SPACE
            # -----------------------------------------------

            mapped_mask = (
                map_template_mask_to_subject(
                    template_mask=
                        template_occipital_mask,

                    subject=
                        subject,

                    final_transform=
                        final_transform,
                )
            )

            # -----------------------------------------------
            # SAVE SUBJECT-SPACE MASK
            # -----------------------------------------------

            sitk.WriteImage(
                mapped_mask,
                str(
                    output_mask_path
                ),
                useCompression=True,
            )

            # -----------------------------------------------
            # EXTRACT ORIGINAL SUBJECT INTENSITIES
            # -----------------------------------------------

            stats = calculate_statistics(
                subject=
                    subject,

                mapped_mask=
                    mapped_mask,

                template_mask_volume_mm3=
                    template_mask_volume_mm3,
            )

            print(
                f"    voxels: "
                f"{stats['occipital_voxel_count']}"
            )

            print(
                f"    positive: "
                f"{stats['positive_fraction']:.4f}"
            )

            print(
                f"    mean: "
                f"{stats['occipital_mean']:.4f}"
            )

            print(
                f"    border: "
                f"{stats['border_touch']}"
            )

            print(
                f"    screening: "
                f"{stats['screening_status']}"
            )

            row = {

                "uid":
                    uid,

                "file_name":
                    file_name,

                # Final registration provenance
                "final_source_type":
                    registration_row[
                        "final_source_type"
                    ],

                "final_candidate":
                    registration_row[
                        "final_candidate"
                    ],

                "final_registration_dice":
                    registration_row[
                        "final_dice"
                    ],

                "final_registration_qc":
                    registration_row[
                        "final_qc"
                    ],

                "final_transform":
                    transform_relative,

                "mapped_occipital_mask":
                    str(
                        output_mask_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                **stats,

                "status":
                    "success",

                "error":
                    "",
            }

        except Exception as exc:

            message = str(exc)

            print(
                f"    ERROR: "
                f"{message}"
            )

            errors.append(
                {
                    "uid":
                        uid,

                    "error":
                        message,
                }
            )

            row = {

                "uid":
                    uid,

                "file_name":
                    file_name,

                "final_source_type":
                    registration_row.get(
                        "final_source_type",
                        "",
                    ),

                "final_candidate":
                    registration_row.get(
                        "final_candidate",
                        "",
                    ),

                "final_registration_dice":
                    registration_row.get(
                        "final_dice",
                        "",
                    ),

                "final_registration_qc":
                    registration_row.get(
                        "final_qc",
                        "",
                    ),

                "final_transform":
                    transform_relative,

                "mapped_occipital_mask":
                    "",

                "status":
                    "failed",

                "error":
                    message,
            }

        rows.append(
            row
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        REFERENCE_CSV,
        index=False,
    )

    successful_df = result_df[
        result_df["status"]
        ==
        "success"
    ].copy()

    failed_df = result_df[
        result_df["status"]
        !=
        "success"
    ].copy()

    # ========================================================
    # VALIDATE MASK FILES
    # ========================================================

    mask_files = list(
        MAPPED_MASK_DIR.glob(
            "*_occipital_mask.nii.gz"
        )
    )

    # ========================================================
    # SCREENING COUNTS
    # ========================================================

    if len(successful_df) > 0:

        screening_counts = (
            successful_df[
                "screening_status"
            ]
            .value_counts()
            .to_dict()
        )

        review_df = successful_df[
            successful_df[
                "screening_status"
            ]
            ==
            "review"
        ]

        review_uids = (
            review_df[
                "uid"
            ]
            .astype(str)
            .tolist()
        )

        border_touch_count = int(
            successful_df[
                "border_touch"
            ]
            .astype(bool)
            .sum()
        )

        low_positive_count = int(
            (
                successful_df[
                    "positive_fraction"
                ]
                <
                MIN_POSITIVE_FRACTION
            ).sum()
        )

    else:

        screening_counts = {}
        review_uids = []
        border_touch_count = 0
        low_positive_count = 0

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "analysis":
            (
                "Step 6C full-dataset "
                "occipital reference extraction"
            ),

        "number_of_input_scans":
            int(
                len(manifest)
            ),

        "successful":
            int(
                len(successful_df)
            ),

        "failed":
            int(
                len(failed_df)
            ),

        "mapped_mask_files":
            int(
                len(mask_files)
            ),

        "occipital_mask":
            str(
                occipital_mask_path.relative_to(
                    PROJECT_ROOT
                )
                if occipital_mask_path.is_relative_to(
                    PROJECT_ROOT
                )
                else occipital_mask_path
            ),

        "template_mask_voxels":
            template_mask_voxels,

        "template_mask_volume_mm3":
            template_mask_volume_mm3,

        "primary_reference_scalar":
            (
                "occipital_mean from original "
                "resampled subject-space intensities"
            ),

        "mapping_rule":
            (
                "Template occipital mask is mapped "
                "to subject space using the inverse "
                "of each frozen final registration "
                "transform and nearest-neighbor "
                "interpolation."
            ),

        "screening_thresholds": {

            "minimum_positive_fraction":
                MIN_POSITIVE_FRACTION,

            "border_touch":
                (
                    "flag for review"
                ),

            "important_note":
                (
                    "These are screening flags. "
                    "They do not automatically "
                    "replace the frozen normalization "
                    "strategy."
                ),
        },

        "screening_counts":
            screening_counts,

        "review_count":
            int(
                len(review_uids)
            ),

        "review_uids":
            review_uids,

        "border_touch_count":
            border_touch_count,

        "positive_fraction_below_0.90":
            low_positive_count,

        "statistics": {

            "occipital_voxel_count":
                numeric_summary(
                    successful_df[
                        "occipital_voxel_count"
                    ]
                )
                if len(successful_df)
                else {},

            "positive_fraction":
                numeric_summary(
                    successful_df[
                        "positive_fraction"
                    ]
                )
                if len(successful_df)
                else {},

            "physical_volume_ratio":
                numeric_summary(
                    successful_df[
                        "physical_volume_ratio"
                    ]
                )
                if len(successful_df)
                else {},

            "occipital_mean":
                numeric_summary(
                    successful_df[
                        "occipital_mean"
                    ]
                )
                if len(successful_df)
                else {},

            "occipital_median":
                numeric_summary(
                    successful_df[
                        "occipital_median"
                    ]
                )
                if len(successful_df)
                else {},

            "occipital_std":
                numeric_summary(
                    successful_df[
                        "occipital_std"
                    ]
                )
                if len(successful_df)
                else {},
        },

        "errors":
            errors,

        "normalization_gate":
            (
                "PENDING_QC_REVIEW"
            ),

        "next_step":
            (
                "Review the Step 6C distribution "
                "and flagged cases before saving "
                "normalized NIfTI images."
            ),
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ========================================================
    # HARD EXECUTION VALIDATION
    # ========================================================

    execution_passed = (
        len(failed_df) == 0
    )

    if args.limit is None:

        execution_passed = (
            execution_passed
            and
            len(result_df)
            ==
            n_scans
            and
            result_df["uid"].nunique()
            ==
            n_scans
            and
            len(mask_files)
            ==
            n_scans
        )

    # ========================================================
    # TERMINAL
    # ========================================================

    print()

    print("=" * 90)

    print(
        "STEP 6C COMPLETED"
    )

    print("=" * 90)

    print(
        f"Rows: "
        f"{len(result_df)}"
    )

    print(
        f"Successful: "
        f"{len(successful_df)}"
    )

    print(
        f"Failed: "
        f"{len(failed_df)}"
    )

    print(
        f"Mapped masks: "
        f"{len(mask_files)}"
    )

    print(
        f"Screening review: "
        f"{len(review_uids)}"
    )

    print(
        f"Border touch: "
        f"{border_touch_count}"
    )

    print(
        f"Positive fraction < 0.90: "
        f"{low_positive_count}"
    )

    print(
        f"Execution validation: "
        f"{execution_passed}"
    )

    print()

    print(
        f"Statistics CSV:"
        f"\n{REFERENCE_CSV}"
    )

    print()

    print(
        f"Summary:"
        f"\n{SUMMARY_JSON}"
    )

    print("=" * 90)

    if not execution_passed:

        raise RuntimeError(
            "Step 6C execution validation failed. "
            "Do not continue to normalization."
        )


if __name__ == "__main__":
    main()
