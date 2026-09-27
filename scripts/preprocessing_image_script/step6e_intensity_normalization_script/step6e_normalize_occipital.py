from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step4_voxel_resampler_data"
)

FINAL_REFERENCE_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6d_occipital_reference_data/step6d4_finalize_reference_values"
)

FINAL_REFERENCE_CSV = (
    FINAL_REFERENCE_DIR
    / "final_reference_values.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6e_intensity_normalization_data/step6e_normalize_occipital"
)

STATISTICS_CSV = (
    OUTPUT_DIR
    / "normalization_statistics.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "normalization_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def numeric_summary(values) -> dict:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {"count": 0}

    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def read_mask(
    path: Path,
) -> sitk.Image:

    return sitk.ReadImage(
        str(path),
        sitk.sitkUInt8,
    )


def validate_same_grid(
    image: sitk.Image,
    mask: sitk.Image,
) -> None:

    if image.GetSize() != mask.GetSize():

        raise RuntimeError(
            "Image/mask size mismatch."
        )

    if not np.allclose(
        image.GetSpacing(),
        mask.GetSpacing(),
        atol=1e-5,
    ):

        raise RuntimeError(
            "Image/mask spacing mismatch."
        )

    if not np.allclose(
        image.GetOrigin(),
        mask.GetOrigin(),
        atol=1e-4,
    ):

        raise RuntimeError(
            "Image/mask origin mismatch."
        )

    if not np.allclose(
        image.GetDirection(),
        mask.GetDirection(),
        atol=1e-5,
    ):

        raise RuntimeError(
            "Image/mask direction mismatch."
        )


# ============================================================
# PROCESS ONE SCAN
# ============================================================

def process_scan(
    row: pd.Series,
) -> dict:

    uid = str(
        row["uid"]
    )

    file_name = str(
        row["file_name"]
    )

    reference_method = str(
        row["reference_method"]
    )

    reference_value = float(
        row["final_reference_value"]
    )

    if (
        not np.isfinite(reference_value)
        or
        reference_value <= 0
    ):

        raise ValueError(
            f"Invalid reference value: "
            f"{reference_value}"
        )

    # ========================================================
    # INPUT PATH
    # ========================================================

    input_path = (
        INPUT_DIR
        / file_name
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input scan missing: "
            f"{input_path}"
        )

    # ========================================================
    # MASK PATH
    # ========================================================

    mask_value = str(
        row[
            "mapped_occipital_mask"
        ]
    )

    mask_path = Path(
        mask_value
    )

    if not mask_path.is_absolute():

        mask_path = (
            PROJECT_ROOT
            / mask_path
        )

    if not mask_path.exists():

        raise FileNotFoundError(
            f"Occipital mask missing: "
            f"{mask_path}"
        )

    # ========================================================
    # READ ORIGINAL RESAMPLED IMAGE
    # ========================================================

    image = sitk.ReadImage(
        str(input_path),
        sitk.sitkFloat32,
    )

    original_array = (
        sitk.GetArrayFromImage(
            image
        ).astype(
            np.float32,
            copy=False,
        )
    )

    # ========================================================
    # VALIDATE ORIGINAL IMAGE
    # ========================================================

    original_nan_count = int(
        np.isnan(
            original_array
        ).sum()
    )

    original_inf_count = int(
        np.isinf(
            original_array
        ).sum()
    )

    if (
        original_nan_count > 0
        or
        original_inf_count > 0
    ):

        raise RuntimeError(
            f"Input contains invalid values: "
            f"NaN={original_nan_count}, "
            f"Inf={original_inf_count}"
        )

    # ========================================================
    # NORMALIZE
    #
    # THIS IS THE ACTUAL STEP 6D OPERATION.
    # ========================================================

    normalized_array = (
        original_array
        /
        np.float32(
            reference_value
        )
    )

    normalized_array = (
        normalized_array.astype(
            np.float32,
            copy=False,
        )
    )

    # ========================================================
    # VALIDATE NORMALIZED ARRAY
    # ========================================================

    nan_count = int(
        np.isnan(
            normalized_array
        ).sum()
    )

    inf_count = int(
        np.isinf(
            normalized_array
        ).sum()
    )

    if (
        nan_count > 0
        or
        inf_count > 0
    ):

        raise RuntimeError(
            f"Normalized output contains "
            f"NaN={nan_count}, "
            f"Inf={inf_count}"
        )

    # ========================================================
    # CREATE NIFTI
    #
    # CopyInformation preserves:
    # spacing
    # origin
    # direction
    # ========================================================

    normalized_image = (
        sitk.GetImageFromArray(
            normalized_array
        )
    )

    normalized_image.CopyInformation(
        image
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / file_name
    )

    sitk.WriteImage(
        normalized_image,
        str(output_path),
        useCompression=True,
    )

    # ========================================================
    # OCCIPITAL SANITY CHECK
    # ========================================================

    mask = read_mask(
        mask_path
    )

    validate_same_grid(
        image,
        mask,
    )

    mask_array = (
        sitk.GetArrayFromImage(
            mask
        ) > 0
    )

    if not np.any(
        mask_array
    ):

        raise RuntimeError(
            "Mapped occipital mask is empty."
        )

    normalized_roi = (
        normalized_array[
            mask_array
        ]
    )

    finite_roi = normalized_roi[
        np.isfinite(
            normalized_roi
        )
    ]

    if finite_roi.size == 0:

        raise RuntimeError(
            "No finite normalized ROI voxels."
        )

    positive_roi = finite_roi[
        finite_roi > 0
    ]

    normalized_full_roi_mean = float(
        np.mean(
            finite_roi
        )
    )

    if positive_roi.size:

        normalized_positive_roi_mean = float(
            np.mean(
                positive_roi
            )
        )

    else:

        normalized_positive_roi_mean = (
            float("nan")
        )

    # ========================================================
    # EXPECTED REFERENCE CHECK
    #
    # For standard scans:
    # full ROI mean should be ~1.
    #
    # For rescued scans:
    # positive-only ROI mean should be ~1.
    # ========================================================

    if (
        reference_method
        ==
        "full_occipital_mean"
    ):

        validation_reference_mean = (
            normalized_full_roi_mean
        )

        validation_type = (
            "normalized_full_roi_mean"
        )

    elif (
        reference_method
        ==
        "positive_only_occipital_mean"
    ):

        validation_reference_mean = (
            normalized_positive_roi_mean
        )

        validation_type = (
            "normalized_positive_roi_mean"
        )

    else:

        raise ValueError(
            f"Unknown reference method: "
            f"{reference_method}"
        )

    reference_error = float(
        abs(
            validation_reference_mean
            -
            1.0
        )
    )

    # ========================================================
    # IMAGE STATISTICS
    # ========================================================

    finite_normalized = (
        normalized_array[
            np.isfinite(
                normalized_array
            )
        ]
    )

    return {

        "uid":
            uid,

        "file_name":
            file_name,

        "reference_method":
            reference_method,

        "reference_value":
            reference_value,

        "reference_rescued":
            bool(
                row[
                    "reference_rescued"
                ]
            ),

        "positive_fraction":
            float(
                row[
                    "positive_fraction"
                ]
            ),

        "normalized_full_roi_mean":
            normalized_full_roi_mean,

        "normalized_positive_roi_mean":
            normalized_positive_roi_mean,

        "validation_reference_type":
            validation_type,

        "validation_reference_mean":
            validation_reference_mean,

        "reference_error_from_1":
            reference_error,

        "normalized_min":
            float(
                np.min(
                    finite_normalized
                )
            ),

        "normalized_mean":
            float(
                np.mean(
                    finite_normalized
                )
            ),

        "normalized_median":
            float(
                np.median(
                    finite_normalized
                )
            ),

        "normalized_p01":
            float(
                np.percentile(
                    finite_normalized,
                    1,
                )
            ),

        "normalized_p05":
            float(
                np.percentile(
                    finite_normalized,
                    5,
                )
            ),

        "normalized_p95":
            float(
                np.percentile(
                    finite_normalized,
                    95,
                )
            ),

        "normalized_p99":
            float(
                np.percentile(
                    finite_normalized,
                    99,
                )
            ),

        "normalized_max":
            float(
                np.max(
                    finite_normalized
                )
            ),

        "nan_count":
            nan_count,

        "inf_count":
            inf_count,

        "output_path":
            str(
                output_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "status":
            "success",

        "error":
            "",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 6D: normalize every resampled "
            "DaT scan using the frozen final "
            "occipital reference scalar."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only first N scans "
            "for technical testing."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete existing normalized output."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    for required in [
        INPUT_DIR,
        FINAL_REFERENCE_CSV,
    ]:

        if not required.exists():

            raise FileNotFoundError(
                f"Required path missing: "
                f"{required}"
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

    # ========================================================
    # LOAD FINAL REFERENCES
    # ========================================================

    df = pd.read_csv(
        FINAL_REFERENCE_CSV,
        dtype={
            "uid": str,
            "file_name": str,
        },
    )

    if df[
        "uid"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate UIDs in final "
            "reference table."
        )

    if args.limit is not None:

        df = df.head(
            args.limit
        ).copy()

    df = df.reset_index(
        drop=True
    )

    n_scans = len(df)

    # ========================================================
    # PROCESS
    # ========================================================

    print("=" * 90)

    print(
        "STEP 6D — OCCIPITAL INTENSITY NORMALIZATION"
    )

    print("=" * 90)

    print(
        f"Scans: "
        f"{len(df)}"
    )

    print("=" * 90)

    results = []
    errors = []

    total = len(df)

    for index, row in df.iterrows():

        uid = str(
            row["uid"]
        )

        print(
            f"[{index + 1}/{total}] "
            f"{uid}"
        )

        try:

            result = process_scan(
                row
            )

            results.append(
                result
            )

            print(
                f"    method: "
                f"{result['reference_method']}"
            )

            print(
                f"    denominator: "
                f"{result['reference_value']:.6f}"
            )

            print(
                f"    validation mean: "
                f"{result['validation_reference_mean']:.6f}"
            )

            print(
                f"    error from 1: "
                f"{result['reference_error_from_1']:.8f}"
            )

        except Exception as exc:

            message = str(
                exc
            )

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

            results.append(
                {
                    "uid":
                        uid,

                    "file_name":
                        str(
                            row[
                                "file_name"
                            ]
                        ),

                    "status":
                        "failed",

                    "error":
                        message,
                }
            )

    # ========================================================
    # SAVE STATISTICS
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        STATISTICS_CSV,
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
    # COUNT OUTPUT FILES
    # ========================================================

    normalized_files = list(
        OUTPUT_DIR.glob(
            "*.nii.gz"
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(successful_df):

        reference_errors = (
            successful_df[
                "reference_error_from_1"
            ].astype(float)
        )

        max_reference_error = float(
            reference_errors.max()
        )

        reference_validation_failures = int(
            (
                reference_errors
                >
                1e-4
            ).sum()
        )

        nan_total = int(
            successful_df[
                "nan_count"
            ].sum()
        )

        inf_total = int(
            successful_df[
                "inf_count"
            ].sum()
        )

    else:

        max_reference_error = None

        reference_validation_failures = 0

        nan_total = 0
        inf_total = 0

    # ========================================================
    # HARD PASS
    # ========================================================

    validation_passed = (
        len(failed_df) == 0
        and
        reference_validation_failures == 0
        and
        nan_total == 0
        and
        inf_total == 0
    )

    if args.limit is None:

        validation_passed = (
            validation_passed
            and
            len(result_df)
            ==
            n_scans
            and
            result_df[
                "uid"
            ].nunique()
            ==
            n_scans
            and
            len(
                normalized_files
            )
            ==
            n_scans
        )

    # ========================================================
    # METHOD COUNTS
    # ========================================================

    if len(successful_df):

        method_counts = (
            successful_df[
                "reference_method"
            ]
            .value_counts()
            .to_dict()
        )

    else:

        method_counts = {}

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "analysis":
            (
                "Step 6D full-dataset "
                "occipital intensity normalization"
            ),

        "number_of_rows":
            int(
                len(result_df)
            ),

        "successful":
            int(
                len(successful_df)
            ),

        "failed":
            int(
                len(failed_df)
            ),

        "normalized_nifti_files":
            int(
                len(normalized_files)
            ),

        "reference_method_counts":
            method_counts,

        "formula":
            (
                "normalized_image = "
                "original_resampled_image / "
                "final_reference_value"
            ),

        "reference_validation": {

            "expected_value":
                1.0,

            "tolerance":
                1e-4,

            "failures":
                reference_validation_failures,

            "maximum_absolute_error":
                max_reference_error,
        },

        "invalid_values": {

            "nan_total":
                nan_total,

            "inf_total":
                inf_total,
        },

        "normalized_image_mean":
            (
                numeric_summary(
                    successful_df[
                        "normalized_mean"
                    ]
                )
                if len(successful_df)
                else {}
            ),

        "normalized_image_p99":
            (
                numeric_summary(
                    successful_df[
                        "normalized_p99"
                    ]
                )
                if len(successful_df)
                else {}
            ),

        "normalized_image_max":
            (
                numeric_summary(
                    successful_df[
                        "normalized_max"
                    ]
                )
                if len(successful_df)
                else {}
            ),

        "validation_passed":
            bool(
                validation_passed
            ),

        "output_directory":
            str(
                OUTPUT_DIR.relative_to(
                    PROJECT_ROOT
                )
            ),

        "statistics_csv":
            str(
                STATISTICS_CSV.relative_to(
                    PROJECT_ROOT
                )
            ),

        "errors":
            errors,

        "important_note":
            (
                "Normalization was performed on "
                "the original resampled subject "
                "images. Registered/template-space "
                "intensity images were not used."
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
    # TERMINAL
    # ========================================================

    print()

    print("=" * 90)

    print(
        "STEP 6D COMPLETED"
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
        f"Normalized files: "
        f"{len(normalized_files)}"
    )

    print(
        f"Reference validation failures: "
        f"{reference_validation_failures}"
    )

    print(
        f"NaN total: "
        f"{nan_total}"
    )

    print(
        f"Inf total: "
        f"{inf_total}"
    )

    print(
        f"Validation passed: "
        f"{validation_passed}"
    )

    print()

    print(
        f"Statistics:"
        f"\n{STATISTICS_CSV}"
    )

    print()

    print(
        f"Summary:"
        f"\n{SUMMARY_JSON}"
    )

    print("=" * 90)

    if not validation_passed:

        raise RuntimeError(
            "Step 6D validation failed."
        )


if __name__ == "__main__":
    main()
