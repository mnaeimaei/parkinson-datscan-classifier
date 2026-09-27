from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

STEP6C_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6d_occipital_reference_data/step6d1_extract_occipital_reference"
)

REFERENCE_CSV = (
    STEP6C_DIR
    / "occipital_reference_statistics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6d_occipital_reference_data/step6d4_finalize_reference_values"
)

FINAL_REFERENCE_CSV = (
    OUTPUT_DIR
    / "final_reference_values.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "final_reference_summary.json"
)


POSITIVE_FRACTION_THRESHOLD = 0.90


# ============================================================
# HELPERS
# ============================================================

def numeric_summary(
    values: pd.Series,
) -> dict:

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ].to_numpy(
        dtype=np.float64
    )

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


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 6C-4: freeze one final occipital "
            "reference scalar for every scan."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if not REFERENCE_CSV.exists():

        raise FileNotFoundError(
            f"Missing Step-6C CSV: {REFERENCE_CSV}"
        )

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
    # READ STEP 6C
    # ========================================================

    df = pd.read_csv(
        REFERENCE_CSV,
        dtype={
            "uid": str,
            "file_name": str,
        },
    )

    df = df[
        df["status"] == "success"
    ].copy()

    if df["uid"].duplicated().any():

        raise RuntimeError(
            "Duplicate UIDs found."
        )

    n_scans = len(df)

    numeric_columns = [
        "positive_fraction",
        "occipital_mean",
        "positive_only_mean",
        "positive_only_median",
        "occipital_voxel_count",
        "positive_voxel_count",
        "physical_volume_ratio",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    # ========================================================
    # FINAL REFERENCE RULE
    # ========================================================

    use_positive = (
        df["positive_fraction"]
        <
        POSITIVE_FRACTION_THRESHOLD
    )

    df["reference_method"] = np.where(
        use_positive,
        "positive_only_occipital_mean",
        "full_occipital_mean",
    )

    df["final_reference_value"] = np.where(
        use_positive,
        df["positive_only_mean"],
        df["occipital_mean"],
    )

    df["reference_rescued"] = (
        use_positive
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    df["full_reference_value"] = (
        df["occipital_mean"]
    )

    df["positive_reference_value"] = (
        df["positive_only_mean"]
    )

    df["reference_change_absolute"] = (
        df["final_reference_value"]
        -
        df["occipital_mean"]
    )

    df["reference_change_percent"] = (
        (
            df["final_reference_value"]
            -
            df["occipital_mean"]
        )
        /
        df["occipital_mean"]
        *
        100.0
    )

    # ========================================================
    # HARD VALIDATION
    # ========================================================

    invalid_reference = (
        ~np.isfinite(
            df["final_reference_value"]
        )
        |
        (
            df["final_reference_value"]
            <= 0
        )
    )

    invalid_count = int(
        invalid_reference.sum()
    )

    if invalid_count:

        bad = df.loc[
            invalid_reference,
            [
                "uid",
                "final_reference_value",
            ],
        ]

        raise RuntimeError(
            "Invalid final reference values:\n"
            + bad.to_string(
                index=False
            )
        )

    if df["final_reference_value"].isna().any():

        raise RuntimeError(
            "Missing final reference values."
        )

    # ========================================================
    # SELECT OUTPUT COLUMNS
    # ========================================================

    output_columns = [
        "uid",
        "file_name",

        "reference_method",
        "final_reference_value",
        "reference_rescued",

        "positive_fraction",

        "full_reference_value",
        "positive_reference_value",

        "reference_change_absolute",
        "reference_change_percent",

        "occipital_voxel_count",
        "positive_voxel_count",

        "border_touch",
        "physical_volume_ratio",

        "final_source_type",
        "final_candidate",
        "final_registration_dice",
        "final_registration_qc",

        "mapped_occipital_mask",
        "final_transform",
    ]

    output_df = df[
        output_columns
    ].copy()

    output_df.to_csv(
        FINAL_REFERENCE_CSV,
        index=False,
    )

    # ========================================================
    # COUNTS
    # ========================================================

    method_counts = (
        output_df[
            "reference_method"
        ]
        .value_counts()
        .to_dict()
    )

    rescued_df = output_df[
        output_df[
            "reference_rescued"
        ]
    ].copy()

    nonrescued_df = output_df[
        ~output_df[
            "reference_rescued"
        ]
    ].copy()

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    validation_passed = all(
        [
            len(output_df)
            == n_scans,

            output_df[
                "uid"
            ].nunique()
            == n_scans,

            invalid_count
            == 0,

            output_df[
                "final_reference_value"
            ].notna().all(),
        ]
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "analysis":
            (
                "Step 6C-4 definitive occipital "
                "reference-value selection"
            ),

        "number_of_scans":
            int(
                len(output_df)
            ),

        "unique_uids":
            int(
                output_df[
                    "uid"
                ].nunique()
            ),

        "policy": {

            "positive_fraction_threshold":
                POSITIVE_FRACTION_THRESHOLD,

            "positive_fraction_gte_0.90":
                (
                    "Use full mapped occipital "
                    "ROI mean."
                ),

            "positive_fraction_lt_0.90":
                (
                    "Use mean of finite positive "
                    "voxels inside the same mapped "
                    "occipital ROI."
                ),

            "border_touch":
                (
                    "Border contact alone does not "
                    "change the reference method."
                ),
        },

        "reference_method_counts":
            method_counts,

        "rescued_reference_count":
            int(
                len(rescued_df)
            ),

        "standard_reference_count":
            int(
                len(nonrescued_df)
            ),

        "final_reference_value":
            numeric_summary(
                output_df[
                    "final_reference_value"
                ]
            ),

        "rescued_reference_change_percent":
            numeric_summary(
                rescued_df[
                    "reference_change_percent"
                ]
            ),

        "standard_reference_change_percent":
            numeric_summary(
                nonrescued_df[
                    "reference_change_percent"
                ]
            ),

        "invalid_reference_count":
            invalid_count,

        "final_validation_passed":
            bool(
                validation_passed
            ),

        "final_reference_csv":
            str(
                FINAL_REFERENCE_CSV.relative_to(
                    PROJECT_ROOT
                )
            ),

        "important_note":
            (
                "No normalized images are created "
                "in this step. This file freezes "
                "the denominator that Step 6D "
                "will use for every scan."
            ),

        "next_step":
            (
                "Proceed to Step 6D image "
                "normalization only if "
                "final_validation_passed is true."
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

    print("=" * 90)

    print(
        "STEP 6C-4 — FINAL REFERENCE VALUES"
    )

    print("=" * 90)

    print(
        f"Rows: "
        f"{len(output_df)}"
    )

    print(
        f"Unique UIDs: "
        f"{output_df['uid'].nunique()}"
    )

    print()

    print("Reference methods:")

    for method, count in (
        method_counts.items()
    ):

        print(
            f"    {method}: "
            f"{count}"
        )

    print()

    print(
        f"Rescued references: "
        f"{len(rescued_df)}"
    )

    print(
        f"Invalid references: "
        f"{invalid_count}"
    )

    print(
        f"Validation passed: "
        f"{validation_passed}"
    )

    print()

    print(
        f"Final reference CSV:"
        f"\n{FINAL_REFERENCE_CSV}"
    )

    print()

    print(
        f"Summary:"
        f"\n{SUMMARY_JSON}"
    )

    print("=" * 90)

    if not validation_passed:

        raise RuntimeError(
            "Final reference validation failed. "
            "Do not continue to Step 6D."
        )


if __name__ == "__main__":
    main()
