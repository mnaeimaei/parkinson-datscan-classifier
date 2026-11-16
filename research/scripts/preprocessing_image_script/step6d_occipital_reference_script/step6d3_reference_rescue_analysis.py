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

STEP6C2_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6d_occipital_reference_data/step6d2_occipital_mask_qc"
)

QC_DECISION_CSV = (
    STEP6C2_DIR
    / "occipital_qc_decisions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6d_occipital_reference_data/step6d3_reference_rescue_analysis"
)

ANALYSIS_CSV = (
    OUTPUT_DIR
    / "reference_rescue_analysis.csv"
)

REVIEW_CSV = (
    OUTPUT_DIR
    / "review_reference_candidates.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "reference_rescue_summary.json"
)


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
        return {
            "count": 0
        }

    return {
        "count":
            int(len(values)),

        "min":
            float(np.min(values)),

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
                np.median(values)
            ),

        "mean":
            float(
                np.mean(values)
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
                np.max(values)
            ),

        "std":
            float(
                np.std(values)
            ),
    }


def bool_series(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(numerator)
        &
        np.isfinite(denominator)
        &
        (denominator != 0)
    )

    result.loc[valid] = (
        numerator.loc[valid]
        /
        denominator.loc[valid]
    )

    return result


def percentage_difference(
    candidate: pd.Series,
    baseline: pd.Series,
) -> pd.Series:
    """
    Relative change:

        (candidate - baseline) / baseline * 100
    """

    ratio = safe_ratio(
        candidate - baseline,
        baseline,
    )

    return ratio * 100.0


# ============================================================
# REVIEW CATEGORY
# ============================================================

def classify_review_category(
    positive_fraction: float,
    border_touch: bool,
) -> str:

    low_positive = (
        positive_fraction < 0.90
    )

    if low_positive and border_touch:

        return (
            "both_low_positive_and_border"
        )

    if low_positive:

        return (
            "low_positive_only"
        )

    if border_touch:

        return (
            "border_touch_only"
        )

    return "not_flagged"


# ============================================================
# MANUAL QC DECISIONS
# ============================================================

def load_manual_decisions() -> pd.DataFrame:
    """
    Decisions are optional at this stage.

    The script can run even if you have not yet
    classified all 55 PNGs.
    """

    if not QC_DECISION_CSV.exists():

        return pd.DataFrame(
            columns=[
                "uid",
                "visual_decision",
                "visual_confidence",
                "visual_notes",
            ]
        )

    qc_df = pd.read_csv(
        QC_DECISION_CSV,
        dtype={
            "uid": str,
        },
    )

    required = [
        "uid",
        "visual_decision",
        "visual_confidence",
        "visual_notes",
    ]

    for column in required:

        if column not in qc_df.columns:

            qc_df[column] = ""

    qc_df = qc_df[
        required
    ].copy()

    qc_df = qc_df.drop_duplicates(
        subset=["uid"],
        keep="last",
    )

    return qc_df


# ============================================================
# GROUP ANALYSIS
# ============================================================

def group_summary(
    df: pd.DataFrame,
) -> dict:

    if len(df) == 0:

        return {
            "count": 0
        }

    difference = (
        pd.to_numeric(
            df[
                "positive_mean_change_percent"
            ],
            errors="coerce",
        )
    )

    return {

        "count":
            int(len(df)),

        "positive_fraction":
            numeric_summary(
                df[
                    "positive_fraction"
                ]
            ),

        "full_occipital_mean":
            numeric_summary(
                df[
                    "occipital_mean"
                ]
            ),

        "positive_only_mean":
            numeric_summary(
                df[
                    "positive_only_mean"
                ]
            ),

        "positive_mean_change_percent":
            numeric_summary(
                difference
            ),

        "full_to_positive_ratio":
            numeric_summary(
                df[
                    "full_to_positive_ratio"
                ]
            ),

        "change_gt_1_percent":
            int(
                (
                    difference.abs()
                    >
                    1.0
                ).sum()
            ),

        "change_gt_5_percent":
            int(
                (
                    difference.abs()
                    >
                    5.0
                ).sum()
            ),

        "change_gt_10_percent":
            int(
                (
                    difference.abs()
                    >
                    10.0
                ).sum()
            ),

        "change_gt_25_percent":
            int(
                (
                    difference.abs()
                    >
                    25.0
                ).sum()
            ),

        "change_gt_50_percent":
            int(
                (
                    difference.abs()
                    >
                    50.0
                ).sum()
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 6C-3: compare full occipital "
            "mean with positive-only candidate "
            "reference statistics before "
            "normalization."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE
    # ========================================================

    if not REFERENCE_CSV.exists():

        raise FileNotFoundError(
            f"Missing Step-6C CSV: "
            f"{REFERENCE_CSV}"
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
        df["status"]
        ==
        "success"
    ].copy()

    if df["uid"].duplicated().any():

        raise RuntimeError(
            "Duplicate UIDs found."
        )

    # ========================================================
    # TYPES
    # ========================================================

    numeric_columns = [
        "positive_fraction",
        "occipital_mean",
        "occipital_median",
        "positive_only_mean",
        "positive_only_median",
        "physical_volume_ratio",
        "occipital_voxel_count",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["border_touch"] = bool_series(
        df["border_touch"]
    )

    # ========================================================
    # REVIEW CATEGORY
    # ========================================================

    df["review_category"] = [
        classify_review_category(
            positive_fraction=float(
                positive_fraction
            ),
            border_touch=bool(
                border_touch
            ),
        )
        for positive_fraction, border_touch
        in zip(
            df["positive_fraction"],
            df["border_touch"],
        )
    ]

    # ========================================================
    # CANDIDATE REFERENCE COMPARISONS
    # ========================================================

    # Current R1
    df[
        "reference_full_mean"
    ] = df[
        "occipital_mean"
    ]

    # Candidate fallback A
    df[
        "reference_positive_mean"
    ] = df[
        "positive_only_mean"
    ]

    # Candidate fallback B
    df[
        "reference_positive_median"
    ] = df[
        "positive_only_median"
    ]

    # --------------------------------------------------------
    # How much did zeros/non-positive values dilute
    # the current full ROI mean?
    # --------------------------------------------------------

    df[
        "full_to_positive_ratio"
    ] = safe_ratio(
        df[
            "occipital_mean"
        ],
        df[
            "positive_only_mean"
        ],
    )

    df[
        "positive_to_full_ratio"
    ] = safe_ratio(
        df[
            "positive_only_mean"
        ],
        df[
            "occipital_mean"
        ],
    )

    df[
        "positive_mean_change_percent"
    ] = percentage_difference(
        df[
            "positive_only_mean"
        ],
        df[
            "occipital_mean"
        ],
    )

    df[
        "positive_median_change_percent"
    ] = percentage_difference(
        df[
            "positive_only_median"
        ],
        df[
            "occipital_mean"
        ],
    )

    # --------------------------------------------------------
    # Expected relationship if excluded values are
    # essentially zero:
    #
    # full mean ≈ positive mean × positive fraction
    #
    # This is useful to confirm that zero-padding is
    # what is causing the difference.
    # --------------------------------------------------------

    df[
        "expected_full_from_positive"
    ] = (
        df[
            "positive_only_mean"
        ]
        *
        df[
            "positive_fraction"
        ]
    )

    df[
        "zero_dilution_consistency_error_percent"
    ] = percentage_difference(
        df[
            "expected_full_from_positive"
        ],
        df[
            "occipital_mean"
        ],
    )

    # ========================================================
    # QC DECISIONS, IF AVAILABLE
    # ========================================================

    decisions = load_manual_decisions()

    df = df.merge(
        decisions,
        on="uid",
        how="left",
    )

    for column in [
        "visual_decision",
        "visual_confidence",
        "visual_notes",
    ]:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
        )

    # ========================================================
    # ANALYSIS FLAGS
    # ========================================================

    df[
        "positive_mean_change_abs_percent"
    ] = (
        df[
            "positive_mean_change_percent"
        ].abs()
    )

    df[
        "large_reference_change"
    ] = (
        df[
            "positive_mean_change_abs_percent"
        ]
        >=
        10.0
    )

    df[
        "severe_reference_change"
    ] = (
        df[
            "positive_mean_change_abs_percent"
        ]
        >=
        25.0
    )

    # This means current full mean is less than
    # half of the positive-only mean.
    df[
        "very_severe_zero_dilution"
    ] = (
        df[
            "full_to_positive_ratio"
        ]
        <
        0.50
    )

    # ========================================================
    # SORT
    # ========================================================

    category_order = {
        "both_low_positive_and_border": 1,
        "low_positive_only": 2,
        "border_touch_only": 3,
        "not_flagged": 4,
    }

    df[
        "category_order"
    ] = (
        df[
            "review_category"
        ]
        .map(
            category_order
        )
        .fillna(99)
    )

    df = df.sort_values(
        [
            "category_order",
            "positive_fraction",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SAVE ALL-SCAN ANALYSIS
    # ========================================================

    df.to_csv(
        ANALYSIS_CSV,
        index=False,
    )

    # ========================================================
    # SAVE FLAGGED / REVIEW SUBSET
    # ========================================================

    review_df = df[
        df[
            "review_category"
        ]
        !=
        "not_flagged"
    ].copy()

    review_df = review_df.sort_values(
        [
            "category_order",
            "positive_fraction",
        ],
        ascending=[
            True,
            True,
        ],
    )

    review_df.to_csv(
        REVIEW_CSV,
        index=False,
    )

    # ========================================================
    # COUNTS
    # ========================================================

    category_counts = (
        df[
            "review_category"
        ]
        .value_counts()
        .to_dict()
    )

    decision_counts = (
        review_df[
            "visual_decision"
        ]
        .replace(
            "",
            "not_reviewed",
        )
        .value_counts()
        .to_dict()
    )

    # ========================================================
    # SUMMARIES
    # ========================================================

    clean_df = df[
        df[
            "review_category"
        ]
        ==
        "not_flagged"
    ]

    both_df = df[
        df[
            "review_category"
        ]
        ==
        "both_low_positive_and_border"
    ]

    low_only_df = df[
        df[
            "review_category"
        ]
        ==
        "low_positive_only"
    ]

    border_only_df = df[
        df[
            "review_category"
        ]
        ==
        "border_touch_only"
    ]

    # ========================================================
    # EXTREME CASES
    # ========================================================

    largest_changes = (
        df.sort_values(
            "positive_mean_change_abs_percent",
            ascending=False,
        )
        .head(25)
    )

    largest_change_records = []

    for _, row in (
        largest_changes.iterrows()
    ):

        largest_change_records.append(
            {
                "uid":
                    str(
                        row["uid"]
                    ),

                "category":
                    str(
                        row[
                            "review_category"
                        ]
                    ),

                "positive_fraction":
                    float(
                        row[
                            "positive_fraction"
                        ]
                    ),

                "full_mean":
                    float(
                        row[
                            "occipital_mean"
                        ]
                    ),

                "positive_mean":
                    float(
                        row[
                            "positive_only_mean"
                        ]
                    ),

                "change_percent":
                    float(
                        row[
                            "positive_mean_change_percent"
                        ]
                    ),

                "border_touch":
                    bool(
                        row[
                            "border_touch"
                        ]
                    ),

                "visual_decision":
                    str(
                        row[
                            "visual_decision"
                        ]
                    ),
            }
        )

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {

        "analysis":
            (
                "Step 6C-3 occipital reference "
                "rescue analysis"
            ),

        "number_of_scans":
            int(
                len(df)
            ),

        "review_cases":
            int(
                len(review_df)
            ),

        "category_counts":
            category_counts,

        "manual_visual_decision_counts":
            decision_counts,

        "reference_candidates": {

            "current_R1":
                (
                    "Full mapped occipital ROI mean, "
                    "including zero/non-positive "
                    "voxels."
                ),

            "candidate_positive_mean":
                (
                    "Mean of finite voxels with "
                    "intensity > 0 inside the "
                    "same mapped occipital ROI."
                ),

            "candidate_positive_median":
                (
                    "Median of finite voxels with "
                    "intensity > 0 inside the "
                    "same mapped occipital ROI."
                ),
        },

        "groups": {

            "clean_not_flagged":
                group_summary(
                    clean_df
                ),

            "both_low_positive_and_border":
                group_summary(
                    both_df
                ),

            "low_positive_only":
                group_summary(
                    low_only_df
                ),

            "border_touch_only":
                group_summary(
                    border_only_df
                ),

            "all_review_cases":
                group_summary(
                    review_df
                ),

            "all_scans":
                group_summary(
                    df
                ),
        },

        "reference_change_counts_all_scans": {

            "positive_mean_change_gt_1_percent":
                int(
                    (
                        df[
                            "positive_mean_change_abs_percent"
                        ]
                        >
                        1.0
                    ).sum()
                ),

            "positive_mean_change_gt_5_percent":
                int(
                    (
                        df[
                            "positive_mean_change_abs_percent"
                        ]
                        >
                        5.0
                    ).sum()
                ),

            "positive_mean_change_gt_10_percent":
                int(
                    (
                        df[
                            "positive_mean_change_abs_percent"
                        ]
                        >
                        10.0
                    ).sum()
                ),

            "positive_mean_change_gt_25_percent":
                int(
                    (
                        df[
                            "positive_mean_change_abs_percent"
                        ]
                        >
                        25.0
                    ).sum()
                ),

            "full_mean_lt_half_positive_mean":
                int(
                    df[
                        "very_severe_zero_dilution"
                    ].sum()
                ),
        },

        "largest_reference_changes":
            largest_change_records,

        "output_analysis_csv":
            str(
                ANALYSIS_CSV.relative_to(
                    PROJECT_ROOT
                )
            ),

        "output_review_csv":
            str(
                REVIEW_CSV.relative_to(
                    PROJECT_ROOT
                )
            ),

        "important_note":
            (
                "This step does NOT choose the "
                "final normalization reference and "
                "does NOT create normalized images."
            ),

        "next_step":
            (
                "Use these distributions together "
                "with Step 6C-2 visual QC to choose "
                "a consistent final reference policy."
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
        "STEP 6C-3 — REFERENCE RESCUE ANALYSIS"
    )

    print("=" * 90)

    print(
        f"All scans: "
        f"{len(df)}"
    )

    print(
        f"Review scans: "
        f"{len(review_df)}"
    )

    print()

    print("Review categories:")

    for category, count in (
        category_counts.items()
    ):

        print(
            f"    {category}: "
            f"{count}"
        )

    print()

    print("Current visual decisions:")

    for decision, count in (
        decision_counts.items()
    ):

        print(
            f"    {decision}: "
            f"{count}"
        )

    print()

    print(
        "Positive-only mean differs from "
        "full mean by:"
    )

    differences = (
        df[
            "positive_mean_change_abs_percent"
        ]
    )

    print(
        f"    > 1%:  "
        f"{int((differences > 1).sum())}"
    )

    print(
        f"    > 5%:  "
        f"{int((differences > 5).sum())}"
    )

    print(
        f"    > 10%: "
        f"{int((differences > 10).sum())}"
    )

    print(
        f"    > 25%: "
        f"{int((differences > 25).sum())}"
    )

    print()

    print(
        f"Analysis CSV:"
        f"\n{ANALYSIS_CSV}"
    )

    print()

    print(
        f"Review candidates:"
        f"\n{REVIEW_CSV}"
    )

    print()

    print(
        f"Summary:"
        f"\n{SUMMARY_JSON}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
