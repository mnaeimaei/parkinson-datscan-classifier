from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

def _safe_float(value: Any) -> float | None:
    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def classify_registration_qc(
    row: pd.Series,
    high_confidence_dice: float = 0.70,
    fail_dice: float = 0.50,
    minimum_occipital_positive: float = 0.90,
    minimum_volume_ratio: float = 0.80,
    maximum_volume_ratio: float = 2.70,
) -> tuple[str, list[str]]:
    """
    Create an ANALYSIS QC category.

    These thresholds are provisional and must not yet be treated
    as the final inference rule.
    """
    reasons: list[str] = []

    dice = float(
        row["head_mask_dice"]
    )

    positive_fraction = float(
        row["occipital_positive_fraction"]
    )

    volume_ratio = float(
        row["occipital_volume_ratio"]
    )

    touches_border = bool(
        row["occipital_touches_border"]
    )

    # Strong failure indicators.
    if dice < fail_dice:
        reasons.append(
            "head_mask_dice_below_fail_threshold"
        )

    if positive_fraction < minimum_occipital_positive:
        reasons.append(
            "low_occipital_positive_fraction"
        )

    if touches_border:
        reasons.append(
            "occipital_mask_touches_border"
        )

    if volume_ratio < minimum_volume_ratio:
        reasons.append(
            "occipital_volume_ratio_too_low"
        )

    if volume_ratio > maximum_volume_ratio:
        reasons.append(
            "occipital_volume_ratio_too_high"
        )

    if reasons:
        return "fail_or_fallback", reasons

    # High-confidence registration.
    if dice >= high_confidence_dice:
        return "high_confidence", []

    # Everything between fail and high-confidence threshold.
    return "review", [
        "intermediate_head_mask_dice"
    ]


def describe(
    dataframe: pd.DataFrame,
    column: str,
) -> dict[str, float | int | None]:

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return {
            "count": 0
        }

    return {
        "count": int(len(values)),
        "min": _safe_float(values.min()),
        "p05": _safe_float(
            values.quantile(0.05)
        ),
        "p25": _safe_float(
            values.quantile(0.25)
        ),
        "median": _safe_float(
            values.median()
        ),
        "mean": _safe_float(
            values.mean()
        ),
        "p75": _safe_float(
            values.quantile(0.75)
        ),
        "p95": _safe_float(
            values.quantile(0.95)
        ),
        "max": _safe_float(values.max()),
    }


def coefficient_of_variation_log(
    values: pd.Series,
) -> float | None:
    """
    Standard deviation of log10 values.

    More useful here than raw CV because reference intensities
    span several orders of magnitude.
    """
    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
        & (values > 0)
    ]

    if len(values) < 2:
        return None

    log_values = np.log10(
        values.to_numpy()
    )

    return float(
        np.std(
            log_values,
            ddof=0,
        )
    )


def analyze_reference_strategy(
    pilot_results_path: str | Path,
    output_dir: str | Path,
    high_confidence_dice: float = 0.70,
    fail_dice: float = 0.50,
    minimum_occipital_positive: float = 0.90,
    minimum_volume_ratio: float = 0.80,
    maximum_volume_ratio: float = 2.70,
) -> dict:

    pilot_results_path = Path(
        pilot_results_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        pilot_results_path
    )

    required_columns = [
        "file_name",
        "geometry_cohort",
        "scale_decade",

        "head_mask_dice",

        "occipital_positive_fraction",
        "occipital_volume_ratio",
        "occipital_touches_border",

        "occipital_mean",
        "occipital_median",

        "scale_foreground_trimmed_mean_10_90",
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------
    # Assign provisional QC categories
    # --------------------------------------------------

    categories = []
    reasons = []

    for _, row in dataframe.iterrows():

        category, row_reasons = (
            classify_registration_qc(
                row=row,
                high_confidence_dice=(
                    high_confidence_dice
                ),
                fail_dice=fail_dice,
                minimum_occipital_positive=(
                    minimum_occipital_positive
                ),
                minimum_volume_ratio=(
                    minimum_volume_ratio
                ),
                maximum_volume_ratio=(
                    maximum_volume_ratio
                ),
            )
        )

        categories.append(
            category
        )

        reasons.append(
            ";".join(row_reasons)
        )

    dataframe[
        "reference_qc_category"
    ] = categories

    dataframe[
        "reference_qc_reasons"
    ] = reasons

    # --------------------------------------------------
    # Candidate reference values
    # --------------------------------------------------

    dataframe[
        "reference_occipital_mean"
    ] = dataframe[
        "occipital_mean"
    ]

    dataframe[
        "reference_occipital_median"
    ] = dataframe[
        "occipital_median"
    ]

    dataframe[
        "reference_global_trimmed_mean"
    ] = dataframe[
        "scale_foreground_trimmed_mean_10_90"
    ]

    # --------------------------------------------------
    # Ratios between methods
    # --------------------------------------------------

    dataframe[
        "occipital_mean_to_global_ratio"
    ] = (
        dataframe[
            "reference_occipital_mean"
        ]
        /
        dataframe[
            "reference_global_trimmed_mean"
        ]
    )

    dataframe[
        "occipital_median_to_global_ratio"
    ] = (
        dataframe[
            "reference_occipital_median"
        ]
        /
        dataframe[
            "reference_global_trimmed_mean"
        ]
    )

    # --------------------------------------------------
    # Save enriched table
    # --------------------------------------------------

    dataframe.to_csv(
        output_dir
        / "reference_strategy_per_scan.csv",
        index=False,
    )

    # --------------------------------------------------
    # QC category summary
    # --------------------------------------------------

    qc_counts = (
        dataframe[
            "reference_qc_category"
        ]
        .value_counts()
        .to_dict()
    )

    qc_summary_rows = []

    for category, group in dataframe.groupby(
        "reference_qc_category"
    ):

        qc_summary_rows.append(
            {
                "reference_qc_category": (
                    category
                ),

                "count": int(
                    len(group)
                ),

                "dice_median": _safe_float(
                    group[
                        "head_mask_dice"
                    ].median()
                ),

                "occipital_positive_median": (
                    _safe_float(
                        group[
                            "occipital_positive_fraction"
                        ].median()
                    )
                ),

                "occipital_volume_ratio_median": (
                    _safe_float(
                        group[
                            "occipital_volume_ratio"
                        ].median()
                    )
                ),
            }
        )

    pd.DataFrame(
        qc_summary_rows
    ).to_csv(
        output_dir
        / "reference_qc_category_summary.csv",
        index=False,
    )

    # --------------------------------------------------
    # Candidate reference comparison
    # --------------------------------------------------

    candidates = [
        "reference_occipital_mean",
        "reference_occipital_median",
        "reference_global_trimmed_mean",
    ]

    candidate_statistics = {}

    for candidate in candidates:

        candidate_statistics[
            candidate
        ] = {
            "all_scans": describe(
                dataframe,
                candidate,
            ),

            "log10_standard_deviation_all_scans": (
                coefficient_of_variation_log(
                    dataframe[candidate]
                )
            ),
        }

        high_confidence = dataframe[
            dataframe[
                "reference_qc_category"
            ]
            == "high_confidence"
        ]

        candidate_statistics[
            candidate
        ][
            "high_confidence"
        ] = describe(
            high_confidence,
            candidate,
        )

        candidate_statistics[
            candidate
        ][
            "log10_standard_deviation_high_confidence"
        ] = (
            coefficient_of_variation_log(
                high_confidence[
                    candidate
                ]
            )
        )

    # --------------------------------------------------
    # Geometry/QC interaction
    # --------------------------------------------------

    geometry_qc = pd.crosstab(
        dataframe[
            "geometry_cohort"
        ],
        dataframe[
            "reference_qc_category"
        ],
    )

    geometry_qc.to_csv(
        output_dir
        / "geometry_vs_reference_qc.csv"
    )

    scale_qc = pd.crosstab(
        dataframe[
            "scale_decade"
        ],
        dataframe[
            "reference_qc_category"
        ],
    )

    scale_qc.to_csv(
        output_dir
        / "scale_decade_vs_reference_qc.csv"
    )

    # --------------------------------------------------
    # Cases requiring review
    # --------------------------------------------------

    review_cases = dataframe[
        dataframe[
            "reference_qc_category"
        ]
        != "high_confidence"
    ].copy()

    review_cases = review_cases.sort_values(
        [
            "reference_qc_category",
            "head_mask_dice",
        ]
    )

    review_cases.to_csv(
        output_dir
        / "reference_qc_review_cases.csv",
        index=False,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = {
        "analysis_type": (
            "reference_strategy_analysis"
        ),

        "number_of_scans": int(
            len(dataframe)
        ),

        "provisional_qc_thresholds": {
            "high_confidence_dice": (
                high_confidence_dice
            ),

            "fail_dice": (
                fail_dice
            ),

            "minimum_occipital_positive_fraction": (
                minimum_occipital_positive
            ),

            "minimum_occipital_volume_ratio": (
                minimum_volume_ratio
            ),

            "maximum_occipital_volume_ratio": (
                maximum_volume_ratio
            ),

            "occipital_border_touch_allowed": (
                False
            ),

            "warning": (
                "These thresholds are provisional "
                "analysis thresholds and are not yet "
                "the frozen inference rule."
            ),
        },

        "qc_category_counts": {
            str(key): int(value)
            for key, value
            in qc_counts.items()
        },

        "candidate_reference_statistics": (
            candidate_statistics
        ),

        "candidate_methods": {
            "R1": (
                "occipital mean measured in original "
                "resampled subject space"
            ),

            "R2": (
                "occipital median measured in original "
                "resampled subject space"
            ),

            "R3": (
                "whole-volume robust foreground "
                "10-90 percent trimmed mean"
            ),
        },

        "important_interpretation": [
            (
                "A smaller variation in reference values "
                "does not by itself establish the best "
                "normalization strategy."
            ),

            (
                "Occipital methods require registration "
                "QC."
            ),

            (
                "The global trimmed-mean method remains "
                "a candidate fallback rather than a "
                "clinical anatomical reference."
            ),

            (
                "No NIfTI image is normalized during "
                "this analysis."
            ),
        ],
    }

    with (
        output_dir
        / "reference_strategy_statistics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    return summary

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze candidate DaT-SPECT "
            "reference-normalization strategies."
        )
    )

    parser.add_argument(
        "--pilot-results",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--high-confidence-dice",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--fail-dice",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--minimum-occipital-positive",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--minimum-volume-ratio",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--maximum-volume-ratio",
        type=float,
        default=2.70,
    )

    args = parser.parse_args()

    print("=" * 72)
    print("REFERENCE STRATEGY ANALYSIS")
    print("=" * 72)

    summary = analyze_reference_strategy(
        pilot_results_path=(
            args.pilot_results
        ),
        output_dir=(
            args.output_dir
        ),
        high_confidence_dice=(
            args.high_confidence_dice
        ),
        fail_dice=(
            args.fail_dice
        ),
        minimum_occipital_positive=(
            args.minimum_occipital_positive
        ),
        minimum_volume_ratio=(
            args.minimum_volume_ratio
        ),
        maximum_volume_ratio=(
            args.maximum_volume_ratio
        ),
    )

    print()
    print("QC categories:")

    for category, count in (
        summary[
            "qc_category_counts"
        ].items()
    ):
        print(
            f"  {category}: {count}"
        )

    print()
    print("Candidate reference methods:")

    for name, description in (
        summary[
            "candidate_methods"
        ].items()
    ):
        print(
            f"  {name}: {description}"
        )

    print()
    print("=" * 72)
    print("REFERENCE STRATEGY ANALYSIS COMPLETED")
    print("=" * 72)

    print(
        f"Output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()