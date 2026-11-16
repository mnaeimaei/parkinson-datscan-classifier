from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.registration.reference_region_mapper import (
    map_template_mask_to_subject,
)


STRATEGIES = {
    "R1_occipital_mean": "occipital_mean",
    "R2_occipital_median": "occipital_median",
    "R3_global_trimmed_mean": (
        "scale_foreground_trimmed_mean_10_90"
    ),
}


def _safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def _strip_nifti_suffix(
    file_name: str,
) -> str:
    if file_name.endswith(".nii.gz"):
        return file_name[:-7]

    if file_name.endswith(".nii"):
        return file_name[:-4]

    return file_name


def find_subject_files(
    input_dir: str | Path,
) -> dict[str, Path]:
    """
    Build filename -> NIfTI path lookup.
    """
    input_dir = Path(
        input_dir
    )

    files = (
        list(input_dir.rglob("*.nii"))
        + list(input_dir.rglob("*.nii.gz"))
    )

    lookup: dict[str, Path] = {}

    for path in files:

        if path.name in lookup:
            raise RuntimeError(
                f"Duplicate NIfTI filename: {path.name}"
            )

        lookup[path.name] = path

    return lookup


def validate_mask_geometry(
    reference_mask: sitk.Image,
    other_mask: sitk.Image,
    reference_name: str,
    other_name: str,
) -> None:
    """
    Confirm template-space masks share the same geometry.
    """
    if reference_mask.GetSize() != other_mask.GetSize():
        raise ValueError(
            f"{reference_name} and {other_name} "
            "have different sizes."
        )

    if not np.allclose(
        reference_mask.GetSpacing(),
        other_mask.GetSpacing(),
        atol=1e-5,
    ):
        raise ValueError(
            f"{reference_name} and {other_name} "
            "have different spacing."
        )

    if not np.allclose(
        reference_mask.GetOrigin(),
        other_mask.GetOrigin(),
        atol=1e-4,
    ):
        raise ValueError(
            f"{reference_name} and {other_name} "
            "have different origins."
        )

    if not np.allclose(
        reference_mask.GetDirection(),
        other_mask.GetDirection(),
        atol=1e-5,
    ):
        raise ValueError(
            f"{reference_name} and {other_name} "
            "have different directions."
        )


def analyze_mask_signal(
    subject: sitk.Image,
    subject_mask: sitk.Image,
) -> dict[str, Any]:
    """
    Measure signal inside a mapped subject-space mask.

    IMPORTANT:
    Primary statistics include zero-valued voxels.

    We do NOT remove zeros from the striatal statistics because
    doing so could hide poor coverage or a misplaced mask.

    positive_fraction is reported separately for QC.
    """
    data = sitk.GetArrayFromImage(
        subject
    ).astype(np.float32)

    mask = (
        sitk.GetArrayFromImage(
            subject_mask
        ) > 0
    )

    mask_voxels = int(
        mask.sum()
    )

    if mask_voxels == 0:
        return {
            "valid": False,
            "mask_voxels": 0,
        }

    values = data[
        mask
    ]

    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.size == 0:
        return {
            "valid": False,
            "mask_voxels": mask_voxels,
        }

    positive_fraction = float(
        np.count_nonzero(
            finite_values > 0
        )
        / finite_values.size
    )

    return {
        "valid": True,

        "mask_voxels": mask_voxels,

        "finite_voxels": int(
            finite_values.size
        ),

        "positive_fraction": (
            positive_fraction
        ),

        "mean": float(
            np.mean(finite_values)
        ),

        "median": float(
            np.median(finite_values)
        ),

        "p75": float(
            np.percentile(
                finite_values,
                75,
            )
        ),

        "p90": float(
            np.percentile(
                finite_values,
                90,
            )
        ),

        "p95": float(
            np.percentile(
                finite_values,
                95,
            )
        ),

        "p99": float(
            np.percentile(
                finite_values,
                99,
            )
        ),

        "max": float(
            np.max(finite_values)
        ),
    }


def _add_prefixed(
    row: dict,
    prefix: str,
    values: dict,
) -> None:
    for key, value in values.items():
        row[
            f"{prefix}_{key}"
        ] = value


def calculate_asymmetry(
    left_value: float,
    right_value: float,
) -> float | None:
    """
    Symmetric left-right asymmetry index:

        |L - R| / ((L + R) / 2)

    Scale-invariant.
    """
    left_value = float(
        left_value
    )

    right_value = float(
        right_value
    )

    denominator = (
        left_value + right_value
    ) / 2.0

    if denominator <= 0:
        return None

    return float(
        abs(
            left_value - right_value
        )
        / denominator
    )


def _describe(
    values: pd.Series,
) -> dict[str, Any]:

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            "count": 0,
        }

    return {
        "count": int(
            len(values)
        ),

        "min": _safe_float(
            values.min()
        ),

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

        "max": _safe_float(
            values.max()
        ),

        "std": _safe_float(
            values.std(
                ddof=0
            )
        ),
    }


def _log10_std(
    values: pd.Series,
) -> float | None:

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

    return float(
        np.std(
            np.log10(
                values.to_numpy()
            ),
            ddof=0,
        )
    )


def _spearman_like(
    x: pd.Series,
    y: pd.Series,
) -> float | None:
    """
    Rank correlation without scipy dependency.
    """
    dataframe = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                y,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(dataframe) < 3:
        return None

    if (
        dataframe["x"].nunique() <= 1
        or dataframe["y"].nunique() <= 1
    ):
        return None

    correlation = (
        dataframe["x"]
        .rank(method="average")
        .corr(
            dataframe["y"]
            .rank(method="average")
        )
    )

    return _safe_float(
        correlation
    )


def add_normalized_metrics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulate R1 / R2 / R3.

    We do NOT create normalized image files.
    """
    dataframe = dataframe.copy()

    metrics = [
        "left_mean",
        "left_median",
        "left_p95",

        "right_mean",
        "right_median",
        "right_p95",

        "bilateral_mean",
        "bilateral_median",
        "bilateral_p95",
        "bilateral_p99",
    ]

    for strategy_name, reference_column in (
        STRATEGIES.items()
    ):

        reference = pd.to_numeric(
            dataframe[
                reference_column
            ],
            errors="coerce",
        )

        valid_reference = (
            np.isfinite(reference)
            & (reference > 0)
        )

        for metric in metrics:

            values = pd.to_numeric(
                dataframe[
                    metric
                ],
                errors="coerce",
            )

            output = pd.Series(
                np.nan,
                index=dataframe.index,
                dtype=float,
            )

            valid = (
                valid_reference
                & np.isfinite(values)
            )

            output.loc[valid] = (
                values.loc[valid]
                / reference.loc[valid]
            )

            dataframe[
                f"{strategy_name}_{metric}"
            ] = output

    # --------------------------------------------------
    # Raw asymmetry
    # --------------------------------------------------

    dataframe[
        "raw_mean_asymmetry"
    ] = [
        calculate_asymmetry(
            left,
            right,
        )
        for left, right in zip(
            dataframe[
                "left_mean"
            ],
            dataframe[
                "right_mean"
            ],
        )
    ]

    # Verify the same value after each strategy.
    for strategy_name in STRATEGIES:

        dataframe[
            f"{strategy_name}_mean_asymmetry"
        ] = [
            calculate_asymmetry(
                left,
                right,
            )
            for left, right in zip(
                dataframe[
                    f"{strategy_name}_left_mean"
                ],
                dataframe[
                    f"{strategy_name}_right_mean"
                ],
            )
        ]

    return dataframe


def _save_histogram(
    dataframe: pd.DataFrame,
    column: str,
    title: str,
    output_path: Path,
) -> None:

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
        & (values > 0)
    ]

    if len(values) == 0:
        return

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        np.log10(values),
        bins=30,
    )

    plt.xlabel(
        "log10(normalized bilateral mean)"
    )

    plt.ylabel(
        "Number of scans"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()


def _save_scale_scatter(
    dataframe: pd.DataFrame,
    normalized_column: str,
    title: str,
    output_path: Path,
) -> None:

    x = pd.to_numeric(
        dataframe[
            "scale_foreground_trimmed_mean_10_90"
        ],
        errors="coerce",
    )

    y = pd.to_numeric(
        dataframe[
            normalized_column
        ],
        errors="coerce",
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
        & (y > 0)
    )

    if valid.sum() == 0:
        return

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        np.log10(
            x[valid]
        ),
        np.log10(
            y[valid]
        ),
        s=16,
        alpha=0.6,
    )

    plt.xlabel(
        "log10(original global intensity scale)"
    )

    plt.ylabel(
        "log10(normalized bilateral striatal mean)"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()


def run_normalization_strategy_comparison(
    reference_strategy_csv: str | Path,
    input_dir: str | Path,
    left_mask_path: str | Path,
    right_mask_path: str | Path,
    bilateral_mask_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """
    Compare R1/R2/R3 using the existing 90-scan pilot.

    No normalized NIfTI files are created.
    """
    reference_strategy_csv = Path(
        reference_strategy_csv
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        reference_strategy_csv
    )

    required_columns = [
        "file_name",
        "transform_path",
        "reference_qc_category",

        "occipital_mean",
        "occipital_median",
        "scale_foreground_trimmed_mean_10_90",

        "geometry_cohort",
        "scale_decade",
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

    subject_lookup = find_subject_files(
        input_dir
    )

    # --------------------------------------------------
    # Load template-space striatal masks
    # --------------------------------------------------

    left_template_mask = (
        sitk.ReadImage(
            str(left_mask_path),
            sitk.sitkUInt8,
        )
    )

    right_template_mask = (
        sitk.ReadImage(
            str(right_mask_path),
            sitk.sitkUInt8,
        )
    )

    bilateral_template_mask = (
        sitk.ReadImage(
            str(bilateral_mask_path),
            sitk.sitkUInt8,
        )
    )

    validate_mask_geometry(
        left_template_mask,
        right_template_mask,
        "left mask",
        "right mask",
    )

    validate_mask_geometry(
        left_template_mask,
        bilateral_template_mask,
        "left mask",
        "bilateral mask",
    )

    # --------------------------------------------------
    # Map masks and measure original subject signal
    # --------------------------------------------------

    results: list[dict] = []
    failures: list[dict] = []

    total = len(
        dataframe
    )

    for index, source_row in dataframe.iterrows():

        file_name = str(
            source_row[
                "file_name"
            ]
        )

        print(
            f"[{index + 1}/{total}] "
            f"{file_name}"
        )

        try:
            if file_name not in subject_lookup:
                raise FileNotFoundError(
                    f"NIfTI not found: "
                    f"{file_name}"
                )

            transform_path = Path(
                str(
                    source_row[
                        "transform_path"
                    ]
                )
            )

            if not transform_path.exists():
                raise FileNotFoundError(
                    f"Transform not found: "
                    f"{transform_path}"
                )

            subject = sitk.ReadImage(
                str(
                    subject_lookup[
                        file_name
                    ]
                ),
                sitk.sitkFloat32,
            )

            transform = sitk.ReadTransform(
                str(
                    transform_path
                )
            )

            left_subject_mask = (
                map_template_mask_to_subject(
                    template_mask=(
                        left_template_mask
                    ),
                    subject_image=subject,
                    subject_to_template_transform=(
                        transform
                    ),
                )
            )

            right_subject_mask = (
                map_template_mask_to_subject(
                    template_mask=(
                        right_template_mask
                    ),
                    subject_image=subject,
                    subject_to_template_transform=(
                        transform
                    ),
                )
            )

            bilateral_subject_mask = (
                map_template_mask_to_subject(
                    template_mask=(
                        bilateral_template_mask
                    ),
                    subject_image=subject,
                    subject_to_template_transform=(
                        transform
                    ),
                )
            )

            left_stats = (
                analyze_mask_signal(
                    subject,
                    left_subject_mask,
                )
            )

            right_stats = (
                analyze_mask_signal(
                    subject,
                    right_subject_mask,
                )
            )

            bilateral_stats = (
                analyze_mask_signal(
                    subject,
                    bilateral_subject_mask,
                )
            )

            row = (
                source_row.to_dict()
            )

            row[
                "uid"
            ] = _strip_nifti_suffix(
                file_name
            )

            _add_prefixed(
                row,
                "left",
                left_stats,
            )

            _add_prefixed(
                row,
                "right",
                right_stats,
            )

            _add_prefixed(
                row,
                "bilateral",
                bilateral_stats,
            )

            results.append(
                row
            )

            print(
                "  Left positive="
                f"{left_stats.get('positive_fraction')}"
            )

            print(
                "  Right positive="
                f"{right_stats.get('positive_fraction')}"
            )

            print(
                "  Bilateral mean="
                f"{bilateral_stats.get('mean')}"
            )

        except Exception as exc:

            print(
                f"  FAILED: {exc}"
            )

            failures.append(
                {
                    "file_name": file_name,
                    "error": str(exc),
                }
            )

    results_df = pd.DataFrame(
        results
    )

    if len(results_df) == 0:
        raise RuntimeError(
            "No successful scans."
        )

    # --------------------------------------------------
    # Simulated normalization
    # --------------------------------------------------

    results_df = (
        add_normalized_metrics(
            results_df
        )
    )

    results_df.to_csv(
        output_dir
        / "normalization_strategy_per_scan.csv",
        index=False,
    )

    # --------------------------------------------------
    # High-confidence registrations only
    # for primary R1/R2 comparison.
    # --------------------------------------------------

    high_confidence = results_df[
        results_df[
            "reference_qc_category"
        ]
        == "high_confidence"
    ].copy()

    # --------------------------------------------------
    # Strategy summaries
    # --------------------------------------------------

    strategy_statistics = {}

    original_scale = results_df[
        "scale_foreground_trimmed_mean_10_90"
    ]

    for strategy_name in STRATEGIES:

        bilateral_mean_column = (
            f"{strategy_name}_bilateral_mean"
        )

        bilateral_p95_column = (
            f"{strategy_name}_bilateral_p95"
        )

        strategy_statistics[
            strategy_name
        ] = {
            "all_scans": {
                "bilateral_mean": (
                    _describe(
                        results_df[
                            bilateral_mean_column
                        ]
                    )
                ),

                "bilateral_p95": (
                    _describe(
                        results_df[
                            bilateral_p95_column
                        ]
                    )
                ),

                "log10_std_bilateral_mean": (
                    _log10_std(
                        results_df[
                            bilateral_mean_column
                        ]
                    )
                ),

                "rank_correlation_with_original_scale": (
                    _spearman_like(
                        original_scale,
                        results_df[
                            bilateral_mean_column
                        ],
                    )
                ),
            },

            "high_confidence": {
                "count": int(
                    len(
                        high_confidence
                    )
                ),

                "bilateral_mean": (
                    _describe(
                        high_confidence[
                            bilateral_mean_column
                        ]
                    )
                ),

                "bilateral_p95": (
                    _describe(
                        high_confidence[
                            bilateral_p95_column
                        ]
                    )
                ),

                "log10_std_bilateral_mean": (
                    _log10_std(
                        high_confidence[
                            bilateral_mean_column
                        ]
                    )
                ),

                "rank_correlation_with_original_scale": (
                    _spearman_like(
                        high_confidence[
                            "scale_foreground_trimmed_mean_10_90"
                        ],
                        high_confidence[
                            bilateral_mean_column
                        ],
                    )
                ),
            },
        }

    # --------------------------------------------------
    # Verify asymmetry invariance
    # --------------------------------------------------

    asymmetry_differences = {}

    raw_asymmetry = pd.to_numeric(
        results_df[
            "raw_mean_asymmetry"
        ],
        errors="coerce",
    )

    for strategy_name in STRATEGIES:

        normalized_asymmetry = (
            pd.to_numeric(
                results_df[
                    f"{strategy_name}_mean_asymmetry"
                ],
                errors="coerce",
            )
        )

        valid = (
            np.isfinite(
                raw_asymmetry
            )
            & np.isfinite(
                normalized_asymmetry
            )
        )

        if valid.sum() == 0:
            max_difference = None
        else:
            max_difference = float(
                np.max(
                    np.abs(
                        raw_asymmetry[
                            valid
                        ]
                        - normalized_asymmetry[
                            valid
                        ]
                    )
                )
            )

        asymmetry_differences[
            strategy_name
        ] = max_difference

    # --------------------------------------------------
    # Signal coverage QC
    # --------------------------------------------------

    coverage_statistics = {
        "left_positive_fraction": (
            _describe(
                results_df[
                    "left_positive_fraction"
                ]
            )
        ),

        "right_positive_fraction": (
            _describe(
                results_df[
                    "right_positive_fraction"
                ]
            )
        ),

        "bilateral_positive_fraction": (
            _describe(
                results_df[
                    "bilateral_positive_fraction"
                ]
            )
        ),
    }

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------

    for strategy_name in STRATEGIES:

        column = (
            f"{strategy_name}_bilateral_mean"
        )

        _save_histogram(
            dataframe=results_df,
            column=column,
            title=(
                f"{strategy_name}: "
                "Normalized Bilateral Striatal Mean"
            ),
            output_path=(
                output_dir
                / (
                    f"{strategy_name}_"
                    "bilateral_mean_histogram.png"
                )
            ),
        )

        _save_scale_scatter(
            dataframe=results_df,
            normalized_column=column,
            title=(
                f"{strategy_name}: "
                "Residual Dependence on Original Scale"
            ),
            output_path=(
                output_dir
                / (
                    f"{strategy_name}_"
                    "vs_original_scale.png"
                )
            ),
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = {
        "analysis_type": (
            "normalization_strategy_comparison"
        ),

        "number_of_input_scans": int(
            len(dataframe)
        ),

        "successful": int(
            len(results_df)
        ),

        "failed": int(
            len(failures)
        ),

        "high_confidence_scans": int(
            len(
                high_confidence
            )
        ),

        "strategies": {
            "R1_occipital_mean": (
                "Original subject intensities divided "
                "by subject-specific occipital mean."
            ),

            "R2_occipital_median": (
                "Original subject intensities divided "
                "by subject-specific occipital median."
            ),

            "R3_global_trimmed_mean": (
                "Original subject intensities divided "
                "by whole-volume robust foreground "
                "10-90% trimmed mean."
            ),
        },

        "striatal_definition": {
            "left": [
                "Caudate_L",
                "Putamen_L",
            ],

            "right": [
                "Caudate_R",
                "Putamen_R",
            ],

            "pallidum_included": False,
        },

        "coverage_statistics": (
            coverage_statistics
        ),

        "strategy_statistics": (
            strategy_statistics
        ),

        "asymmetry_invariance": {
            "definition": (
                "|L-R| / ((L+R)/2)"
            ),

            "maximum_absolute_difference_after_normalization": (
                asymmetry_differences
            ),

            "expectation": (
                "Differences should be approximately "
                "zero because all voxels in a scan "
                "are divided by the same scalar."
            ),
        },

        "interpretation_warning": [
            (
                "Lower normalized variance alone does "
                "not prove that a strategy is superior."
            ),

            (
                "R1 and R2 should primarily be judged "
                "on high-confidence registrations."
            ),

            (
                "R3 can be computed without anatomical "
                "registration, but it is not a clinical "
                "anatomical reference."
            ),

            (
                "No normalized NIfTI images were saved "
                "during this analysis."
            ),
        ],

        "failures": failures,
    }

    with (
        output_dir
        / "normalization_strategy_statistics.json"
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
            "Compare candidate DaT-SPECT intensity "
            "normalization strategies using mapped "
            "striatal masks."
        )
    )

    parser.add_argument(
        "--reference-strategy-csv",
        required=True,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
    )

    parser.add_argument(
        "--left-mask",
        required=True,
    )

    parser.add_argument(
        "--right-mask",
        required=True,
    )

    parser.add_argument(
        "--bilateral-mask",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    print("=" * 72)
    print("NORMALIZATION STRATEGY COMPARISON")
    print("=" * 72)

    summary = (
        run_normalization_strategy_comparison(
            reference_strategy_csv=(
                args.reference_strategy_csv
            ),
            input_dir=(
                args.input_dir
            ),
            left_mask_path=(
                args.left_mask
            ),
            right_mask_path=(
                args.right_mask
            ),
            bilateral_mask_path=(
                args.bilateral_mask
            ),
            output_dir=(
                args.output_dir
            ),
        )
    )

    print()
    print("=" * 72)
    print("NORMALIZATION STRATEGY COMPARISON COMPLETED")
    print("=" * 72)

    print(
        f"Successful: "
        f"{summary['successful']}"
    )

    print(
        f"Failed: "
        f"{summary['failed']}"
    )

    print(
        f"High-confidence registrations: "
        f"{summary['high_confidence_scans']}"
    )

    print()
    print("Strategy summary:")

    for strategy, values in (
        summary[
            "strategy_statistics"
        ].items()
    ):

        print()
        print(
            f"  {strategy}"
        )

        all_scans = (
            values["all_scans"]
        )

        high_confidence = (
            values["high_confidence"]
        )

        print(
            "    All scans:"
        )

        print(
            "      log10 SD = "
            f"{all_scans['log10_std_bilateral_mean']}"
        )

        print(
            "      correlation with original scale = "
            f"{all_scans['rank_correlation_with_original_scale']}"
        )

        print(
            "    High confidence:"
        )

        print(
            "      log10 SD = "
            f"{high_confidence['log10_std_bilateral_mean']}"
        )

        print(
            "      correlation with original scale = "
            f"{high_confidence['rank_correlation_with_original_scale']}"
        )

    print()
    print(
        "Asymmetry preservation check:"
    )

    for strategy, difference in (
        summary[
            "asymmetry_invariance"
        ][
            "maximum_absolute_difference_after_normalization"
        ].items()
    ):

        print(
            f"  {strategy}: "
            f"{difference}"
        )

    print()
    print(
        f"Output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
