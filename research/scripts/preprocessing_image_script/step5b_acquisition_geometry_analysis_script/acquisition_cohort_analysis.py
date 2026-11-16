from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCALE_COLUMN = "scale_foreground_trimmed_mean_10_90"


def _safe_float(value: Any) -> float | None:
    """
    Convert numerical values to JSON-safe floats.
    """
    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def _shape_signature(values: list[int] | tuple[int, ...]) -> str:
    """
    Convert a 3D shape to a readable signature.

    Example:
        [128, 128, 90] -> "128x128x90"
    """
    return "x".join(str(int(v)) for v in values)


def _spacing_signature(
    values: list[float] | tuple[float, ...],
    decimals: int,
) -> str:
    """
    Convert voxel spacing to a rounded signature.

    Example:
        [3.89537, 3.89537, 3.90000]
        -> "3.895x3.895x3.900"
    """
    return "x".join(
        f"{float(v):.{decimals}f}"
        for v in values
    )


def _rank_correlation(
    x: pd.Series,
    y: pd.Series,
) -> float | None:
    """
    Dependency-free Spearman-like rank correlation.

    Uses pandas ranking followed by Pearson correlation.
    """
    dataframe = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if len(dataframe) < 3:
        return None

    if (
        dataframe["x"].nunique() <= 1
        or dataframe["y"].nunique() <= 1
    ):
        return None

    x_rank = dataframe["x"].rank(method="average")
    y_rank = dataframe["y"].rank(method="average")

    correlation = x_rank.corr(y_rank)

    return _safe_float(correlation)


def load_resampling_metadata(
    resampling_summary_path: str | Path,
    spacing_round_decimals: int = 3,
) -> pd.DataFrame:
    """
    Load acquisition geometry information from resampling_summary.json.

    The original shape and original voxel spacing are particularly useful
    because they describe the image geometry before the common 2.46-mm
    resampling step.
    """
    resampling_summary_path = Path(resampling_summary_path)

    if not resampling_summary_path.exists():
        raise FileNotFoundError(
            f"Resampling summary not found: "
            f"{resampling_summary_path}"
        )

    with resampling_summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    files = summary.get("files", [])

    if not files:
        raise ValueError(
            "No per-file records found in resampling summary."
        )

    rows: list[dict[str, Any]] = []

    for record in files:
        original_shape = record["original_shape"]
        original_spacing = record["original_spacing"]
        resampled_shape = record["resampled_shape"]

        original_voxel_volume = float(
            original_spacing[0]
            * original_spacing[1]
            * original_spacing[2]
        )

        original_spacing_signature = _spacing_signature(
            original_spacing,
            decimals=spacing_round_decimals,
        )

        original_shape_signature = _shape_signature(
            original_shape
        )

        geometry_cohort = (
            f"shape={original_shape_signature}"
            f"__spacing={original_spacing_signature}"
        )

        rows.append(
            {
                "file_name": record["file_name"],

                "original_shape_x": int(original_shape[0]),
                "original_shape_y": int(original_shape[1]),
                "original_shape_z": int(original_shape[2]),

                "original_spacing_x": float(
                    original_spacing[0]
                ),
                "original_spacing_y": float(
                    original_spacing[1]
                ),
                "original_spacing_z": float(
                    original_spacing[2]
                ),

                "original_voxel_volume_mm3": (
                    original_voxel_volume
                ),

                "original_shape_signature": (
                    original_shape_signature
                ),

                "original_spacing_signature": (
                    original_spacing_signature
                ),

                "geometry_cohort": geometry_cohort,

                "resampled_shape_x": int(resampled_shape[0]),
                "resampled_shape_y": int(resampled_shape[1]),
                "resampled_shape_z": int(resampled_shape[2]),

                "resampled_shape_signature": (
                    _shape_signature(resampled_shape)
                ),

                "original_isotropic": bool(
                    np.allclose(
                        original_spacing,
                        [
                            original_spacing[0],
                            original_spacing[0],
                            original_spacing[0],
                        ],
                        atol=1e-4,
                    )
                ),
            }
        )

    dataframe = pd.DataFrame(rows)

    if dataframe["file_name"].duplicated().any():
        duplicates = dataframe.loc[
            dataframe["file_name"].duplicated(),
            "file_name",
        ].tolist()

        raise ValueError(
            f"Duplicate file names in resampling summary: "
            f"{duplicates[:10]}"
        )

    return dataframe


def _create_scale_decade(
    scale: pd.Series,
) -> pd.Series:
    """
    Create descriptive order-of-magnitude groups.

    Example:
        8     -> 10^0
        50    -> 10^1
        700   -> 10^2
        5000  -> 10^3

    This is descriptive only.
    It is NOT used for normalization.
    """
    numeric = pd.to_numeric(
        scale,
        errors="coerce",
    )

    output = pd.Series(
        index=numeric.index,
        dtype="object",
    )

    positive = numeric > 0

    decades = np.floor(
        np.log10(numeric.loc[positive])
    ).astype(int)

    output.loc[positive] = decades.map(
        lambda value: f"10^{value}"
    )

    output.loc[~positive] = "non-positive"

    return output


def _summarize_group(
    dataframe: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """
    Produce descriptive statistics by cohort.
    """
    rows: list[dict[str, Any]] = []

    for group_name, group in dataframe.groupby(
        group_column,
        dropna=False,
    ):
        scale = pd.to_numeric(
            group[SCALE_COLUMN],
            errors="coerce",
        ).dropna()

        foreground_fraction = pd.to_numeric(
            group["foreground_fraction"],
            errors="coerce",
        ).dropna()

        fov_x = pd.to_numeric(
            group["fov_mm_x"],
            errors="coerce",
        ).dropna()

        fov_y = pd.to_numeric(
            group["fov_mm_y"],
            errors="coerce",
        ).dropna()

        fov_z = pd.to_numeric(
            group["fov_mm_z"],
            errors="coerce",
        ).dropna()

        negative_scans = (
            int((group["negative_voxels"] > 0).sum())
            if "negative_voxels" in group.columns
            else 0
        )

        rows.append(
            {
                group_column: str(group_name),
                "count": int(len(group)),

                "scale_min": (
                    _safe_float(scale.min())
                    if len(scale)
                    else None
                ),
                "scale_p25": (
                    _safe_float(scale.quantile(0.25))
                    if len(scale)
                    else None
                ),
                "scale_median": (
                    _safe_float(scale.median())
                    if len(scale)
                    else None
                ),
                "scale_mean": (
                    _safe_float(scale.mean())
                    if len(scale)
                    else None
                ),
                "scale_p75": (
                    _safe_float(scale.quantile(0.75))
                    if len(scale)
                    else None
                ),
                "scale_max": (
                    _safe_float(scale.max())
                    if len(scale)
                    else None
                ),

                "foreground_fraction_median": (
                    _safe_float(
                        foreground_fraction.median()
                    )
                    if len(foreground_fraction)
                    else None
                ),

                "fov_x_median_mm": (
                    _safe_float(fov_x.median())
                    if len(fov_x)
                    else None
                ),
                "fov_y_median_mm": (
                    _safe_float(fov_y.median())
                    if len(fov_y)
                    else None
                ),
                "fov_z_median_mm": (
                    _safe_float(fov_z.median())
                    if len(fov_z)
                    else None
                ),

                "scans_with_negative_values": (
                    negative_scans
                ),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            by="count",
            ascending=False,
        ).reset_index(drop=True)

    return result


def _save_cohort_count_plot(
    cohort_summary: pd.DataFrame,
    output_path: Path,
    top_n: int,
) -> None:
    """
    Plot number of scans in the largest geometry cohorts.
    """
    if cohort_summary.empty:
        return

    top = cohort_summary.head(top_n).copy()

    labels = top["geometry_cohort"].astype(str)
    counts = top["count"]

    plt.figure(
        figsize=(12, max(6, 0.45 * len(top)))
    )

    positions = np.arange(len(top))

    plt.barh(
        positions,
        counts,
    )

    plt.yticks(
        positions,
        labels,
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Number of scans")
    plt.ylabel("Geometry cohort")
    plt.title(
        f"Top {len(top)} Acquisition Geometry Cohorts"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def _save_log_scale_histogram(
    dataframe: pd.DataFrame,
    output_path: Path,
    bins: int = 50,
) -> None:
    """
    Plot distribution of normalization scale on log10 scale.
    """
    scale = pd.to_numeric(
        dataframe[SCALE_COLUMN],
        errors="coerce",
    )

    scale = scale[
        np.isfinite(scale)
        & (scale > 0)
    ]

    if len(scale) == 0:
        return

    log_scale = np.log10(scale)

    plt.figure(figsize=(8, 5))

    plt.hist(
        log_scale,
        bins=bins,
    )

    plt.xlabel(
        "log10(candidate normalization scale)"
    )

    plt.ylabel("Number of scans")

    plt.title(
        "Distribution of Candidate Normalization Scale"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()


def _save_scale_by_cohort_boxplot(
    dataframe: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
) -> None:
    """
    Compare intensity scale across the largest geometry cohorts.
    """
    if cohort_summary.empty:
        return

    top_cohorts = (
        cohort_summary
        .head(top_n)["geometry_cohort"]
        .tolist()
    )

    plot_data: list[np.ndarray] = []
    labels: list[str] = []

    for cohort in top_cohorts:
        values = pd.to_numeric(
            dataframe.loc[
                dataframe["geometry_cohort"] == cohort,
                SCALE_COLUMN,
            ],
            errors="coerce",
        )

        values = values[
            np.isfinite(values)
            & (values > 0)
        ]

        if len(values) == 0:
            continue

        plot_data.append(
            np.log10(values.to_numpy())
        )

        labels.append(cohort)

    if not plot_data:
        return

    plt.figure(
        figsize=(13, max(6, len(labels) * 0.6))
    )

    positions = np.arange(
        1,
        len(plot_data) + 1,
    )

    plt.boxplot(
        plot_data,
        positions=positions,
        vert=False,
    )

    plt.yticks(
        positions,
        labels,
    )

    plt.xlabel(
        "log10(candidate normalization scale)"
    )

    plt.ylabel("Geometry cohort")

    plt.title(
        "Intensity Scale by Acquisition Geometry Cohort"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def _save_scatter(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    log_y: bool = False,
) -> None:
    """
    Save a simple scatter plot.
    """
    x = pd.to_numeric(
        dataframe[x_column],
        errors="coerce",
    )

    y = pd.to_numeric(
        dataframe[y_column],
        errors="coerce",
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if log_y:
        valid &= y > 0

    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return

    if log_y:
        y = np.log10(y)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        x,
        y,
        s=12,
        alpha=0.5,
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()


def analyze_acquisition_cohorts(
    reference_csv_path: str | Path,
    resampling_summary_path: str | Path,
    output_dir: str | Path,
    spacing_round_decimals: int = 3,
    top_cohorts: int = 15,
) -> dict[str, Any]:
    """
    Analyze whether intensity/FOV differences correspond to
    acquisition geometry cohorts.

    IMPORTANT:
    - This is analysis only.
    - No NIfTI images are modified.
    - Geometry cohorts are proxies based on image shape/spacing.
    - They are not guaranteed to correspond to scanner manufacturers
      or reconstruction protocols because DICOM acquisition metadata
      are not available here.
    """
    reference_csv_path = Path(reference_csv_path)
    resampling_summary_path = Path(
        resampling_summary_path
    )
    output_dir = Path(output_dir)

    if not reference_csv_path.exists():
        raise FileNotFoundError(
            f"Reference-normalization CSV not found: "
            f"{reference_csv_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_df = pd.read_csv(
        reference_csv_path
    )

    geometry_df = load_resampling_metadata(
        resampling_summary_path,
        spacing_round_decimals=(
            spacing_round_decimals
        ),
    )

    if reference_df["file_name"].duplicated().any():
        raise ValueError(
            "Duplicate file names found in reference analysis CSV."
        )

    merged = reference_df.merge(
        geometry_df,
        on="file_name",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(reference_df):
        missing = sorted(
            set(reference_df["file_name"])
            - set(merged["file_name"])
        )

        raise ValueError(
            "Some reference-analysis scans did not match the "
            "resampling summary. "
            f"Missing examples: {missing[:10]}"
        )

    if SCALE_COLUMN not in merged.columns:
        raise ValueError(
            f"Required scale column missing: "
            f"{SCALE_COLUMN}"
        )

    # Order-of-magnitude group for descriptive analysis.
    merged["scale_decade"] = _create_scale_decade(
        merged[SCALE_COLUMN]
    )

    # Save enriched per-scan table.
    per_scan_output = (
        output_dir
        / "acquisition_cohort_per_scan.csv"
    )

    merged.to_csv(
        per_scan_output,
        index=False,
    )

    # --------------------------------------------
    # Summaries
    # --------------------------------------------

    cohort_summary = _summarize_group(
        merged,
        "geometry_cohort",
    )

    cohort_summary.to_csv(
        output_dir
        / "acquisition_cohort_summary.csv",
        index=False,
    )

    spacing_summary = _summarize_group(
        merged,
        "original_spacing_signature",
    )

    spacing_summary.to_csv(
        output_dir
        / "spacing_family_summary.csv",
        index=False,
    )

    shape_summary = _summarize_group(
        merged,
        "original_shape_signature",
    )

    shape_summary.to_csv(
        output_dir
        / "shape_family_summary.csv",
        index=False,
    )

    scale_decade_summary = (
        merged["scale_decade"]
        .value_counts(dropna=False)
        .rename_axis("scale_decade")
        .reset_index(name="count")
    )

    scale_decade_summary.to_csv(
        output_dir
        / "scale_decade_summary.csv",
        index=False,
    )

    geometry_scale_crosstab = pd.crosstab(
        merged["geometry_cohort"],
        merged["scale_decade"],
    )

    geometry_scale_crosstab.to_csv(
        output_dir
        / "geometry_scale_decade_crosstab.csv"
    )

    # --------------------------------------------
    # Correlations
    # --------------------------------------------

    correlation_variables = {
        "foreground_fraction": (
            merged["foreground_fraction"]
        ),
        "fov_mm_x": merged["fov_mm_x"],
        "fov_mm_y": merged["fov_mm_y"],
        "fov_mm_z": merged["fov_mm_z"],
        "original_voxel_volume_mm3": (
            merged["original_voxel_volume_mm3"]
        ),
        "original_spacing_x": (
            merged["original_spacing_x"]
        ),
        "original_spacing_z": (
            merged["original_spacing_z"]
        ),
    }

    correlations: dict[str, float | None] = {}

    for name, values in correlation_variables.items():
        correlations[
            f"{name}_vs_scale"
        ] = _rank_correlation(
            values,
            merged[SCALE_COLUMN],
        )

    # --------------------------------------------
    # Dataset-level summary
    # --------------------------------------------

    geometry_counts = (
        merged["geometry_cohort"]
        .value_counts()
    )

    spacing_counts = (
        merged["original_spacing_signature"]
        .value_counts()
    )

    shape_counts = (
        merged["original_shape_signature"]
        .value_counts()
    )

    top_geometry_cohorts = []

    for cohort, count in geometry_counts.head(
        top_cohorts
    ).items():
        cohort_rows = cohort_summary[
            cohort_summary["geometry_cohort"]
            == cohort
        ]

        record: dict[str, Any] = {
            "geometry_cohort": str(cohort),
            "count": int(count),
        }

        if len(cohort_rows) == 1:
            row = cohort_rows.iloc[0]

            record.update(
                {
                    "scale_median": _safe_float(
                        row["scale_median"]
                    ),
                    "scale_p25": _safe_float(
                        row["scale_p25"]
                    ),
                    "scale_p75": _safe_float(
                        row["scale_p75"]
                    ),
                    "foreground_fraction_median": (
                        _safe_float(
                            row[
                                "foreground_fraction_median"
                            ]
                        )
                    ),
                }
            )

        top_geometry_cohorts.append(record)

    summary = {
        "analysis_type": (
            "acquisition_reconstruction_cohort_analysis"
        ),

        "reference_csv": str(
            reference_csv_path
        ),

        "resampling_summary": str(
            resampling_summary_path
        ),

        "number_of_reference_records": int(
            len(reference_df)
        ),

        "number_of_resampling_records": int(
            len(geometry_df)
        ),

        "number_of_matched_scans": int(
            len(merged)
        ),

        "geometry_definition": {
            "geometry_cohort": (
                "original_shape + rounded_original_spacing"
            ),
            "spacing_round_decimals": int(
                spacing_round_decimals
            ),
            "warning": (
                "These cohorts are acquisition-geometry proxies. "
                "Without scanner/reconstruction DICOM metadata, "
                "they cannot be interpreted as confirmed scanner "
                "or reconstruction-protocol groups."
            ),
        },

        "scale_analyzed": SCALE_COLUMN,

        "number_of_unique_geometry_cohorts": int(
            merged["geometry_cohort"].nunique()
        ),

        "number_of_unique_original_shapes": int(
            merged[
                "original_shape_signature"
            ].nunique()
        ),

        "number_of_unique_original_spacing_families": int(
            merged[
                "original_spacing_signature"
            ].nunique()
        ),

        "largest_geometry_cohort_count": (
            int(geometry_counts.iloc[0])
            if len(geometry_counts)
            else 0
        ),

        "largest_spacing_family_count": (
            int(spacing_counts.iloc[0])
            if len(spacing_counts)
            else 0
        ),

        "largest_shape_family_count": (
            int(shape_counts.iloc[0])
            if len(shape_counts)
            else 0
        ),

        "scale_decade_counts": {
            str(key): int(value)
            for key, value
            in merged[
                "scale_decade"
            ].value_counts().items()
        },

        "rank_correlations_with_candidate_scale": (
            correlations
        ),

        "top_geometry_cohorts": (
            top_geometry_cohorts
        ),

        "interpretation_notes": [
            (
                "A strong association between geometry cohort "
                "and candidate scale suggests that intensity "
                "heterogeneity may be related to acquisition or "
                "reconstruction families."
            ),
            (
                "Geometry alone cannot establish the scanner or "
                "reconstruction protocol."
            ),
            (
                "No normalization strategy is selected by this "
                "analysis."
            ),
            (
                "Do not normalize images during this step."
            ),
        ],
    }

    with (
        output_dir
        / "acquisition_cohort_statistics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    # --------------------------------------------
    # Plots
    # --------------------------------------------

    _save_cohort_count_plot(
        cohort_summary=cohort_summary,
        output_path=(
            output_dir
            / "geometry_cohort_counts.png"
        ),
        top_n=top_cohorts,
    )

    _save_log_scale_histogram(
        dataframe=merged,
        output_path=(
            output_dir
            / "candidate_scale_log10_histogram.png"
        ),
    )

    _save_scale_by_cohort_boxplot(
        dataframe=merged,
        cohort_summary=cohort_summary,
        output_path=(
            output_dir
            / "candidate_scale_by_geometry_cohort.png"
        ),
        top_n=min(top_cohorts, 10),
    )

    _save_scatter(
        dataframe=merged,
        x_column="foreground_fraction",
        y_column=SCALE_COLUMN,
        xlabel="Foreground fraction",
        ylabel=(
            "log10(candidate normalization scale)"
        ),
        title=(
            "Foreground Fraction vs Candidate Intensity Scale"
        ),
        output_path=(
            output_dir
            / "foreground_fraction_vs_scale.png"
        ),
        log_y=True,
    )

    _save_scatter(
        dataframe=merged,
        x_column="original_voxel_volume_mm3",
        y_column=SCALE_COLUMN,
        xlabel="Original voxel volume (mm³)",
        ylabel=(
            "log10(candidate normalization scale)"
        ),
        title=(
            "Original Voxel Volume vs Candidate Intensity Scale"
        ),
        output_path=(
            output_dir
            / "voxel_volume_vs_scale.png"
        ),
        log_y=True,
    )

    _save_scatter(
        dataframe=merged,
        x_column="fov_mm_z",
        y_column="foreground_fraction",
        xlabel="Resampled Z-axis FOV (mm)",
        ylabel="Foreground fraction",
        title="Z-axis FOV vs Foreground Fraction",
        output_path=(
            output_dir
            / "fov_z_vs_foreground_fraction.png"
        ),
        log_y=False,
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze acquisition/reconstruction geometry cohorts "
            "and their relationship to DaT-SPECT intensity scale."
        )
    )

    parser.add_argument(
        "--reference-csv",
        type=str,
        required=True,
        help=(
            "CSV created by the reference-normalization "
            "analysis step."
        ),
    )

    parser.add_argument(
        "--resampling-summary",
        type=str,
        required=True,
        help=(
            "resampling_summary.json containing original "
            "shape and voxel-spacing information."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory in which analysis outputs are saved.",
    )

    parser.add_argument(
        "--spacing-round-decimals",
        type=int,
        default=3,
        help=(
            "Decimals used to group original voxel spacing. "
            "Default: 3"
        ),
    )

    parser.add_argument(
        "--top-cohorts",
        type=int,
        default=15,
        help=(
            "Number of largest geometry cohorts included "
            "in plots and summary. Default: 15"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("ACQUISITION / RECONSTRUCTION COHORT ANALYSIS")
    print("=" * 72)

    print(
        f"Reference analysis CSV: "
        f"{args.reference_csv}"
    )

    print(
        f"Resampling summary: "
        f"{args.resampling_summary}"
    )

    print(
        f"Output directory: "
        f"{args.output_dir}"
    )

    print(
        f"Spacing rounding: "
        f"{args.spacing_round_decimals} decimals"
    )

    print("-" * 72)

    summary = analyze_acquisition_cohorts(
        reference_csv_path=args.reference_csv,
        resampling_summary_path=args.resampling_summary,
        output_dir=args.output_dir,
        spacing_round_decimals=(
            args.spacing_round_decimals
        ),
        top_cohorts=args.top_cohorts,
    )

    print()
    print("=" * 72)
    print("ACQUISITION COHORT ANALYSIS COMPLETED")
    print("=" * 72)

    print(
        f"Matched scans: "
        f"{summary['number_of_matched_scans']}"
    )

    print(
        f"Unique geometry cohorts: "
        f"{summary['number_of_unique_geometry_cohorts']}"
    )

    print(
        f"Unique original shapes: "
        f"{summary['number_of_unique_original_shapes']}"
    )

    print(
        f"Unique spacing families: "
        f"{summary['number_of_unique_original_spacing_families']}"
    )

    print()
    print("Scale decades:")

    for decade, count in (
        summary["scale_decade_counts"].items()
    ):
        print(
            f"  {decade}: {count}"
        )

    print()
    print(
        "Rank correlations with candidate "
        "normalization scale:"
    )

    for name, value in (
        summary[
            "rank_correlations_with_candidate_scale"
        ].items()
    ):
        print(
            f"  {name}: {value}"
        )

    print()
    print("Largest geometry cohorts:")

    for cohort in summary[
        "top_geometry_cohorts"
    ]:
        print(
            f"  {cohort['count']:4d} scans | "
            f"{cohort['geometry_cohort']} | "
            f"scale median="
            f"{cohort.get('scale_median')}"
        )

    print()
    print(
        "NOTE: Geometry cohorts are proxies for acquisition/"
        "reconstruction families. They are not confirmed scanner "
        "protocols without acquisition metadata."
    )

    print()
    print(
        "No images were normalized or modified."
    )

    print()
    print(
        f"Results saved to: "
        f"{args.output_dir}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()