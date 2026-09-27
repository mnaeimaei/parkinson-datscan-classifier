# dat-scan-classifier/scripts/preprocessing_image_script/step3_voxel_spacing_analyzer_script/analyzing.py

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_spacing_data(
    orientation_summary_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Load standardized voxel spacing for all scans from
    orientation_summary.json.

    No resampling is performed.
    """

    orientation_summary_path = Path(
        orientation_summary_path
    )

    if not orientation_summary_path.exists():
        raise FileNotFoundError(
            f"File not found: {orientation_summary_path}"
        )

    with orientation_summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    if "files" not in summary:
        raise ValueError(
            "orientation_summary.json does not contain "
            "the 'files' field."
        )

    records = []

    for item in summary["files"]:

        spacing = item.get(
            "standardized_spacing"
        )

        if spacing is None:
            raise ValueError(
                f"Missing standardized_spacing for "
                f"{item.get('file_name')}"
            )

        if len(spacing) < 3:
            raise ValueError(
                f"Invalid spacing for "
                f"{item.get('file_name')}: {spacing}"
            )

        records.append(
            {
                "file_name": item["file_name"],
                "spacing_x": float(spacing[0]),
                "spacing_y": float(spacing[1]),
                "spacing_z": float(spacing[2]),
            }
        )

    if not records:
        raise ValueError(
            "No spacing records were found."
        )

    return records


def calculate_axis_statistics(
    values: np.ndarray,
) -> dict[str, float]:
    """
    Calculate descriptive statistics for one voxel-spacing axis.
    """

    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def calculate_statistics(
    records: list[dict[str, Any]],
    round_decimals: int = 4,
) -> dict[str, Any]:
    """
    Calculate voxel-spacing distribution statistics.

    Exact values are used for descriptive statistics.

    Rounded values are used only for frequency counting,
    because NIfTI spacing values may contain small floating
    point differences.
    """

    x = np.asarray(
        [item["spacing_x"] for item in records],
        dtype=np.float64,
    )

    y = np.asarray(
        [item["spacing_y"] for item in records],
        dtype=np.float64,
    )

    z = np.asarray(
        [item["spacing_z"] for item in records],
        dtype=np.float64,
    )

    rounded_x = [
        round(float(value), round_decimals)
        for value in x
    ]

    rounded_y = [
        round(float(value), round_decimals)
        for value in y
    ]

    rounded_z = [
        round(float(value), round_decimals)
        for value in z
    ]

    x_counter = Counter(rounded_x)
    y_counter = Counter(rounded_y)
    z_counter = Counter(rounded_z)

    triplets = [
        (
            round(
                item["spacing_x"],
                round_decimals,
            ),
            round(
                item["spacing_y"],
                round_decimals,
            ),
            round(
                item["spacing_z"],
                round_decimals,
            ),
        )
        for item in records
    ]

    triplet_counter = Counter(triplets)

    most_common_triplet, most_common_count = (
        triplet_counter.most_common(1)[0]
    )

    isotropic_count = sum(
        1
        for sx, sy, sz in triplets
        if sx == sy == sz
    )

    total = len(records)

    return {
        "number_of_scans": total,

        "round_decimals_for_frequency": (
            round_decimals
        ),

        "x_axis": calculate_axis_statistics(x),

        "y_axis": calculate_axis_statistics(y),

        "z_axis": calculate_axis_statistics(z),

        "mode": {
            "x": x_counter.most_common(1)[0][0],
            "y": y_counter.most_common(1)[0][0],
            "z": z_counter.most_common(1)[0][0],
        },

        "unique_spacing_values": {
            "x": len(x_counter),
            "y": len(y_counter),
            "z": len(z_counter),
        },

        "unique_spacing_triplets": len(
            triplet_counter
        ),

        "most_common_spacing_triplet": {
            "spacing_x": most_common_triplet[0],
            "spacing_y": most_common_triplet[1],
            "spacing_z": most_common_triplet[2],
            "count": most_common_count,
            "percentage": (
                most_common_count / total
            ) * 100.0,
        },

        "isotropic_scans": {
            "count": isotropic_count,
            "percentage": (
                isotropic_count / total
            ) * 100.0,
        },
    }


def save_per_scan_csv(
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Save voxel spacing for every scan.
    """

    output_path = Path(output_path)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file_name",
                "spacing_x",
                "spacing_y",
                "spacing_z",
            ],
        )

        writer.writeheader()
        writer.writerows(records)


def save_frequency_csv(
    records: list[dict[str, Any]],
    output_path: str | Path,
    round_decimals: int = 4,
) -> None:
    """
    Save frequency of XYZ voxel-spacing combinations.
    """

    output_path = Path(output_path)

    triplets = [
        (
            round(
                item["spacing_x"],
                round_decimals,
            ),
            round(
                item["spacing_y"],
                round_decimals,
            ),
            round(
                item["spacing_z"],
                round_decimals,
            ),
        )
        for item in records
    ]

    counter = Counter(triplets)

    total = len(records)

    rows = []

    for (
        spacing_x,
        spacing_y,
        spacing_z,
    ), count in counter.most_common():

        rows.append(
            {
                "spacing_x": spacing_x,
                "spacing_y": spacing_y,
                "spacing_z": spacing_z,
                "count": count,
                "percentage": (
                    count / total
                ) * 100.0,
            }
        )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "spacing_x",
                "spacing_y",
                "spacing_z",
                "count",
                "percentage",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def save_histogram(
    values: list[float],
    axis_name: str,
    output_path: str | Path,
    bins: int = 30,
) -> None:
    """
    Save voxel-spacing histogram for one axis.
    """

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        values,
        bins=bins,
    )

    plt.xlabel(
        f"{axis_name}-axis voxel spacing (mm)"
    )

    plt.ylabel(
        "Number of scans"
    )

    plt.title(
        f"{axis_name}-axis voxel spacing distribution"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def save_spacing_scatter(
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Save X versus Y voxel-spacing scatter plot.
    """

    x = [
        item["spacing_x"]
        for item in records
    ]

    y = [
        item["spacing_y"]
        for item in records
    ]

    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        x,
        y,
        alpha=0.6,
    )

    plt.xlabel(
        "X-axis spacing (mm)"
    )

    plt.ylabel(
        "Y-axis spacing (mm)"
    )

    plt.title(
        "X vs Y voxel spacing"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def analyze_voxel_spacing(
    orientation_summary_path: str | Path,
    output_dir: str | Path,
    round_decimals: int = 4,
    histogram_bins: int = 30,
) -> dict[str, Any]:
    """
    Analyze voxel-spacing distribution for all scans.

    This function ONLY analyzes voxel spacing.

    It does NOT:
    - choose a target spacing
    - resample images
    - modify NIfTI files
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_spacing_data(
        orientation_summary_path
    )

    statistics = calculate_statistics(
        records=records,
        round_decimals=round_decimals,
    )

    # -------------------------
    # Statistics JSON
    # -------------------------

    statistics_path = (
        output_dir
        / "voxel_spacing_statistics.json"
    )

    with statistics_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            statistics,
            file,
            indent=4,
        )

    # -------------------------
    # Per-scan CSV
    # -------------------------

    save_per_scan_csv(
        records=records,
        output_path=(
            output_dir
            / "voxel_spacing_per_scan.csv"
        ),
    )

    # -------------------------
    # Frequency CSV
    # -------------------------

    save_frequency_csv(
        records=records,
        output_path=(
            output_dir
            / "voxel_spacing_frequency.csv"
        ),
        round_decimals=round_decimals,
    )

    # -------------------------
    # Histograms
    # -------------------------

    save_histogram(
        values=[
            item["spacing_x"]
            for item in records
        ],
        axis_name="X",
        output_path=(
            output_dir
            / "voxel_spacing_x_histogram.png"
        ),
        bins=histogram_bins,
    )

    save_histogram(
        values=[
            item["spacing_y"]
            for item in records
        ],
        axis_name="Y",
        output_path=(
            output_dir
            / "voxel_spacing_y_histogram.png"
        ),
        bins=histogram_bins,
    )

    save_histogram(
        values=[
            item["spacing_z"]
            for item in records
        ],
        axis_name="Z",
        output_path=(
            output_dir
            / "voxel_spacing_z_histogram.png"
        ),
        bins=histogram_bins,
    )

    # -------------------------
    # X-Y scatter
    # -------------------------

    save_spacing_scatter(
        records=records,
        output_path=(
            output_dir
            / "voxel_spacing_xy_scatter.png"
        ),
    )

    return statistics


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze voxel-spacing distribution "
            "of the NIfTI dataset."
        )
    )

    parser.add_argument(
        "--orientation-summary",
        type=Path,
        required=True,
        help=(
            "Path to orientation_summary.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory where analysis results "
            "will be saved."
        ),
    )

    parser.add_argument(
        "--round-decimals",
        type=int,
        default=4,
        help=(
            "Decimal precision used for "
            "spacing-frequency grouping."
        ),
    )

    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=30,
        help=(
            "Number of bins used in histograms."
        ),
    )

    args = parser.parse_args()

    statistics = analyze_voxel_spacing(
        orientation_summary_path=(
            args.orientation_summary
        ),
        output_dir=args.output_dir,
        round_decimals=args.round_decimals,
        histogram_bins=args.histogram_bins,
    )

    print()
    print("=" * 60)
    print("VOXEL SPACING ANALYSIS")
    print("=" * 60)

    print()
    print(
        "Number of scans:",
        statistics["number_of_scans"],
    )

    print()
    print("X-axis spacing")
    print(
        f"  Min:    "
        f"{statistics['x_axis']['min']:.4f}"
    )
    print(
        f"  P05:    "
        f"{statistics['x_axis']['p05']:.4f}"
    )
    print(
        f"  P25:    "
        f"{statistics['x_axis']['p25']:.4f}"
    )
    print(
        f"  Median: "
        f"{statistics['x_axis']['median']:.4f}"
    )
    print(
        f"  Mean:   "
        f"{statistics['x_axis']['mean']:.4f}"
    )
    print(
        f"  P75:    "
        f"{statistics['x_axis']['p75']:.4f}"
    )
    print(
        f"  P95:    "
        f"{statistics['x_axis']['p95']:.4f}"
    )
    print(
        f"  Max:    "
        f"{statistics['x_axis']['max']:.4f}"
    )
    print(
        f"  Mode:   "
        f"{statistics['mode']['x']}"
    )

    print()
    print("Y-axis spacing")
    print(
        f"  Min:    "
        f"{statistics['y_axis']['min']:.4f}"
    )
    print(
        f"  P05:    "
        f"{statistics['y_axis']['p05']:.4f}"
    )
    print(
        f"  P25:    "
        f"{statistics['y_axis']['p25']:.4f}"
    )
    print(
        f"  Median: "
        f"{statistics['y_axis']['median']:.4f}"
    )
    print(
        f"  Mean:   "
        f"{statistics['y_axis']['mean']:.4f}"
    )
    print(
        f"  P75:    "
        f"{statistics['y_axis']['p75']:.4f}"
    )
    print(
        f"  P95:    "
        f"{statistics['y_axis']['p95']:.4f}"
    )
    print(
        f"  Max:    "
        f"{statistics['y_axis']['max']:.4f}"
    )
    print(
        f"  Mode:   "
        f"{statistics['mode']['y']}"
    )

    print()
    print("Z-axis spacing")
    print(
        f"  Min:    "
        f"{statistics['z_axis']['min']:.4f}"
    )
    print(
        f"  P05:    "
        f"{statistics['z_axis']['p05']:.4f}"
    )
    print(
        f"  P25:    "
        f"{statistics['z_axis']['p25']:.4f}"
    )
    print(
        f"  Median: "
        f"{statistics['z_axis']['median']:.4f}"
    )
    print(
        f"  Mean:   "
        f"{statistics['z_axis']['mean']:.4f}"
    )
    print(
        f"  P75:    "
        f"{statistics['z_axis']['p75']:.4f}"
    )
    print(
        f"  P95:    "
        f"{statistics['z_axis']['p95']:.4f}"
    )
    print(
        f"  Max:    "
        f"{statistics['z_axis']['max']:.4f}"
    )
    print(
        f"  Mode:   "
        f"{statistics['mode']['z']}"
    )

    common = statistics[
        "most_common_spacing_triplet"
    ]

    print()
    print("Most common spacing triplet")
    print(
        f"  {common['spacing_x']} × "
        f"{common['spacing_y']} × "
        f"{common['spacing_z']} mm"
    )

    print(
        f"  Count: "
        f"{common['count']}"
    )

    print(
        f"  Percentage: "
        f"{common['percentage']:.2f}%"
    )

    print()
    print(
        "Unique spacing triplets:",
        statistics[
            "unique_spacing_triplets"
        ],
    )

    print(
        "Isotropic scans:",
        statistics[
            "isotropic_scans"
        ]["count"],
    )

    print(
        "Isotropic percentage:",
        f"{statistics['isotropic_scans']['percentage']:.2f}%",
    )

    print()
    print(
        "Results saved to:",
        args.output_dir,
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
