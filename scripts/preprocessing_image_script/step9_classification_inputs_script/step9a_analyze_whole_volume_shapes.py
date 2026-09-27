#!/usr/bin/env python3

"""
Step 9A — Analyze normalized whole-volume shapes.

Purpose
-------
Analyze the spatial dimensions and voxel spacing of the normalized
whole-volume DaT scans before selecting a fixed input shape for
classification Scenario A and Scenario C.

This step is READ-ONLY:
    - no resampling
    - no cropping
    - no padding
    - no intensity modification

Outputs
-------
1. whole_volume_shapes.csv
2. whole_volume_shape_statistics.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


EXPECTED_SPACING_MM = 2.46
SPACING_TOLERANCE_MM = 1e-3


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def uid_from_path(path: Path) -> str:
    """Return UID from .nii or .nii.gz filename."""
    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return path.stem


def find_nifti_files(input_dir: Path) -> list[Path]:
    """Find all .nii and .nii.gz files."""
    files = list(input_dir.glob("*.nii"))
    files.extend(input_dir.glob("*.nii.gz"))

    return sorted(set(files))


def percentile_summary(values: list[float]) -> dict[str, float]:
    """Calculate descriptive statistics."""
    array = np.asarray(values, dtype=np.float64)

    percentiles = {
        "min": np.min(array),
        "p05": np.percentile(array, 5),
        "p10": np.percentile(array, 10),
        "p25": np.percentile(array, 25),
        "median": np.percentile(array, 50),
        "p75": np.percentile(array, 75),
        "p90": np.percentile(array, 90),
        "p95": np.percentile(array, 95),
        "max": np.max(array),
        "mean": np.mean(array),
        "std": np.std(array),
    }

    return {
        key: round(float(value), 4)
        for key, value in percentiles.items()
    }


def dimension_summary(values: list[int]) -> dict[str, Any]:
    """Summary specifically for voxel dimensions."""
    result = percentile_summary([float(v) for v in values])

    # Dimension statistics are more readable as integers where exact.
    result["min"] = int(min(values))
    result["max"] = int(max(values))

    return result


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------


def analyze_scans(
    input_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = find_nifti_files(input_dir)

    if not files:
        raise RuntimeError(
            f"No NIfTI files found in:\n{input_dir}"
        )

    records: list[dict[str, Any]] = []

    shape_x: list[int] = []
    shape_y: list[int] = []
    shape_z: list[int] = []

    spacing_x: list[float] = []
    spacing_y: list[float] = []
    spacing_z: list[float] = []

    fov_x: list[float] = []
    fov_y: list[float] = []
    fov_z: list[float] = []

    shape_counter: Counter[tuple[int, int, int]] = Counter()

    spacing_validation_failures = 0
    invalid_dimension_scans = 0
    failed_files: list[dict[str, str]] = []

    print()
    print("=" * 72)
    print("STEP 9A — ANALYZE NORMALIZED WHOLE-VOLUME SHAPES")
    print("=" * 72)
    print(f"Input directory : {input_dir}")
    print(f"NIfTI files     : {len(files)}")
    print("=" * 72)

    for index, path in enumerate(files, start=1):
        try:
            image = nib.load(str(path))

            shape = image.shape
            zooms = image.header.get_zooms()

            if len(shape) != 3:
                invalid_dimension_scans += 1
                failed_files.append(
                    {
                        "uid": uid_from_path(path),
                        "file": str(path),
                        "reason": f"Expected 3D image, found shape {shape}",
                    }
                )
                continue

            x, y, z = map(int, shape)
            sx, sy, sz = map(float, zooms[:3])

            spacing_ok = all(
                abs(value - EXPECTED_SPACING_MM)
                <= SPACING_TOLERANCE_MM
                for value in (sx, sy, sz)
            )

            if not spacing_ok:
                spacing_validation_failures += 1

            physical_x = x * sx
            physical_y = y * sy
            physical_z = z * sz

            voxel_count = int(x * y * z)

            record = {
                "uid": uid_from_path(path),
                "file": str(path),
                "shape_x": x,
                "shape_y": y,
                "shape_z": z,
                "spacing_x_mm": round(sx, 6),
                "spacing_y_mm": round(sy, 6),
                "spacing_z_mm": round(sz, 6),
                "fov_x_mm": round(physical_x, 3),
                "fov_y_mm": round(physical_y, 3),
                "fov_z_mm": round(physical_z, 3),
                "voxel_count": voxel_count,
                "spacing_valid": spacing_ok,
            }

            records.append(record)

            shape_x.append(x)
            shape_y.append(y)
            shape_z.append(z)

            spacing_x.append(sx)
            spacing_y.append(sy)
            spacing_z.append(sz)

            fov_x.append(physical_x)
            fov_y.append(physical_y)
            fov_z.append(physical_z)

            shape_counter[(x, y, z)] += 1

            if (
                index == 1
                or index % 100 == 0
                or index == len(files)
            ):
                print(
                    f"[{index:4d}/{len(files)}] "
                    f"{uid_from_path(path):20s} "
                    f"shape=({x:3d}, {y:3d}, {z:3d}) "
                    f"spacing=({sx:.4f}, {sy:.4f}, {sz:.4f})"
                )

        except Exception as exc:
            failed_files.append(
                {
                    "uid": uid_from_path(path),
                    "file": str(path),
                    "reason": str(exc),
                }
            )

    if not records:
        raise RuntimeError("No valid 3D scans could be analyzed.")

    most_common_shapes = [
        {
            "shape": list(shape),
            "count": int(count),
            "percentage": round(
                100.0 * count / len(records),
                3,
            ),
        }
        for shape, count in shape_counter.most_common(20)
    ]

    statistics = {
        "step": "9A",
        "description": "Normalized whole-volume shape analysis",
        "input_directory": str(input_dir),
        "number_of_files_found": len(files),
        "number_of_scans_analyzed": len(records),
        "number_of_failed_scans": len(failed_files),
        "invalid_dimension_scans": invalid_dimension_scans,
        "spacing_validation": {
            "expected_spacing_mm": EXPECTED_SPACING_MM,
            "tolerance_mm": SPACING_TOLERANCE_MM,
            "number_of_failures": spacing_validation_failures,
        },
        "shape_statistics_voxels": {
            "x": dimension_summary(shape_x),
            "y": dimension_summary(shape_y),
            "z": dimension_summary(shape_z),
        },
        "spacing_statistics_mm": {
            "x": percentile_summary(spacing_x),
            "y": percentile_summary(spacing_y),
            "z": percentile_summary(spacing_z),
        },
        "physical_fov_statistics_mm": {
            "x": percentile_summary(fov_x),
            "y": percentile_summary(fov_y),
            "z": percentile_summary(fov_z),
        },
        "number_of_unique_shapes": len(shape_counter),
        "most_common_shapes": most_common_shapes,
        "failed_files": failed_files,
    }

    return records, statistics


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------


def save_csv(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "uid",
        "file",
        "shape_x",
        "shape_y",
        "shape_z",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "fov_x_mm",
        "fov_y_mm",
        "fov_z_mm",
        "voxel_count",
        "spacing_valid",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(records)


def save_json(
    statistics: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            statistics,
            handle,
            indent=2,
            sort_keys=False,
        )


def print_summary(statistics: dict[str, Any]) -> None:
    shape_stats = statistics["shape_statistics_voxels"]

    print()
    print("=" * 72)
    print("STEP 9A SUMMARY")
    print("=" * 72)

    print(
        f"Scans analyzed              : "
        f"{statistics['number_of_scans_analyzed']}"
    )
    print(
        f"Failed scans                : "
        f"{statistics['number_of_failed_scans']}"
    )
    print(
        f"Spacing validation failures : "
        f"{statistics['spacing_validation']['number_of_failures']}"
    )
    print(
        f"Unique shapes               : "
        f"{statistics['number_of_unique_shapes']}"
    )

    print()
    print("X DIMENSION")
    print(json.dumps(shape_stats["x"], indent=2))

    print()
    print("Y DIMENSION")
    print(json.dumps(shape_stats["y"], indent=2))

    print()
    print("Z DIMENSION")
    print(json.dumps(shape_stats["z"], indent=2))

    print()
    print("MOST COMMON SHAPES")

    for item in statistics["most_common_shapes"][:10]:
        shape = tuple(item["shape"])

        print(
            f"  {str(shape):20s} "
            f"{item['count']:4d} scans "
            f"({item['percentage']:.2f}%)"
        )

    print("=" * 72)
    print(
        "IMPORTANT: No fixed classification shape is selected in Step 9A."
    )
    print(
        "Use these results to choose the Step 9B crop/padding dimensions."
    )
    print("=" * 72)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 9A: analyze normalized whole-volume "
            "DaT scan shapes."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing normalized NIfTI scans.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where Step 9A reports are saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist:\n{input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records, statistics = analyze_scans(input_dir)

    csv_path = (
        output_dir
        / "whole_volume_shapes.csv"
    )

    json_path = (
        output_dir
        / "whole_volume_shape_statistics.json"
    )

    save_csv(records, csv_path)
    save_json(statistics, json_path)

    print_summary(statistics)

    print()
    print(f"CSV saved  : {csv_path}")
    print(f"JSON saved : {json_path}")
    print()


if __name__ == "__main__":
    main()
