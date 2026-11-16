#!/usr/bin/env python3

"""
Step 9B — Analyze candidate fixed whole-volume shapes.

This step DOES NOT modify or save image volumes.

For each normalized whole-volume scan, it simulates deterministic
center crop/padding using several candidate shapes and measures how
much above-background image signal would be retained.

The purpose is to choose the fixed whole-volume shape for Step 9C.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


DEFAULT_CANDIDATES = [
    (128, 128, 96),
    (160, 160, 128),
    (192, 192, 160),
    (224, 224, 192),
]


def uid_from_path(path: Path) -> str:
    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return path.stem


def find_nifti_files(input_dir: Path) -> list[Path]:
    files = list(input_dir.glob("*.nii"))
    files.extend(input_dir.glob("*.nii.gz"))

    return sorted(set(files))


def parse_candidate(text: str) -> tuple[int, int, int]:
    cleaned = text.lower().replace("x", ",")
    parts = [p.strip() for p in cleaned.split(",")]

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid candidate shape: {text}"
        )

    try:
        values = tuple(int(v) for v in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid candidate shape: {text}"
        ) from exc

    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(
            "Candidate dimensions must be positive."
        )

    return values


def centered_source_slices(
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    """
    Return the portion of the source that would survive a centered
    crop/pad operation.

    If the source dimension is smaller than the target dimension,
    the entire source axis is retained and Step 9C would pad it.
    """

    slices = []

    for source_size, target_size in zip(
        source_shape,
        target_shape,
    ):
        if source_size <= target_size:
            slices.append(slice(0, source_size))
        else:
            start = (source_size - target_size) // 2
            stop = start + target_size
            slices.append(slice(start, stop))

    return tuple(slices)


def percentile_summary(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)

    return {
        "min": round(float(np.min(array)), 6),
        "p05": round(float(np.percentile(array, 5)), 6),
        "p10": round(float(np.percentile(array, 10)), 6),
        "p25": round(float(np.percentile(array, 25)), 6),
        "median": round(float(np.percentile(array, 50)), 6),
        "p75": round(float(np.percentile(array, 75)), 6),
        "p90": round(float(np.percentile(array, 90)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "max": round(float(np.max(array)), 6),
        "mean": round(float(np.mean(array)), 6),
        "std": round(float(np.std(array)), 6),
    }


def intensity_center_of_mass(
    weights: np.ndarray,
) -> tuple[float, float, float]:
    total = float(weights.sum())

    if total <= 0:
        return tuple(
            (size - 1) / 2.0
            for size in weights.shape
        )

    projection_x = weights.sum(axis=(1, 2))
    projection_y = weights.sum(axis=(0, 2))
    projection_z = weights.sum(axis=(0, 1))

    x = float(
        np.dot(
            np.arange(weights.shape[0]),
            projection_x,
        )
        / total
    )

    y = float(
        np.dot(
            np.arange(weights.shape[1]),
            projection_y,
        )
        / total
    )

    z = float(
        np.dot(
            np.arange(weights.shape[2]),
            projection_z,
        )
        / total
    )

    return x, y, z


def analyze(
    input_dir: Path,
    candidates: list[tuple[int, int, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:

    files = find_nifti_files(input_dir)

    if not files:
        raise RuntimeError(
            f"No NIfTI files found in {input_dir}"
        )

    records: list[dict[str, Any]] = []

    candidate_retention: dict[
        tuple[int, int, int],
        list[float],
    ] = {
        candidate: []
        for candidate in candidates
    }

    center_offsets_x_mm: list[float] = []
    center_offsets_y_mm: list[float] = []
    center_offsets_z_mm: list[float] = []

    failures: list[dict[str, str]] = []

    print()
    print("=" * 76)
    print("STEP 9B — ANALYZE FIXED WHOLE-VOLUME CANDIDATES")
    print("=" * 76)
    print(f"Input scans : {len(files)}")
    print()
    print("Candidates:")

    for candidate in candidates:
        print(f"  {candidate}")

    print("=" * 76)

    for index, path in enumerate(files, start=1):
        uid = uid_from_path(path)

        try:
            image = nib.load(
                str(path),
                mmap="r",
            )

            if len(image.shape) != 3:
                raise ValueError(
                    f"Expected 3D image, got {image.shape}"
                )

            data = image.get_fdata(
                dtype=np.float32
            )

            data = np.nan_to_num(
                data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            spacing = tuple(
                float(v)
                for v in image.header.get_zooms()[:3]
            )

            #
            # Estimate a low-level background baseline.
            #
            # Subtracting P10 prevents large low-intensity/background
            # regions from dominating the retention measurement.
            #
            finite_values = data[np.isfinite(data)]

            background = float(
                np.percentile(
                    finite_values,
                    10,
                )
            )

            weights = np.clip(
                data - background,
                a_min=0.0,
                a_max=None,
            )

            total_weight = float(weights.sum())

            if total_weight <= 0:
                #
                # Extremely defensive fallback.
                #
                weights = np.abs(data)
                total_weight = float(weights.sum())

            if total_weight <= 0:
                raise ValueError(
                    "Image has no measurable signal."
                )

            com = intensity_center_of_mass(
                weights
            )

            geometric_center = tuple(
                (size - 1) / 2.0
                for size in data.shape
            )

            offset_x_mm = (
                com[0] - geometric_center[0]
            ) * spacing[0]

            offset_y_mm = (
                com[1] - geometric_center[1]
            ) * spacing[1]

            offset_z_mm = (
                com[2] - geometric_center[2]
            ) * spacing[2]

            center_offsets_x_mm.append(
                abs(offset_x_mm)
            )
            center_offsets_y_mm.append(
                abs(offset_y_mm)
            )
            center_offsets_z_mm.append(
                abs(offset_z_mm)
            )

            record: dict[str, Any] = {
                "uid": uid,
                "shape_x": data.shape[0],
                "shape_y": data.shape[1],
                "shape_z": data.shape[2],
                "background_p10": round(
                    background,
                    6,
                ),
                "center_offset_x_mm": round(
                    offset_x_mm,
                    4,
                ),
                "center_offset_y_mm": round(
                    offset_y_mm,
                    4,
                ),
                "center_offset_z_mm": round(
                    offset_z_mm,
                    4,
                ),
            }

            for candidate in candidates:
                source_slices = centered_source_slices(
                    data.shape,
                    candidate,
                )

                retained = float(
                    weights[source_slices].sum()
                )

                fraction = retained / total_weight

                fraction = min(
                    max(fraction, 0.0),
                    1.0,
                )

                candidate_retention[
                    candidate
                ].append(fraction)

                name = (
                    f"{candidate[0]}x"
                    f"{candidate[1]}x"
                    f"{candidate[2]}"
                )

                record[
                    f"retention_{name}"
                ] = round(
                    fraction,
                    6,
                )

            records.append(record)

            if (
                index == 1
                or index % 100 == 0
                or index == len(files)
            ):
                retention_text = " | ".join(
                    (
                        f"{candidate}: "
                        f"{candidate_retention[candidate][-1] * 100:.2f}%"
                    )
                    for candidate in candidates
                )

                print(
                    f"[{index:4d}/{len(files)}] "
                    f"{uid:18s} "
                    f"shape={data.shape} | "
                    f"{retention_text}"
                )

        except Exception as exc:
            failures.append(
                {
                    "uid": uid,
                    "file": str(path),
                    "reason": str(exc),
                }
            )

    candidate_summary: dict[str, Any] = {}

    for candidate in candidates:
        key = (
            f"{candidate[0]}x"
            f"{candidate[1]}x"
            f"{candidate[2]}"
        )

        values = candidate_retention[
            candidate
        ]

        array = np.asarray(
            values,
            dtype=np.float64,
        )

        candidate_summary[key] = {
            "shape": list(candidate),
            "physical_fov_mm": [
                round(
                    candidate[0] * 2.46,
                    2,
                ),
                round(
                    candidate[1] * 2.46,
                    2,
                ),
                round(
                    candidate[2] * 2.46,
                    2,
                ),
            ],
            "retention_fraction_statistics":
                percentile_summary(values),

            "scans_retaining_at_least_99_percent":
                int(
                    np.sum(
                        array >= 0.99
                    )
                ),

            "scans_retaining_at_least_99_5_percent":
                int(
                    np.sum(
                        array >= 0.995
                    )
                ),

            "scans_retaining_at_least_99_9_percent":
                int(
                    np.sum(
                        array >= 0.999
                    )
                ),

            "percentage_retaining_at_least_99_percent":
                round(
                    100.0
                    * float(
                        np.mean(
                            array >= 0.99
                        )
                    ),
                    3,
                ),

            "percentage_retaining_at_least_99_5_percent":
                round(
                    100.0
                    * float(
                        np.mean(
                            array >= 0.995
                        )
                    ),
                    3,
                ),

            "percentage_retaining_at_least_99_9_percent":
                round(
                    100.0
                    * float(
                        np.mean(
                            array >= 0.999
                        )
                    ),
                    3,
                ),
        }

    summary = {
        "step": "9B",
        "description":
            "Candidate fixed whole-volume FOV retention analysis",

        "number_of_files_found": len(files),

        "number_of_scans_analyzed":
            len(records),

        "number_of_failed_scans":
            len(failures),

        "method": {
            "crop_alignment":
                "geometric image center",

            "padding":
                "conceptual only; no output image generated",

            "background_estimate":
                "10th intensity percentile",

            "signal_weight":
                "max(voxel - p10, 0)",

            "retention_metric":
                "fraction of above-background signal "
                "remaining after centered crop",
        },

        "absolute_signal_center_offset_mm": {
            "x": percentile_summary(
                center_offsets_x_mm
            ),
            "y": percentile_summary(
                center_offsets_y_mm
            ),
            "z": percentile_summary(
                center_offsets_z_mm
            ),
        },

        "candidate_shapes":
            candidate_summary,

        "failed_files":
            failures,
    }

    return records, summary


def save_csv(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not records:
        return

    fieldnames = list(
        records[0].keys()
    )

    with path.open(
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
    summary: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )


def print_summary(
    summary: dict[str, Any],
) -> None:
    print()
    print("=" * 76)
    print("STEP 9B SUMMARY")
    print("=" * 76)

    print(
        "Scans analyzed : "
        f"{summary['number_of_scans_analyzed']}"
    )

    print(
        "Failed scans   : "
        f"{summary['number_of_failed_scans']}"
    )

    print()
    print("SIGNAL-CENTER ABSOLUTE OFFSET FROM IMAGE CENTER")

    offsets = summary[
        "absolute_signal_center_offset_mm"
    ]

    for axis in ("x", "y", "z"):
        values = offsets[axis]

        print(
            f"  {axis.upper()}: "
            f"median={values['median']:.2f} mm | "
            f"P95={values['p95']:.2f} mm | "
            f"max={values['max']:.2f} mm"
        )

    print()
    print("CANDIDATE RETENTION")
    print()

    for name, result in summary[
        "candidate_shapes"
    ].items():

        stats = result[
            "retention_fraction_statistics"
        ]

        print(f"{name}")
        print(
            f"  FOV mm       : "
            f"{result['physical_fov_mm']}"
        )

        print(
            f"  median       : "
            f"{stats['median'] * 100:.3f}%"
        )

        print(
            f"  P05          : "
            f"{stats['p05'] * 100:.3f}%"
        )

        print(
            f"  minimum      : "
            f"{stats['min'] * 100:.3f}%"
        )

        print(
            f"  >=99.0%      : "
            f"{result['percentage_retaining_at_least_99_percent']:.2f}% "
            "of scans"
        )

        print(
            f"  >=99.5%      : "
            f"{result['percentage_retaining_at_least_99_5_percent']:.2f}% "
            "of scans"
        )

        print(
            f"  >=99.9%      : "
            f"{result['percentage_retaining_at_least_99_9_percent']:.2f}% "
            "of scans"
        )

        print()

    print("=" * 76)
    print(
        "No candidate is automatically selected."
    )
    print(
        "Review these results before Step 9C."
    )
    print("=" * 76)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--candidate",
        type=parse_candidate,
        action="append",
        dest="candidates",
        help=(
            "Candidate shape, e.g. 160x160x128. "
            "Can be specified multiple times."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    candidates = (
        args.candidates
        if args.candidates
        else DEFAULT_CANDIDATES
    )

    records, summary = analyze(
        input_dir=args.input_dir.resolve(),
        candidates=candidates,
    )

    output_dir = args.output_dir.resolve()

    csv_path = (
        output_dir
        / "whole_volume_candidate_retention.csv"
    )

    json_path = (
        output_dir
        / "whole_volume_candidate_retention_statistics.json"
    )

    save_csv(
        records,
        csv_path,
    )

    save_json(
        summary,
        json_path,
    )

    print_summary(summary)

    print()
    print(f"CSV saved  : {csv_path}")
    print(f"JSON saved : {json_path}")
    print()


if __name__ == "__main__":
    main()
