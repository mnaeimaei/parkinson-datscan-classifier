#!/usr/bin/env python3

"""
STEP 9D — VALIDATE FIXED WHOLE-VOLUME CLASSIFICATION DATASET

Performs:

1. Full numerical validation of all Step 9C outputs.
2. Exact crop/padding correspondence checks against normalized source.
3. Affine, spacing, orientation and datatype validation.
4. Targeted visual QC selection using Step 9B retention statistics.
5. Source-vs-output maximum-intensity-projection QC images.

This step is READ-ONLY with respect to the NIfTI datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np


EXPECTED_SHAPE = (192, 192, 160)
EXPECTED_SPACING = (2.46, 2.46, 2.46)
EXPECTED_ORIENTATION = ("R", "A", "S")

SPACING_TOLERANCE = 1e-3
AFFINE_TOLERANCE_MM = 1e-4
VOXEL_TOLERANCE = 1e-6
RETENTION_COLUMN = "retention_192x192x160"


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------


def uid_from_path(path: Path) -> str:
    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return path.stem


def find_nifti_files(directory: Path) -> list[Path]:
    files = list(directory.glob("*.nii"))
    files.extend(directory.glob("*.nii.gz"))

    return sorted(set(files))


def read_csv_by_uid(
    path: Path,
) -> dict[str, dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        rows = list(
            csv.DictReader(handle)
        )

    result = {}

    for row in rows:
        uid = row["uid"]

        if uid in result:
            raise RuntimeError(
                f"Duplicate UID in {path}: {uid}"
            )

        result[uid] = row

    return result


def write_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def write_json(
    data: dict[str, Any],
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
            data,
            handle,
            indent=2,
        )


# ---------------------------------------------------------------------
# Crop / pad reconstruction
# ---------------------------------------------------------------------


def axis_plan(
    source_size: int,
    target_size: int,
) -> dict[str, int]:

    if source_size >= target_size:

        crop_before = (
            source_size - target_size
        ) // 2

        crop_after = (
            source_size
            - target_size
            - crop_before
        )

        return {
            "crop_before": crop_before,
            "crop_after": crop_after,

            "pad_before": 0,
            "pad_after": 0,

            "source_start": crop_before,
            "source_stop":
                source_size - crop_after,

            "output_start": 0,
            "output_stop": target_size,

            "offset_voxels":
                crop_before,
        }

    pad_before = (
        target_size - source_size
    ) // 2

    pad_after = (
        target_size
        - source_size
        - pad_before
    )

    return {
        "crop_before": 0,
        "crop_after": 0,

        "pad_before": pad_before,
        "pad_after": pad_after,

        "source_start": 0,
        "source_stop": source_size,

        "output_start": pad_before,
        "output_stop":
            pad_before + source_size,

        "offset_voxels":
            -pad_before,
    }


def get_plans(
    source_shape: tuple[int, int, int],
) -> list[dict[str, int]]:

    return [
        axis_plan(source, target)
        for source, target in zip(
            source_shape,
            EXPECTED_SHAPE,
        )
    ]


def expected_affine(
    source_affine: np.ndarray,
    plans: list[dict[str, int]],
) -> np.ndarray:

    translation = np.eye(
        4,
        dtype=np.float64,
    )

    translation[:3, 3] = [
        plan["offset_voxels"]
        for plan in plans
    ]

    return (
        source_affine
        @ translation
    )


# ---------------------------------------------------------------------
# Numerical validation
# ---------------------------------------------------------------------


def maximum_padding_value(
    output: np.ndarray,
    plans: list[dict[str, int]],
) -> float:

    maxima: list[float] = []

    for axis, plan in enumerate(plans):

        if plan["pad_before"] > 0:

            selection = [
                slice(None),
                slice(None),
                slice(None),
            ]

            selection[axis] = slice(
                0,
                plan["pad_before"],
            )

            slab = output[
                tuple(selection)
            ]

            if slab.size:
                maxima.append(
                    float(
                        np.max(
                            np.abs(slab)
                        )
                    )
                )

        if plan["pad_after"] > 0:

            selection = [
                slice(None),
                slice(None),
                slice(None),
            ]

            start = (
                EXPECTED_SHAPE[axis]
                - plan["pad_after"]
            )

            selection[axis] = slice(
                start,
                EXPECTED_SHAPE[axis],
            )

            slab = output[
                tuple(selection)
            ]

            if slab.size:
                maxima.append(
                    float(
                        np.max(
                            np.abs(slab)
                        )
                    )
                )

    if not maxima:
        return 0.0

    return max(maxima)


def validate_scan(
    source_path: Path,
    output_path: Path,
) -> dict[str, Any]:

    uid = uid_from_path(
        source_path
    )

    source_image = nib.load(
        str(source_path),
        mmap="r",
    )

    output_image = nib.load(
        str(output_path),
        mmap="r",
    )

    source_shape = tuple(
        int(v)
        for v in source_image.shape
    )

    output_shape = tuple(
        int(v)
        for v in output_image.shape
    )

    plans = get_plans(
        source_shape
    )

    output_spacing = tuple(
        float(v)
        for v in nib.affines.voxel_sizes(
            output_image.affine
        )
    )

    output_orientation = tuple(
        nib.aff2axcodes(
            output_image.affine
        )
    )

    output_dtype = str(
        output_image.header.get_data_dtype()
    )

    source_data = source_image.get_fdata(
        dtype=np.float32
    )

    output_data = output_image.get_fdata(
        dtype=np.float32
    )

    finite_valid = bool(
        np.isfinite(
            output_data
        ).all()
    )

    source_slices = tuple(
        slice(
            plan["source_start"],
            plan["source_stop"],
        )
        for plan in plans
    )

    output_slices = tuple(
        slice(
            plan["output_start"],
            plan["output_stop"],
        )
        for plan in plans
    )

    source_retained = source_data[
        source_slices
    ]

    output_copied = output_data[
        output_slices
    ]

    if (
        source_retained.shape
        != output_copied.shape
    ):
        raise RuntimeError(
            f"{uid}: copied region "
            "shape mismatch."
        )

    voxel_difference = np.abs(
        source_retained
        - output_copied
    )

    if voxel_difference.size:

        max_voxel_difference = float(
            np.max(
                voxel_difference
            )
        )

        mean_voxel_difference = float(
            np.mean(
                voxel_difference
            )
        )

    else:

        max_voxel_difference = float(
            "inf"
        )

        mean_voxel_difference = float(
            "inf"
        )

    max_padding_abs = (
        maximum_padding_value(
            output_data,
            plans,
        )
    )

    predicted_affine = expected_affine(
        source_image.affine,
        plans,
    )

    affine_difference = np.abs(
        output_image.affine
        - predicted_affine
    )

    affine_max_error = float(
        np.max(
            affine_difference
        )
    )

    shape_valid = (
        output_shape
        == EXPECTED_SHAPE
    )

    spacing_valid = all(
        abs(actual - expected)
        <= SPACING_TOLERANCE
        for actual, expected in zip(
            output_spacing,
            EXPECTED_SPACING,
        )
    )

    orientation_valid = (
        output_orientation
        == EXPECTED_ORIENTATION
    )

    dtype_valid = (
        output_image.header.get_data_dtype()
        == np.dtype(np.float32)
    )

    copied_values_valid = (
        max_voxel_difference
        <= VOXEL_TOLERANCE
    )

    padding_valid = (
        max_padding_abs
        <= VOXEL_TOLERANCE
    )

    affine_valid = (
        affine_max_error
        <= AFFINE_TOLERANCE_MM
    )

    validation_passed = all(
        [
            shape_valid,
            spacing_valid,
            orientation_valid,
            dtype_valid,
            finite_valid,
            copied_values_valid,
            padding_valid,
            affine_valid,
        ]
    )

    crop_total = sum(
        plan["crop_before"]
        + plan["crop_after"]
        for plan in plans
    )

    padding_total = sum(
        plan["pad_before"]
        + plan["pad_after"]
        for plan in plans
    )

    return {
        "uid": uid,

        "source_shape":
            "x".join(
                str(v)
                for v in source_shape
            ),

        "output_shape":
            "x".join(
                str(v)
                for v in output_shape
            ),

        "crop_total_voxels":
            crop_total,

        "padding_total_voxels":
            padding_total,

        "shape_valid":
            shape_valid,

        "spacing_valid":
            spacing_valid,

        "orientation_valid":
            orientation_valid,

        "dtype":
            output_dtype,

        "dtype_valid":
            dtype_valid,

        "finite_valid":
            finite_valid,

        "max_copied_voxel_difference":
            max_voxel_difference,

        "mean_copied_voxel_difference":
            mean_voxel_difference,

        "copied_values_valid":
            copied_values_valid,

        "max_padding_absolute_value":
            max_padding_abs,

        "padding_valid":
            padding_valid,

        "affine_max_absolute_error":
            affine_max_error,

        "affine_valid":
            affine_valid,

        "validation_passed":
            validation_passed,
    }


# ---------------------------------------------------------------------
# QC selection
# ---------------------------------------------------------------------


def crop_total(
    row: dict[str, str],
) -> int:

    return sum(
        int(row[f"crop_{side}_{axis}"])
        for axis in ("x", "y", "z")
        for side in ("before", "after")
    )


def padding_total(
    row: dict[str, str],
) -> int:

    return sum(
        int(row[f"pad_{side}_{axis}"])
        for axis in ("x", "y", "z")
        for side in ("before", "after")
    )


def build_qc_selection(
    retention_rows:
        dict[str, dict[str, str]],

    manifest_rows:
        dict[str, dict[str, str]],

    max_cases: int,
) -> list[dict[str, Any]]:

    selection: dict[
        str,
        set[str],
    ] = {}

    retention_values = []

    for uid, row in retention_rows.items():

        value = float(
            row[RETENTION_COLUMN]
        )

        retention_values.append(
            (uid, value)
        )

    retention_values.sort(
        key=lambda item: item[1]
    )

    def add(
        uid: str,
        reason: str,
    ) -> None:

        selection.setdefault(
            uid,
            set(),
        ).add(reason)

    #
    # Every case below the 99% threshold.
    #
    for uid, retention in retention_values:

        if retention < 0.99:
            add(
                uid,
                "retention_below_99_percent",
            )

    #
    # 12 lowest overall.
    #
    for uid, _ in retention_values[:12]:

        add(
            uid,
            "lowest_retention",
        )

    #
    # Largest amount of cropping.
    #
    crop_ranked = sorted(
        manifest_rows.items(),
        key=lambda item:
            crop_total(item[1]),
        reverse=True,
    )

    added = 0

    for uid, row in crop_ranked:

        if crop_total(row) <= 0:
            continue

        add(
            uid,
            "largest_total_crop",
        )

        added += 1

        if added >= 5:
            break

    #
    # Largest amount of padding.
    #
    padding_ranked = sorted(
        manifest_rows.items(),
        key=lambda item:
            padding_total(item[1]),
        reverse=True,
    )

    added = 0

    for uid, row in padding_ranked:

        if padding_total(row) <= 0:
            continue

        add(
            uid,
            "largest_total_padding",
        )

        added += 1

        if added >= 5:
            break

    #
    # Crop + pad on different axes.
    #
    crop_pad_cases = []

    for uid, row in manifest_rows.items():

        c = crop_total(row)
        p = padding_total(row)

        if c > 0 and p > 0:

            retention = float(
                retention_rows[
                    uid
                ][RETENTION_COLUMN]
            )

            crop_pad_cases.append(
                (
                    uid,
                    retention,
                )
            )

    crop_pad_cases.sort(
        key=lambda item: item[1]
    )

    for uid, _ in crop_pad_cases[:5]:

        add(
            uid,
            "crop_and_padding",
        )

    #
    # Keep most important cases first.
    #
    ordered_uids = sorted(
        selection,
        key=lambda uid: (
            float(
                retention_rows[
                    uid
                ][RETENTION_COLUMN]
            ),
            uid,
        ),
    )

    ordered_uids = ordered_uids[
        :max_cases
    ]

    rows: list[
        dict[str, Any]
    ] = []

    for uid in ordered_uids:

        manifest = manifest_rows[
            uid
        ]

        retention = float(
            retention_rows[
                uid
            ][RETENTION_COLUMN]
        )

        rows.append(
            {
                "uid": uid,

                "retention_fraction":
                    retention,

                "retention_percent":
                    round(
                        retention * 100.0,
                        4,
                    ),

                "source_shape":
                    (
                        f"{manifest['source_shape_x']}"
                        "x"
                        f"{manifest['source_shape_y']}"
                        "x"
                        f"{manifest['source_shape_z']}"
                    ),

                "crop_total_voxels":
                    crop_total(
                        manifest
                    ),

                "padding_total_voxels":
                    padding_total(
                        manifest
                    ),

                "selection_reason":
                    ";".join(
                        sorted(
                            selection[uid]
                        )
                    ),
            }
        )

    return rows


# ---------------------------------------------------------------------
# Visual QC
# ---------------------------------------------------------------------


def mip_views(
    data: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    axial = np.max(
        data,
        axis=2,
    ).T

    coronal = np.max(
        data,
        axis=1,
    ).T

    sagittal = np.max(
        data,
        axis=0,
    ).T

    return (
        axial,
        coronal,
        sagittal,
    )


def display_limits(
    source: np.ndarray,
    output: np.ndarray,
) -> tuple[float, float]:

    combined = np.concatenate(
        [
            source.ravel(),
            output.ravel(),
        ]
    )

    combined = combined[
        np.isfinite(combined)
    ]

    if combined.size == 0:
        return 0.0, 1.0

    vmax = float(
        np.percentile(
            combined,
            99.8,
        )
    )

    if vmax <= 0:
        vmax = float(
            np.max(combined)
        )

    if vmax <= 0:
        vmax = 1.0

    return 0.0, vmax


def add_crop_rectangle(
    axis,
    plane: str,
    plans: list[dict[str, int]],
) -> None:

    x = plans[0]
    y = plans[1]
    z = plans[2]

    if plane == "axial":

        horizontal_start = x[
            "source_start"
        ]

        vertical_start = y[
            "source_start"
        ]

        width = (
            x["source_stop"]
            - x["source_start"]
        )

        height = (
            y["source_stop"]
            - y["source_start"]
        )

    elif plane == "coronal":

        horizontal_start = x[
            "source_start"
        ]

        vertical_start = z[
            "source_start"
        ]

        width = (
            x["source_stop"]
            - x["source_start"]
        )

        height = (
            z["source_stop"]
            - z["source_start"]
        )

    else:

        horizontal_start = y[
            "source_start"
        ]

        vertical_start = z[
            "source_start"
        ]

        width = (
            y["source_stop"]
            - y["source_start"]
        )

        height = (
            z["source_stop"]
            - z["source_start"]
        )

    rectangle = Rectangle(
        (
            horizontal_start,
            vertical_start,
        ),
        width,
        height,
        fill=False,
        edgecolor="red",
        linewidth=1.5,
    )

    axis.add_patch(
        rectangle
    )


def create_qc_image(
    uid: str,
    source_path: Path,
    output_path: Path,
    qc_path: Path,
    retention: float,
    reason: str,
) -> None:

    source_image = nib.load(
        str(source_path),
        mmap="r",
    )

    output_image = nib.load(
        str(output_path),
        mmap="r",
    )

    source = source_image.get_fdata(
        dtype=np.float32
    )

    output = output_image.get_fdata(
        dtype=np.float32
    )

    plans = get_plans(
        tuple(
            int(v)
            for v in source.shape
        )
    )

    source_views = mip_views(
        source
    )

    output_views = mip_views(
        output
    )

    vmin, vmax = display_limits(
        source,
        output,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13, 8),
    )

    plane_names = [
        "Axial MIP",
        "Coronal MIP",
        "Sagittal MIP",
    ]

    plane_keys = [
        "axial",
        "coronal",
        "sagittal",
    ]

    for column in range(3):

        axes[0, column].imshow(
            source_views[column],
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

        add_crop_rectangle(
            axes[0, column],
            plane_keys[column],
            plans,
        )

        axes[0, column].set_title(
            "Source — "
            + plane_names[column]
        )

        axes[1, column].imshow(
            output_views[column],
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )

        axes[1, column].set_title(
            "Fixed — "
            + plane_names[column]
        )

        axes[0, column].axis(
            "off"
        )

        axes[1, column].axis(
            "off"
        )

    figure.suptitle(
        (
            f"{uid} | "
            f"source={source.shape} | "
            "target=192x192x160 | "
            f"retention={retention * 100:.3f}%\n"
            f"{reason}\n"
            "Red box = source region retained by Step 9C"
        ),
        fontsize=11,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.91)
    )

    qc_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        qc_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fixed-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--retention-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--qc-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-qc-cases",
        type=int,
        default=25,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    source_dir = (
        args.source_dir.resolve()
    )

    fixed_dir = (
        args.fixed_dir.resolve()
    )

    report_dir = (
        args.report_dir.resolve()
    )

    qc_dir = (
        args.qc_dir.resolve()
    )

    source_files = find_nifti_files(
        source_dir
    )

    fixed_files = find_nifti_files(
        fixed_dir
    )

    source_by_uid = {
        uid_from_path(path): path
        for path in source_files
    }

    fixed_by_uid = {
        uid_from_path(path): path
        for path in fixed_files
    }

    expected_uids = set(
        source_by_uid
    )

    actual_uids = set(
        fixed_by_uid
    )

    missing_outputs = sorted(
        expected_uids
        - actual_uids
    )

    unexpected_outputs = sorted(
        actual_uids
        - expected_uids
    )

    retention_rows = read_csv_by_uid(
        args.retention_csv.resolve()
    )

    manifest_rows = read_csv_by_uid(
        args.manifest_csv.resolve()
    )

    if RETENTION_COLUMN not in next(
        iter(
            retention_rows.values()
        )
    ):
        raise RuntimeError(
            f"Missing column "
            f"{RETENTION_COLUMN}"
        )

    print()
    print("=" * 78)
    print(
        "STEP 9D — VALIDATE FIXED WHOLE-VOLUME DATASET"
    )
    print("=" * 78)

    print(
        f"Source scans : {len(source_files)}"
    )

    print(
        f"Fixed scans  : {len(fixed_files)}"
    )

    print(
        f"Target shape : {EXPECTED_SHAPE}"
    )

    print("=" * 78)
    print()

    validation_rows: list[
        dict[str, Any]
    ] = []

    failed_validation_uids: list[
        str
    ] = []

    for index, uid in enumerate(
        sorted(expected_uids),
        start=1,
    ):

        if uid not in fixed_by_uid:
            continue

        row = validate_scan(
            source_by_uid[uid],
            fixed_by_uid[uid],
        )

        validation_rows.append(
            row
        )

        if not row[
            "validation_passed"
        ]:
            failed_validation_uids.append(
                uid
            )

        if (
            index == 1
            or index % 100 == 0
            or index == len(expected_uids)
        ):

            print(
                f"[{index:4d}/{len(expected_uids)}] "
                f"{uid:18s} "
                f"pass={row['validation_passed']} "
                f"voxel_diff="
                f"{row['max_copied_voxel_difference']:.3e} "
                f"pad_max="
                f"{row['max_padding_absolute_value']:.3e} "
                f"affine_err="
                f"{row['affine_max_absolute_error']:.3e}"
            )

    validation_csv = (
        report_dir
        / "fixed_whole_volume_validation.csv"
    )

    write_csv(
        validation_rows,
        validation_csv,
    )

    #
    # Visual QC selection
    #
    qc_selection = build_qc_selection(
        retention_rows=retention_rows,
        manifest_rows=manifest_rows,
        max_cases=args.max_qc_cases,
    )

    qc_selection_csv = (
        report_dir
        / "fixed_whole_volume_qc_selection.csv"
    )

    write_csv(
        qc_selection,
        qc_selection_csv,
    )

    print()
    print(
        f"Generating {len(qc_selection)} "
        "targeted QC images..."
    )

    qc_failures = []

    for index, row in enumerate(
        qc_selection,
        start=1,
    ):

        uid = row["uid"]

        try:

            qc_path = (
                qc_dir
                / f"{uid}_whole_volume_qc.png"
            )

            create_qc_image(
                uid=uid,
                source_path=
                    source_by_uid[uid],
                output_path=
                    fixed_by_uid[uid],
                qc_path=qc_path,
                retention=float(
                    row[
                        "retention_fraction"
                    ]
                ),
                reason=row[
                    "selection_reason"
                ],
            )

            print(
                f"[QC {index:2d}/"
                f"{len(qc_selection)}] "
                f"{uid} "
                f"retention="
                f"{row['retention_percent']:.3f}%"
            )

        except Exception as exc:

            qc_failures.append(
                {
                    "uid": uid,
                    "reason": str(exc),
                }
            )

    max_voxel_difference = max(
        (
            float(
                row[
                    "max_copied_voxel_difference"
                ]
            )
            for row in validation_rows
        ),
        default=float("inf"),
    )

    max_padding_value = max(
        (
            float(
                row[
                    "max_padding_absolute_value"
                ]
            )
            for row in validation_rows
        ),
        default=float("inf"),
    )

    max_affine_error = max(
        (
            float(
                row[
                    "affine_max_absolute_error"
                ]
            )
            for row in validation_rows
        ),
        default=float("inf"),
    )

    below_99 = sum(
        float(
            row[
                RETENTION_COLUMN
            ]
        ) < 0.99
        for row in retention_rows.values()
    )

    below_995 = sum(
        float(
            row[
                RETENTION_COLUMN
            ]
        ) < 0.995
        for row in retention_rows.values()
    )

    numerical_validation_passed = (
        len(source_files) > 0
        and len(fixed_files)
            == len(source_files)
        and not missing_outputs
        and not unexpected_outputs
        and len(validation_rows)
            == len(source_files)
        and not failed_validation_uids
        and max_voxel_difference
            <= VOXEL_TOLERANCE
        and max_padding_value
            <= VOXEL_TOLERANCE
        and max_affine_error
            <= AFFINE_TOLERANCE_MM
    )

    summary = {
        "step": "9D",

        "description":
            "Numerical and visual QC of "
            "fixed whole-volume inputs",

        "target_shape":
            list(EXPECTED_SHAPE),

        "expected_spacing_mm":
            list(EXPECTED_SPACING),

        "number_of_source_scans":
            len(source_files),

        "number_of_fixed_scans":
            len(fixed_files),

        "number_numerically_validated":
            len(validation_rows),

        "missing_outputs":
            missing_outputs,

        "unexpected_outputs":
            unexpected_outputs,

        "failed_validation_uids":
            failed_validation_uids,

        "maximum_copied_voxel_difference":
            max_voxel_difference,

        "maximum_padding_absolute_value":
            max_padding_value,

        "maximum_affine_absolute_error":
            max_affine_error,

        "retention": {
            "scans_below_99_percent":
                below_99,

            "scans_below_99_5_percent":
                below_995,
        },

        "visual_qc": {
            "number_selected":
                len(qc_selection),

            "qc_directory":
                str(qc_dir),

            "generation_failures":
                qc_failures,

            "manual_review_required":
                True,
        },

        "numerical_validation_passed":
            numerical_validation_passed,
    }

    summary_json = (
        report_dir
        / "fixed_whole_volume_qc_summary.json"
    )

    write_json(
        summary,
        summary_json,
    )

    print()
    print("=" * 78)
    print("STEP 9D SUMMARY")
    print("=" * 78)

    print(
        f"Source scans                 : "
        f"{len(source_files)}"
    )

    print(
        f"Fixed scans                  : "
        f"{len(fixed_files)}"
    )

    print(
        f"Numerically validated        : "
        f"{len(validation_rows)}"
    )

    print(
        f"Failed numerical validation  : "
        f"{len(failed_validation_uids)}"
    )

    print(
        f"Missing outputs              : "
        f"{len(missing_outputs)}"
    )

    print(
        f"Unexpected outputs           : "
        f"{len(unexpected_outputs)}"
    )

    print(
        f"Maximum copied voxel diff    : "
        f"{max_voxel_difference:.3e}"
    )

    print(
        f"Maximum padding value        : "
        f"{max_padding_value:.3e}"
    )

    print(
        f"Maximum affine error         : "
        f"{max_affine_error:.3e}"
    )

    print(
        f"Retention <99%               : "
        f"{below_99}"
    )

    print(
        f"Retention <99.5%             : "
        f"{below_995}"
    )

    print(
        f"Visual QC cases              : "
        f"{len(qc_selection)}"
    )

    print(
        f"QC image failures            : "
        f"{len(qc_failures)}"
    )

    print(
        f"Numerical validation passed  : "
        f"{numerical_validation_passed}"
    )

    print("=" * 78)

    print()
    print(
        f"Validation CSV : "
        f"{validation_csv}"
    )

    print(
        f"QC selection   : "
        f"{qc_selection_csv}"
    )

    print(
        f"Summary JSON   : "
        f"{summary_json}"
    )

    print(
        f"QC images      : "
        f"{qc_dir}"
    )

    print()

    if not numerical_validation_passed:

        raise RuntimeError(
            "Step 9D numerical validation failed."
        )

    if qc_failures:

        raise RuntimeError(
            "Some Step 9D QC images "
            "could not be generated."
        )


if __name__ == "__main__":
    main()
