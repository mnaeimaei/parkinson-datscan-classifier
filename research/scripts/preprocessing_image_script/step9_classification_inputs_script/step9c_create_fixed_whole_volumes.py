#!/usr/bin/env python3

"""
STEP 9C — CREATE FIXED WHOLE-VOLUME CLASSIFICATION DATASET

Converts all normalized whole-volume DaT scans to one fixed spatial shape
using deterministic center cropping and symmetric zero padding.

Frozen target:
    Shape   : 192 x 192 x 160
    Spacing : 2.46 x 2.46 x 2.46 mm

Important:
    - NO resampling
    - NO interpolation
    - NO new intensity normalization
    - NO augmentation
    - intensities are preserved
    - NIfTI affine is updated after crop/padding
    - RAS orientation is preserved
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


EXPECTED_SPACING = (2.46, 2.46, 2.46)
SPACING_TOLERANCE = 1e-3
EXPECTED_ORIENTATION = ("R", "A", "S")


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


def axis_plan(
    source_size: int,
    target_size: int,
) -> dict[str, int]:
    """
    Determine deterministic center crop/pad parameters.

    offset_voxels defines:

        source_index = output_index + offset_voxels

    and is used for affine correction.
    """

    if source_size >= target_size:
        crop_before = (source_size - target_size) // 2
        crop_after = source_size - target_size - crop_before

        return {
            "crop_before": crop_before,
            "crop_after": crop_after,
            "pad_before": 0,
            "pad_after": 0,
            "source_start": crop_before,
            "source_stop": source_size - crop_after,
            "output_start": 0,
            "output_stop": target_size,
            "offset_voxels": crop_before,
        }

    pad_before = (target_size - source_size) // 2
    pad_after = target_size - source_size - pad_before

    return {
        "crop_before": 0,
        "crop_after": 0,
        "pad_before": pad_before,
        "pad_after": pad_after,
        "source_start": 0,
        "source_stop": source_size,
        "output_start": pad_before,
        "output_stop": pad_before + source_size,
        "offset_voxels": -pad_before,
    }


def crop_pad_volume(
    data: np.ndarray,
    target_shape: tuple[int, int, int],
) -> tuple[
    np.ndarray,
    list[dict[str, int]],
]:
    plans = [
        axis_plan(source, target)
        for source, target in zip(
            data.shape,
            target_shape,
        )
    ]

    output = np.zeros(
        target_shape,
        dtype=np.float32,
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

    output[output_slices] = data[source_slices]

    return output, plans


def corrected_affine(
    old_affine: np.ndarray,
    plans: list[dict[str, int]],
) -> np.ndarray:
    """
    Correct affine after crop/padding.

    For output voxel coordinate v_out:

        v_source = v_out + offset

    Therefore:

        world = old_affine @ translation(offset) @ v_out
    """

    translation = np.eye(
        4,
        dtype=np.float64,
    )

    translation[:3, 3] = [
        plan["offset_voxels"]
        for plan in plans
    ]

    return old_affine @ translation


def validate_affine_correspondence(
    old_affine: np.ndarray,
    new_affine: np.ndarray,
    plans: list[dict[str, int]],
) -> float:
    """
    Verify that the first copied source voxel and corresponding output voxel
    map to the same physical world coordinate.
    """

    source_voxel = np.array(
        [
            plans[0]["source_start"],
            plans[1]["source_start"],
            plans[2]["source_start"],
            1.0,
        ],
        dtype=np.float64,
    )

    output_voxel = np.array(
        [
            plans[0]["output_start"],
            plans[1]["output_start"],
            plans[2]["output_start"],
            1.0,
        ],
        dtype=np.float64,
    )

    old_world = old_affine @ source_voxel
    new_world = new_affine @ output_voxel

    return float(
        np.linalg.norm(
            old_world[:3] - new_world[:3]
        )
    )


def spacing_is_valid(
    spacing: tuple[float, float, float],
) -> bool:
    return all(
        abs(actual - expected)
        <= SPACING_TOLERANCE
        for actual, expected in zip(
            spacing,
            EXPECTED_SPACING,
        )
    )


def save_nifti_atomic(
    image: nib.Nifti1Image,
    output_path: Path,
) -> None:
    temp_path = output_path.with_name(
        f"{output_path.stem}.tmp.nii.gz"
    )

    try:
        nib.save(
            image,
            str(temp_path),
        )

        os.replace(
            temp_path,
            output_path,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


def process_scan(
    input_path: Path,
    output_path: Path,
    target_shape: tuple[int, int, int],
) -> dict[str, Any]:

    uid = uid_from_path(input_path)

    image = nib.load(
        str(input_path),
        mmap="r",
    )

    if len(image.shape) != 3:
        raise ValueError(
            f"Expected 3D NIfTI, found {image.shape}"
        )

    source_shape = tuple(
        int(v)
        for v in image.shape
    )

    source_spacing = tuple(
        float(v)
        for v in nib.affines.voxel_sizes(
            image.affine
        )
    )

    source_orientation = tuple(
        nib.aff2axcodes(
            image.affine
        )
    )

    if not spacing_is_valid(
        source_spacing
    ):
        raise ValueError(
            f"Unexpected spacing: {source_spacing}"
        )

    if source_orientation != EXPECTED_ORIENTATION:
        raise ValueError(
            f"Expected RAS orientation, "
            f"found {source_orientation}"
        )

    data = image.get_fdata(
        dtype=np.float32
    )

    if not np.isfinite(data).all():
        raise ValueError(
            "Input contains NaN or Inf."
        )

    source_min = float(
        np.min(data)
    )

    source_max = float(
        np.max(data)
    )

    source_mean = float(
        np.mean(data)
    )

    output_data, plans = crop_pad_volume(
        data=data,
        target_shape=target_shape,
    )

    new_affine = corrected_affine(
        image.affine,
        plans,
    )

    affine_error = validate_affine_correspondence(
        old_affine=image.affine,
        new_affine=new_affine,
        plans=plans,
    )

    if affine_error > 1e-5:
        raise ValueError(
            f"Affine correspondence error: "
            f"{affine_error}"
        )

    header = image.header.copy()

    header.set_data_dtype(
        np.float32
    )

    # We already materialized the real intensity values.
    # Prevent old NIfTI scaling from being re-applied.
    header.set_slope_inter(
        1.0,
        0.0,
    )

    output_image = nib.Nifti1Image(
        output_data,
        new_affine,
        header=header,
    )

    qform_code = int(
        image.header["qform_code"]
    )

    sform_code = int(
        image.header["sform_code"]
    )

    output_image.set_qform(
        new_affine,
        code=qform_code,
    )

    output_image.set_sform(
        new_affine,
        code=sform_code,
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

    if tuple(output_data.shape) != target_shape:
        raise ValueError(
            f"Incorrect output shape: "
            f"{output_data.shape}"
        )

    if not spacing_is_valid(
        output_spacing
    ):
        raise ValueError(
            f"Output spacing changed: "
            f"{output_spacing}"
        )

    if output_orientation != EXPECTED_ORIENTATION:
        raise ValueError(
            f"Output orientation changed: "
            f"{output_orientation}"
        )

    if not np.isfinite(output_data).all():
        raise ValueError(
            "Output contains NaN or Inf."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_nifti_atomic(
        output_image,
        output_path,
    )

    axis_names = ("x", "y", "z")

    record: dict[str, Any] = {
        "uid": uid,
        "input_file": str(input_path),
        "output_file": str(output_path),

        "source_shape_x": source_shape[0],
        "source_shape_y": source_shape[1],
        "source_shape_z": source_shape[2],

        "target_shape_x": target_shape[0],
        "target_shape_y": target_shape[1],
        "target_shape_z": target_shape[2],

        "source_min": round(
            source_min,
            8,
        ),
        "source_max": round(
            source_max,
            8,
        ),
        "source_mean": round(
            source_mean,
            8,
        ),

        "affine_correspondence_error_mm":
            round(
                affine_error,
                10,
            ),

        "input_orientation":
            "".join(source_orientation),

        "output_orientation":
            "".join(output_orientation),

        "spacing_x_mm":
            round(
                output_spacing[0],
                6,
            ),

        "spacing_y_mm":
            round(
                output_spacing[1],
                6,
            ),

        "spacing_z_mm":
            round(
                output_spacing[2],
                6,
            ),
    }

    for axis, plan in zip(
        axis_names,
        plans,
    ):
        record[
            f"crop_before_{axis}"
        ] = plan["crop_before"]

        record[
            f"crop_after_{axis}"
        ] = plan["crop_after"]

        record[
            f"pad_before_{axis}"
        ] = plan["pad_before"]

        record[
            f"pad_after_{axis}"
        ] = plan["pad_after"]

    return record


def save_csv(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not records:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                records[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(records)


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            data,
            handle,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 9C: create fixed whole-volume "
            "classification images."
        )
    )

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
        "--report-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=(192, 192, 160),
        metavar=("X", "Y", "Z"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()

    target_shape = tuple(
        int(v)
        for v in args.target_shape
    )

    files = find_nifti_files(
        input_dir
    )

    if not files:
        raise RuntimeError(
            f"No NIfTI files found in "
            f"{input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 76)
    print(
        "STEP 9C — CREATE FIXED WHOLE-VOLUME DATASET"
    )
    print("=" * 76)
    print(
        f"Input scans     : {len(files)}"
    )
    print(
        f"Target shape    : {target_shape}"
    )
    print(
        f"Target spacing  : {EXPECTED_SPACING}"
    )
    print(
        f"Output directory: {output_dir}"
    )
    print("=" * 76)
    print()

    records: list[
        dict[str, Any]
    ] = []

    failures: list[
        dict[str, str]
    ] = []

    cropped_any = 0
    padded_any = 0
    crop_and_pad = 0

    for index, input_path in enumerate(
        files,
        start=1,
    ):
        uid = uid_from_path(
            input_path
        )

        output_path = (
            output_dir
            / f"{uid}.nii.gz"
        )

        try:
            if (
                output_path.exists()
                and not args.overwrite
            ):
                raise FileExistsError(
                    f"Output already exists: "
                    f"{output_path}"
                )

            record = process_scan(
                input_path=input_path,
                output_path=output_path,
                target_shape=target_shape,
            )

            records.append(
                record
            )

            has_crop = any(
                record[
                    f"crop_before_{axis}"
                ] > 0
                or record[
                    f"crop_after_{axis}"
                ] > 0
                for axis in (
                    "x",
                    "y",
                    "z",
                )
            )

            has_pad = any(
                record[
                    f"pad_before_{axis}"
                ] > 0
                or record[
                    f"pad_after_{axis}"
                ] > 0
                for axis in (
                    "x",
                    "y",
                    "z",
                )
            )

            if has_crop:
                cropped_any += 1

            if has_pad:
                padded_any += 1

            if has_crop and has_pad:
                crop_and_pad += 1

            if (
                index == 1
                or index % 100 == 0
                or index == len(files)
            ):
                print(
                    f"[{index:4d}/{len(files)}] "
                    f"{uid:18s} "
                    f"{record['source_shape_x']}x"
                    f"{record['source_shape_y']}x"
                    f"{record['source_shape_z']}"
                    " -> "
                    f"{target_shape[0]}x"
                    f"{target_shape[1]}x"
                    f"{target_shape[2]}"
                )

        except Exception as exc:
            failures.append(
                {
                    "uid": uid,
                    "input_file":
                        str(input_path),
                    "reason":
                        str(exc),
                }
            )

            print(
                f"FAILED: {uid}: {exc}"
            )

    output_files = find_nifti_files(
        output_dir
    )

    successful_uids = {
        record["uid"]
        for record in records
    }

    expected_uids = {
        uid_from_path(path)
        for path in files
    }

    output_uids = {
        uid_from_path(path)
        for path in output_files
    }

    missing_outputs = sorted(
        expected_uids
        - output_uids
    )

    unexpected_outputs = sorted(
        output_uids
        - expected_uids
    )

    validation_passed = (
        len(files) == len(records)
        and len(failures) == 0
        and len(output_files) == len(files)
        and successful_uids == expected_uids
        and not missing_outputs
        and not unexpected_outputs
    )

    summary = {
        "step": "9C",

        "description":
            "Create fixed whole-volume "
            "classification dataset",

        "input_directory":
            str(input_dir),

        "output_directory":
            str(output_dir),

        "target_shape":
            list(target_shape),

        "expected_spacing_mm":
            list(EXPECTED_SPACING),

        "method": {
            "crop":
                "deterministic center crop",

            "padding":
                "symmetric zero padding",

            "interpolation":
                "none",

            "resampling":
                "none",

            "intensity_normalization":
                "none; preserve Step 6D "
                "normalized values",

            "output_dtype":
                "float32",

            "affine":
                "updated to preserve "
                "physical coordinates",

            "orientation":
                "RAS",
        },

        "number_of_input_scans":
            len(files),

        "number_of_successful_scans":
            len(records),

        "number_of_failed_scans":
            len(failures),

        "number_of_output_files":
            len(output_files),

        "scans_requiring_crop":
            cropped_any,

        "scans_requiring_padding":
            padded_any,

        "scans_requiring_both_crop_and_padding":
            crop_and_pad,

        "missing_outputs":
            missing_outputs,

        "unexpected_outputs":
            unexpected_outputs,

        "failures":
            failures,

        "validation_passed":
            validation_passed,
    }

    csv_path = (
        report_dir
        / "fixed_whole_volume_manifest.csv"
    )

    json_path = (
        report_dir
        / "fixed_whole_volume_creation_summary.json"
    )

    save_csv(
        records,
        csv_path,
    )

    save_json(
        summary,
        json_path,
    )

    print()
    print("=" * 76)
    print("STEP 9C SUMMARY")
    print("=" * 76)

    print(
        f"Input scans                  : "
        f"{len(files)}"
    )

    print(
        f"Successful                   : "
        f"{len(records)}"
    )

    print(
        f"Failed                       : "
        f"{len(failures)}"
    )

    print(
        f"Output files                 : "
        f"{len(output_files)}"
    )

    print(
        f"Scans requiring crop         : "
        f"{cropped_any}"
    )

    print(
        f"Scans requiring padding      : "
        f"{padded_any}"
    )

    print(
        f"Scans requiring crop + pad   : "
        f"{crop_and_pad}"
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
        f"Validation passed            : "
        f"{validation_passed}"
    )

    print("=" * 76)

    print()
    print(
        f"Manifest saved: {csv_path}"
    )

    print(
        f"Summary saved : {json_path}"
    )

    print()

    if not validation_passed:
        raise RuntimeError(
            "Step 9C validation failed."
        )


if __name__ == "__main__":
    main()
