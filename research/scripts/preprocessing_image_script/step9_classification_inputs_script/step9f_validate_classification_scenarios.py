#!/usr/bin/env python3

"""
STEP 9F — VALIDATE CLASSIFICATION SCENARIOS

Validates Scenario A, B and C manifests and all referenced
whole-volume / striatal NIfTI files.

This is an integration validation step.

It checks:
    - row counts
    - UID uniqueness
    - subject consistency
    - label consistency
    - relative paths
    - file existence
    - whole-volume shape / spacing / orientation / dtype
    - striatal shape / spacing / orientation
    - Scenario C pairing consistency

No files are modified.
No image data are copied.
No augmentation is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


WHOLE_SHAPE = (192, 192, 160)
STRIATAL_SHAPE = (44, 44, 36)

EXPECTED_SPACING = (2.46, 2.46, 2.46)
EXPECTED_ORIENTATION = ("R", "A", "S")

SPACING_TOLERANCE = 1e-3


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------


def read_manifest(
    path: Path,
    required_columns: set[str],
) -> list[dict[str, str]]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        columns = set(
            reader.fieldnames or []
        )

        missing = (
            required_columns
            - columns
        )

        if missing:
            raise RuntimeError(
                f"{path} is missing columns: "
                f"{sorted(missing)}"
            )

        rows = list(reader)

    return rows


def write_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:

    if not rows:
        raise RuntimeError(
            "Cannot write empty validation CSV."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
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
# Manifest helpers
# ---------------------------------------------------------------------


def parse_label(
    value: str,
) -> int:

    text = str(value).strip()

    if text == "0":
        return 0

    if text == "1":
        return 1

    raise ValueError(
        f"Invalid label: {value!r}"
    )


def index_by_uid(
    rows: list[dict[str, str]],
    scenario_name: str,
) -> dict[str, dict[str, str]]:

    result = {}

    for row in rows:

        uid = row["uid"].strip()

        if not uid:
            raise RuntimeError(
                f"{scenario_name}: empty UID."
            )

        if uid in result:
            raise RuntimeError(
                f"{scenario_name}: duplicate UID "
                f"{uid}"
            )

        result[uid] = row

    return result


def resolve_project_path(
    project_root: Path,
    relative_text: str,
) -> Path:

    raw = Path(
        relative_text
    )

    if raw.is_absolute():

        raise RuntimeError(
            f"Manifest contains absolute path: "
            f"{relative_text}"
        )

    resolved = (
        project_root
        / raw
    ).resolve()

    try:

        resolved.relative_to(
            project_root
        )

    except ValueError as exc:

        raise RuntimeError(
            "Manifest path escapes project root: "
            f"{relative_text}"
        ) from exc

    return resolved


# ---------------------------------------------------------------------
# NIfTI validation
# ---------------------------------------------------------------------


def validate_nifti_header(
    path: Path,
    expected_shape: tuple[int, int, int],
    require_float32: bool,
) -> dict[str, Any]:

    if not path.exists():

        return {
            "exists": False,
            "shape_valid": False,
            "spacing_valid": False,
            "orientation_valid": False,
            "dtype_valid": False,
            "dtype": "",
            "validation_passed": False,
            "error": "file_not_found",
        }

    try:

        image = nib.load(
            str(path),
            mmap="r",
        )

        shape = tuple(
            int(v)
            for v in image.shape
        )

        spacing = tuple(
            float(v)
            for v in nib.affines.voxel_sizes(
                image.affine
            )
        )

        orientation = tuple(
            nib.aff2axcodes(
                image.affine
            )
        )

        dtype = np.dtype(
            image.header.get_data_dtype()
        )

        shape_valid = (
            shape == expected_shape
        )

        spacing_valid = all(
            abs(actual - expected)
            <= SPACING_TOLERANCE
            for actual, expected in zip(
                spacing,
                EXPECTED_SPACING,
            )
        )

        orientation_valid = (
            orientation
            == EXPECTED_ORIENTATION
        )

        if require_float32:

            dtype_valid = (
                dtype
                == np.dtype(np.float32)
            )

        else:

            dtype_valid = bool(
                np.issubdtype(
                    dtype,
                    np.number,
                )
            )

        passed = all(
            [
                shape_valid,
                spacing_valid,
                orientation_valid,
                dtype_valid,
            ]
        )

        return {
            "exists": True,

            "shape":
                "x".join(
                    str(v)
                    for v in shape
                ),

            "shape_valid":
                shape_valid,

            "spacing":
                "x".join(
                    f"{v:.6f}"
                    for v in spacing
                ),

            "spacing_valid":
                spacing_valid,

            "orientation":
                "".join(orientation),

            "orientation_valid":
                orientation_valid,

            "dtype":
                str(dtype),

            "dtype_valid":
                dtype_valid,

            "validation_passed":
                passed,

            "error":
                "",
        }

    except Exception as exc:

        return {
            "exists": True,
            "shape_valid": False,
            "spacing_valid": False,
            "orientation_valid": False,
            "dtype_valid": False,
            "dtype": "",
            "validation_passed": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--scenario-a",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--scenario-b",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--scenario-c",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    project_root = (
        args.project_root.resolve()
    )

    scenario_a_path = (
        args.scenario_a.resolve()
    )

    scenario_b_path = (
        args.scenario_b.resolve()
    )

    scenario_c_path = (
        args.scenario_c.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    print()
    print("=" * 78)
    print(
        "STEP 9F — VALIDATE CLASSIFICATION SCENARIOS"
    )
    print("=" * 78)

    print(
        f"Scenario A : {scenario_a_path}"
    )

    print(
        f"Scenario B : {scenario_b_path}"
    )

    print(
        f"Scenario C : {scenario_c_path}"
    )

    print("=" * 78)
    print()

    #
    # Read manifests
    #
    scenario_a_rows = read_manifest(
        scenario_a_path,
        {
            "uid",
            "is_pathologic",
            "whole_volume_path",
        },
    )

    scenario_b_rows = read_manifest(
        scenario_b_path,
        {
            "uid",
            "is_pathologic",
            "striatal_path",
        },
    )

    scenario_c_rows = read_manifest(
        scenario_c_path,
        {
            "uid",
            "is_pathologic",
            "whole_volume_path",
            "striatal_path",
        },
    )

    a = index_by_uid(
        scenario_a_rows,
        "Scenario A",
    )

    b = index_by_uid(
        scenario_b_rows,
        "Scenario B",
    )

    c = index_by_uid(
        scenario_c_rows,
        "Scenario C",
    )

    a_uids = set(a)
    b_uids = set(b)
    c_uids = set(c)

    uid_sets_identical = (
        a_uids
        == b_uids
        == c_uids
    )

    n_subjects = len(a)

    row_counts_valid = (
        n_subjects > 0
        and len(a) == len(b)
        and len(b) == len(c)
    )

    #
    # Check deterministic row ordering as well.
    #
    a_order = [
        row["uid"]
        for row in scenario_a_rows
    ]

    b_order = [
        row["uid"]
        for row in scenario_b_rows
    ]

    c_order = [
        row["uid"]
        for row in scenario_c_rows
    ]

    order_identical = (
        a_order
        == b_order
        == c_order
    )

    if not uid_sets_identical:

        raise RuntimeError(
            "Scenario UID sets do not match."
        )

    ordered_uids = sorted(
        a_uids
    )

    validation_rows = []

    whole_cache = {}
    striatal_cache = {}

    normal_count = 0
    pathologic_count = 0

    label_mismatch_count = 0
    scenario_c_path_mismatch_count = 0

    whole_failures = 0
    striatal_failures = 0
    subject_failures = 0

    print(
        f"Subjects to validate: "
        f"{len(ordered_uids)}"
    )

    print()

    for index, uid in enumerate(
        ordered_uids,
        start=1,
    ):

        row_a = a[uid]
        row_b = b[uid]
        row_c = c[uid]

        try:

            label_a = parse_label(
                row_a["is_pathologic"]
            )

            label_b = parse_label(
                row_b["is_pathologic"]
            )

            label_c = parse_label(
                row_c["is_pathologic"]
            )

            label_valid = (
                label_a
                == label_b
                == label_c
            )

        except Exception:

            label_a = -1
            label_b = -1
            label_c = -1
            label_valid = False

        if not label_valid:
            label_mismatch_count += 1

        if label_valid:

            if label_a == 0:
                normal_count += 1

            elif label_a == 1:
                pathologic_count += 1

        #
        # Scenario C must point to exactly the same
        # physical datasets as A and B.
        #
        whole_path_consistent = (
            row_a["whole_volume_path"]
            == row_c["whole_volume_path"]
        )

        striatal_path_consistent = (
            row_b["striatal_path"]
            == row_c["striatal_path"]
        )

        scenario_c_pair_valid = (
            whole_path_consistent
            and striatal_path_consistent
        )

        if not scenario_c_pair_valid:
            scenario_c_path_mismatch_count += 1

        #
        # Resolve paths.
        #
        try:

            whole_path = resolve_project_path(
                project_root,
                row_a["whole_volume_path"],
            )

            whole_path_relative_valid = True

        except Exception:

            whole_path = Path(
                "__invalid_whole_path__"
            )

            whole_path_relative_valid = False

        try:

            striatal_path = resolve_project_path(
                project_root,
                row_b["striatal_path"],
            )

            striatal_path_relative_valid = True

        except Exception:

            striatal_path = Path(
                "__invalid_striatal_path__"
            )

            striatal_path_relative_valid = False

        #
        # Validate each physical image once.
        #
        whole_key = str(
            whole_path
        )

        if whole_key not in whole_cache:

            whole_cache[
                whole_key
            ] = validate_nifti_header(
                whole_path,
                WHOLE_SHAPE,
                require_float32=True,
            )

        whole_result = whole_cache[
            whole_key
        ]

        striatal_key = str(
            striatal_path
        )

        if striatal_key not in striatal_cache:

            striatal_cache[
                striatal_key
            ] = validate_nifti_header(
                striatal_path,
                STRIATAL_SHAPE,
                require_float32=False,
            )

        striatal_result = striatal_cache[
            striatal_key
        ]

        whole_valid = (
            whole_path_relative_valid
            and whole_result[
                "validation_passed"
            ]
        )

        striatal_valid = (
            striatal_path_relative_valid
            and striatal_result[
                "validation_passed"
            ]
        )

        if not whole_valid:
            whole_failures += 1

        if not striatal_valid:
            striatal_failures += 1

        subject_valid = all(
            [
                label_valid,
                scenario_c_pair_valid,
                whole_valid,
                striatal_valid,
            ]
        )

        if not subject_valid:
            subject_failures += 1

        validation_rows.append(
            {
                "uid":
                    uid,

                "is_pathologic":
                    label_a,

                "label_consistent":
                    label_valid,

                "whole_volume_path":
                    row_a[
                        "whole_volume_path"
                    ],

                "whole_path_relative":
                    whole_path_relative_valid,

                "whole_exists":
                    whole_result[
                        "exists"
                    ],

                "whole_shape":
                    whole_result.get(
                        "shape",
                        "",
                    ),

                "whole_shape_valid":
                    whole_result[
                        "shape_valid"
                    ],

                "whole_spacing_valid":
                    whole_result[
                        "spacing_valid"
                    ],

                "whole_orientation_valid":
                    whole_result[
                        "orientation_valid"
                    ],

                "whole_dtype":
                    whole_result[
                        "dtype"
                    ],

                "whole_dtype_valid":
                    whole_result[
                        "dtype_valid"
                    ],

                "whole_valid":
                    whole_valid,

                "striatal_path":
                    row_b[
                        "striatal_path"
                    ],

                "striatal_path_relative":
                    striatal_path_relative_valid,

                "striatal_exists":
                    striatal_result[
                        "exists"
                    ],

                "striatal_shape":
                    striatal_result.get(
                        "shape",
                        "",
                    ),

                "striatal_shape_valid":
                    striatal_result[
                        "shape_valid"
                    ],

                "striatal_spacing_valid":
                    striatal_result[
                        "spacing_valid"
                    ],

                "striatal_orientation_valid":
                    striatal_result[
                        "orientation_valid"
                    ],

                "striatal_dtype":
                    striatal_result[
                        "dtype"
                    ],

                "striatal_dtype_numeric":
                    striatal_result[
                        "dtype_valid"
                    ],

                "striatal_valid":
                    striatal_valid,

                "scenario_c_whole_matches_a":
                    whole_path_consistent,

                "scenario_c_striatal_matches_b":
                    striatal_path_consistent,

                "scenario_c_pair_valid":
                    scenario_c_pair_valid,

                "subject_validation_passed":
                    subject_valid,
            }
        )

        if (
            index == 1
            or index % 100 == 0
            or index == len(ordered_uids)
        ):

            print(
                f"[{index:4d}/"
                f"{len(ordered_uids)}] "
                f"{uid:18s} "
                f"whole={whole_valid} "
                f"roi={striatal_valid} "
                f"pair={scenario_c_pair_valid} "
                f"pass={subject_valid}"
            )

    #
    # Global checks
    #
    class_distribution_valid = (
        normal_count
        + pathologic_count
        == n_subjects
    )

    unique_whole_paths = {
        row[
            "whole_volume_path"
        ]
        for row in scenario_a_rows
    }

    unique_striatal_paths = {
        row[
            "striatal_path"
        ]
        for row in scenario_b_rows
    }

    image_path_uniqueness_valid = (
        len(unique_whole_paths)
        == n_subjects
        and len(unique_striatal_paths)
        == n_subjects
    )

    validation_passed = all(
        [
            row_counts_valid,
            uid_sets_identical,
            order_identical,
            class_distribution_valid,
            image_path_uniqueness_valid,
            label_mismatch_count == 0,
            scenario_c_path_mismatch_count == 0,
            whole_failures == 0,
            striatal_failures == 0,
            subject_failures == 0,
        ]
    )

    #
    # Save reports.
    #
    validation_csv = (
        output_dir
        / "classification_scenarios_validation.csv"
    )

    summary_json = (
        output_dir
        / "classification_scenarios_validation_summary.json"
    )

    write_csv(
        validation_rows,
        validation_csv,
    )

    summary = {
        "step":
            "9F",

        "description":
            "Validate classification scenarios A, B and C",

        "expected_subjects":
            n_subjects,

        "scenario_rows": {
            "scenario_a":
                len(scenario_a_rows),

            "scenario_b":
                len(scenario_b_rows),

            "scenario_c":
                len(scenario_c_rows),
        },

        "uids": {
            "scenario_a_unique":
                len(a_uids),

            "scenario_b_unique":
                len(b_uids),

            "scenario_c_unique":
                len(c_uids),

            "sets_identical":
                uid_sets_identical,

            "row_order_identical":
                order_identical,
        },

        "class_distribution": {
            "normal_0":
                normal_count,

            "pathologic_1":
                pathologic_count,

            "valid":
                class_distribution_valid,
        },

        "whole_volume": {
            "expected_shape":
                list(WHOLE_SHAPE),

            "expected_spacing":
                list(EXPECTED_SPACING),

            "expected_orientation":
                "RAS",

            "expected_dtype":
                "float32",

            "unique_paths":
                len(
                    unique_whole_paths
                ),

            "validation_failures":
                whole_failures,
        },

        "striatal_roi": {
            "expected_shape":
                list(STRIATAL_SHAPE),

            "expected_spacing":
                list(EXPECTED_SPACING),

            "expected_orientation":
                "RAS",

            "unique_paths":
                len(
                    unique_striatal_paths
                ),

            "validation_failures":
                striatal_failures,
        },

        "scenario_c": {
            "whole_path_mismatches":
                scenario_c_path_mismatch_count,

            "separate_inputs":
                True,

            "image_duplication":
                False,
        },

        "label_mismatches":
            label_mismatch_count,

        "subject_validation_failures":
            subject_failures,

        "validation_passed":
            validation_passed,
    }

    write_json(
        summary,
        summary_json,
    )

    print()
    print("=" * 78)
    print(
        "STEP 9F SUMMARY"
    )
    print("=" * 78)

    print(
        f"Scenario A rows             : "
        f"{len(scenario_a_rows)}"
    )

    print(
        f"Scenario B rows             : "
        f"{len(scenario_b_rows)}"
    )

    print(
        f"Scenario C rows             : "
        f"{len(scenario_c_rows)}"
    )

    print()

    print(
        f"Unique UIDs                 : "
        f"{len(a_uids)}"
    )

    print(
        f"UID sets identical          : "
        f"{uid_sets_identical}"
    )

    print(
        f"Row order identical         : "
        f"{order_identical}"
    )

    print()

    print(
        f"Normal (0)                  : "
        f"{normal_count}"
    )

    print(
        f"Pathologic (1)              : "
        f"{pathologic_count}"
    )

    print(
        f"Class distribution valid    : "
        f"{class_distribution_valid}"
    )

    print()

    print(
        f"Whole-volume failures       : "
        f"{whole_failures}"
    )

    print(
        f"Striatal ROI failures       : "
        f"{striatal_failures}"
    )

    print(
        f"Label mismatches             : "
        f"{label_mismatch_count}"
    )

    print(
        f"Scenario C pair mismatches  : "
        f"{scenario_c_path_mismatch_count}"
    )

    print(
        f"Subject failures            : "
        f"{subject_failures}"
    )

    print()

    print(
        f"Unique whole image paths    : "
        f"{len(unique_whole_paths)}"
    )

    print(
        f"Unique striatal paths       : "
        f"{len(unique_striatal_paths)}"
    )

    print()

    print(
        f"VALIDATION PASSED           : "
        f"{validation_passed}"
    )

    print("=" * 78)

    print()
    print(
        f"Validation CSV : "
        f"{validation_csv}"
    )

    print(
        f"Summary JSON   : "
        f"{summary_json}"
    )

    print()

    if not validation_passed:

        raise RuntimeError(
            "Step 9F validation failed."
        )

    print("=" * 78)
    print(
        "STEP 9F PASSED"
    )
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
