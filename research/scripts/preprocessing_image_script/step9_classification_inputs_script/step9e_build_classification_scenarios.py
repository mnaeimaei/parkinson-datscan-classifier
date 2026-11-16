#!/usr/bin/env python3

"""
STEP 9E — BUILD CLASSIFICATION SCENARIO MANIFESTS

Builds the three frozen-input classification scenarios:

Scenario A
----------
Normalized fixed whole volume:
    192 x 192 x 160

Scenario B
----------
Normalized frozen bilateral striatal crop:
    44 x 44 x 36

Scenario C
----------
Two separate inputs:
    whole volume
    +
    striatal crop

IMPORTANT
---------
This step:
    - DOES NOT copy images
    - DOES NOT modify images
    - DOES NOT perform augmentation
    - DOES NOT make train/validation splits
    - stores project-relative paths for portability
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


UID_COLUMN = "uid"
LABEL_COLUMN = "is_pathologic"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def uid_from_path(path: Path) -> str:
    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return path.stem


def find_nifti_files(
    directory: Path,
) -> list[Path]:

    files = list(
        directory.glob("*.nii")
    )

    files.extend(
        directory.glob("*.nii.gz")
    )

    return sorted(
        set(files)
    )


def build_uid_map(
    directory: Path,
    dataset_name: str,
) -> dict[str, Path]:

    files = find_nifti_files(
        directory
    )

    if not files:
        raise RuntimeError(
            f"No NIfTI files found for "
            f"{dataset_name}:\n"
            f"{directory}"
        )

    result: dict[
        str,
        Path
    ] = {}

    duplicates: list[str] = []

    for path in files:

        uid = uid_from_path(
            path
        )

        if uid in result:
            duplicates.append(
                uid
            )
            continue

        result[uid] = path

    if duplicates:
        raise RuntimeError(
            f"Duplicate UIDs in "
            f"{dataset_name}: "
            f"{sorted(set(duplicates))}"
        )

    return result


def normalize_label(
    value: str,
) -> int:

    text = str(value).strip().lower()

    if text in {
        "1",
        "1.0",
        "true",
        "yes",
    }:
        return 1

    if text in {
        "0",
        "0.0",
        "false",
        "no",
    }:
        return 0

    raise ValueError(
        f"Invalid binary label: {value!r}"
    )


def inspect_label_csv(
    path: Path,
) -> bool:

    try:

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:

            reader = csv.DictReader(
                handle
            )

            fields = set(
                reader.fieldnames or []
            )

        return {
            UID_COLUMN,
            LABEL_COLUMN,
        }.issubset(fields)

    except Exception:

        return False


def discover_labels_csv(
    project_root: Path,
) -> Path:

    data_root = (
        project_root
        / "data"
    )

    candidates = []

    for path in sorted(
        data_root.rglob("*.csv")
    ):

        if inspect_label_csv(
            path
        ):
            candidates.append(
                path
            )

    if not candidates:

        raise RuntimeError(
            "Could not automatically find "
            "a CSV containing columns:\n"
            "  uid\n"
            "  is_pathologic"
        )

    #
    # Prefer conventional source label filenames.
    #
    preferred_names = [
        "train_labels_JNDlMjr.csv",
        "train_labels.csv",
        "labels.csv",
        "training_labels.csv",
    ]

    for preferred in preferred_names:

        matches = [
            path
            for path in candidates
            if path.name == preferred
        ]

        if len(matches) == 1:
            return matches[0]

    raw_label_matches = [
        path
        for path in candidates
        if "raw_label_data" in path.parts
    ]

    if len(raw_label_matches) == 1:
        return raw_label_matches[0]

    if len(candidates) == 1:
        return candidates[0]

    candidate_text = "\n".join(
        f"  {path}"
        for path in candidates
    )

    raise RuntimeError(
        "Multiple possible label CSV files "
        "were found.\n"
        "Please specify --labels-csv.\n\n"
        f"Candidates:\n{candidate_text}"
    )


def read_labels(
    path: Path,
) -> dict[str, int]:

    result: dict[
        str,
        int
    ] = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        fields = set(
            reader.fieldnames or []
        )

        required = {
            UID_COLUMN,
            LABEL_COLUMN,
        }

        missing = (
            required
            - fields
        )

        if missing:

            raise RuntimeError(
                f"Missing columns in "
                f"{path}: "
                f"{sorted(missing)}"
            )

        for row in reader:

            uid = str(
                row[UID_COLUMN]
            ).strip()

            if not uid:

                raise RuntimeError(
                    "Found empty UID "
                    f"in {path}"
                )

            if uid in result:

                raise RuntimeError(
                    f"Duplicate label UID: "
                    f"{uid}"
                )

            result[uid] = (
                normalize_label(
                    row[LABEL_COLUMN]
                )
            )

    return result


def relative_path(
    path: Path,
    project_root: Path,
) -> str:

    resolved = path.resolve()

    try:

        relative = (
            resolved.relative_to(
                project_root
            )
        )

    except ValueError as exc:

        raise RuntimeError(
            "Dataset file is outside "
            "the project root:\n"
            f"{resolved}"
        ) from exc

    return relative.as_posix()


def write_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:

        raise RuntimeError(
            "Cannot write empty manifest."
        )

    with output_path.open(
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
        writer.writerows(
            rows
        )


def write_json(
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Step 9E: build classification "
            "scenario manifests."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--whole-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--striatal-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--expected-subjects",
        type=int,
        default=None,
        help=(
            "Expected subject count. Defaults to the number of "
            "unique UIDs in the labels CSV; whole-volume and "
            "striatal folders must match."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    project_root = (
        args.project_root.resolve()
    )

    whole_dir = (
        args.whole_dir.resolve()
    )

    striatal_dir = (
        args.striatal_dir.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    if args.labels_csv is None:

        labels_csv = (
            discover_labels_csv(
                project_root
            )
        )

    else:

        labels_csv = (
            args.labels_csv.resolve()
        )

    print()
    print("=" * 76)
    print(
        "STEP 9E — BUILD CLASSIFICATION SCENARIOS"
    )
    print("=" * 76)

    print(
        f"Project root : {project_root}"
    )

    print(
        f"Whole volume : {whole_dir}"
    )

    print(
        f"Striatal ROI : {striatal_dir}"
    )

    print(
        f"Labels       : {labels_csv}"
    )

    print(
        f"Output       : {output_dir}"
    )

    print("=" * 76)
    print()

    #
    # Read the three sources.
    #
    whole_by_uid = build_uid_map(
        whole_dir,
        "fixed whole-volume dataset",
    )

    striatal_by_uid = build_uid_map(
        striatal_dir,
        "frozen striatal dataset",
    )

    labels = read_labels(
        labels_csv
    )

    expected_subjects = (
        int(args.expected_subjects)
        if args.expected_subjects is not None
        else len(labels)
    )

    whole_uids = set(
        whole_by_uid
    )

    striatal_uids = set(
        striatal_by_uid
    )

    label_uids = set(
        labels
    )

    #
    # Cross-source validation.
    #
    missing_whole = sorted(
        label_uids
        - whole_uids
    )

    missing_striatal = sorted(
        label_uids
        - striatal_uids
    )

    whole_without_label = sorted(
        whole_uids
        - label_uids
    )

    striatal_without_label = sorted(
        striatal_uids
        - label_uids
    )

    paired_uids = (
        whole_uids
        & striatal_uids
        & label_uids
    )

    source_validation_passed = all(
        [
            len(labels)
                == expected_subjects,

            len(whole_by_uid)
                == expected_subjects,

            len(striatal_by_uid)
                == expected_subjects,

            len(paired_uids)
                == expected_subjects,

            not missing_whole,
            not missing_striatal,
            not whole_without_label,
            not striatal_without_label,

            whole_uids
                == striatal_uids
                == label_uids,
        ]
    )

    if not source_validation_passed:

        raise RuntimeError(
            "Step 9E source validation failed.\n"
            f"Labels: {len(labels)}\n"
            f"Whole: {len(whole_by_uid)}\n"
            f"Striatal: {len(striatal_by_uid)}\n"
            f"Paired: {len(paired_uids)}\n"
            f"Missing whole: "
            f"{len(missing_whole)}\n"
            f"Missing striatal: "
            f"{len(missing_striatal)}"
        )

    #
    # Deterministic UID order.
    #
    ordered_uids = sorted(
        paired_uids
    )

    scenario_a = []
    scenario_b = []
    scenario_c = []

    for uid in ordered_uids:

        label = labels[
            uid
        ]

        whole_path = relative_path(
            whole_by_uid[uid],
            project_root,
        )

        striatal_path = relative_path(
            striatal_by_uid[uid],
            project_root,
        )

        #
        # Scenario A
        #
        scenario_a.append(
            {
                "uid": uid,
                "is_pathologic": label,
                "whole_volume_path":
                    whole_path,
            }
        )

        #
        # Scenario B
        #
        scenario_b.append(
            {
                "uid": uid,
                "is_pathologic": label,
                "striatal_path":
                    striatal_path,
            }
        )

        #
        # Scenario C
        #
        scenario_c.append(
            {
                "uid": uid,
                "is_pathologic": label,
                "whole_volume_path":
                    whole_path,
                "striatal_path":
                    striatal_path,
            }
        )

    #
    # Final manifest-level checks.
    #
    assert len(
        scenario_a
    ) == expected_subjects

    assert len(
        scenario_b
    ) == expected_subjects

    assert len(
        scenario_c
    ) == expected_subjects

    assert [
        row["uid"]
        for row in scenario_a
    ] == [
        row["uid"]
        for row in scenario_b
    ] == [
        row["uid"]
        for row in scenario_c
    ]

    #
    # Save manifests.
    #
    scenario_a_path = (
        output_dir
        / "scenario_a.csv"
    )

    scenario_b_path = (
        output_dir
        / "scenario_b.csv"
    )

    scenario_c_path = (
        output_dir
        / "scenario_c.csv"
    )

    write_csv(
        scenario_a,
        scenario_a_path,
    )

    write_csv(
        scenario_b,
        scenario_b_path,
    )

    write_csv(
        scenario_c,
        scenario_c_path,
    )

    normal_count = sum(
        label == 0
        for label in labels.values()
    )

    pathologic_count = sum(
        label == 1
        for label in labels.values()
    )

    summary = {
        "step": "9E",

        "description":
            "Build classification scenario manifests",

        "number_of_subjects":
            expected_subjects,

        "labels": {
            "source":
                relative_path(
                    labels_csv,
                    project_root,
                ),

            "normal_0":
                normal_count,

            "pathologic_1":
                pathologic_count,

            "total":
                len(labels),
        },

        "scenario_a": {
            "description":
                "Normalized fixed whole volume",

            "input_count":
                1,

            "whole_volume_shape":
                [192, 192, 160],

            "manifest":
                relative_path(
                    scenario_a_path,
                    project_root,
                ),

            "rows":
                len(scenario_a),
        },

        "scenario_b": {
            "description":
                "Normalized frozen bilateral "
                "striatal ROI",

            "input_count":
                1,

            "striatal_shape":
                [44, 44, 36],

            "manifest":
                relative_path(
                    scenario_b_path,
                    project_root,
                ),

            "rows":
                len(scenario_b),
        },

        "scenario_c": {
            "description":
                "Dual-input whole volume + "
                "striatal ROI",

            "input_count":
                2,

            "whole_volume_shape":
                [192, 192, 160],

            "striatal_shape":
                [44, 44, 36],

            "image_duplication":
                False,

            "manifest":
                relative_path(
                    scenario_c_path,
                    project_root,
                ),

            "rows":
                len(scenario_c),
        },

        "cross_source_validation": {
            "whole_volume_uids":
                len(whole_uids),

            "striatal_uids":
                len(striatal_uids),

            "label_uids":
                len(label_uids),

            "fully_paired_uids":
                len(paired_uids),

            "missing_whole":
                missing_whole,

            "missing_striatal":
                missing_striatal,

            "whole_without_label":
                whole_without_label,

            "striatal_without_label":
                striatal_without_label,

            "validation_passed":
                source_validation_passed,
        },
    }

    summary_path = (
        output_dir
        / "classification_scenarios_summary.json"
    )

    write_json(
        summary,
        summary_path,
    )

    print(
        "Source validation"
    )

    print(
        f"  Labels             : "
        f"{len(labels)}"
    )

    print(
        f"  Whole volumes      : "
        f"{len(whole_by_uid)}"
    )

    print(
        f"  Striatal crops     : "
        f"{len(striatal_by_uid)}"
    )

    print(
        f"  Fully paired       : "
        f"{len(paired_uids)}"
    )

    print()

    print(
        "Class distribution"
    )

    print(
        f"  Normal (0)         : "
        f"{normal_count}"
    )

    print(
        f"  Pathologic (1)     : "
        f"{pathologic_count}"
    )

    print()

    print(
        "Scenario A"
    )

    print(
        f"  Rows               : "
        f"{len(scenario_a)}"
    )

    print(
        "  Input              : "
        "whole volume"
    )

    print(
        "  Shape              : "
        "192 x 192 x 160"
    )

    print()

    print(
        "Scenario B"
    )

    print(
        f"  Rows               : "
        f"{len(scenario_b)}"
    )

    print(
        "  Input              : "
        "striatal ROI"
    )

    print(
        "  Shape              : "
        "44 x 44 x 36"
    )

    print()

    print(
        "Scenario C"
    )

    print(
        f"  Rows               : "
        f"{len(scenario_c)}"
    )

    print(
        "  Inputs             : "
        "whole + striatal"
    )

    print(
        "  Separate tensors   : True"
    )

    print(
        "  Image duplication  : False"
    )

    print()

    print("=" * 76)
    print(
        f"Scenario A : {scenario_a_path}"
    )

    print(
        f"Scenario B : {scenario_b_path}"
    )

    print(
        f"Scenario C : {scenario_c_path}"
    )

    print(
        f"Summary    : {summary_path}"
    )

    print("=" * 76)
    print(
        "STEP 9E PASSED"
    )
    print("=" * 76)
    print()


if __name__ == "__main__":
    main()
