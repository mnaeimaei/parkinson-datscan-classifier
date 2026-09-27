#!/usr/bin/env python3

"""
STEP 9G — FREEZE FINAL CLASSIFICATION DATASETS

Final reproducibility freeze for classification Scenarios A/B/C.

This step:
    - does NOT modify images
    - does NOT copy images
    - verifies previous Step 9 validation status
    - SHA256-hashes every whole-volume input
    - SHA256-hashes every striatal input
    - SHA256-hashes manifests / labels / important Step 9 reports
    - creates a deterministic final freeze manifest
    - calculates one aggregate Step-9 SHA256

Frozen classification inputs
----------------------------
Scenario A:
    whole volume
    shape = 192 x 192 x 160

Scenario B:
    bilateral striatal crop
    shape = 44 x 44 x 36

Scenario C:
    whole + striatal as two separate inputs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


WHOLE_SHAPE = [192, 192, 160]
STRIATAL_SHAPE = [44, 44, 36]
SPACING_MM = [2.46, 2.46, 2.46]


# ---------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        while True:

            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(
    text: str,
) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------
# CSV / JSON
# ---------------------------------------------------------------------


def read_csv(
    path: Path,
) -> list[dict[str, str]]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def write_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:

    if not rows:
        raise RuntimeError(
            "Cannot write empty freeze manifest."
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


def read_json(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(handle)


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
            sort_keys=False,
        )


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------


def resolve_project_path(
    project_root: Path,
    relative_path: str,
) -> Path:

    path = Path(relative_path)

    if path.is_absolute():

        raise RuntimeError(
            f"Expected project-relative path, "
            f"found absolute path:\n{path}"
        )

    resolved = (
        project_root
        / path
    ).resolve()

    try:

        resolved.relative_to(
            project_root
        )

    except ValueError as exc:

        raise RuntimeError(
            f"Path escapes project root: "
            f"{relative_path}"
        ) from exc

    return resolved


def relative_path(
    path: Path,
    project_root: Path,
) -> str:

    return (
        path.resolve()
        .relative_to(project_root)
        .as_posix()
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


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


def verify_previous_validation(
    creation_summary: Path,
    qc_summary: Path,
    scenario_validation_summary: Path,
) -> None:

    creation = read_json(
        creation_summary
    )

    qc = read_json(
        qc_summary
    )

    scenarios = read_json(
        scenario_validation_summary
    )

    if creation.get(
        "validation_passed"
    ) is not True:

        raise RuntimeError(
            "Step 9C validation is not PASS."
        )

    if qc.get(
        "numerical_validation_passed"
    ) is not True:

        raise RuntimeError(
            "Step 9D numerical QC is not PASS."
        )

    if scenarios.get(
        "validation_passed"
    ) is not True:

        raise RuntimeError(
            "Step 9F validation is not PASS."
        )


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
        "--manifest-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--labels-csv",
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

    manifest_dir = (
        args.manifest_dir.resolve()
    )

    report_dir = (
        args.report_dir.resolve()
    )

    labels_csv = (
        args.labels_csv.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    scenario_a_path = (
        manifest_dir
        / "scenario_a.csv"
    )

    scenario_b_path = (
        manifest_dir
        / "scenario_b.csv"
    )

    scenario_c_path = (
        manifest_dir
        / "scenario_c.csv"
    )

    scenario_summary_path = (
        manifest_dir
        / "classification_scenarios_summary.json"
    )

    creation_summary = (
        report_dir
        / "step9c_create_fixed_whole_volumes"
        / "fixed_whole_volume_creation_summary.json"
    )

    qc_summary = (
        report_dir
        / "step9d_validate_fixed_whole_volumes"
        / "fixed_whole_volume_qc_summary.json"
    )

    scenario_validation_csv = (
        report_dir
        / "step9f_validate_classification_scenarios"
        / "classification_scenarios_validation.csv"
    )

    scenario_validation_summary = (
        report_dir
        / "step9f_validate_classification_scenarios"
        / "classification_scenarios_validation_summary.json"
    )

    required_files = [
        scenario_a_path,
        scenario_b_path,
        scenario_c_path,
        scenario_summary_path,
        creation_summary,
        qc_summary,
        scenario_validation_csv,
        scenario_validation_summary,
        labels_csv,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required Step 9 file missing:\n"
                f"{path}"
            )

    #
    # Refuse to freeze anything that did not previously pass.
    #
    verify_previous_validation(
        creation_summary,
        qc_summary,
        scenario_validation_summary,
    )

    #
    # Read final manifests.
    #
    scenario_a = index_by_uid(
        read_csv(
            scenario_a_path
        ),
        "Scenario A",
    )

    scenario_b = index_by_uid(
        read_csv(
            scenario_b_path
        ),
        "Scenario B",
    )

    scenario_c = index_by_uid(
        read_csv(
            scenario_c_path
        ),
        "Scenario C",
    )

    uids_a = set(
        scenario_a
    )

    uids_b = set(
        scenario_b
    )

    uids_c = set(
        scenario_c
    )

    if not (
        uids_a
        == uids_b
        == uids_c
    ):

        raise RuntimeError(
            "Scenario UID sets differ."
        )

    n_subjects = len(uids_a)

    if n_subjects == 0:

        raise RuntimeError(
            "No subjects found in Scenario A."
        )

    ordered_uids = sorted(
        uids_a
    )

    #
    # Confirm labels and Scenario C references one last time.
    #
    normal_count = 0
    pathologic_count = 0

    for uid in ordered_uids:

        a = scenario_a[uid]
        b = scenario_b[uid]
        c = scenario_c[uid]

        labels = {
            int(a["is_pathologic"]),
            int(b["is_pathologic"]),
            int(c["is_pathologic"]),
        }

        if len(labels) != 1:

            raise RuntimeError(
                f"Label mismatch for {uid}"
            )

        label = next(
            iter(labels)
        )

        if label == 0:
            normal_count += 1

        elif label == 1:
            pathologic_count += 1

        else:

            raise RuntimeError(
                f"Invalid label for {uid}: "
                f"{label}"
            )

        if (
            a["whole_volume_path"]
            != c["whole_volume_path"]
        ):

            raise RuntimeError(
                f"Scenario C whole-volume "
                f"path mismatch for {uid}"
            )

        if (
            b["striatal_path"]
            != c["striatal_path"]
        ):

            raise RuntimeError(
                f"Scenario C striatal "
                f"path mismatch for {uid}"
            )

    if (
        normal_count
        + pathologic_count
        != n_subjects
    ):

        raise RuntimeError(
            "Class counts do not cover all subjects."
        )

    print()
    print("=" * 78)
    print(
        "STEP 9G — FREEZE CLASSIFICATION SCENARIOS"
    )
    print("=" * 78)

    print(
        f"Subjects expected       : "
        f"{n_subjects}"
    )

    print(
        f"Normal                  : "
        f"{normal_count}"
    )

    print(
        f"Pathologic              : "
        f"{pathologic_count}"
    )

    print(
        "Whole-volume shape      : "
        "192 x 192 x 160"
    )

    print(
        "Striatal shape          : "
        "44 x 44 x 36"
    )

    print(
        "Expected spacing        : "
        "2.46 x 2.46 x 2.46 mm"
    )

    print()
    print(
        "Previous validations    : PASS"
    )

    print("=" * 78)
    print()

    #
    # Hash every physical image.
    #
    freeze_rows: list[
        dict[str, Any]
    ] = []

    aggregate_lines: list[str] = []

    for index, uid in enumerate(
        ordered_uids,
        start=1,
    ):

        row_a = scenario_a[uid]
        row_b = scenario_b[uid]

        label = int(
            row_a["is_pathologic"]
        )

        whole_relative = (
            row_a["whole_volume_path"]
        )

        striatal_relative = (
            row_b["striatal_path"]
        )

        whole_path = resolve_project_path(
            project_root,
            whole_relative,
        )

        striatal_path = resolve_project_path(
            project_root,
            striatal_relative,
        )

        if not whole_path.is_file():

            raise FileNotFoundError(
                f"Missing whole volume: "
                f"{whole_path}"
            )

        if not striatal_path.is_file():

            raise FileNotFoundError(
                f"Missing striatal crop: "
                f"{striatal_path}"
            )

        whole_hash = sha256_file(
            whole_path
        )

        striatal_hash = sha256_file(
            striatal_path
        )

        whole_size = (
            whole_path.stat().st_size
        )

        striatal_size = (
            striatal_path.stat().st_size
        )

        freeze_rows.append(
            {
                "uid":
                    uid,

                "is_pathologic":
                    label,

                "whole_volume_path":
                    whole_relative,

                "whole_volume_sha256":
                    whole_hash,

                "whole_volume_size_bytes":
                    whole_size,

                "striatal_path":
                    striatal_relative,

                "striatal_sha256":
                    striatal_hash,

                "striatal_size_bytes":
                    striatal_size,
            }
        )

        #
        # Canonical aggregate representation.
        #
        aggregate_lines.append(
            "\t".join(
                [
                    "SUBJECT",
                    uid,
                    str(label),
                    whole_relative,
                    whole_hash,
                    striatal_relative,
                    striatal_hash,
                ]
            )
        )

        if (
            index == 1
            or index % 100 == 0
            or index == n_subjects
        ):

            print(
                f"Verified and hashed "
                f"{index}/{n_subjects}"
            )

    #
    # Hash Step 9 metadata/artifacts.
    #
    artifact_paths = [
        labels_csv,

        scenario_a_path,
        scenario_b_path,
        scenario_c_path,
        scenario_summary_path,

        report_dir
        / "step9a_analyze_whole_volume_shapes"
        / "whole_volume_shapes.csv",

        report_dir
        / "step9a_analyze_whole_volume_shapes"
        / "whole_volume_shape_statistics.json",

        report_dir
        / "step9b_analyze_fixed_volume_candidates"
        / "whole_volume_candidate_retention.csv",

        report_dir
        / "step9b_analyze_fixed_volume_candidates"
        / "whole_volume_candidate_retention_statistics.json",

        report_dir
        / "step9c_create_fixed_whole_volumes"
        / "fixed_whole_volume_manifest.csv",

        creation_summary,

        report_dir
        / "step9d_validate_fixed_whole_volumes"
        / "fixed_whole_volume_validation.csv",

        report_dir
        / "step9d_validate_fixed_whole_volumes"
        / "fixed_whole_volume_qc_selection.csv",

        qc_summary,

        scenario_validation_csv,
        scenario_validation_summary,
    ]

    artifact_hashes = []

    print()
    print(
        "Hashing Step 9 metadata..."
    )

    for path in artifact_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Step 9 artifact missing:\n"
                f"{path}"
            )

        digest = sha256_file(
            path
        )

        relative = relative_path(
            path,
            project_root,
        )

        artifact_hashes.append(
            {
                "path":
                    relative,

                "sha256":
                    digest,

                "size_bytes":
                    path.stat().st_size,
            }
        )

        aggregate_lines.append(
            "\t".join(
                [
                    "ARTIFACT",
                    relative,
                    digest,
                ]
            )
        )

    #
    # Frozen configuration.
    #
    frozen_config = {
        "step":
            "9G",

        "status":
            "FROZEN",

        "subjects":
            n_subjects,

        "class_distribution": {
            "normal_0":
                normal_count,

            "pathologic_1":
                pathologic_count,
        },

        "spacing_mm":
            SPACING_MM,

        "scenario_a": {
            "input":
                "whole_volume",

            "shape":
                WHOLE_SHAPE,

            "subjects":
                n_subjects,
        },

        "scenario_b": {
            "input":
                "bilateral_striatal_crop",

            "shape":
                STRIATAL_SHAPE,

            "subjects":
                n_subjects,
        },

        "scenario_c": {
            "inputs": [
                "whole_volume",
                "bilateral_striatal_crop",
            ],

            "separate_inputs":
                True,

            "image_duplication":
                False,

            "subjects":
                n_subjects,
        },

        "whole_volume_preparation": {
            "shape":
                WHOLE_SHAPE,

            "method":
                "deterministic center crop "
                "plus symmetric zero padding",

            "additional_resampling":
                False,

            "additional_normalization":
                False,

            "orientation":
                "RAS",

            "dtype":
                "float32",
        },

        "striatal_preparation": {
            "shape":
                STRIATAL_SHAPE,

            "source":
                "frozen Step 8 striatal crop pipeline",

            "orientation":
                "RAS",
        },

        "previous_validation": {
            "step9c":
                True,

            "step9d_numerical":
                True,

            "step9d_visual":
                True,

            "step9f":
                True,
        },
    }

    config_path = (
        output_dir
        / "frozen_configuration.json"
    )

    write_json(
        frozen_config,
        config_path,
    )

    config_hash = sha256_file(
        config_path
    )

    config_relative = relative_path(
        config_path,
        project_root,
    )

    aggregate_lines.append(
        "\t".join(
            [
                "CONFIG",
                config_relative,
                config_hash,
            ]
        )
    )

    #
    # Freeze manifest.
    #
    freeze_manifest_path = (
        output_dir
        / "frozen_dataset_manifest.csv"
    )

    write_csv(
        freeze_rows,
        freeze_manifest_path,
    )

    freeze_manifest_hash = (
        sha256_file(
            freeze_manifest_path
        )
    )

    #
    # Deterministic aggregate freeze SHA256.
    #
    aggregate_text = (
        "\n".join(
            aggregate_lines
        )
        + "\n"
    )

    overall_sha256 = (
        sha256_text(
            aggregate_text
        )
    )

    #
    # Save canonical digest input as well.
    #
    digest_input_path = (
        output_dir
        / "freeze_digest_input.txt"
    )

    digest_input_path.write_text(
        aggregate_text,
        encoding="utf-8",
    )

    digest_input_sha256 = (
        sha256_file(
            digest_input_path
        )
    )

    #
    # Final summary.
    #
    total_whole_bytes = sum(
        int(
            row[
                "whole_volume_size_bytes"
            ]
        )
        for row in freeze_rows
    )

    total_striatal_bytes = sum(
        int(
            row[
                "striatal_size_bytes"
            ]
        )
        for row in freeze_rows
    )

    summary = {
        "step":
            "9G",

        "status":
            "FROZEN",

        "validation_passed":
            True,

        "subjects":
            n_subjects,

        "normal_0":
            normal_count,

        "pathologic_1":
            pathologic_count,

        "whole_volume_files":
            n_subjects,

        "striatal_files":
            n_subjects,

        "whole_volume_shape":
            WHOLE_SHAPE,

        "striatal_shape":
            STRIATAL_SHAPE,

        "spacing_mm":
            SPACING_MM,

        "whole_volume_total_bytes":
            total_whole_bytes,

        "striatal_total_bytes":
            total_striatal_bytes,

        "freeze_manifest": {
            "path":
                relative_path(
                    freeze_manifest_path,
                    project_root,
                ),

            "sha256":
                freeze_manifest_hash,
        },

        "frozen_configuration": {
            "path":
                config_relative,

            "sha256":
                config_hash,
        },

        "digest_input": {
            "path":
                relative_path(
                    digest_input_path,
                    project_root,
                ),

            "sha256":
                digest_input_sha256,
        },

        "artifacts":
            artifact_hashes,

        "overall_step9_sha256":
            overall_sha256,
    }

    summary_path = (
        output_dir
        / "freeze_summary.json"
    )

    write_json(
        summary,
        summary_path,
    )

    #
    # Convenient plain-text checksum.
    #
    checksum_path = (
        output_dir
        / "SHA256.txt"
    )

    checksum_path.write_text(
        overall_sha256 + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "STEP 9G — FREEZE SUMMARY"
    )
    print("=" * 78)

    print(
        f"Subjects                     : "
        f"{n_subjects}"
    )

    print(
        f"Whole-volume files           : "
        f"{n_subjects}"
    )

    print(
        f"Striatal files               : "
        f"{n_subjects}"
    )

    print(
        f"Normal (0)                   : "
        f"{normal_count}"
    )

    print(
        f"Pathologic (1)               : "
        f"{pathologic_count}"
    )

    print()

    print(
        "Whole-volume shape           : "
        "192 x 192 x 160"
    )

    print(
        "Striatal shape               : "
        "44 x 44 x 36"
    )

    print(
        "Spacing                      : "
        "2.46 x 2.46 x 2.46 mm"
    )

    print()

    print(
        f"Frozen manifest SHA256       : "
        f"{freeze_manifest_hash}"
    )

    print(
        f"Frozen config SHA256         : "
        f"{config_hash}"
    )

    print()

    print(
        "OVERALL STEP 9 SHA256"
    )

    print(
        overall_sha256
    )

    print()

    print(
        f"Freeze manifest : "
        f"{freeze_manifest_path}"
    )

    print(
        f"Freeze summary  : "
        f"{summary_path}"
    )

    print(
        f"Checksum file   : "
        f"{checksum_path}"
    )

    print("=" * 78)

    print(
        "STEP 9 FROZEN SUCCESSFULLY"
    )

    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
