#!/usr/bin/env python3

"""
STEP 10 — BUILD & FREEZE SUPERVISED DATASET MANIFEST

10A — Read frozen Step-9 outputs
10B — Read labels.csv
10C — Validate labels
10D — Match labels with image inputs
10E — Create master manifest
10F — Freeze manifest + validation + SHA256 hashes

IMPORTANT:
    No image preprocessing happens here.

    This script does NOT modify:
        - images
        - normalization
        - crops
        - shapes
        - registration

Steps 1–9 are treated as frozen.

NOTE:
    The frozen Step-9 manifest contains an outdated historical ROI path.
    Step 10 therefore resolves each ROI by UID from the actual frozen
    Step-8 striatal crop directory.

    No Step-9 file is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Frozen dataset expectations
# ---------------------------------------------------------------------

EXPECTED_TOTAL = 1362
EXPECTED_NORMAL = 615
EXPECTED_PATHOLOGIC = 747

VALID_LABELS = {0, 1}


# ---------------------------------------------------------------------
# Actual frozen ROI source
# ---------------------------------------------------------------------

FROZEN_ROI_DIR = Path(
    "data/preprocessing_image_data/"
    "step8_bilateral_striatal_crop_data/"
    "step8c_striatal_crop_generator/crops"
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA256 of a file without modifying it."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    sys.exit(1)


def resolve_image_path(
    path_string: str,
    repo_root: Path,
) -> Path:
    """
    Resolve paths stored in manifests.

    Absolute paths are used directly.
    Relative paths are resolved relative to repository root.
    """
    path = Path(path_string)

    if path.is_absolute():
        return path

    return repo_root / path


def relative_to_repo(
    path: Path,
    repo_root: Path,
) -> str:
    """
    Store paths relative to repository root where possible.

    This makes the frozen manifest portable across machines while
    preserving deterministic path structure.
    """
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def detect_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    """
    Detect a canonical Step-9 path column while allowing minor naming
    differences from previous pipeline stages.
    """
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    fail(
        f"Could not find {description} column.\n"
        f"Accepted names: {candidates}\n"
        f"Available columns: {list(dataframe.columns)}"
    )

    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------
# STEP 10A
# ---------------------------------------------------------------------

def read_step9_manifest(
    step9_manifest_path: Path,
) -> pd.DataFrame:

    print("\n# STEP 10A — READ FROZEN STEP-9 OUTPUTS")
    print()

    if not step9_manifest_path.exists():
        fail(
            f"Step-9 manifest not found: "
            f"{step9_manifest_path}"
        )

    df = pd.read_csv(
        step9_manifest_path,
        dtype={"uid": str},
    )

    if "uid" not in df.columns:
        fail(
            "Step-9 manifest does not contain a 'uid' column."
        )

    whole_column = detect_column(
        df,
        [
            "whole_path",
            "whole_volume_path",
            "normalized_whole_path",
            "scenario_a_path",
        ],
        "whole-volume path",
    )

    roi_column = detect_column(
        df,
        [
            "roi_path",
            "striatal_path",
            "striatal_crop_path",
            "normalized_roi_path",
            "scenario_b_path",
        ],
        "striatal ROI path",
    )

    # Preserve original Step-9 ROI path for provenance/audit.
    df["step9_original_roi_path"] = df[roi_column].astype(str)

    # Canonical Step-10 representation.
    df = df.rename(
        columns={
            whole_column: "whole_path",
            roi_column: "roi_path",
        }
    )

    df = df[
        [
            "uid",
            "whole_path",
            "roi_path",
            "step9_original_roi_path",
        ]
    ].copy()

    print(f"Step-9 manifest          : {step9_manifest_path}")
    print(f"Subjects found           : {len(df)}")
    print(f"Whole-volume column      : {whole_column}")
    print(f"ROI column               : {roi_column}")
    print()
    print("Scenario A               : whole volume")
    print("Expected whole shape     : 192 x 192 x 160")
    print()
    print("Scenario B               : striatal ROI")
    print("Expected ROI shape       : 44 x 44 x 36")
    print()
    print("Scenario C               : whole + ROI")
    print("                           two separate inputs")

    return df


# ---------------------------------------------------------------------
# STEP 10B
# ---------------------------------------------------------------------

def read_labels(
    labels_path: Path,
) -> pd.DataFrame:

    print("\n# STEP 10B — READ LABELS")
    print()

    if not labels_path.exists():
        fail(
            f"Labels file not found: {labels_path}"
        )

    labels = pd.read_csv(
        labels_path,
        dtype={"uid": str},
    )

    required_columns = {
        "uid",
        "is_pathologic",
    }

    missing = required_columns - set(labels.columns)

    if missing:
        fail(
            "labels.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    labels = labels[
        [
            "uid",
            "is_pathologic",
        ]
    ].copy()

    print(f"Labels file              : {labels_path}")
    print(f"Rows found               : {len(labels)}")
    print()
    print("Frozen label meaning:")
    print("    0 = Normal")
    print("    1 = Pathologic")

    return labels


# ---------------------------------------------------------------------
# STEP 10C
# ---------------------------------------------------------------------

def validate_labels(
    labels: pd.DataFrame,
) -> dict:

    print("\n# STEP 10C — VALIDATE LABELS")
    print()

    if labels["uid"].isna().any():
        fail(
            "Missing UID found in labels.csv."
        )

    labels["uid"] = (
        labels["uid"]
        .astype(str)
        .str.strip()
    )

    if (labels["uid"] == "").any():
        fail(
            "Empty UID found in labels.csv."
        )

    duplicate_count = int(
        labels["uid"].duplicated().sum()
    )

    if duplicate_count != 0:
        duplicate_uids = (
            labels.loc[
                labels["uid"].duplicated(keep=False),
                "uid",
            ]
            .unique()
            .tolist()
        )

        fail(
            f"Duplicate label UIDs found: "
            f"{duplicate_count}\n"
            f"Examples: {duplicate_uids[:10]}"
        )

    numeric_labels = pd.to_numeric(
        labels["is_pathologic"],
        errors="coerce",
    )

    if numeric_labels.isna().any():
        fail(
            "Missing or non-numeric labels found."
        )

    if not numeric_labels.isin(VALID_LABELS).all():
        invalid = sorted(
            set(
                numeric_labels[
                    ~numeric_labels.isin(
                        VALID_LABELS
                    )
                ].tolist()
            )
        )

        fail(
            f"Invalid labels found: {invalid}"
        )

    labels["is_pathologic"] = (
        numeric_labels.astype(int)
    )

    total = len(labels)

    unique_uids = int(
        labels["uid"].nunique()
    )

    normal = int(
        (labels["is_pathologic"] == 0).sum()
    )

    pathologic = int(
        (labels["is_pathologic"] == 1).sum()
    )

    if total != EXPECTED_TOTAL:
        fail(
            f"Expected {EXPECTED_TOTAL} labels, "
            f"found {total}."
        )

    if unique_uids != EXPECTED_TOTAL:
        fail(
            f"Expected {EXPECTED_TOTAL} unique UIDs, "
            f"found {unique_uids}."
        )

    if normal != EXPECTED_NORMAL:
        fail(
            f"Expected {EXPECTED_NORMAL} Normal subjects, "
            f"found {normal}."
        )

    if pathologic != EXPECTED_PATHOLOGIC:
        fail(
            f"Expected {EXPECTED_PATHOLOGIC} "
            f"Pathologic subjects, "
            f"found {pathologic}."
        )

    print(f"Total                     : {total}")
    print(f"Unique UIDs               : {unique_uids}")
    print(f"Normal                    : {normal}")
    print(f"Pathologic                : {pathologic}")
    print()
    print("✓ row count valid")
    print("✓ UID uniqueness valid")
    print("✓ labels only {0,1}")
    print("✓ no missing labels")
    print("✓ no duplicate UIDs")
    print("✓ no invalid labels")

    return {
        "total": total,
        "unique_uids": unique_uids,
        "normal": normal,
        "pathologic": pathologic,
        "labels_valid": True,
    }


# ---------------------------------------------------------------------
# STEP 10D
# ---------------------------------------------------------------------

def validate_and_match_inputs(
    step9: pd.DataFrame,
    labels: pd.DataFrame,
    repo_root: Path,
) -> tuple[pd.DataFrame, dict]:

    print(
        "\n# STEP 10D — MATCH LABELS WITH IMAGE INPUTS"
    )
    print()

    # -------------------------------------------------------------
    # Validate Step-9 UID field
    # -------------------------------------------------------------

    if step9["uid"].isna().any():
        fail(
            "Missing UID found in Step-9 manifest."
        )

    step9["uid"] = (
        step9["uid"]
        .astype(str)
        .str.strip()
    )

    if (step9["uid"] == "").any():
        fail(
            "Empty UID found in Step-9 manifest."
        )

    if step9["uid"].duplicated().any():
        duplicates = (
            step9.loc[
                step9["uid"].duplicated(keep=False),
                "uid",
            ]
            .unique()
            .tolist()
        )

        fail(
            "Duplicate UIDs found in Step-9 manifest.\n"
            f"Examples: {duplicates[:10]}"
        )

    # -------------------------------------------------------------
    # Validate whole-volume paths from Step 9
    # -------------------------------------------------------------

    if step9["whole_path"].isna().any():
        fail(
            "Missing values found in whole_path."
        )

    if (
        step9["whole_path"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        fail(
            "Empty paths found in whole_path."
        )

    # -------------------------------------------------------------
    # Compare image UIDs with label UIDs
    # -------------------------------------------------------------

    image_uids = set(step9["uid"])
    label_uids = set(labels["uid"])

    images_without_labels = sorted(
        image_uids - label_uids
    )

    labels_without_images = sorted(
        label_uids - image_uids
    )

    if images_without_labels:
        fail(
            f"{len(images_without_labels)} image subjects "
            f"have no label.\n"
            f"Examples: {images_without_labels[:10]}"
        )

    if labels_without_images:
        fail(
            f"{len(labels_without_images)} labels "
            f"have no images.\n"
            f"Examples: {labels_without_images[:10]}"
        )

    if len(step9) != EXPECTED_TOTAL:
        fail(
            f"Expected {EXPECTED_TOTAL} Step-9 subjects, "
            f"found {len(step9)}."
        )

    # -------------------------------------------------------------
    # Actual frozen ROI directory
    # -------------------------------------------------------------

    frozen_roi_dir = (
        repo_root / FROZEN_ROI_DIR
    )

    if not frozen_roi_dir.is_dir():
        fail(
            "Frozen ROI directory not found:\n"
            f"{frozen_roi_dir}"
        )

    print(
        f"Actual frozen ROI source  : {frozen_roi_dir}"
    )
    print()

    # -------------------------------------------------------------
    # Validate actual frozen files
    # -------------------------------------------------------------

    missing_whole = []
    missing_roi = []

    canonical_whole_paths = []
    canonical_roi_paths = []

    stale_step9_roi_paths = 0

    for row in step9.itertuples(index=False):

        # Whole volume remains exactly the frozen Step-9 input.
        whole_file = resolve_image_path(
            str(row.whole_path),
            repo_root,
        )

        # ROI is resolved from the actual frozen Step-8 crop location.
        roi_file = (
            frozen_roi_dir
            / f"{row.uid}.nii.gz"
        )

        if not whole_file.is_file():
            missing_whole.append(
                (
                    row.uid,
                    str(whole_file),
                )
            )

        if not roi_file.is_file():
            missing_roi.append(
                (
                    row.uid,
                    str(roi_file),
                )
            )

        canonical_whole_paths.append(
            relative_to_repo(
                whole_file,
                repo_root,
            )
        )

        canonical_roi_paths.append(
            relative_to_repo(
                roi_file,
                repo_root,
            )
        )

        # Audit historical Step-9 ROI path.
        original_roi_file = resolve_image_path(
            str(row.step9_original_roi_path),
            repo_root,
        )

        if not original_roi_file.is_file():
            stale_step9_roi_paths += 1

    if missing_whole:
        fail(
            f"{len(missing_whole)} whole-volume files "
            f"are missing.\n"
            f"Examples: {missing_whole[:5]}"
        )

    if missing_roi:
        fail(
            f"{len(missing_roi)} ROI files are missing.\n"
            f"Examples: {missing_roi[:5]}"
        )

    # -------------------------------------------------------------
    # Canonical Step-10 paths
    #
    # We do NOT edit Step 9.
    # We only construct correct paths in the new Step-10 dataframe.
    # -------------------------------------------------------------

    step10_inputs = step9.copy()

    step10_inputs["whole_path"] = (
        canonical_whole_paths
    )

    step10_inputs["roi_path"] = (
        canonical_roi_paths
    )

    # -------------------------------------------------------------
    # Join with labels
    # -------------------------------------------------------------

    master = step10_inputs.merge(
        labels,
        on="uid",
        how="inner",
        validate="one_to_one",
    )

    master = master[
        [
            "uid",
            "is_pathologic",
            "whole_path",
            "roi_path",
        ]
    ].copy()

    # Deterministic manifest ordering
    master = master.sort_values(
        "uid",
        kind="stable",
    ).reset_index(drop=True)

    if len(master) != EXPECTED_TOTAL:
        fail(
            f"Joined manifest contains {len(master)} "
            f"subjects; expected {EXPECTED_TOTAL}."
        )

    # -------------------------------------------------------------
    # Final UID-set validation
    # -------------------------------------------------------------

    master_uids = set(master["uid"])

    if master_uids != image_uids:
        fail(
            "Final master-manifest UID set differs "
            "from Step-9 UID set."
        )

    if master_uids != label_uids:
        fail(
            "Final master-manifest UID set differs "
            "from label UID set."
        )

    # -------------------------------------------------------------
    # Report
    # -------------------------------------------------------------

    print(f"Step-9 subjects           : {len(step9)}")
    print(f"Label subjects            : {len(labels)}")
    print(f"Matched subjects          : {len(master)}")
    print(
        f"Images without labels     : "
        f"{len(images_without_labels)}"
    )
    print(
        f"Labels without images     : "
        f"{len(labels_without_images)}"
    )
    print(
        f"Missing whole files       : "
        f"{len(missing_whole)}"
    )
    print(
        f"Missing ROI files         : "
        f"{len(missing_roi)}"
    )
    print(
        f"Stale Step-9 ROI paths    : "
        f"{stale_step9_roi_paths}"
    )

    print()
    print("✓ whole UID = ROI UID = label UID")
    print("✓ images without labels = 0")
    print("✓ labels without images = 0")
    print("✓ Scenario A/B/C contain same subjects")
    print("✓ all frozen whole-volume files exist")
    print("✓ all frozen ROI files exist")
    print(
        "✓ stale Step-9 ROI paths resolved "
        "without modifying Step 9"
    )

    validation = {
        "step9_subjects": len(step9),
        "label_subjects": len(labels),
        "matched_subjects": len(master),
        "images_without_labels": len(
            images_without_labels
        ),
        "labels_without_images": len(
            labels_without_images
        ),
        "missing_whole_files": len(
            missing_whole
        ),
        "missing_roi_files": len(
            missing_roi
        ),
        "stale_step9_roi_paths": (
            stale_step9_roi_paths
        ),
        "actual_frozen_roi_directory": str(
            FROZEN_ROI_DIR
        ),
        "subject_matching_valid": True,
    }

    return master, validation


# ---------------------------------------------------------------------
# STEP 10E
# ---------------------------------------------------------------------

def save_master_manifest(
    master: pd.DataFrame,
    output_path: Path,
) -> None:

    print(
        "\n# STEP 10E — CREATE MASTER MANIFEST"
    )
    print()

    master.to_csv(
        output_path,
        index=False,
        lineterminator="\n",
    )

    print(
        f"Manifest rows             : {len(master)}"
    )
    print("Columns                   :")
    print("    uid")
    print("    is_pathologic")
    print("    whole_path")
    print("    roi_path")
    print()
    print(
        f"Saved                     : {output_path}"
    )


# ---------------------------------------------------------------------
# STEP 10F
# ---------------------------------------------------------------------

def freeze_dataset(
    master: pd.DataFrame,
    repo_root: Path,
    step9_manifest_path: Path,
    labels_path: Path,
    manifest_path: Path,
    validation_summary_path: Path,
    image_hashes_path: Path,
    freeze_hashes_path: Path,
    label_validation: dict,
    matching_validation: dict,
) -> None:

    print("\n# STEP 10F — FREEZE")
    print()
    print("Calculating SHA256 hashes...")

    # -------------------------------------------------------------
    # Hash every referenced frozen input image.
    # -------------------------------------------------------------

    hash_rows = []

    total = len(master)

    for i, row in enumerate(
        master.itertuples(index=False),
        start=1,
    ):

        whole_file = resolve_image_path(
            str(row.whole_path),
            repo_root,
        )

        roi_file = resolve_image_path(
            str(row.roi_path),
            repo_root,
        )

        if not whole_file.is_file():
            fail(
                "Whole-volume file disappeared during "
                f"freezing: {whole_file}"
            )

        if not roi_file.is_file():
            fail(
                "ROI file disappeared during freezing: "
                f"{roi_file}"
            )

        hash_rows.append(
            {
                "uid": row.uid,
                "whole_sha256": sha256_file(
                    whole_file
                ),
                "roi_sha256": sha256_file(
                    roi_file
                ),
            }
        )

        if (
            i == 1
            or i % 100 == 0
            or i == total
        ):
            print(
                f"Hashed {i}/{total}"
            )

    image_hashes_df = pd.DataFrame(
        hash_rows
    )

    image_hashes_df.to_csv(
        image_hashes_path,
        index=False,
        lineterminator="\n",
    )

    # -------------------------------------------------------------
    # Top-level frozen artifact hashes
    # -------------------------------------------------------------

    hashes = {
        "sha256_algorithm": "SHA256",

        "step9_manifest": {
            "path": relative_to_repo(
                step9_manifest_path,
                repo_root,
            ),
            "sha256": sha256_file(
                step9_manifest_path
            ),
        },

        "labels": {
            "path": relative_to_repo(
                labels_path,
                repo_root,
            ),
            "sha256": sha256_file(
                labels_path
            ),
        },

        "supervised_dataset_manifest": {
            "path": relative_to_repo(
                manifest_path,
                repo_root,
            ),
            "sha256": sha256_file(
                manifest_path
            ),
        },

        "image_hashes": {
            "path": relative_to_repo(
                image_hashes_path,
                repo_root,
            ),
            "sha256": sha256_file(
                image_hashes_path
            ),
        },
    }

    with freeze_hashes_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            hashes,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    # -------------------------------------------------------------
    # Final validation summary
    # -------------------------------------------------------------

    summary = {
        "step": (
            "STEP 10 — BUILD & FREEZE "
            "SUPERVISED DATASET MANIFEST"
        ),

        "validation": True,

        "expected": {
            "subjects": EXPECTED_TOTAL,
            "normal": EXPECTED_NORMAL,
            "pathologic": EXPECTED_PATHOLOGIC,

            "whole_shape": [
                192,
                192,
                160,
            ],

            "roi_shape": [
                44,
                44,
                36,
            ],

            "label_meaning": {
                "0": "Normal",
                "1": "Pathologic",
            },
        },

        "labels": label_validation,

        "matching": matching_validation,

        "scenarios": {
            "scenario_a": (
                "whole volume"
            ),

            "scenario_b": (
                "striatal ROI"
            ),

            "scenario_c": (
                "whole volume + striatal ROI "
                "as separate inputs"
            ),
        },

        "path_resolution": {
            "whole_volume_source": (
                "Frozen Step-9 manifest"
            ),

            "roi_source": str(
                FROZEN_ROI_DIR
            ),

            "reason": (
                "Step-9 manifest contains historical "
                "ROI paths that no longer correspond "
                "to the actual frozen crop directory."
            ),

            "step9_manifest_modified": False,

            "image_files_modified": False,
        },

        "frozen_subjects": len(master),

        "artifacts": {
            "manifest": relative_to_repo(
                manifest_path,
                repo_root,
            ),

            "image_hashes": relative_to_repo(
                image_hashes_path,
                repo_root,
            ),

            "freeze_hashes": relative_to_repo(
                freeze_hashes_path,
                repo_root,
            ),
        },

        "steps_1_to_9_modified": False,
    }

    with validation_summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    print()
    print(
        f"Subjects frozen           : "
        f"{len(master)}"
    )
    print(
        f"Image hash rows           : "
        f"{len(image_hashes_df)}"
    )
    print()
    print(
        f"Manifest                  : "
        f"{manifest_path}"
    )
    print(
        f"Validation summary        : "
        f"{validation_summary_path}"
    )
    print(
        f"Image SHA256 hashes       : "
        f"{image_hashes_path}"
    )
    print(
        f"Freeze SHA256 hashes      : "
        f"{freeze_hashes_path}"
    )
    print()
    print("✓ supervised dataset frozen")
    print("✓ Steps 1–9 unchanged")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build and freeze the Step-10 "
            "supervised dataset manifest."
        )
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--step9-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    repo_root = (
        args.repo_root.resolve()
    )

    step9_manifest_path = (
        args.step9_manifest
        if args.step9_manifest.is_absolute()
        else repo_root / args.step9_manifest
    )

    labels_path = (
        args.labels
        if args.labels.is_absolute()
        else repo_root / args.labels
    )

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_dir
        / "supervised_dataset_manifest.csv"
    )

    validation_summary_path = (
        output_dir
        / "supervised_dataset_validation_summary.json"
    )

    image_hashes_path = (
        output_dir
        / "supervised_dataset_image_sha256.csv"
    )

    freeze_hashes_path = (
        output_dir
        / "supervised_dataset_sha256.json"
    )

    print(
        "============================================================"
    )
    print(
        "STEP 10 — BUILD & FREEZE SUPERVISED DATASET MANIFEST"
    )
    print(
        "============================================================"
    )

    # -------------------------------------------------------------
    # 10A
    # -------------------------------------------------------------

    step9 = read_step9_manifest(
        step9_manifest_path
    )

    # -------------------------------------------------------------
    # 10B
    # -------------------------------------------------------------

    labels = read_labels(
        labels_path
    )

    # -------------------------------------------------------------
    # 10C
    # -------------------------------------------------------------

    label_validation = validate_labels(
        labels
    )

    # -------------------------------------------------------------
    # 10D
    # -------------------------------------------------------------

    master, matching_validation = (
        validate_and_match_inputs(
            step9,
            labels,
            repo_root,
        )
    )

    # -------------------------------------------------------------
    # 10E
    # -------------------------------------------------------------

    save_master_manifest(
        master,
        manifest_path,
    )

    # -------------------------------------------------------------
    # 10F
    # -------------------------------------------------------------

    freeze_dataset(
        master=master,
        repo_root=repo_root,
        step9_manifest_path=step9_manifest_path,
        labels_path=labels_path,
        manifest_path=manifest_path,
        validation_summary_path=validation_summary_path,
        image_hashes_path=image_hashes_path,
        freeze_hashes_path=freeze_hashes_path,
        label_validation=label_validation,
        matching_validation=matching_validation,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "STEP 10 COMPLETE — PASS"
    )
    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()