from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import SimpleITK as sitk


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step4_voxel_resampler_data"
)

TEMPLATE_PATH = (
    PROJECT_ROOT
    / "data/template/dat_spect/fpcit_template_mni.nii"
)

# Step 6B-4
STEP6B4_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6b_registration_rescue_data/step6b1_finalize_registration_transforms"
)

STEP6B4_MANIFEST = (
    STEP6B4_DIR
    / "final_transform_manifest.csv"
)

SIMILARITY_RESCUE_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6b_registration_rescue_data/step6b3_rescue_pnsm_similarity"
)

SIMILARITY_RESCUE_CSV = (
    SIMILARITY_RESCUE_DIR
    / "similarity_rescue_result.csv"
)

SIMILARITY_TRANSFORM_DIR = (
    SIMILARITY_RESCUE_DIR
    / "transforms"
)

# ============================================================
# FINAL OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6c_finalize_registration_data/step6c_finalize_all_registration_transforms"
)

FINAL_TRANSFORM_DIR = (
    OUTPUT_DIR
    / "transforms"
)

FINAL_MANIFEST = (
    OUTPUT_DIR
    / "final_registration_manifest.csv"
)

FINAL_SUMMARY = (
    OUTPUT_DIR
    / "final_registration_summary.json"
)


# ============================================================
# INITIAL TRANSFORM RECONSTRUCTION
# ============================================================

def create_initial_transform(
    fixed: sitk.Image,
    moving: sitk.Image,
    candidate: str,
) -> sitk.Transform:
    """
    Reconstruct the manually accepted initial transform.

    Offline, all 6b1 manual_review cases were visually
    reviewed and the INITIAL transform was selected.
    Inference uses the same rule automatically.

    candidate:
        moments_initial
        geometry_initial
    """

    if candidate == "moments_initial":

        mode = (
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )

    elif candidate == "geometry_initial":

        mode = (
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

    else:

        raise ValueError(
            f"Unsupported initial candidate: {candidate}"
        )

    transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        mode,
    )

    return transform


# ============================================================
# TRANSFORM VALIDATION
# ============================================================

def validate_transform_file(
    path: Path,
) -> tuple[bool, str]:
    """
    Confirm:
        1. file exists
        2. file can be read by SimpleITK
    """

    if not path.exists():

        return (
            False,
            "transform_file_missing",
        )

    try:

        transform = sitk.ReadTransform(
            str(path)
        )

        if transform is None:

            return (
                False,
                "transform_read_returned_none",
            )

    except Exception as exc:

        return (
            False,
            f"transform_read_error: {exc}",
        )

    return (
        True,
        "",
    )


# ============================================================
# COPY EXISTING FINAL TRANSFORM
# ============================================================

def copy_step6b4_transform(
    source_relative: str,
    destination: Path,
) -> None:

    if not source_relative:

        raise ValueError(
            "Step 6B-4 final_transform path is empty."
        )

    source = Path(
        source_relative
    )

    if not source.is_absolute():

        source = (
            PROJECT_ROOT
            / source
        )

    if not source.exists():

        raise FileNotFoundError(
            f"Step 6B-4 transform missing: "
            f"{source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )


# ============================================================
# MANUAL INITIAL TRANSFORM
# ============================================================

def save_manual_initial_transform(
    uid: str,
    file_name: str,
    candidate: str,
    fixed: sitk.Image,
    destination: Path,
) -> None:

    moving_path = (
        INPUT_DIR
        / file_name
    )

    if not moving_path.exists():

        raise FileNotFoundError(
            f"Moving scan missing: "
            f"{moving_path}"
        )

    moving = sitk.ReadImage(
        str(moving_path),
        sitk.sitkFloat32,
    )

    transform = create_initial_transform(
        fixed=fixed,
        moving=moving,
        candidate=candidate,
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sitk.WriteTransform(
        transform,
        str(destination),
    )


# ============================================================
# SIMILARITY RESCUE TRANSFORMS
# ============================================================

def similarity_rescue_transform_path(
    uid: str,
) -> Path:

    return (
        SIMILARITY_TRANSFORM_DIR
        / f"{uid}_similarity_rescue.h5"
    )


def copy_similarity_rescue_transform(
    uid: str,
    destination: Path,
) -> Path:

    source = similarity_rescue_transform_path(
        uid
    )

    if not source.exists():

        raise FileNotFoundError(
            "Similarity rescue transform "
            f"not found for {uid}: {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    return source


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create the definitive registration "
            "transform set from 6b1, using "
            "similarity rescue for every "
            "unresolved scan."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete previous final_registration "
            "output before rebuilding."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Technical test only. Process first N rows."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    required_paths = [
        INPUT_DIR,
        TEMPLATE_PATH,
        STEP6B4_MANIFEST,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path missing: {path}"
            )

    # ========================================================
    # HANDLE OUTPUT DIRECTORY
    # ========================================================

    if args.overwrite and OUTPUT_DIR.exists():

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_TRANSFORM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD STEP 6B-4 MANIFEST
    # ========================================================

    df = pd.read_csv(
        STEP6B4_MANIFEST,
        dtype={
            "uid": str,
            "file_name": str,
        },
    )

    # --------------------------------------------------------
    # Basic manifest validation
    # --------------------------------------------------------

    if df["uid"].duplicated().any():

        duplicates = (
            df.loc[
                df["uid"].duplicated(
                    keep=False
                ),
                "uid",
            ]
            .tolist()
        )

        raise RuntimeError(
            f"Duplicate UIDs in Step 6B-4 manifest: "
            f"{duplicates}"
        )

    if args.limit is not None:

        df = df.head(
            args.limit
        ).copy()

    df = df.reset_index(
        drop=True
    )

    n_scans = len(df)

    rescue_by_uid: dict[str, pd.Series] = {}

    if SIMILARITY_RESCUE_CSV.exists():

        rescue_df = pd.read_csv(
            SIMILARITY_RESCUE_CSV,
            dtype={"uid": str},
        )

        for _, rescue_row in rescue_df.iterrows():

            rescue_by_uid[
                str(rescue_row["uid"])
            ] = rescue_row

    unresolved_uids = (
        df.loc[
            df["selection_status"].astype(str)
            == "unresolved",
            "uid",
        ]
        .astype(str)
        .tolist()
    )

    missing_rescue = [
        uid
        for uid in unresolved_uids
        if not similarity_rescue_transform_path(
            uid
        ).exists()
    ]

    if missing_rescue:

        raise FileNotFoundError(
            "Similarity rescue transform missing "
            "for unresolved UID(s): "
            + ", ".join(missing_rescue)
            + f"\nExpected under: {SIMILARITY_TRANSFORM_DIR}"
        )

    if unresolved_uids and not SIMILARITY_RESCUE_CSV.exists():

        raise FileNotFoundError(
            "Unresolved scans require "
            f"{SIMILARITY_RESCUE_CSV}"
        )

    # ========================================================
    # LOAD TEMPLATE ONCE
    # ========================================================

    fixed = sitk.ReadImage(
        str(TEMPLATE_PATH),
        sitk.sitkFloat32,
    )

    # ========================================================
    # PROCESS
    # ========================================================

    print("=" * 90)

    print(
        "STEP 6B-6 — FINALIZE ALL REGISTRATION TRANSFORMS"
    )

    print("=" * 90)

    print(
        f"Rows to process: {len(df)}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print("=" * 90)

    result_rows = []

    errors = []

    total = len(df)

    for index, row in df.iterrows():

        uid = str(
            row["uid"]
        )

        file_name = str(
            row["file_name"]
        )

        selection_status = str(
            row["selection_status"]
        )

        final_uid_dir = (
            FINAL_TRANSFORM_DIR
            / uid
        )

        final_path = (
            final_uid_dir
            / "final_transform.h5"
        )

        print(
            f"[{index + 1}/{total}] {uid}"
        )

        source_type = ""
        source_candidate = ""
        decision_source = ""
        final_dice = None
        final_qc = ""
        error_message = ""

        try:

            # =================================================
            # UNRESOLVED RIGID: SIMILARITY RESCUE
            # =================================================

            if selection_status == "unresolved":

                copy_similarity_rescue_transform(
                    uid=uid,
                    destination=final_path,
                )

                rescue_row = rescue_by_uid.get(
                    uid
                )

                if rescue_row is None:

                    raise RuntimeError(
                        "Unresolved UID has a rescue "
                        "transform but no row in "
                        f"{SIMILARITY_RESCUE_CSV}: {uid}"
                    )

                init_name = str(
                    rescue_row[
                        "selected_initialization"
                    ]
                )

                source_type = (
                    "similarity_rescue"
                )

                source_candidate = (
                    f"{init_name}_similarity"
                )

                decision_source = (
                    "similarity_rescue"
                )

                final_dice = float(
                    rescue_row[
                        "similarity_dice"
                    ]
                )

                final_qc = str(
                    rescue_row[
                        "qc_category"
                    ]
                )

                print(
                    "    source: similarity rescue"
                )

                print(
                    f"    Dice: {final_dice:.4f}"
                )

            # =================================================
            # 25 MANUALLY REVIEWED CASES
            #
            # Visual decision from Step 6B-5:
            # INITIAL selected for all 25.
            # =================================================

            elif selection_status == "manual_review":

                candidate = str(
                    row[
                        "best_initial_candidate"
                    ]
                )

                save_manual_initial_transform(
                    uid=uid,
                    file_name=file_name,
                    candidate=candidate,
                    fixed=fixed,
                    destination=final_path,
                )

                source_type = (
                    "manual_initial"
                )

                source_candidate = candidate

                decision_source = (
                    "step6b5_visual_qc"
                )

                final_dice = float(
                    row[
                        "best_initial_dice"
                    ]
                )

                final_qc = (
                    "high_confidence"
                    if final_dice >= 0.70
                    else "review"
                )

                print(
                    f"    source: "
                    f"{candidate}"
                )

                print(
                    f"    Dice: "
                    f"{final_dice:.4f}"
                )

            # =================================================
            # AUTOMATICALLY FINALIZED CASES
            # =================================================

            elif selection_status in (
                "auto_final",
                "auto_final_review_recommended",
            ):

                existing_transform = str(
                    row["final_transform"]
                )

                copy_step6b4_transform(
                    source_relative=
                        existing_transform,

                    destination=
                        final_path,
                )

                source_type = (
                    "step6b4_final"
                )

                source_candidate = str(
                    row[
                        "selected_candidate"
                    ]
                )

                decision_source = (
                    "step6b4_policy"
                )

                final_dice = float(
                    row[
                        "selected_dice"
                    ]
                )

                final_qc = str(
                    row[
                        "selected_qc"
                    ]
                )

                print(
                    f"    source: "
                    f"{source_candidate}"
                )

                print(
                    f"    Dice: "
                    f"{final_dice:.4f}"
                )

            # =================================================
            # OTHER UNRESOLVED CASE
            # =================================================

            else:

                raise RuntimeError(
                    f"Unexpected unresolved case: "
                    f"{uid} "
                    f"status={selection_status}"
                )

            # =================================================
            # VALIDATE SAVED TRANSFORM
            # =================================================

            valid, validation_error = (
                validate_transform_file(
                    final_path
                )
            )

            if not valid:

                raise RuntimeError(
                    validation_error
                )

            print(
                "    final transform: OK"
            )

            transform_relative = str(
                final_path.relative_to(
                    PROJECT_ROOT
                )
            )

            status = "success"

        except Exception as exc:

            error_message = str(exc)

            status = "failed"

            transform_relative = ""

            errors.append(
                {
                    "uid": uid,
                    "error": error_message,
                }
            )

            print(
                f"    ERROR: "
                f"{error_message}"
            )

        # =====================================================
        # MANIFEST ROW
        # =====================================================

        result_rows.append(
            {
                "uid":
                    uid,

                "file_name":
                    file_name,

                "source_selection_status":
                    selection_status,

                "final_source_type":
                    source_type,

                "final_candidate":
                    source_candidate,

                "decision_source":
                    decision_source,

                "final_dice":
                    final_dice,

                "final_qc":
                    final_qc,

                "final_transform":
                    transform_relative,

                "status":
                    status,

                "error":
                    error_message,
            }
        )

    # ========================================================
    # SAVE FINAL MANIFEST
    # ========================================================

    result_df = pd.DataFrame(
        result_rows
    )

    result_df.to_csv(
        FINAL_MANIFEST,
        index=False,
    )

    # ========================================================
    # HARD VALIDATION
    # ========================================================

    successful = result_df[
        result_df["status"] == "success"
    ].copy()

    failed = result_df[
        result_df["status"] != "success"
    ].copy()

    duplicate_uids = (
        result_df["uid"]
        .duplicated()
        .sum()
    )

    missing_transform_entries = int(
        (
            successful[
                "final_transform"
            ]
            .fillna("")
            ==
            ""
        ).sum()
    )

    # --------------------------------------------------------
    # Check every referenced transform exists
    # --------------------------------------------------------

    missing_transform_files = []

    unreadable_transform_files = []

    for _, row in successful.iterrows():

        transform_path = (
            PROJECT_ROOT
            / str(
                row["final_transform"]
            )
        )

        valid, reason = (
            validate_transform_file(
                transform_path
            )
        )

        if not transform_path.exists():

            missing_transform_files.append(
                str(row["uid"])
            )

        elif not valid:

            unreadable_transform_files.append(
                {
                    "uid":
                        str(
                            row["uid"]
                        ),

                    "reason":
                        reason,
                }
            )

    # --------------------------------------------------------
    # Count actual files
    # --------------------------------------------------------

    actual_transform_files = list(
        FINAL_TRANSFORM_DIR.glob(
            "*/final_transform.h5"
        )
    )

    actual_transform_count = len(
        actual_transform_files
    )

    # ========================================================
    # DISTRIBUTIONS
    # ========================================================

    source_counts = Counter(
        successful[
            "final_source_type"
        ]
        .fillna("")
        .tolist()
    )

    candidate_counts = Counter(
        successful[
            "final_candidate"
        ]
        .fillna("")
        .tolist()
    )

    qc_counts = Counter(
        successful[
            "final_qc"
        ]
        .fillna("")
        .tolist()
    )

    # ========================================================
    # FINAL PASS / FAIL
    # ========================================================

    if args.limit is None:

        final_validation_passed = all(
            [
                len(result_df)
                == n_scans,

                len(successful)
                == n_scans,

                len(failed)
                == 0,

                duplicate_uids
                == 0,

                missing_transform_entries
                == 0,

                len(
                    missing_transform_files
                )
                == 0,

                len(
                    unreadable_transform_files
                )
                == 0,

                actual_transform_count
                == n_scans,
            ]
        )

    else:

        final_validation_passed = (
            len(failed) == 0
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "analysis":
            (
                "Step 6B-6 definitive final "
                "registration transform set"
            ),

        "expected_scans":
            n_scans,

        "manifest_rows":
            int(
                len(result_df)
            ),

        "successful":
            int(
                len(successful)
            ),

        "failed":
            int(
                len(failed)
            ),

        "unique_uids":
            int(
                result_df[
                    "uid"
                ].nunique()
            ),

        "duplicate_uid_count":
            int(
                duplicate_uids
            ),

        "actual_final_transform_files":
            int(
                actual_transform_count
            ),

        "missing_transform_entries":
            int(
                missing_transform_entries
            ),

        "missing_transform_files":
            missing_transform_files,

        "unreadable_transform_files":
            unreadable_transform_files,

        "final_source_counts":
            dict(
                source_counts
            ),

        "final_candidate_counts":
            dict(
                candidate_counts
            ),

        "final_qc_counts":
            dict(
                qc_counts
            ),

        "manual_review_policy":
            (
                "All 6b1 manual_review cases use "
                "the best initial rigid transform "
                "(offline visual QC; the same rule "
                "is used automatically at inference)."
            ),

        "similarity_rescue_policy":
            (
                "Every 6b1 unresolved scan uses "
                "its Step 6b3 isotropic similarity "
                "rescue transform, if present."
            ),

        "similarity_rescue_uids":
            unresolved_uids,

        "final_validation_passed":
            bool(
                final_validation_passed
            ),

        "final_manifest":
            str(
                FINAL_MANIFEST.relative_to(
                    PROJECT_ROOT
                )
            ),

        "final_transform_directory":
            str(
                FINAL_TRANSFORM_DIR.relative_to(
                    PROJECT_ROOT
                )
            ),

        "errors":
            errors,

        "next_step":
            (
                "Proceed to Step 6C occipital "
                "reference-mask mapping only if "
                "final_validation_passed is true."
            ),
    }

    with FINAL_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()
    print("=" * 90)

    print(
        "STEP 6B-6 COMPLETED"
    )

    print("=" * 90)

    print(
        f"Manifest rows: "
        f"{len(result_df)}"
    )

    print(
        f"Successful: "
        f"{len(successful)}"
    )

    print(
        f"Failed: "
        f"{len(failed)}"
    )

    print(
        f"Unique UIDs: "
        f"{result_df['uid'].nunique()}"
    )

    print(
        f"Final transform files: "
        f"{actual_transform_count}"
    )

    print(
        f"Missing transforms: "
        f"{len(missing_transform_files)}"
    )

    print(
        f"Unreadable transforms: "
        f"{len(unreadable_transform_files)}"
    )

    print(
        f"Validation passed: "
        f"{final_validation_passed}"
    )

    print()

    print(
        f"Final manifest:"
        f"\n{FINAL_MANIFEST}"
    )

    print()

    print(
        f"Summary:"
        f"\n{FINAL_SUMMARY}"
    )

    print("=" * 90)

    # --------------------------------------------------------
    # HARD FAILURE
    # --------------------------------------------------------

    if (
        args.limit is None
        and
        not final_validation_passed
    ):

        raise RuntimeError(
            "Final registration validation FAILED. "
            "Do not continue to Step 6C."
        )


if __name__ == "__main__":
    main()
