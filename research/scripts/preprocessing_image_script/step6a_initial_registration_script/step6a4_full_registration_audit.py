from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# PATHS
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

STEP6A_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6a_initial_registration_data/step6a1_full_rigid_registration"
)

REGISTRATION_CSV = (
    STEP6A_DIR
    / "registration_qc.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6a_initial_registration_data/step6a4_full_registration_audit"
)

TRANSFORM_OUTPUT_DIR = (
    OUTPUT_DIR
    / "transforms"
)

AUDIT_CSV = (
    OUTPUT_DIR
    / "registration_audit.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "registration_audit_summary.json"
)


# ============================================================
# QC THRESHOLDS
# ============================================================

HIGH_CONFIDENCE_DICE = 0.70
REVIEW_DICE = 0.50


# ============================================================
# BASIC HELPERS
# ============================================================

def foreground_mask(
    image: sitk.Image,
) -> sitk.Image:
    """
    Same foreground definition used in Step 6A.

    Keep this unchanged so Dice values remain
    directly comparable with Step 6A.
    """

    return sitk.Cast(
        image > 0,
        sitk.sitkUInt8,
    )


def qc_category(
    dice: float,
) -> str:

    if dice >= HIGH_CONFIDENCE_DICE:
        return "high_confidence"

    if dice >= REVIEW_DICE:
        return "review"

    return "failed"


def dice_score(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    transform: sitk.Transform,
) -> float:
    """
    Calculate foreground-mask Dice after moving
    the subject mask to template space.
    """

    moved_mask = sitk.Resample(
        moving_mask,
        fixed_mask,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    fixed_array = (
        sitk.GetArrayViewFromImage(
            fixed_mask
        ) > 0
    )

    moved_array = (
        sitk.GetArrayViewFromImage(
            moved_mask
        ) > 0
    )

    intersection = np.count_nonzero(
        fixed_array & moved_array
    )

    denominator = (
        np.count_nonzero(fixed_array)
        +
        np.count_nonzero(moved_array)
    )

    if denominator == 0:
        return 0.0

    return float(
        2.0
        * intersection
        / denominator
    )


# ============================================================
# INITIAL TRANSFORMS
# ============================================================

def create_initial_transform(
    fixed: sitk.Image,
    moving: sitk.Image,
    initialization: str,
) -> sitk.Transform:

    if initialization == "moments":

        mode = (
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )

    elif initialization == "geometry":

        mode = (
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

    else:

        raise ValueError(
            f"Unknown initialization: {initialization}"
        )

    transform = (
        sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            mode,
        )
    )

    return transform


# ============================================================
# PATH HELPERS
# ============================================================

def get_step6a_transform_path(
    uid: str,
    name: str,
) -> Path:

    return (
        STEP6A_DIR
        / "transforms"
        / uid
        / f"{name}.h5"
    )


# ============================================================
# PROCESS ONE SCAN
# ============================================================

def process_scan(
    row: pd.Series,
    fixed: sitk.Image,
    fixed_mask: sitk.Image,
) -> dict:

    uid = str(
        row["uid"]
    )

    file_name = str(
        row["file_name"]
    )

    moving_path = (
        INPUT_DIR
        / file_name
    )

    if not moving_path.exists():

        raise FileNotFoundError(
            f"Moving image missing: "
            f"{moving_path}"
        )

    # --------------------------------------------------------
    # Read moving scan
    # --------------------------------------------------------

    moving = sitk.ReadImage(
        str(moving_path),
        sitk.sitkFloat32,
    )

    moving_mask = foreground_mask(
        moving
    )

    # ========================================================
    # Candidate 1 — MOMENTS INITIAL
    # ========================================================

    moments_initial = (
        create_initial_transform(
            fixed=fixed,
            moving=moving,
            initialization="moments",
        )
    )

    moments_initial_dice = (
        dice_score(
            fixed_mask,
            moving_mask,
            moments_initial,
        )
    )

    # ========================================================
    # Candidate 2 — GEOMETRY INITIAL
    # ========================================================

    geometry_initial = (
        create_initial_transform(
            fixed=fixed,
            moving=moving,
            initialization="geometry",
        )
    )

    geometry_initial_dice = (
        dice_score(
            fixed_mask,
            moving_mask,
            geometry_initial,
        )
    )

    # ========================================================
    # Candidate 3 — MOMENTS OPTIMIZED
    # ========================================================

    moments_optimized_path = (
        get_step6a_transform_path(
            uid,
            "moments",
        )
    )

    if not moments_optimized_path.exists():

        raise FileNotFoundError(
            f"Missing MOMENTS optimized transform: "
            f"{moments_optimized_path}"
        )

    moments_optimized = (
        sitk.ReadTransform(
            str(
                moments_optimized_path
            )
        )
    )

    moments_optimized_dice = (
        dice_score(
            fixed_mask,
            moving_mask,
            moments_optimized,
        )
    )

    # ========================================================
    # Candidate 4 — GEOMETRY OPTIMIZED
    # ========================================================

    geometry_optimized_path = (
        get_step6a_transform_path(
            uid,
            "geometry",
        )
    )

    if not geometry_optimized_path.exists():

        raise FileNotFoundError(
            f"Missing GEOMETRY optimized transform: "
            f"{geometry_optimized_path}"
        )

    geometry_optimized = (
        sitk.ReadTransform(
            str(
                geometry_optimized_path
            )
        )
    )

    geometry_optimized_dice = (
        dice_score(
            fixed_mask,
            moving_mask,
            geometry_optimized,
        )
    )

    # ========================================================
    # Candidate dictionary
    # ========================================================

    candidates = {

        "moments_initial": {
            "transform":
                moments_initial,

            "dice":
                moments_initial_dice,
        },

        "geometry_initial": {
            "transform":
                geometry_initial,

            "dice":
                geometry_initial_dice,
        },

        "moments_optimized": {
            "transform":
                moments_optimized,

            "dice":
                moments_optimized_dice,
        },

        "geometry_optimized": {
            "transform":
                geometry_optimized,

            "dice":
                geometry_optimized_dice,
        },
    }

    # ========================================================
    # Best initial / optimized
    # ========================================================

    best_initial_dice = max(
        moments_initial_dice,
        geometry_initial_dice,
    )

    best_optimized_dice = max(
        moments_optimized_dice,
        geometry_optimized_dice,
    )

    # ========================================================
    # Best of all 4
    # ========================================================

    selected_candidate = max(
        candidates,
        key=lambda name:
            candidates[name]["dice"],
    )

    selected_transform = (
        candidates[
            selected_candidate
        ]["transform"]
    )

    audit_dice = float(
        candidates[
            selected_candidate
        ]["dice"]
    )

    # --------------------------------------------------------
    # Step 6A old result
    # --------------------------------------------------------

    old_dice = float(
        row["selected_dice"]
    )

    old_initialization = str(
        row["selected_initialization"]
    )

    old_candidate = (
        f"{old_initialization}_optimized"
    )

    old_qc = str(
        row["qc_category"]
    )

    audit_qc = qc_category(
        audit_dice
    )

    dice_gain = (
        audit_dice
        -
        old_dice
    )

    initial_gain_over_optimized = (
        best_initial_dice
        -
        best_optimized_dice
    )

    initial_beats_optimized = bool(
        best_initial_dice
        >
        best_optimized_dice
    )

    changed_from_step6a = bool(
        selected_candidate
        != old_candidate
    )

    # ========================================================
    # Save selected audit transform
    # ========================================================

    uid_output_dir = (
        TRANSFORM_OUTPUT_DIR
        / uid
    )

    uid_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_transform_path = (
        uid_output_dir
        / "selected_audit.h5"
    )

    sitk.WriteTransform(
        selected_transform,
        str(
            selected_transform_path
        ),
    )

    # ========================================================
    # Result
    # ========================================================

    return {

        "uid":
            uid,

        "file_name":
            file_name,

        # --------------------------------------------
        # Step 6A
        # --------------------------------------------

        "step6a_candidate":
            old_candidate,

        "step6a_dice":
            old_dice,

        "step6a_qc":
            old_qc,

        # --------------------------------------------
        # Four candidates
        # --------------------------------------------

        "moments_initial_dice":
            moments_initial_dice,

        "geometry_initial_dice":
            geometry_initial_dice,

        "moments_optimized_dice":
            moments_optimized_dice,

        "geometry_optimized_dice":
            geometry_optimized_dice,

        # --------------------------------------------
        # Initial vs optimized
        # --------------------------------------------

        "best_initial_dice":
            best_initial_dice,

        "best_optimized_dice":
            best_optimized_dice,

        "initial_beats_optimized":
            initial_beats_optimized,

        "initial_gain_over_optimized":
            initial_gain_over_optimized,

        # --------------------------------------------
        # Audit result
        # --------------------------------------------

        "audit_selected_candidate":
            selected_candidate,

        "audit_dice":
            audit_dice,

        "audit_qc":
            audit_qc,

        "dice_gain_vs_step6a":
            dice_gain,

        "changed_from_step6a":
            changed_from_step6a,

        # --------------------------------------------
        # Helpful review flags
        # --------------------------------------------

        "gain_gt_0_01":
            bool(
                dice_gain > 0.01
            ),

        "gain_gt_0_05":
            bool(
                dice_gain > 0.05
            ),

        "gain_gt_0_10":
            bool(
                dice_gain > 0.10
            ),

        "needs_visual_review":
            bool(
                audit_dice
                <
                HIGH_CONFIDENCE_DICE
            ),

        # --------------------------------------------
        # Saved transform
        # --------------------------------------------

        "selected_audit_transform":
            str(
                selected_transform_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "status":
            "success",

        "error":
            "",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 6B-3: audit all registrations "
            "by comparing initial and optimized "
            "MOMENTS/GEOMETRY transforms."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only first N scans. "
            "Useful for technical testing."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Validate paths
    # ========================================================

    for required_path in [
        INPUT_DIR,
        TEMPLATE_PATH,
        REGISTRATION_CSV,
    ]:

        if not required_path.exists():

            raise FileNotFoundError(
                f"Required path not found: "
                f"{required_path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRANSFORM_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Read Step 6A CSV
    # ========================================================

    df = pd.read_csv(
        REGISTRATION_CSV,
        dtype={
            "uid": str,
            "file_name": str,
        },
    )

    df["selected_dice"] = (
        pd.to_numeric(
            df["selected_dice"],
            errors="raise",
        )
    )

    # Only technically successful Step-6A runs
    if "status" in df.columns:

        df = df[
            df["status"] == "success"
        ].copy()

    df = df.reset_index(
        drop=True
    )

    if args.limit is not None:

        df = df.head(
            args.limit
        ).copy()

    # ========================================================
    # Load template once
    # ========================================================

    fixed = sitk.ReadImage(
        str(TEMPLATE_PATH),
        sitk.sitkFloat32,
    )

    fixed_mask = foreground_mask(
        fixed
    )

    print("=" * 84)

    print(
        "STEP 6B-3 — FULL REGISTRATION "
        "INITIAL-vs-OPTIMIZED AUDIT"
    )

    print("=" * 84)

    print(
        f"Input scans: "
        f"{len(df)}"
    )

    print(
        f"Template: "
        f"{TEMPLATE_PATH}"
    )

    print(
        f"Step 6A CSV: "
        f"{REGISTRATION_CSV}"
    )

    print(
        f"Output: "
        f"{OUTPUT_DIR}"
    )

    print("=" * 84)

    results = []
    errors = []

    # ========================================================
    # Process
    # ========================================================

    total = len(df)

    for index, row in df.iterrows():

        uid = str(
            row["uid"]
        )

        print(
            f"[{index + 1}/{total}] "
            f"{uid}"
        )

        try:

            result = process_scan(
                row=row,
                fixed=fixed,
                fixed_mask=fixed_mask,
            )

            results.append(
                result
            )

            print(
                f"    Step6A: "
                f"{result['step6a_candidate']} "
                f"Dice={result['step6a_dice']:.4f}"
            )

            print(
                f"    MOM init: "
                f"{result['moments_initial_dice']:.4f}"
            )

            print(
                f"    GEO init: "
                f"{result['geometry_initial_dice']:.4f}"
            )

            print(
                f"    MOM opt:  "
                f"{result['moments_optimized_dice']:.4f}"
            )

            print(
                f"    GEO opt:  "
                f"{result['geometry_optimized_dice']:.4f}"
            )

            print(
                f"    Audit: "
                f"{result['audit_selected_candidate']} "
                f"Dice={result['audit_dice']:.4f}"
            )

            print(
                f"    Gain: "
                f"{result['dice_gain_vs_step6a']:+.4f}"
            )

            print(
                f"    QC: "
                f"{result['audit_qc']}"
            )

        except Exception as exc:

            message = str(exc)

            print(
                f"    ERROR: "
                f"{message}"
            )

            errors.append(
                {
                    "uid": uid,
                    "error": message,
                }
            )

            results.append(
                {
                    "uid":
                        uid,

                    "file_name":
                        str(
                            row["file_name"]
                        ),

                    "status":
                        "failed",

                    "error":
                        message,
                }
            )

    # ========================================================
    # Save audit CSV
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        AUDIT_CSV,
        index=False,
    )

    # ========================================================
    # Build summary from successful cases
    # ========================================================

    successful_df = result_df[
        result_df["status"]
        ==
        "success"
    ].copy()

    if len(successful_df) > 0:

        candidate_counts = (
            successful_df[
                "audit_selected_candidate"
            ]
            .value_counts()
            .to_dict()
        )

        qc_counts = (
            successful_df[
                "audit_qc"
            ]
            .value_counts()
            .to_dict()
        )

        changed_count = int(
            successful_df[
                "changed_from_step6a"
            ].sum()
        )

        initial_selected_count = int(
            successful_df[
                "audit_selected_candidate"
            ]
            .isin(
                [
                    "moments_initial",
                    "geometry_initial",
                ]
            )
            .sum()
        )

        optimized_selected_count = int(
            successful_df[
                "audit_selected_candidate"
            ]
            .isin(
                [
                    "moments_optimized",
                    "geometry_optimized",
                ]
            )
            .sum()
        )

        gain_gt_001 = int(
            successful_df[
                "gain_gt_0_01"
            ].sum()
        )

        gain_gt_005 = int(
            successful_df[
                "gain_gt_0_05"
            ].sum()
        )

        gain_gt_010 = int(
            successful_df[
                "gain_gt_0_10"
            ].sum()
        )

        remaining_review = (
            successful_df[
                successful_df[
                    "audit_qc"
                ]
                ==
                "review"
            ]
        )

        remaining_failed = (
            successful_df[
                successful_df[
                    "audit_qc"
                ]
                ==
                "failed"
            ]
        )

        old_failed = successful_df[
            successful_df[
                "step6a_qc"
            ]
            ==
            "failed"
        ]

        rescued_old_failures = int(
            (
                old_failed[
                    "audit_qc"
                ]
                !=
                "failed"
            ).sum()
        )

        old_review = successful_df[
            successful_df[
                "step6a_qc"
            ]
            ==
            "review"
        ]

        review_to_high_confidence = int(
            (
                old_review[
                    "audit_qc"
                ]
                ==
                "high_confidence"
            ).sum()
        )

        audit_dice_values = (
            successful_df[
                "audit_dice"
            ]
            .astype(float)
            .to_numpy()
        )

        dice_summary = {

            "min":
                float(
                    np.min(
                        audit_dice_values
                    )
                ),

            "p05":
                float(
                    np.percentile(
                        audit_dice_values,
                        5,
                    )
                ),

            "median":
                float(
                    np.median(
                        audit_dice_values
                    )
                ),

            "p95":
                float(
                    np.percentile(
                        audit_dice_values,
                        95,
                    )
                ),

            "max":
                float(
                    np.max(
                        audit_dice_values
                    )
                ),
        }

        remaining_review_uids = (
            remaining_review[
                "uid"
            ]
            .astype(str)
            .tolist()
        )

        remaining_failed_uids = (
            remaining_failed[
                "uid"
            ]
            .astype(str)
            .tolist()
        )

    else:

        candidate_counts = {}
        qc_counts = {}

        changed_count = 0

        initial_selected_count = 0
        optimized_selected_count = 0

        gain_gt_001 = 0
        gain_gt_005 = 0
        gain_gt_010 = 0

        rescued_old_failures = 0
        review_to_high_confidence = 0

        dice_summary = {}

        remaining_review_uids = []
        remaining_failed_uids = []

    # ========================================================
    # Summary JSON
    # ========================================================

    summary = {

        "analysis":
            (
                "Step 6B-3 full registration "
                "initial-vs-optimized safety audit"
            ),

        "number_of_scans_requested":
            int(
                len(df)
            ),

        "successful":
            int(
                len(successful_df)
            ),

        "failed_execution":
            int(
                len(errors)
            ),

        "audit_selection_rule":
            (
                "Maximum Dice among "
                "MOMENTS initial, "
                "GEOMETRY initial, "
                "MOMENTS optimized, "
                "GEOMETRY optimized."
            ),

        "selected_candidate_counts":
            candidate_counts,

        "audit_qc_counts":
            qc_counts,

        "changed_from_step6a":
            changed_count,

        "initial_transform_selected":
            initial_selected_count,

        "optimized_transform_selected":
            optimized_selected_count,

        "meaningful_improvements": {

            "gain_gt_0.01":
                gain_gt_001,

            "gain_gt_0.05":
                gain_gt_005,

            "gain_gt_0.10":
                gain_gt_010,
        },

        "step6a_failed_rescued_to_review_or_better":
            rescued_old_failures,

        "step6a_review_upgraded_to_high_confidence":
            review_to_high_confidence,

        "audit_dice":
            dice_summary,

        "remaining_review_count":
            len(
                remaining_review_uids
            ),

        "remaining_review_uids":
            remaining_review_uids,

        "remaining_failed_count":
            len(
                remaining_failed_uids
            ),

        "remaining_failed_uids":
            remaining_failed_uids,

        "qc_thresholds": {

            "high_confidence":
                "Dice >= 0.70",

            "review":
                "0.50 <= Dice < 0.70",

            "failed":
                "Dice < 0.50",
        },

        "errors":
            errors,

        "important_note":
            (
                "Audit-selected transforms are "
                "saved separately and do not "
                "overwrite Step 6A transforms."
            ),
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ========================================================
    # Finish
    # ========================================================

    print()

    print("=" * 84)

    print(
        "STEP 6B-3 AUDIT COMPLETED"
    )

    print("=" * 84)

    print(
        f"Audit CSV: "
        f"{AUDIT_CSV}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_JSON}"
    )

    print(
        f"Audit transforms: "
        f"{TRANSFORM_OUTPUT_DIR}"
    )

    print(
        f"Successful: "
        f"{len(successful_df)}"
    )

    print(
        f"Errors: "
        f"{len(errors)}"
    )

    if len(successful_df) > 0:

        print(
            f"Changed from Step 6A: "
            f"{changed_count}"
        )

        print(
            f"Initial selected: "
            f"{initial_selected_count}"
        )

        print(
            f"Optimized selected: "
            f"{optimized_selected_count}"
        )

        print(
            f"Remaining review: "
            f"{len(remaining_review_uids)}"
        )

        print(
            f"Remaining failed: "
            f"{len(remaining_failed_uids)}"
        )

        if remaining_failed_uids:

            print(
                "Remaining failed UIDs: "
                + ", ".join(
                    remaining_failed_uids
                )
            )

    print("=" * 84)


if __name__ == "__main__":
    main()
