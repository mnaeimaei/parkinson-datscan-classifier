from __future__ import annotations

import argparse
import json
import shutil
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

STEP6B3_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6a_initial_registration_data/step6a4_full_registration_audit"
)

AUDIT_CSV = (
    STEP6B3_DIR
    / "registration_audit.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6b_registration_rescue_data/step6b1_finalize_registration_transforms"
)

FINAL_TRANSFORM_DIR = (
    OUTPUT_DIR
    / "transforms"
)

FINAL_MANIFEST_CSV = (
    OUTPUT_DIR
    / "final_transform_manifest.csv"
)

MANUAL_REVIEW_CSV = (
    OUTPUT_DIR
    / "manual_review_required.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "final_transform_summary.json"
)


# ============================================================
# POLICY
# ============================================================

HIGH_CONFIDENCE_DICE = 0.70
FAILED_DICE = 0.50

# Small differences in whole-volume foreground Dice
# are not considered sufficient evidence to replace
# an intensity-optimized registration.
SMALL_GAIN_THRESHOLD = 0.05

# Large improvements must be visually reviewed before
# replacing an otherwise non-catastrophic optimized transform.
MANUAL_REVIEW_GAIN_THRESHOLD = 0.10


# ============================================================
# HELPERS
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

    return sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        mode,
    )


def get_best_initial_candidate(
    row: pd.Series,
) -> tuple[str, float]:

    moments = float(
        row["moments_initial_dice"]
    )

    geometry = float(
        row["geometry_initial_dice"]
    )

    if moments >= geometry:

        return (
            "moments_initial",
            moments,
        )

    return (
        "geometry_initial",
        geometry,
    )


def get_best_optimized_candidate(
    row: pd.Series,
) -> tuple[str, float]:

    moments = float(
        row["moments_optimized_dice"]
    )

    geometry = float(
        row["geometry_optimized_dice"]
    )

    if moments >= geometry:

        return (
            "moments_optimized",
            moments,
        )

    return (
        "geometry_optimized",
        geometry,
    )


def qc_category(
    dice: float,
) -> str:

    if dice >= 0.70:
        return "high_confidence"

    if dice >= 0.50:
        return "review"

    return "failed"


def get_optimized_transform_path(
    uid: str,
    candidate: str,
) -> Path:

    if candidate == "moments_optimized":

        filename = "moments.h5"

    elif candidate == "geometry_optimized":

        filename = "geometry.h5"

    else:

        raise ValueError(
            f"Not an optimized candidate: "
            f"{candidate}"
        )

    return (
        STEP6A_DIR
        / "transforms"
        / uid
        / filename
    )


# ============================================================
# POLICY DECISION
# ============================================================

def decide_transform(
    row: pd.Series,
) -> dict:

    best_initial_name, best_initial_dice = (
        get_best_initial_candidate(row)
    )

    best_optimized_name, best_optimized_dice = (
        get_best_optimized_candidate(row)
    )

    gain = (
        best_initial_dice
        -
        best_optimized_dice
    )

    # --------------------------------------------------------
    # CASE 1
    # Catastrophic optimizer failure
    #
    # Optimized result is bad but initial transform
    # already has good foreground overlap.
    # --------------------------------------------------------

    if (
        best_optimized_dice < FAILED_DICE
        and
        best_initial_dice >= HIGH_CONFIDENCE_DICE
    ):

        return {

            "selection_status":
                "auto_final",

            "selection_reason":
                "catastrophic_optimizer_failure",

            "selected_candidate":
                best_initial_name,

            "selected_dice":
                best_initial_dice,

            "gain":
                gain,
        }

    # --------------------------------------------------------
    # CASE 2
    # Still unresolved after all four candidates
    # --------------------------------------------------------

    if (
        max(
            best_initial_dice,
            best_optimized_dice,
        )
        <
        FAILED_DICE
    ):

        return {

            "selection_status":
                "unresolved",

            "selection_reason":
                "all_candidates_failed",

            "selected_candidate":
                "",

            "selected_dice":
                max(
                    best_initial_dice,
                    best_optimized_dice,
                ),

            "gain":
                gain,
        }

    # --------------------------------------------------------
    # CASE 3
    # Initial transform substantially better.
    #
    # Require visual confirmation rather than
    # automatically replacing optimization.
    # --------------------------------------------------------

    if gain >= MANUAL_REVIEW_GAIN_THRESHOLD:

        return {

            "selection_status":
                "manual_review",

            "selection_reason":
                "initial_gain_ge_0.10",

            "selected_candidate":
                "",

            "selected_dice":
                max(
                    best_initial_dice,
                    best_optimized_dice,
                ),

            "gain":
                gain,
        }

    # --------------------------------------------------------
    # CASE 4
    # Moderate gain: 0.05 to <0.10
    #
    # Keep optimized transform conservatively,
    # but flag it for optional QC.
    # --------------------------------------------------------

    if gain >= SMALL_GAIN_THRESHOLD:

        return {

            "selection_status":
                "auto_final_review_recommended",

            "selection_reason":
                "moderate_initial_gain_keep_optimized",

            "selected_candidate":
                best_optimized_name,

            "selected_dice":
                best_optimized_dice,

            "gain":
                gain,
        }

    # --------------------------------------------------------
    # CASE 5
    # Small / negligible difference
    #
    # Keep image-intensity optimized transform.
    # --------------------------------------------------------

    return {

        "selection_status":
            "auto_final",

        "selection_reason":
            "small_gain_keep_optimized",

        "selected_candidate":
            best_optimized_name,

        "selected_dice":
            best_optimized_dice,

        "gain":
            gain,
    }


# ============================================================
# SAVE FINAL TRANSFORM
# ============================================================

def save_selected_transform(
    uid: str,
    file_name: str,
    candidate: str,
    fixed: sitk.Image,
) -> Path:

    uid_output_dir = (
        FINAL_TRANSFORM_DIR
        / uid
    )

    uid_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        uid_output_dir
        / "final_transform.h5"
    )

    # --------------------------------------------------------
    # Optimized transform
    # --------------------------------------------------------

    if candidate in (
        "moments_optimized",
        "geometry_optimized",
    ):

        source_path = (
            get_optimized_transform_path(
                uid,
                candidate,
            )
        )

        if not source_path.exists():

            raise FileNotFoundError(
                f"Missing optimized transform: "
                f"{source_path}"
            )

        shutil.copy2(
            source_path,
            output_path,
        )

        return output_path

    # --------------------------------------------------------
    # Initial transform
    #
    # Reconstruct exactly from fixed + moving
    # --------------------------------------------------------

    if candidate in (
        "moments_initial",
        "geometry_initial",
    ):

        moving_path = (
            INPUT_DIR
            / file_name
        )

        if not moving_path.exists():

            raise FileNotFoundError(
                f"Moving image missing: "
                f"{moving_path}"
            )

        moving = sitk.ReadImage(
            str(moving_path),
            sitk.sitkFloat32,
        )

        initialization = (
            "moments"
            if candidate == "moments_initial"
            else "geometry"
        )

        transform = (
            create_initial_transform(
                fixed=fixed,
                moving=moving,
                initialization=initialization,
            )
        )

        sitk.WriteTransform(
            transform,
            str(output_path),
        )

        return output_path

    raise ValueError(
        f"Unknown candidate: "
        f"{candidate}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 6B-4 - conservative final "
            "registration transform selection."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional technical-test limit.",
    )

    args = parser.parse_args()

    # ========================================================
    # Validate
    # ========================================================

    for required in [
        INPUT_DIR,
        TEMPLATE_PATH,
        AUDIT_CSV,
    ]:

        if not required.exists():

            raise FileNotFoundError(
                f"Missing required path: "
                f"{required}"
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
    # Read audit
    # ========================================================

    df = pd.read_csv(
        AUDIT_CSV,
        dtype={
            "uid": str,
            "file_name": str,
        },
    )

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

    fixed = sitk.ReadImage(
        str(TEMPLATE_PATH),
        sitk.sitkFloat32,
    )

    print("=" * 86)

    print(
        "STEP 6B-4 — FINAL REGISTRATION "
        "TRANSFORM SELECTION"
    )

    print("=" * 86)

    print(
        f"Scans: {len(df)}"
    )

    print("=" * 86)

    output_rows = []

    errors = []

    # ========================================================
    # Process
    # ========================================================

    for index, row in df.iterrows():

        uid = str(
            row["uid"]
        )

        file_name = str(
            row["file_name"]
        )

        decision = decide_transform(
            row
        )

        best_initial_name, best_initial_dice = (
            get_best_initial_candidate(
                row
            )
        )

        best_optimized_name, best_optimized_dice = (
            get_best_optimized_candidate(
                row
            )
        )

        print(
            f"[{index + 1}/{len(df)}] "
            f"{uid}"
        )

        print(
            f"    best initial:   "
            f"{best_initial_name} "
            f"{best_initial_dice:.4f}"
        )

        print(
            f"    best optimized: "
            f"{best_optimized_name} "
            f"{best_optimized_dice:.4f}"
        )

        print(
            f"    gain: "
            f"{decision['gain']:+.4f}"
        )

        print(
            f"    status: "
            f"{decision['selection_status']}"
        )

        print(
            f"    reason: "
            f"{decision['selection_reason']}"
        )

        final_transform_path = ""

        save_status = "not_saved"

        error_message = ""

        # ----------------------------------------------------
        # Save transform only when automatically final
        # ----------------------------------------------------

        if decision[
            "selection_status"
        ] in (
            "auto_final",
            "auto_final_review_recommended",
        ):

            try:

                saved_path = (
                    save_selected_transform(
                        uid=uid,
                        file_name=file_name,
                        candidate=decision[
                            "selected_candidate"
                        ],
                        fixed=fixed,
                    )
                )

                final_transform_path = (
                    str(
                        saved_path.relative_to(
                            PROJECT_ROOT
                        )
                    )
                )

                save_status = "saved"

            except Exception as exc:

                error_message = str(exc)

                save_status = "error"

                errors.append(
                    {
                        "uid": uid,
                        "error": error_message,
                    }
                )

        output_rows.append(
            {
                "uid":
                    uid,

                "file_name":
                    file_name,

                # ----------------------------------------
                # Step 6A
                # ----------------------------------------

                "step6a_candidate":
                    row[
                        "step6a_candidate"
                    ],

                "step6a_dice":
                    float(
                        row[
                            "step6a_dice"
                        ]
                    ),

                # ----------------------------------------
                # Initial candidate
                # ----------------------------------------

                "best_initial_candidate":
                    best_initial_name,

                "best_initial_dice":
                    best_initial_dice,

                # ----------------------------------------
                # Optimized candidate
                # ----------------------------------------

                "best_optimized_candidate":
                    best_optimized_name,

                "best_optimized_dice":
                    best_optimized_dice,

                # ----------------------------------------
                # Comparison
                # ----------------------------------------

                "initial_gain_over_optimized":
                    decision["gain"],

                # ----------------------------------------
                # Final policy decision
                # ----------------------------------------

                "selection_status":
                    decision[
                        "selection_status"
                    ],

                "selection_reason":
                    decision[
                        "selection_reason"
                    ],

                "selected_candidate":
                    decision[
                        "selected_candidate"
                    ],

                "selected_dice":
                    decision[
                        "selected_dice"
                    ],

                "selected_qc":
                    qc_category(
                        float(
                            decision[
                                "selected_dice"
                            ]
                        )
                    ),

                "final_transform":
                    final_transform_path,

                "transform_save_status":
                    save_status,

                # ----------------------------------------
                # Manual fields
                # ----------------------------------------

                "manual_decision":
                    "",

                "manual_selected_candidate":
                    "",

                "manual_notes":
                    "",

                "error":
                    error_message,
            }
        )

    # ========================================================
    # Save manifest
    # ========================================================

    final_df = pd.DataFrame(
        output_rows
    )

    final_df.to_csv(
        FINAL_MANIFEST_CSV,
        index=False,
    )

    # ========================================================
    # Manual review subset
    # ========================================================

    manual_df = final_df[
        final_df[
            "selection_status"
        ].isin(
            [
                "manual_review",
                "unresolved",
                "auto_final_review_recommended",
            ]
        )
    ].copy()

    manual_df = (
        manual_df.sort_values(
            [
                "selection_status",
                "initial_gain_over_optimized",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    manual_df.to_csv(
        MANUAL_REVIEW_CSV,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    status_counts = (
        final_df[
            "selection_status"
        ]
        .value_counts()
        .to_dict()
    )

    reason_counts = (
        final_df[
            "selection_reason"
        ]
        .value_counts()
        .to_dict()
    )

    candidate_counts = (
        final_df[
            "selected_candidate"
        ]
        .replace(
            "",
            np.nan,
        )
        .dropna()
        .value_counts()
        .to_dict()
    )

    auto_final_mask = (
        final_df[
            "selection_status"
        ].isin(
            [
                "auto_final",
                "auto_final_review_recommended",
            ]
        )
    )

    automatic_final_count = int(
        auto_final_mask.sum()
    )

    manual_required_count = int(
        (
            final_df[
                "selection_status"
            ]
            ==
            "manual_review"
        ).sum()
    )

    unresolved_count = int(
        (
            final_df[
                "selection_status"
            ]
            ==
            "unresolved"
        ).sum()
    )

    catastrophic_rescues = int(
        (
            final_df[
                "selection_reason"
            ]
            ==
            "catastrophic_optimizer_failure"
        ).sum()
    )

    moderate_review_count = int(
        (
            final_df[
                "selection_status"
            ]
            ==
            "auto_final_review_recommended"
        ).sum()
    )

    summary = {

        "analysis":
            (
                "Step 6B-4 conservative final "
                "registration transform selection"
            ),

        "number_of_scans":
            int(
                len(final_df)
            ),

        "automatic_final_count":
            automatic_final_count,

        "manual_review_required":
            manual_required_count,

        "review_recommended_but_optimized_kept":
            moderate_review_count,

        "unresolved_count":
            unresolved_count,

        "catastrophic_optimizer_failures_auto_rescued":
            catastrophic_rescues,

        "selection_status_counts":
            status_counts,

        "selection_reason_counts":
            reason_counts,

        "automatic_selected_candidate_counts":
            candidate_counts,

        "policy": {

            "catastrophic_failure":
                (
                    "If best optimized Dice < 0.50 "
                    "and best initial Dice >= 0.70, "
                    "automatically use best initial."
                ),

            "unresolved":
                (
                    "If all candidates have Dice "
                    "< 0.50, do not finalize."
                ),

            "large_gain":
                (
                    "If initial gain >= 0.10, "
                    "require manual visual review."
                ),

            "moderate_gain":
                (
                    "If 0.05 <= initial gain < 0.10, "
                    "keep optimized transform but "
                    "recommend visual QC."
                ),

            "small_gain":
                (
                    "If initial gain < 0.05, "
                    "keep best optimized transform."
                ),
        },

        "output_manifest":
            str(
                FINAL_MANIFEST_CSV.relative_to(
                    PROJECT_ROOT
                )
            ),

        "manual_review_csv":
            str(
                MANUAL_REVIEW_CSV.relative_to(
                    PROJECT_ROOT
                )
            ),

        "important_note":
            (
                "Step 6C must not be started until "
                "manual_review and unresolved cases "
                "have been resolved and every scan "
                "has exactly one final transform."
            ),

        "errors":
            errors,
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
    # Terminal summary
    # ========================================================

    print()

    print("=" * 86)

    print(
        "STEP 6B-4 COMPLETED"
    )

    print("=" * 86)

    print(
        f"Manifest: "
        f"{FINAL_MANIFEST_CSV}"
    )

    print(
        f"Manual review: "
        f"{MANUAL_REVIEW_CSV}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_JSON}"
    )

    print()

    print(
        f"Automatic final: "
        f"{automatic_final_count}"
    )

    print(
        f"Manual review required: "
        f"{manual_required_count}"
    )

    print(
        f"Review recommended: "
        f"{moderate_review_count}"
    )

    print(
        f"Unresolved: "
        f"{unresolved_count}"
    )

    print(
        f"Catastrophic rescues: "
        f"{catastrophic_rescues}"
    )

    print(
        f"Errors: "
        f"{len(errors)}"
    )

    print("=" * 86)


if __name__ == "__main__":
    main()
