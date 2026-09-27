from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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

STEP6B4_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6b_registration_rescue_data/step6b1_finalize_registration_transforms"
)

MANUAL_REVIEW_CSV = (
    STEP6B4_DIR
    / "manual_review_required.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6b_registration_rescue_data/step6b2_manual_registration_qc"
)

QC_IMAGE_DIR = (
    OUTPUT_DIR
    / "qc_images"
)

DECISION_CSV = (
    OUTPUT_DIR
    / "manual_registration_decisions.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "manual_qc_summary.json"
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

    return sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        mode,
    )


def get_initial_transform(
    fixed: sitk.Image,
    moving: sitk.Image,
    candidate: str,
) -> sitk.Transform:

    if candidate == "moments_initial":

        return create_initial_transform(
            fixed,
            moving,
            "moments",
        )

    if candidate == "geometry_initial":

        return create_initial_transform(
            fixed,
            moving,
            "geometry",
        )

    raise ValueError(
        f"Invalid initial candidate: {candidate}"
    )


# ============================================================
# OPTIMIZED TRANSFORMS
# ============================================================

def get_optimized_transform(
    uid: str,
    candidate: str,
) -> sitk.Transform:

    if candidate == "moments_optimized":

        filename = "moments.h5"

    elif candidate == "geometry_optimized":

        filename = "geometry.h5"

    else:

        raise ValueError(
            f"Invalid optimized candidate: "
            f"{candidate}"
        )

    transform_path = (
        STEP6A_DIR
        / "transforms"
        / uid
        / filename
    )

    if not transform_path.exists():

        raise FileNotFoundError(
            f"Transform not found: "
            f"{transform_path}"
        )

    return sitk.ReadTransform(
        str(transform_path)
    )


# ============================================================
# IMAGE HELPERS
# ============================================================

def resample_to_template(
    moving: sitk.Image,
    fixed: sitk.Image,
    transform: sitk.Transform,
) -> sitk.Image:

    return sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def robust_normalize(
    array: np.ndarray,
) -> np.ndarray:

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    valid = np.isfinite(array)

    positive = array[
        valid & (array > 0)
    ]

    if positive.size >= 10:

        lo, hi = np.percentile(
            positive,
            [1.0, 99.5],
        )

    else:

        finite = array[valid]

        if finite.size == 0:

            return np.zeros_like(
                array,
                dtype=np.float32,
            )

        lo, hi = np.percentile(
            finite,
            [1.0, 99.5],
        )

    if (
        not np.isfinite(lo)
        or not np.isfinite(hi)
        or hi <= lo
    ):

        return np.zeros_like(
            array,
            dtype=np.float32,
        )

    normalized = (
        array - lo
    ) / (
        hi - lo
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    )


def foreground_center(
    array: np.ndarray,
) -> tuple[int, int, int]:

    mask = (
        np.isfinite(array)
        & (array > 0)
    )

    if np.count_nonzero(mask) > 0:

        coordinates = np.argwhere(
            mask
        )

        z, y, x = np.round(
            coordinates.mean(axis=0)
        ).astype(int)

    else:

        z, y, x = (
            np.asarray(array.shape)
            // 2
        )

    z = int(
        np.clip(
            z,
            0,
            array.shape[0] - 1,
        )
    )

    y = int(
        np.clip(
            y,
            0,
            array.shape[1] - 1,
        )
    )

    x = int(
        np.clip(
            x,
            0,
            array.shape[2] - 1,
        )
    )

    return z, y, x


def extract_views(
    array: np.ndarray,
    indices: tuple[int, int, int],
) -> dict[str, np.ndarray]:

    z, y, x = indices

    return {
        "Axial":
            np.rot90(
                array[z, :, :]
            ),

        "Coronal":
            np.rot90(
                array[:, y, :]
            ),

        "Sagittal":
            np.rot90(
                array[:, :, x]
            ),
    }


# ============================================================
# CREATE QC FIGURE
# ============================================================

def make_comparison_figure(
    uid: str,
    moving: sitk.Image,
    fixed: sitk.Image,
    initial_registered: sitk.Image,
    optimized_registered: sitk.Image,
    initial_candidate: str,
    optimized_candidate: str,
    initial_dice: float,
    optimized_dice: float,
    selection_status: str,
    output_path: Path,
) -> None:

    # --------------------------------------------------------
    # NumPy arrays
    # --------------------------------------------------------

    moving_array = sitk.GetArrayFromImage(
        moving
    )

    fixed_array = sitk.GetArrayFromImage(
        fixed
    )

    initial_array = sitk.GetArrayFromImage(
        initial_registered
    )

    optimized_array = sitk.GetArrayFromImage(
        optimized_registered
    )

    # --------------------------------------------------------
    # Display normalization
    # --------------------------------------------------------

    moving_display = robust_normalize(
        moving_array
    )

    fixed_display = robust_normalize(
        fixed_array
    )

    initial_display = robust_normalize(
        initial_array
    )

    optimized_display = robust_normalize(
        optimized_array
    )

    # --------------------------------------------------------
    # Centers
    # --------------------------------------------------------

    moving_center = foreground_center(
        moving_array
    )

    template_center = foreground_center(
        fixed_array
    )

    # --------------------------------------------------------
    # Views
    # --------------------------------------------------------

    moving_views = extract_views(
        moving_display,
        moving_center,
    )

    fixed_views = extract_views(
        fixed_display,
        template_center,
    )

    initial_views = extract_views(
        initial_display,
        template_center,
    )

    optimized_views = extract_views(
        optimized_display,
        template_center,
    )

    planes = [
        "Axial",
        "Coronal",
        "Sagittal",
    ]

    # ========================================================
    # Plot
    #
    # Rows:
    # axial / coronal / sagittal
    #
    # Columns:
    # original
    # template
    # initial
    # initial overlay
    # optimized
    # optimized overlay
    # ========================================================

    fig, axes = plt.subplots(
        nrows=3,
        ncols=6,
        figsize=(22, 12),
    )

    for row_index, plane in enumerate(
        planes
    ):

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        axes[
            row_index,
            0,
        ].imshow(
            moving_views[plane],
            cmap="gray",
        )

        axes[
            row_index,
            0,
        ].set_title(
            f"Original\n{plane}"
        )

        # ----------------------------------------------------
        # Template
        # ----------------------------------------------------

        axes[
            row_index,
            1,
        ].imshow(
            fixed_views[plane],
            cmap="gray",
        )

        axes[
            row_index,
            1,
        ].set_title(
            f"Template\n{plane}"
        )

        # ----------------------------------------------------
        # Initial
        # ----------------------------------------------------

        axes[
            row_index,
            2,
        ].imshow(
            initial_views[plane],
            cmap="gray",
        )

        axes[
            row_index,
            2,
        ].set_title(
            f"INITIAL\n{plane}"
        )

        # ----------------------------------------------------
        # Initial overlay
        # ----------------------------------------------------

        axes[
            row_index,
            3,
        ].imshow(
            fixed_views[plane],
            cmap="gray",
        )

        axes[
            row_index,
            3,
        ].imshow(
            initial_views[plane],
            cmap="magma",
            alpha=0.45,
        )

        axes[
            row_index,
            3,
        ].set_title(
            f"INITIAL overlay\n{plane}"
        )

        # ----------------------------------------------------
        # Optimized
        # ----------------------------------------------------

        axes[
            row_index,
            4,
        ].imshow(
            optimized_views[plane],
            cmap="gray",
        )

        axes[
            row_index,
            4,
        ].set_title(
            f"OPTIMIZED\n{plane}"
        )

        # ----------------------------------------------------
        # Optimized overlay
        # ----------------------------------------------------

        axes[
            row_index,
            5,
        ].imshow(
            fixed_views[plane],
            cmap="gray",
        )

        axes[
            row_index,
            5,
        ].imshow(
            optimized_views[plane],
            cmap="magma",
            alpha=0.45,
        )

        axes[
            row_index,
            5,
        ].set_title(
            f"OPTIMIZED overlay\n{plane}"
        )

    for ax in axes.ravel():

        ax.axis("off")

    gain = (
        initial_dice
        -
        optimized_dice
    )

    fig.suptitle(
        (
            f"{uid}\n"
            f"INITIAL: {initial_candidate} "
            f"Dice={initial_dice:.4f}    |    "
            f"OPTIMIZED: {optimized_candidate} "
            f"Dice={optimized_dice:.4f}    |    "
            f"Gain={gain:+.4f}    |    "
            f"{selection_status}"
        ),
        fontsize=15,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 6B-5: create side-by-side "
            "manual QC images comparing best "
            "initial and best optimized transforms."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only first N selected cases."
        ),
    )

    parser.add_argument(
        "--include-unresolved",
        action="store_true",
        help=(
            "Also include unresolved cases "
            "(all rigid candidates failed)."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Validate
    # ========================================================

    for required in [
        INPUT_DIR,
        TEMPLATE_PATH,
        MANUAL_REVIEW_CSV,
    ]:

        if not required.exists():

            raise FileNotFoundError(
                f"Required path missing: "
                f"{required}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    QC_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Read Step 6B-4 manual review CSV
    # ========================================================

    df = pd.read_csv(
        MANUAL_REVIEW_CSV,
        dtype={
            "uid": str,
            "file_name": str,
        },
    )

    # --------------------------------------------------------
    # Genuinely blocking manual-review cases
    # --------------------------------------------------------

    statuses = [
        "manual_review"
    ]

    if args.include_unresolved:

        statuses.append(
            "unresolved"
        )

    df = df[
        df[
            "selection_status"
        ].isin(statuses)
    ].copy()

    # Largest gains first
    df = df.sort_values(
        "initial_gain_over_optimized",
        ascending=False,
    ).reset_index(
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

    print("=" * 90)

    print(
        "STEP 6B-5 — MANUAL REGISTRATION QC"
    )

    print("=" * 90)

    print(
        f"Cases selected: {len(df)}"
    )

    print(
        f"Include unresolved: "
        f"{args.include_unresolved}"
    )

    print(
        f"QC images: {QC_IMAGE_DIR}"
    )

    print("=" * 90)

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

        initial_candidate = str(
            row["best_initial_candidate"]
        )

        optimized_candidate = str(
            row["best_optimized_candidate"]
        )

        initial_dice = float(
            row["best_initial_dice"]
        )

        optimized_dice = float(
            row["best_optimized_dice"]
        )

        selection_status = str(
            row["selection_status"]
        )

        print(
            f"[{index + 1}/{len(df)}] "
            f"{uid}"
        )

        print(
            f"    initial:   "
            f"{initial_candidate} "
            f"{initial_dice:.4f}"
        )

        print(
            f"    optimized: "
            f"{optimized_candidate} "
            f"{optimized_dice:.4f}"
        )

        print(
            f"    gain: "
            f"{initial_dice - optimized_dice:+.4f}"
        )

        qc_path = (
            QC_IMAGE_DIR
            / f"{uid}_initial_vs_optimized.png"
        )

        error_message = ""

        try:

            # ------------------------------------------------
            # Moving image
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Best initial transform
            # ------------------------------------------------

            initial_transform = (
                get_initial_transform(
                    fixed=fixed,
                    moving=moving,
                    candidate=initial_candidate,
                )
            )

            # ------------------------------------------------
            # Best optimized transform
            # ------------------------------------------------

            optimized_transform = (
                get_optimized_transform(
                    uid=uid,
                    candidate=optimized_candidate,
                )
            )

            # ------------------------------------------------
            # Reconstruct both registrations
            # ------------------------------------------------

            initial_registered = (
                resample_to_template(
                    moving=moving,
                    fixed=fixed,
                    transform=initial_transform,
                )
            )

            optimized_registered = (
                resample_to_template(
                    moving=moving,
                    fixed=fixed,
                    transform=optimized_transform,
                )
            )

            # ------------------------------------------------
            # QC PNG
            # ------------------------------------------------

            make_comparison_figure(
                uid=uid,

                moving=moving,
                fixed=fixed,

                initial_registered=
                    initial_registered,

                optimized_registered=
                    optimized_registered,

                initial_candidate=
                    initial_candidate,

                optimized_candidate=
                    optimized_candidate,

                initial_dice=
                    initial_dice,

                optimized_dice=
                    optimized_dice,

                selection_status=
                    selection_status,

                output_path=
                    qc_path,
            )

            print(
                f"    saved: "
                f"{qc_path.relative_to(PROJECT_ROOT)}"
            )

        except Exception as exc:

            error_message = str(exc)

            errors.append(
                {
                    "uid": uid,
                    "error": error_message,
                }
            )

            print(
                f"    ERROR: {error_message}"
            )

        # ====================================================
        # Decision row
        # ====================================================

        output_rows.append(
            {
                "uid":
                    uid,

                "file_name":
                    file_name,

                "selection_status":
                    selection_status,

                "best_initial_candidate":
                    initial_candidate,

                "best_initial_dice":
                    initial_dice,

                "best_optimized_candidate":
                    optimized_candidate,

                "best_optimized_dice":
                    optimized_dice,

                "initial_gain_over_optimized":
                    (
                        initial_dice
                        -
                        optimized_dice
                    ),

                "qc_image":
                    (
                        str(
                            qc_path.relative_to(
                                PROJECT_ROOT
                            )
                        )
                        if not error_message
                        else ""
                    ),

                # --------------------------------------------
                # Fill these after visual inspection
                # --------------------------------------------

                "visual_choice":
                    "",

                "final_candidate":
                    "",

                "visual_confidence":
                    "",

                "notes":
                    "",

                "error":
                    error_message,
            }
        )

    # ========================================================
    # Save decision CSV
    # ========================================================

    decision_df = pd.DataFrame(
        output_rows
    )

    decision_df.to_csv(
        DECISION_CSV,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "analysis":
            (
                "Step 6B-5 manual registration "
                "initial-vs-optimized visual QC"
            ),

        "number_of_cases":
            int(
                len(decision_df)
            ),

        "manual_review_cases":
            int(
                (
                    decision_df[
                        "selection_status"
                    ]
                    ==
                    "manual_review"
                ).sum()
            ),

        "unresolved_cases":
            int(
                (
                    decision_df[
                        "selection_status"
                    ]
                    ==
                    "unresolved"
                ).sum()
            ),

        "qc_images_generated":
            int(
                (
                    decision_df[
                        "error"
                    ]
                    ==
                    ""
                ).sum()
            ),

        "errors":
            errors,

        "decision_csv":
            str(
                DECISION_CSV.relative_to(
                    PROJECT_ROOT
                )
            ),

        "valid_visual_choices": [
            "initial",
            "optimized",
            "uncertain",
        ],

        "next_action": (
            "Visually inspect each comparison PNG. "
            "For manual_review cases, fill visual_choice "
            "with initial, optimized, or uncertain. "
            "Do not start Step 6C until every blocking "
            "case has been finalized."
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

    print("=" * 90)

    print(
        "STEP 6B-5 QC GENERATION COMPLETED"
    )

    print("=" * 90)

    print(
        f"QC images: "
        f"{QC_IMAGE_DIR}"
    )

    print(
        f"Decision CSV: "
        f"{DECISION_CSV}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_JSON}"
    )

    print(
        f"Generated: "
        f"{len(decision_df) - len(errors)}"
    )

    print(
        f"Errors: "
        f"{len(errors)}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
