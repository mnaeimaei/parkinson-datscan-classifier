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

REGISTRATION_CSV = (
    STEP6A_DIR
    / "registration_qc.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6a_initial_registration_data/step6a3_rescue_failed_registration"
)

TRANSFORM_OUTPUT_DIR = (
    OUTPUT_DIR
    / "transforms"
)

QC_IMAGE_DIR = (
    OUTPUT_DIR
    / "qc_images"
)

RESULTS_CSV = (
    OUTPUT_DIR
    / "rescue_results.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "rescue_summary.json"
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
        fixed_array
        &
        moved_array
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
            f"Unknown initialization: "
            f"{initialization}"
        )

    return sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        mode,
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def robust_display_normalize(
    array: np.ndarray,
) -> np.ndarray:

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    valid = np.isfinite(array)

    positive = array[
        valid
        &
        (array > 0)
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
        or
        not np.isfinite(hi)
        or
        hi <= lo
    ):

        return np.zeros_like(
            array,
            dtype=np.float32,
        )

    output = (
        array - lo
    ) / (
        hi - lo
    )

    return np.clip(
        output,
        0.0,
        1.0,
    )


def center_indices(
    array: np.ndarray,
) -> tuple[int, int, int]:

    mask = (
        np.isfinite(array)
        &
        (array > 0)
    )

    if np.count_nonzero(mask) > 0:

        coords = np.argwhere(mask)

        z, y, x = np.round(
            coords.mean(axis=0)
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
# QC FIGURE
# ============================================================

def make_qc_figure(
    uid: str,
    moving: sitk.Image,
    fixed: sitk.Image,
    registered: sitk.Image,
    old_dice: float,
    new_dice: float,
    selected_candidate: str,
    output_path: Path,
) -> None:

    moving_array = (
        sitk.GetArrayFromImage(
            moving
        )
    )

    fixed_array = (
        sitk.GetArrayFromImage(
            fixed
        )
    )

    registered_array = (
        sitk.GetArrayFromImage(
            registered
        )
    )

    moving_display = (
        robust_display_normalize(
            moving_array
        )
    )

    fixed_display = (
        robust_display_normalize(
            fixed_array
        )
    )

    registered_display = (
        robust_display_normalize(
            registered_array
        )
    )

    moving_center = (
        center_indices(
            moving_array
        )
    )

    fixed_center = (
        center_indices(
            fixed_array
        )
    )

    moving_views = extract_views(
        moving_display,
        moving_center,
    )

    fixed_views = extract_views(
        fixed_display,
        fixed_center,
    )

    registered_views = extract_views(
        registered_display,
        fixed_center,
    )

    planes = [
        "Axial",
        "Coronal",
        "Sagittal",
    ]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(12, 14),
    )

    for col, plane in enumerate(
        planes
    ):

        # Original
        axes[0, col].imshow(
            moving_views[plane],
            cmap="gray",
        )

        axes[0, col].set_title(
            f"Original — {plane}"
        )

        # Template
        axes[1, col].imshow(
            fixed_views[plane],
            cmap="gray",
        )

        axes[1, col].set_title(
            f"Template — {plane}"
        )

        # Rescued registration
        axes[2, col].imshow(
            registered_views[plane],
            cmap="gray",
        )

        axes[2, col].set_title(
            f"Rescued registration — {plane}"
        )

        # Overlay
        axes[3, col].imshow(
            fixed_views[plane],
            cmap="gray",
        )

        axes[3, col].imshow(
            registered_views[plane],
            cmap="magma",
            alpha=0.45,
        )

        axes[3, col].set_title(
            f"Template + rescue — {plane}"
        )

    for ax in axes.ravel():
        ax.axis("off")

    gain = (
        new_dice
        -
        old_dice
    )

    fig.suptitle(
        (
            f"{uid} | "
            f"old={old_dice:.4f} | "
            f"rescue={new_dice:.4f} | "
            f"gain={gain:+.4f} | "
            f"{selected_candidate}"
        ),
        fontsize=14,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.97)
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
# PROCESS ONE FAILED SCAN
# ============================================================

def process_case(
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
            f"Moving image not found: "
            f"{moving_path}"
        )

    moving = sitk.ReadImage(
        str(moving_path),
        sitk.sitkFloat32,
    )

    moving_mask = foreground_mask(
        moving
    )

    # ========================================================
    # Candidate 1: MOMENTS INITIAL
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
    # Candidate 2: GEOMETRY INITIAL
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
    # Candidate 3: MOMENTS OPTIMIZED FROM STEP 6A
    # ========================================================

    moments_optimized_path = (
        STEP6A_DIR
        / "transforms"
        / uid
        / "moments.h5"
    )

    if not moments_optimized_path.exists():

        raise FileNotFoundError(
            f"Missing Step-6A transform: "
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
    # Candidate 4: GEOMETRY OPTIMIZED FROM STEP 6A
    # ========================================================

    geometry_optimized_path = (
        STEP6A_DIR
        / "transforms"
        / uid
        / "geometry.h5"
    )

    if not geometry_optimized_path.exists():

        raise FileNotFoundError(
            f"Missing Step-6A transform: "
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
    # Compare candidates
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

    selected_name = max(
        candidates,
        key=lambda name:
            candidates[name]["dice"],
    )

    selected = (
        candidates[
            selected_name
        ]
    )

    selected_transform = (
        selected["transform"]
    )

    selected_dice = float(
        selected["dice"]
    )

    old_dice = float(
        row["selected_dice"]
    )

    new_qc = qc_category(
        selected_dice
    )

    # ========================================================
    # Save rescue transform
    # ========================================================

    uid_transform_dir = (
        TRANSFORM_OUTPUT_DIR
        / uid
    )

    uid_transform_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_transform_path = (
        uid_transform_dir
        / "selected_rescue.h5"
    )

    sitk.WriteTransform(
        selected_transform,
        str(
            selected_transform_path
        ),
    )

    # ========================================================
    # Registered image for QC only
    # ========================================================

    registered = sitk.Resample(
        moving,
        fixed,
        selected_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    qc_path = (
        QC_IMAGE_DIR
        / f"{uid}_rescue_qc.png"
    )

    make_qc_figure(
        uid=uid,
        moving=moving,
        fixed=fixed,
        registered=registered,

        old_dice=old_dice,
        new_dice=selected_dice,

        selected_candidate=(
            selected_name
        ),

        output_path=qc_path,
    )

    # ========================================================
    # Return row
    # ========================================================

    return {

        "uid":
            uid,

        "file_name":
            file_name,

        "old_selected_dice":
            old_dice,

        "moments_initial_dice":
            moments_initial_dice,

        "geometry_initial_dice":
            geometry_initial_dice,

        "moments_optimized_dice":
            moments_optimized_dice,

        "geometry_optimized_dice":
            geometry_optimized_dice,

        "selected_candidate":
            selected_name,

        "rescue_dice":
            selected_dice,

        "dice_gain":
            selected_dice
            -
            old_dice,

        "old_qc":
            str(
                row["qc_category"]
            ),

        "rescue_qc":
            new_qc,

        "selected_rescue_transform":
            str(
                selected_transform_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "qc_image":
            str(
                qc_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "visual_decision":
            "",

        "notes":
            "",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only first N failed cases "
            "for testing."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Validate
    # ========================================================

    for required in [
        INPUT_DIR,
        TEMPLATE_PATH,
        REGISTRATION_CSV,
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

    TRANSFORM_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    QC_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Read Step-6A results
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
            errors="coerce",
        )
    )

    failed_df = df[
        (
            df["qc_category"]
            == "failed"
        )
        |
        (
            df["selected_dice"]
            < REVIEW_DICE
        )
    ].copy()

    failed_df = (
        failed_df.sort_values(
            "selected_dice"
        )
        .reset_index(
            drop=True
        )
    )

    if args.limit is not None:

        failed_df = (
            failed_df.head(
                args.limit
            )
        )

    # ========================================================
    # Load template
    # ========================================================

    fixed = sitk.ReadImage(
        str(TEMPLATE_PATH),
        sitk.sitkFloat32,
    )

    fixed_mask = foreground_mask(
        fixed
    )

    print("=" * 80)

    print(
        "STEP 6B-2 — RESCUE FAILED "
        "RIGID REGISTRATIONS"
    )

    print("=" * 80)

    print(
        f"Failed cases: "
        f"{len(failed_df)}"
    )

    print("=" * 80)

    results = []

    errors = []

    # ========================================================
    # Process
    # ========================================================

    for index, row in (
        failed_df.iterrows()
    ):

        uid = str(
            row["uid"]
        )

        print(
            f"[{index + 1}/"
            f"{len(failed_df)}] "
            f"{uid}"
        )

        try:

            result = process_case(
                row=row,
                fixed=fixed,
                fixed_mask=fixed_mask,
            )

            results.append(
                result
            )

            print(
                f"    old Dice: "
                f"{result['old_selected_dice']:.4f}"
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
                f"    selected: "
                f"{result['selected_candidate']}"
            )

            print(
                f"    rescue Dice: "
                f"{result['rescue_dice']:.4f}"
            )

            print(
                f"    QC: "
                f"{result['rescue_qc']}"
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

    # ========================================================
    # Save CSV
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    if len(result_df) > 0:

        qc_counts = (
            result_df[
                "rescue_qc"
            ]
            .value_counts()
            .to_dict()
        )

        candidate_counts = (
            result_df[
                "selected_candidate"
            ]
            .value_counts()
            .to_dict()
        )

        rescued_to_review_or_better = int(
            (
                result_df[
                    "rescue_dice"
                ]
                >= REVIEW_DICE
            ).sum()
        )

        rescued_to_high_confidence = int(
            (
                result_df[
                    "rescue_dice"
                ]
                >= HIGH_CONFIDENCE_DICE
            ).sum()
        )

    else:

        qc_counts = {}

        candidate_counts = {}

        rescued_to_review_or_better = 0
        rescued_to_high_confidence = 0

    summary = {

        "analysis":
            "Step 6B-2 failed registration rescue",

        "number_of_failed_cases_processed":
            int(
                len(failed_df)
            ),

        "successful_rescue_executions":
            int(
                len(result_df)
            ),

        "execution_errors":
            int(
                len(errors)
            ),

        "rescue_qc_counts":
            qc_counts,

        "selected_candidate_counts":
            candidate_counts,

        "rescued_to_review_or_better":
            rescued_to_review_or_better,

        "rescued_to_high_confidence":
            rescued_to_high_confidence,

        "errors":
            errors,

        "important_note": (
            "Rescue transforms do not overwrite "
            "Step-6A transforms. "
            "They must be visually inspected "
            "before becoming final transforms."
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

    print("=" * 80)

    print(
        "STEP 6B-2 RESCUE COMPLETED"
    )

    print("=" * 80)

    print(
        f"Results CSV: "
        f"{RESULTS_CSV}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_JSON}"
    )

    print(
        f"QC images: "
        f"{QC_IMAGE_DIR}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
