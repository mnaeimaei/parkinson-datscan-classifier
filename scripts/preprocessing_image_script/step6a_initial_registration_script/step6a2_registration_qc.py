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

INPUT_DIR = PROJECT_ROOT / "data/preprocessing_image_data/step4_voxel_resampler_data"

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
    / "data/preprocessing_image_data/step6a_initial_registration_data/step6a2_registration_qc"
)

QC_IMAGE_DIR = OUTPUT_DIR / "qc_images"

SELECTION_CSV = (
    OUTPUT_DIR
    / "qc_selection.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "qc_summary.json"
)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def robust_display_normalize(
    array: np.ndarray,
) -> np.ndarray:
    """
    Normalize intensities only for PNG visualization.

    IMPORTANT:
    This does NOT change the NIfTI scan.
    """

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


def center_indices(
    array: np.ndarray,
) -> tuple[int, int, int]:
    """
    Find approximate image center based on
    positive foreground.

    SimpleITK NumPy order:
        z, y, x
    """

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
            np.array(array.shape) // 2
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
    """
    Extract axial, coronal and sagittal views.
    """

    z, y, x = indices

    return {
        "Axial": np.rot90(
            array[z, :, :]
        ),

        "Coronal": np.rot90(
            array[:, y, :]
        ),

        "Sagittal": np.rot90(
            array[:, :, x]
        ),
    }


# ============================================================
# REGISTRATION RECONSTRUCTION
# ============================================================

def resample_to_template(
    moving: sitk.Image,
    fixed: sitk.Image,
    transform: sitk.Transform,
) -> sitk.Image:
    """
    Reconstruct the registered image using
    the selected Step-6A rigid transform.

    This image is ONLY for visual QC.
    """

    return sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


# ============================================================
# CASE SELECTION
# ============================================================

def add_rows(
    selected: dict[int, list[str]],
    rows: pd.DataFrame,
    reason: str,
) -> None:

    for idx in rows.index:

        selected.setdefault(
            int(idx),
            [],
        )

        if reason not in selected[int(idx)]:

            selected[int(idx)].append(
                reason
            )


def build_selection(
    df: pd.DataFrame,
    lowest_review: int,
    sample_mid: int,
    sample_upper: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Select cases for manual visual QC.

    1. ALL failures:
           Dice < 0.50

    2. Lowest review cases:
           0.50 <= Dice < 0.70

    3. Random sample:
           0.60 <= Dice < 0.65

    4. Random sample:
           0.65 <= Dice < 0.70
    """

    df = df.copy()

    # Only technically successful registrations
    if "status" in df.columns:

        df = df[
            df["status"] == "success"
        ].copy()

    df["selected_dice"] = pd.to_numeric(
        df["selected_dice"],
        errors="coerce",
    )

    selected: dict[
        int,
        list[str],
    ] = {}

    # --------------------------------------------------------
    # 1. All failed QC cases
    # --------------------------------------------------------

    failures = df[
        (
            df["qc_category"]
            == "failed"
        )
        |
        (
            df["selected_dice"]
            < 0.50
        )
    ].sort_values(
        "selected_dice"
    )

    add_rows(
        selected,
        failures,
        "all_failed_dice_lt_0.50",
    )

    # --------------------------------------------------------
    # Review cases
    # --------------------------------------------------------

    review = df[
        (
            df["selected_dice"]
            >= 0.50
        )
        &
        (
            df["selected_dice"]
            < 0.70
        )
    ].sort_values(
        "selected_dice"
    )

    # --------------------------------------------------------
    # 2. Lowest review cases
    # --------------------------------------------------------

    lowest = review.head(
        lowest_review
    )

    add_rows(
        selected,
        lowest,
        f"lowest_{lowest_review}_review",
    )

    # --------------------------------------------------------
    # 3. Random cases: Dice 0.60–0.65
    # --------------------------------------------------------

    mid = review[
        (
            review["selected_dice"]
            >= 0.60
        )
        &
        (
            review["selected_dice"]
            < 0.65
        )
    ]

    if (
        sample_mid > 0
        and len(mid) > 0
    ):

        mid_sample = mid.sample(
            n=min(
                sample_mid,
                len(mid),
            ),
            random_state=random_seed,
        )

        add_rows(
            selected,
            mid_sample,
            "random_review_0.60_to_0.65",
        )

    # --------------------------------------------------------
    # 4. Random cases: Dice 0.65–0.70
    # --------------------------------------------------------

    upper = review[
        (
            review["selected_dice"]
            >= 0.65
        )
        &
        (
            review["selected_dice"]
            < 0.70
        )
    ]

    if (
        sample_upper > 0
        and len(upper) > 0
    ):

        upper_sample = upper.sample(
            n=min(
                sample_upper,
                len(upper),
            ),
            random_state=(
                random_seed + 1
            ),
        )

        add_rows(
            selected,
            upper_sample,
            "random_review_0.65_to_0.70",
        )

    # --------------------------------------------------------
    # Combine selections
    # --------------------------------------------------------

    selected_indices = sorted(
        selected,
        key=lambda idx: float(
            df.loc[
                idx,
                "selected_dice",
            ]
        ),
    )

    selection = df.loc[
        selected_indices
    ].copy()

    selection[
        "selection_reason"
    ] = [
        ";".join(
            selected[int(idx)]
        )
        for idx in selection.index
    ]

    # Fields we will fill manually after looking
    # at the PNGs.
    selection["qc_image"] = ""
    selection["visual_decision"] = ""
    selection["notes"] = ""

    return selection.reset_index(
        drop=True
    )


# ============================================================
# QC FIGURE
# ============================================================

def make_qc_figure(
    uid: str,
    moving: sitk.Image,
    fixed: sitk.Image,
    registered: sitk.Image,
    selected_dice: float,
    qc_category: str,
    initialization: str,
    output_path: Path,
) -> None:

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Display normalization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Centers
    # --------------------------------------------------------

    moving_center = center_indices(
        moving_array
    )

    fixed_center = center_indices(
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
        fixed_center,
    )

    # Registered image is already in template space,
    # therefore use template center.
    registered_views = extract_views(
        registered_display,
        fixed_center,
    )

    plane_names = [
        "Axial",
        "Coronal",
        "Sagittal",
    ]

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(12, 14),
    )

    for col, plane in enumerate(
        plane_names
    ):

        # Original resampled scan
        axes[0, col].imshow(
            moving_views[plane],
            cmap="gray",
        )

        axes[0, col].set_title(
            f"Original resampled — {plane}"
        )

        # Template
        axes[1, col].imshow(
            fixed_views[plane],
            cmap="gray",
        )

        axes[1, col].set_title(
            f"DaT template — {plane}"
        )

        # Registered
        axes[2, col].imshow(
            registered_views[plane],
            cmap="gray",
        )

        axes[2, col].set_title(
            f"Rigid registered — {plane}"
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
            f"Template + registered — {plane}"
        )

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        (
            f"{uid} | "
            f"Dice={selected_dice:.4f} | "
            f"QC={qc_category} | "
            f"start={initialization}"
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
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 6B - visual QC of "
            "full-dataset rigid registration."
        )
    )

    parser.add_argument(
        "--lowest-review",
        type=int,
        default=20,
        help=(
            "Number of lowest-Dice "
            "review cases."
        ),
    )

    parser.add_argument(
        "--sample-mid",
        type=int,
        default=5,
        help=(
            "Random cases from "
            "0.60 <= Dice < 0.65."
        ),
    )

    parser.add_argument(
        "--sample-upper",
        type=int,
        default=5,
        help=(
            "Random cases from "
            "0.65 <= Dice < 0.70."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help=(
            "Optional test limit. "
            "Example: --max-cases 3"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    required_paths = [
        INPUT_DIR,
        TEMPLATE_PATH,
        REGISTRATION_CSV,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path not found: "
                f"{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    QC_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load Step 6A CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        REGISTRATION_CSV
    )

    # --------------------------------------------------------
    # Select cases
    # --------------------------------------------------------

    selection = build_selection(
        df=df,
        lowest_review=args.lowest_review,
        sample_mid=args.sample_mid,
        sample_upper=args.sample_upper,
        random_seed=args.seed,
    )

    if args.max_cases is not None:

        selection = selection.head(
            args.max_cases
        ).copy()

    # --------------------------------------------------------
    # Load template once
    # --------------------------------------------------------

    fixed = sitk.ReadImage(
        str(TEMPLATE_PATH),
        sitk.sitkFloat32,
    )

    print("=" * 78)

    print(
        "STEP 6B — REGISTRATION VISUAL QC"
    )

    print("=" * 78)

    print(
        f"Registration CSV: "
        f"{REGISTRATION_CSV}"
    )

    print(
        f"Template: "
        f"{TEMPLATE_PATH}"
    )

    print(
        f"Selected cases: "
        f"{len(selection)}"
    )

    print(
        f"QC output: "
        f"{QC_IMAGE_DIR}"
    )

    print("=" * 78)

    # --------------------------------------------------------
    # Generate QC
    # --------------------------------------------------------

    generated = 0
    generation_errors = 0

    errors = []

    for idx, row in selection.iterrows():

        uid = str(
            row["uid"]
        )

        file_name = str(
            row["file_name"]
        )

        moving_path = (
            INPUT_DIR / file_name
        )

        # Path was stored in Step 6A CSV.
        transform_value = str(
            row["selected_transform"]
        )

        transform_path = Path(
            transform_value
        )

        if not transform_path.is_absolute():

            transform_path = (
                PROJECT_ROOT
                / transform_path
            )

        output_path = (
            QC_IMAGE_DIR
            / f"{uid}_registration_qc.png"
        )

        print(
            f"[{idx + 1}/{len(selection)}] "
            f"{file_name} "
            f"Dice="
            f"{float(row['selected_dice']):.4f}"
        )

        try:

            # -----------------------------------------------
            # Check source files
            # -----------------------------------------------

            if not moving_path.exists():

                raise FileNotFoundError(
                    f"Moving image missing: "
                    f"{moving_path}"
                )

            if not transform_path.exists():

                raise FileNotFoundError(
                    f"Transform missing: "
                    f"{transform_path}"
                )

            # -----------------------------------------------
            # Read scan
            # -----------------------------------------------

            moving = sitk.ReadImage(
                str(moving_path),
                sitk.sitkFloat32,
            )

            # -----------------------------------------------
            # Read selected Step-6A transform
            # -----------------------------------------------

            transform = sitk.ReadTransform(
                str(transform_path)
            )

            # -----------------------------------------------
            # Recreate registered image
            # -----------------------------------------------

            registered = resample_to_template(
                moving=moving,
                fixed=fixed,
                transform=transform,
            )

            # -----------------------------------------------
            # Create PNG
            # -----------------------------------------------

            make_qc_figure(
                uid=uid,
                moving=moving,
                fixed=fixed,
                registered=registered,

                selected_dice=float(
                    row["selected_dice"]
                ),

                qc_category=str(
                    row["qc_category"]
                ),

                initialization=str(
                    row[
                        "selected_initialization"
                    ]
                ),

                output_path=output_path,
            )

            selection.loc[
                idx,
                "qc_image",
            ] = str(
                output_path.relative_to(
                    PROJECT_ROOT
                )
            )

            generated += 1

            print(
                f"    Saved: "
                f"{output_path.relative_to(PROJECT_ROOT)}"
            )

        except Exception as exc:

            generation_errors += 1

            message = str(exc)

            print(
                f"    ERROR: {message}"
            )

            selection.loc[
                idx,
                "notes",
            ] = (
                "QC generation error: "
                + message
            )

            errors.append(
                {
                    "uid": uid,
                    "error": message,
                }
            )

    # --------------------------------------------------------
    # Save selection CSV
    # --------------------------------------------------------

    selection.to_csv(
        SELECTION_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    dice_values = pd.to_numeric(
        selection["selected_dice"],
        errors="coerce",
    )

    selected_failed = int(
        (
            dice_values < 0.50
        ).sum()
    )

    selected_review = int(
        (
            (dice_values >= 0.50)
            &
            (dice_values < 0.70)
        ).sum()
    )

    summary = {

        "analysis":
            "Step 6B registration visual QC",

        "source_csv":
            str(
                REGISTRATION_CSV.relative_to(
                    PROJECT_ROOT
                )
            ),

        "template":
            str(
                TEMPLATE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),

        "selected_cases":
            int(len(selection)),

        "selected_failed_cases":
            selected_failed,

        "selected_review_cases":
            selected_review,

        "qc_images_generated":
            generated,

        "qc_image_generation_errors":
            generation_errors,

        "selection_parameters": {

            "lowest_review":
                args.lowest_review,

            "sample_mid_0.60_to_0.65":
                args.sample_mid,

            "sample_upper_0.65_to_0.70":
                args.sample_upper,

            "random_seed":
                args.seed,

            "max_cases":
                args.max_cases,
        },

        "errors":
            errors,

        "next_action": (
            "Visually inspect each PNG and "
            "fill visual_decision in "
            "qc_selection.csv with "
            "accept, rescue, or reject."
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

    print()

    print("=" * 78)

    print(
        "STEP 6B QC GENERATION COMPLETED"
    )

    print("=" * 78)

    print(
        f"QC images: "
        f"{QC_IMAGE_DIR}"
    )

    print(
        f"Selection CSV: "
        f"{SELECTION_CSV}"
    )

    print(
        f"Summary JSON: "
        f"{SUMMARY_JSON}"
    )

    print(
        f"Generated: "
        f"{generated}"
    )

    print(
        f"Errors: "
        f"{generation_errors}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
