from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESAMPLED_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step4_voxel_resampler_data"
)

STEP6C_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6d_occipital_reference_data/step6d1_extract_occipital_reference"
)

REFERENCE_CSV = (
    STEP6C_DIR
    / "occipital_reference_statistics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6d_occipital_reference_data/step6d2_occipital_mask_qc"
)

QC_IMAGE_DIR = (
    OUTPUT_DIR
    / "qc_images"
)

QC_DECISION_CSV = (
    OUTPUT_DIR
    / "occipital_qc_decisions.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "occipital_qc_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def robust_normalize(
    array: np.ndarray,
) -> np.ndarray:
    """
    Display-only intensity normalization.
    Does not modify the NIfTI data.
    """

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    finite = array[
        np.isfinite(array)
    ]

    positive = finite[
        finite > 0
    ]

    if positive.size >= 10:

        lo, hi = np.percentile(
            positive,
            [1.0, 99.5],
        )

    elif finite.size > 0:

        lo, hi = np.percentile(
            finite,
            [1.0, 99.5],
        )

    else:

        return np.zeros_like(
            array,
            dtype=np.float32,
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


def validate_same_grid(
    image: sitk.Image,
    mask: sitk.Image,
) -> None:
    """
    Step 6C mapped masks should be exactly on
    the resampled subject grid.
    """

    if image.GetSize() != mask.GetSize():

        raise RuntimeError(
            f"Image/mask size mismatch: "
            f"{image.GetSize()} vs {mask.GetSize()}"
        )

    if not np.allclose(
        image.GetSpacing(),
        mask.GetSpacing(),
        atol=1e-5,
    ):

        raise RuntimeError(
            "Image/mask spacing mismatch."
        )

    if not np.allclose(
        image.GetOrigin(),
        mask.GetOrigin(),
        atol=1e-4,
    ):

        raise RuntimeError(
            "Image/mask origin mismatch."
        )

    if not np.allclose(
        image.GetDirection(),
        mask.GetDirection(),
        atol=1e-5,
    ):

        raise RuntimeError(
            "Image/mask direction mismatch."
        )


def mask_centroid(
    mask_array: np.ndarray,
) -> tuple[int, int, int]:
    """
    Array order is z, y, x.
    """

    coords = np.argwhere(
        mask_array > 0
    )

    if len(coords) == 0:

        raise RuntimeError(
            "Occipital mask is empty."
        )

    z, y, x = np.round(
        coords.mean(axis=0)
    ).astype(int)

    return (
        int(z),
        int(y),
        int(x),
    )


def extract_plane(
    array: np.ndarray,
    plane: str,
    center: tuple[int, int, int],
) -> np.ndarray:

    z, y, x = center

    if plane == "Axial":

        result = array[
            z,
            :,
            :,
        ]

    elif plane == "Coronal":

        result = array[
            :,
            y,
            :,
        ]

    elif plane == "Sagittal":

        result = array[
            :,
            :,
            x,
        ]

    else:

        raise ValueError(
            f"Unknown plane: {plane}"
        )

    return np.rot90(
        result
    )


def crop_around_mask(
    image_2d: np.ndarray,
    mask_2d: np.ndarray,
    margin: int = 18,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    coords = np.argwhere(
        mask_2d > 0
    )

    if len(coords) == 0:

        return (
            image_2d,
            mask_2d,
        )

    r0, c0 = coords.min(
        axis=0
    )

    r1, c1 = coords.max(
        axis=0
    )

    r0 = max(
        0,
        int(r0) - margin,
    )

    c0 = max(
        0,
        int(c0) - margin,
    )

    r1 = min(
        image_2d.shape[0],
        int(r1) + margin + 1,
    )

    c1 = min(
        image_2d.shape[1],
        int(c1) + margin + 1,
    )

    return (
        image_2d[
            r0:r1,
            c0:c1,
        ],
        mask_2d[
            r0:r1,
            c0:c1,
        ],
    )


def add_mask_contour(
    ax,
    mask: np.ndarray,
) -> None:

    if np.any(
        mask > 0
    ):

        ax.contour(
            mask.astype(
                np.float32
            ),
            levels=[0.5],
            colors="red",
            linewidths=1.5,
        )


def add_zero_roi_overlay(
    ax,
    zero_roi: np.ndarray,
) -> None:
    """
    Highlight ROI voxels whose original
    image intensity is <= 0.
    """

    if not np.any(
        zero_roi
    ):
        return

    overlay = np.ma.masked_where(
        ~zero_roi,
        zero_roi.astype(
            np.float32
        ),
    )

    ax.imshow(
        overlay,
        cmap="Blues",
        alpha=0.55,
        vmin=0,
        vmax=1,
    )


# ============================================================
# REVIEW PRIORITY
# ============================================================

def classify_priority(
    positive_fraction: float,
    border_touch: bool,
) -> tuple[int, str]:

    low_positive = (
        positive_fraction < 0.90
    )

    if (
        low_positive
        and
        border_touch
    ):

        return (
            1,
            "both_low_positive_and_border",
        )

    if low_positive:

        return (
            2,
            "low_positive_only",
        )

    if border_touch:

        return (
            3,
            "border_touch_only",
        )

    return (
        4,
        "other_review_reason",
    )


# ============================================================
# QC FIGURE
# ============================================================

def create_qc_figure(
    uid: str,
    image: sitk.Image,
    mask: sitk.Image,
    row: pd.Series,
    output_path: Path,
) -> None:

    image_array = (
        sitk.GetArrayFromImage(
            image
        )
    )

    mask_array = (
        sitk.GetArrayFromImage(
            mask
        ) > 0
    )

    display_array = (
        robust_normalize(
            image_array
        )
    )

    center = mask_centroid(
        mask_array
    )

    planes = [
        "Axial",
        "Coronal",
        "Sagittal",
    ]

    # ========================================================
    # Layout:
    #
    # rows = axial / coronal / sagittal
    #
    # columns:
    # 1 original
    # 2 mask overlay
    # 3 zero-valued ROI voxels
    # 4 zoomed ROI
    # ========================================================

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(16, 12),
    )

    for row_index, plane in enumerate(
        planes
    ):

        image_view = extract_plane(
            display_array,
            plane,
            center,
        )

        original_view = extract_plane(
            image_array,
            plane,
            center,
        )

        mask_view = extract_plane(
            mask_array.astype(
                np.uint8
            ),
            plane,
            center,
        ) > 0

        # ROI voxels where original scan is <= 0
        zero_roi_view = (
            mask_view
            &
            (
                original_view <= 0
            )
        )

        # ----------------------------------------------------
        # 1. Original
        # ----------------------------------------------------

        ax = axes[
            row_index,
            0,
        ]

        ax.imshow(
            image_view,
            cmap="gray",
        )

        ax.set_title(
            f"Original — {plane}"
        )

        # ----------------------------------------------------
        # 2. Mask contour
        # ----------------------------------------------------

        ax = axes[
            row_index,
            1,
        ]

        ax.imshow(
            image_view,
            cmap="gray",
        )

        add_mask_contour(
            ax,
            mask_view,
        )

        ax.set_title(
            f"Occipital ROI — {plane}"
        )

        # ----------------------------------------------------
        # 3. Zero values inside mask
        # ----------------------------------------------------

        ax = axes[
            row_index,
            2,
        ]

        ax.imshow(
            image_view,
            cmap="gray",
        )

        add_mask_contour(
            ax,
            mask_view,
        )

        add_zero_roi_overlay(
            ax,
            zero_roi_view,
        )

        ax.set_title(
            f"ROI zeros — {plane}"
        )

        # ----------------------------------------------------
        # 4. Zoom
        # ----------------------------------------------------

        (
            zoom_image,
            zoom_mask,
        ) = crop_around_mask(
            image_view,
            mask_view,
            margin=18,
        )

        ax = axes[
            row_index,
            3,
        ]

        ax.imshow(
            zoom_image,
            cmap="gray",
        )

        add_mask_contour(
            ax,
            zoom_mask,
        )

        ax.set_title(
            f"ROI zoom — {plane}"
        )

    for ax in axes.ravel():

        ax.axis("off")

    # ========================================================
    # Metadata
    # ========================================================

    title = (
        f"{uid}\n"
        f"occipital_mean="
        f"{float(row['occipital_mean']):.4f}   |   "
        f"positive_fraction="
        f"{float(row['positive_fraction']):.4f}   |   "
        f"border_touch="
        f"{bool(row['border_touch'])}   |   "
        f"voxels="
        f"{int(row['occipital_voxel_count'])}   |   "
        f"volume_ratio="
        f"{float(row['physical_volume_ratio']):.4f}\n"
        f"registration="
        f"{row['final_candidate']}   |   "
        f"Dice="
        f"{float(row['final_registration_dice']):.4f}   |   "
        f"reason="
        f"{row['screening_reason']}"
    )

    fig.suptitle(
        title,
        fontsize=13,
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.92,
        )
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

    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 6C-2: create visual QC "
            "images for Step-6C flagged "
            "occipital reference masks."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Generate only first N review "
            "cases for testing."
        ),
    )

    parser.add_argument(
        "--priority",
        choices=[
            "all",
            "both",
            "low_positive",
            "border",
        ],
        default="all",
        help=(
            "Optional review subgroup."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # Validate
    # ========================================================

    for path in [
        RESAMPLED_DIR,
        REFERENCE_CSV,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path missing: "
                f"{path}"
            )

    if (
        args.overwrite
        and
        OUTPUT_DIR.exists()
    ):

        shutil.rmtree(
            OUTPUT_DIR
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
    # Load Step 6C table
    # ========================================================

    df = pd.read_csv(
        REFERENCE_CSV,
        dtype={
            "uid": str,
            "file_name": str,
        },
    )

    df = df[
        (
            df["status"]
            ==
            "success"
        )
        &
        (
            df["screening_status"]
            ==
            "review"
        )
    ].copy()

    # Ensure proper bool parsing
    df["border_touch"] = (
        df["border_touch"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    df[
        "positive_fraction"
    ] = pd.to_numeric(
        df[
            "positive_fraction"
        ],
        errors="raise",
    )

    # ========================================================
    # Add priority
    # ========================================================

    priorities = []

    categories = []

    for _, row in df.iterrows():

        priority, category = (
            classify_priority(
                positive_fraction=float(
                    row[
                        "positive_fraction"
                    ]
                ),
                border_touch=bool(
                    row[
                        "border_touch"
                    ]
                ),
            )
        )

        priorities.append(
            priority
        )

        categories.append(
            category
        )

    df[
        "review_priority"
    ] = priorities

    df[
        "review_category"
    ] = categories

    # ========================================================
    # Optional subgroup
    # ========================================================

    if args.priority == "both":

        df = df[
            df[
                "review_category"
            ]
            ==
            "both_low_positive_and_border"
        ]

    elif args.priority == "low_positive":

        df = df[
            df[
                "review_category"
            ]
            ==
            "low_positive_only"
        ]

    elif args.priority == "border":

        df = df[
            df[
                "review_category"
            ]
            ==
            "border_touch_only"
        ]

    # ========================================================
    # Sort
    #
    # 1. both flags
    # 2. low-positive only
    # 3. border only
    #
    # Within each category:
    # lowest positive fraction first.
    # ========================================================

    df = df.sort_values(
        [
            "review_priority",
            "positive_fraction",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    if args.limit is not None:

        df = df.head(
            args.limit
        ).copy()

    # ========================================================
    # Process
    # ========================================================

    print("=" * 90)

    print(
        "STEP 6C-2 — OCCIPITAL MASK VISUAL QC"
    )

    print("=" * 90)

    print(
        f"Cases selected: "
        f"{len(df)}"
    )

    print(
        f"Priority filter: "
        f"{args.priority}"
    )

    print("=" * 90)

    output_rows = []

    errors = []

    total = len(df)

    for index, row in df.iterrows():

        uid = str(
            row["uid"]
        )

        file_name = str(
            row["file_name"]
        )

        image_path = (
            RESAMPLED_DIR
            /
            file_name
        )

        mask_relative = str(
            row[
                "mapped_occipital_mask"
            ]
        )

        mask_path = Path(
            mask_relative
        )

        if not mask_path.is_absolute():

            mask_path = (
                PROJECT_ROOT
                /
                mask_path
            )

        output_path = (
            QC_IMAGE_DIR
            /
            f"{uid}_occipital_qc.png"
        )

        print(
            f"[{index + 1}/{total}] "
            f"{uid}"
        )

        print(
            f"    category: "
            f"{row['review_category']}"
        )

        print(
            f"    positive: "
            f"{float(row['positive_fraction']):.4f}"
        )

        print(
            f"    border: "
            f"{bool(row['border_touch'])}"
        )

        error_message = ""

        try:

            if not image_path.exists():

                raise FileNotFoundError(
                    f"Image missing: "
                    f"{image_path}"
                )

            if not mask_path.exists():

                raise FileNotFoundError(
                    f"Mask missing: "
                    f"{mask_path}"
                )

            image = sitk.ReadImage(
                str(image_path),
                sitk.sitkFloat32,
            )

            mask = sitk.ReadImage(
                str(mask_path),
                sitk.sitkUInt8,
            )

            validate_same_grid(
                image,
                mask,
            )

            create_qc_figure(
                uid=uid,
                image=image,
                mask=mask,
                row=row,
                output_path=output_path,
            )

            print(
                f"    saved: "
                f"{output_path.relative_to(PROJECT_ROOT)}"
            )

        except Exception as exc:

            error_message = str(
                exc
            )

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

        # ====================================================
        # QC decision table
        # ====================================================

        output_row = row.to_dict()

        output_row.update(
            {
                "qc_image":
                    (
                        str(
                            output_path.relative_to(
                                PROJECT_ROOT
                            )
                        )
                        if not error_message
                        else ""
                    ),

                # Fill after visual inspection
                "visual_decision":
                    "",

                "visual_confidence":
                    "",

                "visual_notes":
                    "",

                "qc_generation_error":
                    error_message,
            }
        )

        output_rows.append(
            output_row
        )

    # ========================================================
    # Save QC decision table
    # ========================================================

    decision_df = pd.DataFrame(
        output_rows
    )

    decision_df.to_csv(
        QC_DECISION_CSV,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    category_counts = (
        decision_df[
            "review_category"
        ]
        .value_counts()
        .to_dict()
        if len(decision_df)
        else {}
    )

    generated_count = int(
        (
            decision_df[
                "qc_generation_error"
            ]
            ==
            ""
        ).sum()
    )

    summary = {

        "analysis":
            (
                "Step 6C-2 visual QC of "
                "flagged occipital reference masks"
            ),

        "selected_cases":
            int(
                len(decision_df)
            ),

        "category_counts":
            category_counts,

        "qc_images_generated":
            generated_count,

        "qc_generation_errors":
            int(
                len(errors)
            ),

        "priority_order": {
            "1":
                (
                    "positive_fraction < 0.90 "
                    "AND border_touch"
                ),

            "2":
                (
                    "positive_fraction < 0.90 "
                    "only"
                ),

            "3":
                (
                    "border_touch only"
                ),
        },

        "valid_visual_decisions": [
            "accept",
            "review_reference",
            "reject_mask",
            "uncertain",
        ],

        "decision_guidance": {

            "accept":
                (
                    "Mapped ROI appears anatomically "
                    "reasonable despite screening flag."
                ),

            "review_reference":
                (
                    "ROI location appears plausible "
                    "but zero padding/FOV truncation "
                    "may bias the occipital mean."
                ),

            "reject_mask":
                (
                    "ROI is anatomically misplaced "
                    "or clearly unusable."
                ),

            "uncertain":
                (
                    "Visual QC does not provide "
                    "enough confidence."
                ),
        },

        "decision_csv":
            str(
                QC_DECISION_CSV.relative_to(
                    PROJECT_ROOT
                )
            ),

        "errors":
            errors,

        "next_step":
            (
                "Review QC PNGs before freezing "
                "the occipital reference scalar "
                "for normalization."
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
    # Terminal summary
    # ========================================================

    print()

    print("=" * 90)

    print(
        "STEP 6C-2 QC GENERATION COMPLETED"
    )

    print("=" * 90)

    print(
        f"Selected cases: "
        f"{len(decision_df)}"
    )

    print(
        f"Images generated: "
        f"{generated_count}"
    )

    print(
        f"Errors: "
        f"{len(errors)}"
    )

    print()

    for category, count in (
        category_counts.items()
    ):

        print(
            f"{category}: "
            f"{count}"
        )

    print()

    print(
        f"QC images:"
        f"\n{QC_IMAGE_DIR}"
    )

    print()

    print(
        f"Decision CSV:"
        f"\n{QC_DECISION_CSV}"
    )

    print()

    print(
        f"Summary:"
        f"\n{SUMMARY_JSON}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
