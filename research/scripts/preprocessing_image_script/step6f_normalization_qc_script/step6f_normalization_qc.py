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

NORMALIZED_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6e_intensity_normalization_data/step6e_normalize_occipital"
)

NORMALIZATION_STATISTICS_CSV = (
    NORMALIZED_DIR
    / "normalization_statistics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6f_normalization_qc_data/step6f_normalization_qc"
)

QC_IMAGE_DIR = (
    OUTPUT_DIR
    / "qc_images"
)

SELECTION_CSV = (
    OUTPUT_DIR
    / "normalization_qc_selection.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "normalization_qc_summary.json"
)

FINAL_REGISTRATION_MANIFEST = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6c_finalize_registration_data/step6c_finalize_all_registration_transforms"
    / "final_registration_manifest.csv"
)

SIMILARITY_RESCUE_CSV = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6b_registration_rescue_data/step6b3_rescue_pnsm_similarity"
    / "similarity_rescue_result.csv"
)

STEP6B1_MANIFEST = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6b_registration_rescue_data/step6b1_finalize_registration_transforms"
    / "final_transform_manifest.csv"
)


# ============================================================
# HELPERS
# ============================================================

def load_special_registration_uids() -> list[str]:

    if FINAL_REGISTRATION_MANIFEST.is_file():

        manifest = pd.read_csv(
            FINAL_REGISTRATION_MANIFEST,
            dtype={"uid": str},
        )

        if "final_source_type" in manifest.columns:

            uids = (
                manifest.loc[
                    manifest["final_source_type"]
                    ==
                    "similarity_rescue",
                    "uid",
                ]
                .astype(str)
                .tolist()
            )

            if uids:

                return sorted(set(uids))

    if SIMILARITY_RESCUE_CSV.is_file():

        rescue = pd.read_csv(
            SIMILARITY_RESCUE_CSV,
            dtype={"uid": str},
        )

        if "uid" in rescue.columns:

            uids = (
                rescue["uid"]
                .dropna()
                .astype(str)
                .tolist()
            )

            if uids:

                return sorted(set(uids))

    if STEP6B1_MANIFEST.is_file():

        policy = pd.read_csv(
            STEP6B1_MANIFEST,
            dtype={"uid": str},
        )

        if "selection_status" in policy.columns:

            uids = (
                policy.loc[
                    policy["selection_status"]
                    ==
                    "unresolved",
                    "uid",
                ]
                .astype(str)
                .tolist()
            )

            return sorted(set(uids))

    return []


def robust_display_normalize(
    array: np.ndarray,
) -> tuple[np.ndarray, float]:

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

    if positive.size == 0:

        return (
            np.zeros_like(
                array,
                dtype=np.float32,
            ),
            1.0,
        )

    upper = float(
        np.percentile(
            positive,
            99.5,
        )
    )

    if (
        not np.isfinite(upper)
        or
        upper <= 0
    ):

        upper = float(
            np.max(positive)
        )

    if upper <= 0:

        upper = 1.0

    display = (
        array
        /
        upper
    )

    display = np.clip(
        display,
        0.0,
        1.0,
    )

    return (
        display,
        upper,
    )


def foreground_center(
    array: np.ndarray,
) -> tuple[int, int, int]:
    """
    Get an intensity-weighted center of positive activity.

    Array order:
        z, y, x
    """

    values = np.asarray(
        array,
        dtype=np.float64,
    )

    values = np.where(
        np.isfinite(values)
        &
        (values > 0),
        values,
        0.0,
    )

    total = float(
        values.sum()
    )

    shape = values.shape

    if total <= 0:

        return (
            shape[0] // 2,
            shape[1] // 2,
            shape[2] // 2,
        )

    z_axis = np.arange(
        shape[0],
        dtype=np.float64,
    )

    y_axis = np.arange(
        shape[1],
        dtype=np.float64,
    )

    x_axis = np.arange(
        shape[2],
        dtype=np.float64,
    )

    z_weights = values.sum(
        axis=(1, 2)
    )

    y_weights = values.sum(
        axis=(0, 2)
    )

    x_weights = values.sum(
        axis=(0, 1)
    )

    z = int(
        round(
            np.sum(
                z_axis * z_weights
            )
            /
            np.sum(z_weights)
        )
    )

    y = int(
        round(
            np.sum(
                y_axis * y_weights
            )
            /
            np.sum(y_weights)
        )
    )

    x = int(
        round(
            np.sum(
                x_axis * x_weights
            )
            /
            np.sum(x_weights)
        )
    )

    z = int(
        np.clip(
            z,
            0,
            shape[0] - 1,
        )
    )

    y = int(
        np.clip(
            y,
            0,
            shape[1] - 1,
        )
    )

    x = int(
        np.clip(
            x,
            0,
            shape[2] - 1,
        )
    )

    return (
        z,
        y,
        x,
    )


def max_voxel_location(
    array: np.ndarray,
) -> tuple[int, int, int]:

    safe = np.where(
        np.isfinite(array),
        array,
        -np.inf,
    )

    flat_index = int(
        np.argmax(safe)
    )

    z, y, x = np.unravel_index(
        flat_index,
        array.shape,
    )

    return (
        int(z),
        int(y),
        int(x),
    )


def extract_slice(
    array: np.ndarray,
    plane: str,
    center: tuple[int, int, int],
) -> np.ndarray:

    z, y, x = center

    if plane == "Axial":

        image = array[
            z,
            :,
            :,
        ]

    elif plane == "Coronal":

        image = array[
            :,
            y,
            :,
        ]

    elif plane == "Sagittal":

        image = array[
            :,
            :,
            x,
        ]

    else:

        raise ValueError(
            f"Unknown plane: {plane}"
        )

    return np.rot90(
        image
    )


def make_mip(
    array: np.ndarray,
    plane: str,
) -> np.ndarray:
    """
    Maximum-intensity projections.

    array order = z, y, x
    """

    if plane == "Axial":

        mip = np.max(
            array,
            axis=0,
        )

    elif plane == "Coronal":

        mip = np.max(
            array,
            axis=1,
        )

    elif plane == "Sagittal":

        mip = np.max(
            array,
            axis=2,
        )

    else:

        raise ValueError(
            f"Unknown plane: {plane}"
        )

    return np.rot90(
        mip
    )


def nearest_to_value(
    df: pd.DataFrame,
    column: str,
    value: float,
    n: int,
) -> pd.DataFrame:

    temp = df.copy()

    temp[
        "_distance"
    ] = (
        pd.to_numeric(
            temp[column],
            errors="coerce",
        )
        -
        value
    ).abs()

    return (
        temp
        .sort_values(
            "_distance"
        )
        .head(n)
        .drop(
            columns=["_distance"]
        )
    )


# ============================================================
# SELECTION
# ============================================================

def add_selection(
    selections: list[dict],
    rows: pd.DataFrame,
    reason: str,
) -> None:

    for _, row in rows.iterrows():

        selections.append(
            {
                "uid":
                    str(row["uid"]),

                "reason":
                    reason,
            }
        )


def select_qc_cases(
    df: pd.DataFrame,
    n_extreme: int,
    n_rescued: int,
    n_typical: int,
    special_uids: list[str],
) -> pd.DataFrame:

    selections: list[dict] = []

    # --------------------------------------------------------
    # Highest normalized maximum
    # --------------------------------------------------------

    add_selection(
        selections,
        df.nlargest(
            n_extreme,
            "normalized_max",
        ),
        "highest_normalized_max",
    )

    # --------------------------------------------------------
    # Highest p99
    # --------------------------------------------------------

    add_selection(
        selections,
        df.nlargest(
            n_extreme,
            "normalized_p99",
        ),
        "highest_normalized_p99",
    )

    # --------------------------------------------------------
    # Lowest p99
    # --------------------------------------------------------

    add_selection(
        selections,
        df.nsmallest(
            n_extreme,
            "normalized_p99",
        ),
        "lowest_normalized_p99",
    )

    # --------------------------------------------------------
    # Highest whole-volume normalized mean
    # --------------------------------------------------------

    add_selection(
        selections,
        df.nlargest(
            n_extreme,
            "normalized_mean",
        ),
        "highest_normalized_mean",
    )

    # --------------------------------------------------------
    # Lowest whole-volume normalized mean
    # --------------------------------------------------------

    add_selection(
        selections,
        df.nsmallest(
            n_extreme,
            "normalized_mean",
        ),
        "lowest_normalized_mean",
    )

    # --------------------------------------------------------
    # Rescued reference cases
    #
    # Choose the lowest-positive-fraction rescued cases.
    # These are the most aggressive rescues.
    # --------------------------------------------------------

    rescued = df[
        df[
            "reference_method"
        ]
        ==
        "positive_only_occipital_mean"
    ].copy()

    rescued = rescued.sort_values(
        "positive_fraction",
        ascending=True,
    )

    add_selection(
        selections,
        rescued.head(
            n_rescued
        ),
        "positive_only_reference_rescue",
    )

    # --------------------------------------------------------
    # Typical scans around dataset median p99
    # --------------------------------------------------------

    p99_median = float(
        df[
            "normalized_p99"
        ].median()
    )

    typical = nearest_to_value(
        df=df,
        column="normalized_p99",
        value=p99_median,
        n=n_typical,
    )

    add_selection(
        selections,
        typical,
        "typical_median_p99",
    )

    # --------------------------------------------------------
    # Similarity-rescue / unresolved registration cases
    # --------------------------------------------------------

    for uid in special_uids:

        special = df[
            df["uid"]
            ==
            uid
        ]

        if len(special):

            add_selection(
                selections,
                special,
                "special_registration_case",
            )

    # ========================================================
    # COMBINE REASONS FOR DUPLICATES
    # ========================================================

    selection_df = pd.DataFrame(
        selections
    )

    if len(selection_df) == 0:

        raise RuntimeError(
            "No QC scans selected."
        )

    reasons = (
        selection_df
        .groupby("uid")["reason"]
        .apply(
            lambda values:
                ";".join(
                    sorted(
                        set(values)
                    )
                )
        )
        .reset_index()
    )

    selected = reasons.merge(
        df,
        on="uid",
        how="left",
    )

    # --------------------------------------------------------
    # Priority score:
    # more selection reasons = higher priority
    # --------------------------------------------------------

    selected[
        "selection_reason_count"
    ] = (
        selected[
            "reason"
        ]
        .str.count(";")
        +
        1
    )

    selected = selected.sort_values(
        [
            "selection_reason_count",
            "normalized_max",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return selected


# ============================================================
# QC IMAGE
# ============================================================

def create_qc_image(
    row: pd.Series,
    output_path: Path,
) -> dict:

    uid = str(
        row["uid"]
    )

    file_name = str(
        row["file_name"]
    )

    image_path = (
        NORMALIZED_DIR
        / file_name
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Normalized image missing: "
            f"{image_path}"
        )

    image = sitk.ReadImage(
        str(image_path),
        sitk.sitkFloat32,
    )

    array = (
        sitk.GetArrayFromImage(
            image
        )
        .astype(
            np.float32,
            copy=False,
        )
    )

    if np.isnan(array).any():

        raise RuntimeError(
            "Normalized image contains NaN."
        )

    if np.isinf(array).any():

        raise RuntimeError(
            "Normalized image contains Inf."
        )

    display_array, display_upper = (
        robust_display_normalize(
            array
        )
    )

    center = foreground_center(
        array
    )

    max_location = max_voxel_location(
        array
    )

    planes = [
        "Axial",
        "Coronal",
        "Sagittal",
    ]

    # ========================================================
    # Figure:
    #
    # row 1 = slices through activity-weighted center
    # row 2 = maximum-intensity projections
    # row 3 = slices through maximum voxel
    # ========================================================

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12, 12),
    )

    # --------------------------------------------------------
    # Row 1: central activity slices
    # --------------------------------------------------------

    for col, plane in enumerate(
        planes
    ):

        view = extract_slice(
            display_array,
            plane,
            center,
        )

        axes[
            0,
            col,
        ].imshow(
            view,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[
            0,
            col,
        ].set_title(
            f"{plane} — activity center"
        )

    # --------------------------------------------------------
    # Row 2: MIPs
    # --------------------------------------------------------

    for col, plane in enumerate(
        planes
    ):

        view = make_mip(
            display_array,
            plane,
        )

        axes[
            1,
            col,
        ].imshow(
            view,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[
            1,
            col,
        ].set_title(
            f"{plane} — MIP"
        )

    # --------------------------------------------------------
    # Row 3: slices through maximum voxel
    # --------------------------------------------------------

    for col, plane in enumerate(
        planes
    ):

        view = extract_slice(
            display_array,
            plane,
            max_location,
        )

        axes[
            2,
            col,
        ].imshow(
            view,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[
            2,
            col,
        ].set_title(
            f"{plane} — max voxel"
        )

    for ax in axes.ravel():

        ax.axis(
            "off"
        )

    title = (
        f"{uid}\n"
        f"selection={row['reason']}\n"
        f"reference={row['reference_method']} | "
        f"denominator={float(row['reference_value']):.6f} | "
        f"positive_fraction={float(row['positive_fraction']):.4f}\n"
        f"mean={float(row['normalized_mean']):.4f} | "
        f"p99={float(row['normalized_p99']):.4f} | "
        f"max={float(row['normalized_max']):.4f} | "
        f"display_clip_p99.5_positive={display_upper:.4f}"
    )

    fig.suptitle(
        title,
        fontsize=12,
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.91,
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

    return {
        "center_z":
            int(center[0]),

        "center_y":
            int(center[1]),

        "center_x":
            int(center[2]),

        "max_z":
            int(max_location[0]),

        "max_y":
            int(max_location[1]),

        "max_x":
            int(max_location[2]),

        "display_upper":
            float(display_upper),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 6D-2: visual QC of normalized "
            "DaT scans using deterministic "
            "extreme, rescued, typical, and "
            "special-case selection."
        )
    )

    parser.add_argument(
        "--n-extreme",
        type=int,
        default=4,
        help=(
            "Number selected from each "
            "extreme-statistic category."
        ),
    )

    parser.add_argument(
        "--n-rescued",
        type=int,
        default=6,
        help=(
            "Number of positive-only reference "
            "rescues selected for QC."
        ),
    )

    parser.add_argument(
        "--n-typical",
        type=int,
        default=4,
        help=(
            "Number of typical median-p99 cases."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Technical test: generate only first "
            "N selected QC images."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if not NORMALIZATION_STATISTICS_CSV.exists():

        raise FileNotFoundError(
            f"Missing normalization statistics: "
            f"{NORMALIZATION_STATISTICS_CSV}"
        )

    if not NORMALIZED_DIR.exists():

        raise FileNotFoundError(
            f"Missing normalized directory: "
            f"{NORMALIZED_DIR}"
        )

    # ========================================================
    # OUTPUT
    # ========================================================

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
    # READ NORMALIZATION RESULTS
    # ========================================================

    df = pd.read_csv(
        NORMALIZATION_STATISTICS_CSV,
        dtype={
            "uid": str,
            "file_name": str,
        },
    )

    df = df[
        df["status"]
        ==
        "success"
    ].copy()

    if df[
        "uid"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate UIDs found."
        )

    numeric_columns = [
        "reference_value",
        "positive_fraction",
        "normalized_mean",
        "normalized_p99",
        "normalized_max",
        "reference_error_from_1",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    # ========================================================
    # SELECT CASES
    # ========================================================

    special_uids = load_special_registration_uids()

    selected = select_qc_cases(
        df=df,
        n_extreme=args.n_extreme,
        n_rescued=args.n_rescued,
        n_typical=args.n_typical,
        special_uids=special_uids,
    )

    total_selected_before_limit = len(
        selected
    )

    if args.limit is not None:

        selected = selected.head(
            args.limit
        ).copy()

    selected = selected.reset_index(
        drop=True
    )

    # ========================================================
    # GENERATE QC
    # ========================================================

    print("=" * 90)

    print(
        "STEP 6D-2 — NORMALIZATION VISUAL QC"
    )

    print("=" * 90)

    print(
        f"Selected unique cases: "
        f"{total_selected_before_limit}"
    )

    print(
        f"Generating now: "
        f"{len(selected)}"
    )

    print("=" * 90)

    output_rows = []

    errors = []

    total = len(
        selected
    )

    for index, row in (
        selected.iterrows()
    ):

        uid = str(
            row["uid"]
        )

        output_path = (
            QC_IMAGE_DIR
            /
            f"{uid}_normalization_qc.png"
        )

        print(
            f"[{index + 1}/{total}] "
            f"{uid}"
        )

        print(
            f"    reason: "
            f"{row['reason']}"
        )

        print(
            f"    mean: "
            f"{float(row['normalized_mean']):.4f}"
        )

        print(
            f"    p99: "
            f"{float(row['normalized_p99']):.4f}"
        )

        print(
            f"    max: "
            f"{float(row['normalized_max']):.4f}"
        )

        error_message = ""

        extra = {}

        try:

            extra = create_qc_image(
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
                    "uid":
                        uid,

                    "error":
                        error_message,
                }
            )

            print(
                f"    ERROR: "
                f"{error_message}"
            )

        output_row = row.to_dict()

        output_row.update(
            extra
        )

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
    # SAVE SELECTION TABLE
    # ========================================================

    output_df = pd.DataFrame(
        output_rows
    )

    output_df.to_csv(
        SELECTION_CSV,
        index=False,
    )

    generated_count = int(
        (
            output_df[
                "qc_generation_error"
            ]
            ==
            ""
        ).sum()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    reason_counts = {}

    for reason_string in output_df[
        "reason"
    ]:

        for reason in str(
            reason_string
        ).split(";"):

            reason_counts[
                reason
            ] = (
                reason_counts.get(
                    reason,
                    0,
                )
                +
                1
            )

    summary = {

        "analysis":
            (
                "Step 6D-2 final visual QC "
                "of normalized DaT scans"
            ),

        "total_normalized_scans":
            int(len(df)),

        "selected_unique_cases_before_limit":
            int(
                total_selected_before_limit
            ),

        "generated_cases":
            int(
                len(output_df)
            ),

        "qc_images_generated":
            generated_count,

        "qc_generation_errors":
            int(
                len(errors)
            ),

        "selection_reason_counts":
            reason_counts,

        "selection_policy": {
            "highest_normalized_max":
                args.n_extreme,

            "highest_normalized_p99":
                args.n_extreme,

            "lowest_normalized_p99":
                args.n_extreme,

            "highest_normalized_mean":
                args.n_extreme,

            "lowest_normalized_mean":
                args.n_extreme,

            "positive_only_reference_rescue":
                args.n_rescued,

            "typical_median_p99":
                args.n_typical,

            "special_uids":
                special_uids,
        },

        "visual_decision_options": [
            "pass",
            "review",
            "fail",
            "uncertain",
        ],

        "what_to_check": [
            (
                "Overall brain/activity distribution "
                "is visible and not nearly blank."
            ),
            (
                "No obviously catastrophic global "
                "intensity scaling."
            ),
            (
                "Very high maximum values correspond "
                "to localized plausible activity "
                "rather than isolated artifacts."
            ),
            (
                "Maximum-intensity projections do not "
                "show obvious reconstruction or "
                "normalization artifacts."
            ),
            (
                "Positive-only rescued cases remain "
                "visually comparable to standard "
                "normalized scans."
            ),
            (
                "Similarity-rescue registration "
                "cases remain usable after "
                "normalization."
            ),
        ],

        "selection_csv":
            str(
                SELECTION_CSV.relative_to(
                    PROJECT_ROOT
                )
            ),

        "qc_image_directory":
            str(
                QC_IMAGE_DIR.relative_to(
                    PROJECT_ROOT
                )
            ),

        "errors":
            errors,

        "next_step":
            (
                "If visual QC passes, freeze Step 6 "
                "and proceed to creation of model "
                "input representations/crops."
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
    # TERMINAL
    # ========================================================

    print()

    print("=" * 90)

    print(
        "STEP 6D-2 QC GENERATION COMPLETED"
    )

    print("=" * 90)

    print(
        f"Selected unique cases: "
        f"{total_selected_before_limit}"
    )

    print(
        f"Generated this run: "
        f"{len(output_df)}"
    )

    print(
        f"QC images generated: "
        f"{generated_count}"
    )

    print(
        f"Errors: "
        f"{len(errors)}"
    )

    print()

    print(
        "Selection reasons:"
    )

    for reason, count in sorted(
        reason_counts.items()
    ):

        print(
            f"    {reason}: "
            f"{count}"
        )

    print()

    print(
        f"QC images:"
        f"\n{QC_IMAGE_DIR}"
    )

    print()

    print(
        f"Selection CSV:"
        f"\n{SELECTION_CSV}"
    )

    print()

    print(
        f"Summary:"
        f"\n{SUMMARY_JSON}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
