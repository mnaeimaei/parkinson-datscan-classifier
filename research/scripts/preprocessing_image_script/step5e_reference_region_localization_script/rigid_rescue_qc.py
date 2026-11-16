from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def normalize_for_display(
    array: np.ndarray,
) -> np.ndarray:
    """
    Robust visualization-only normalization.

    Does NOT affect preprocessing or normalization.
    """

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    finite = np.isfinite(array)

    positive = array[
        finite & (array > 0)
    ]

    if positive.size == 0:
        return np.zeros_like(
            array,
            dtype=np.float32,
        )

    low = float(
        np.percentile(
            positive,
            1,
        )
    )

    high = float(
        np.percentile(
            positive,
            99,
        )
    )

    if high <= low:
        high = float(
            positive.max()
        )

    output = np.zeros_like(
        array,
        dtype=np.float32,
    )

    if high > low:
        output[finite] = (
            array[finite] - low
        ) / (
            high - low
        )

    return np.clip(
        output,
        0.0,
        1.0,
    )


def validate_same_geometry(
    image_a: sitk.Image,
    image_b: sitk.Image,
    name_a: str,
    name_b: str,
) -> None:

    if image_a.GetSize() != image_b.GetSize():
        raise ValueError(
            f"{name_a} and {name_b} have "
            "different sizes."
        )

    if not np.allclose(
        image_a.GetSpacing(),
        image_b.GetSpacing(),
        atol=1e-5,
    ):
        raise ValueError(
            f"{name_a} and {name_b} have "
            "different spacing."
        )

    if not np.allclose(
        image_a.GetOrigin(),
        image_b.GetOrigin(),
        atol=1e-4,
    ):
        raise ValueError(
            f"{name_a} and {name_b} have "
            "different origins."
        )

    if not np.allclose(
        image_a.GetDirection(),
        image_b.GetDirection(),
        atol=1e-5,
    ):
        raise ValueError(
            f"{name_a} and {name_b} have "
            "different directions."
        )


def resample_subject_to_template(
    subject: sitk.Image,
    template: sitk.Image,
    transform: sitk.Transform,
) -> sitk.Image:
    """
    Resample subject image onto template grid.

    SimpleITK Resample() expects an output->input transform.
    The saved registration transforms have that convention.
    """

    return sitk.Resample(
        subject,
        template,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def mask_center(
    mask_array: np.ndarray,
) -> tuple[int, int, int]:
    """
    SimpleITK array order:
        z, y, x
    """

    coordinates = np.where(
        mask_array > 0
    )

    if coordinates[0].size == 0:
        raise ValueError(
            "Occipital mask is empty."
        )

    z = int(
        (
            coordinates[0].min()
            + coordinates[0].max()
        )
        // 2
    )

    y = int(
        (
            coordinates[1].min()
            + coordinates[1].max()
        )
        // 2
    )

    x = int(
        (
            coordinates[2].min()
            + coordinates[2].max()
        )
        // 2
    )

    return z, y, x


def extract_views(
    array: np.ndarray,
    center: tuple[int, int, int],
) -> dict[str, np.ndarray]:

    z, y, x = center

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


def plot_overlay(
    axis,
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    title: str,
) -> None:

    axis.imshow(
        image_slice,
        cmap="gray",
    )

    if np.any(mask_slice):
        axis.contour(
            mask_slice,
            levels=[0.5],
            linewidths=1.5,
        )

    axis.set_title(
        title
    )

    axis.axis(
        "off"
    )


def save_rigid_rescue_qc(
    subject: sitk.Image,
    template: sitk.Image,
    occipital_mask: sitk.Image,
    affine_transform: sitk.Transform,
    rigid_transform: sitk.Transform,
    output_path: str | Path,
    uid: str,
    affine_dice: float,
    rigid_dice: float,
    affine_qc_category: str,
    rigid_qc_category: str,
    affine_volume_ratio: float | None = None,
    rigid_volume_ratio: float | None = None,
) -> None:
    """
    Create a 3 x 3 comparison:

        Row 1:
            Template + occipital mask

        Row 2:
            Affine-registered subject + same mask

        Row 3:
            Rigid-registered subject + same mask

    Columns:
        axial / coronal / sagittal
    """

    output_path = Path(
        output_path
    )

    affine_registered = (
        resample_subject_to_template(
            subject=subject,
            template=template,
            transform=affine_transform,
        )
    )

    rigid_registered = (
        resample_subject_to_template(
            subject=subject,
            template=template,
            transform=rigid_transform,
        )
    )

    template_array = (
        sitk.GetArrayFromImage(
            template
        ).astype(np.float32)
    )

    affine_array = (
        sitk.GetArrayFromImage(
            affine_registered
        ).astype(np.float32)
    )

    rigid_array = (
        sitk.GetArrayFromImage(
            rigid_registered
        ).astype(np.float32)
    )

    mask_array = (
        sitk.GetArrayFromImage(
            occipital_mask
        ) > 0
    )

    template_display = normalize_for_display(
        template_array
    )

    affine_display = normalize_for_display(
        affine_array
    )

    rigid_display = normalize_for_display(
        rigid_array
    )

    center = mask_center(
        mask_array
    )

    template_views = extract_views(
        template_display,
        center,
    )

    affine_views = extract_views(
        affine_display,
        center,
    )

    rigid_views = extract_views(
        rigid_display,
        center,
    )

    mask_views = extract_views(
        mask_array.astype(np.uint8),
        center,
    )

    orientations = [
        "Axial",
        "Coronal",
        "Sagittal",
    ]

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(13, 12),
    )

    for column, orientation in enumerate(
        orientations
    ):

        plot_overlay(
            axis=axes[0, column],
            image_slice=(
                template_views[
                    orientation
                ]
            ),
            mask_slice=(
                mask_views[
                    orientation
                ]
            ),
            title=(
                f"Template + Occipital "
                f"— {orientation}"
            ),
        )

        plot_overlay(
            axis=axes[1, column],
            image_slice=(
                affine_views[
                    orientation
                ]
            ),
            mask_slice=(
                mask_views[
                    orientation
                ]
            ),
            title=(
                f"Affine — {orientation}"
            ),
        )

        plot_overlay(
            axis=axes[2, column],
            image_slice=(
                rigid_views[
                    orientation
                ]
            ),
            mask_slice=(
                mask_views[
                    orientation
                ]
            ),
            title=(
                f"Rigid — {orientation}"
            ),
        )

    title = (
        f"{uid} — Affine vs Rigid Rescue QC\n"
        f"Affine: Dice={affine_dice:.3f}, "
        f"QC={affine_qc_category}"
        " | "
        f"Rigid: Dice={rigid_dice:.3f}, "
        f"QC={rigid_qc_category}"
    )

    if (
        affine_volume_ratio is not None
        and rigid_volume_ratio is not None
    ):
        title += (
            "\nOccipital volume ratio: "
            f"Affine={affine_volume_ratio:.3f} | "
            f"Rigid={rigid_volume_ratio:.3f}"
        )

    figure.suptitle(
        title,
        fontsize=13,
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.93,
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

def strip_nifti_suffix(
    file_name: str,
) -> str:

    if file_name.endswith(
        ".nii.gz"
    ):
        return file_name[:-7]

    if file_name.endswith(
        ".nii"
    ):
        return file_name[:-4]

    return file_name


def select_rescued_cases(
    dataframe: pd.DataFrame,
    number_of_cases: int,
) -> pd.DataFrame:
    """
    Select strongest rigid rescue cases.

    Primary rescue:
        affine fail -> rigid high confidence

    Secondary:
        affine fail -> rigid review

    Within groups:
        prioritize larger Dice improvement.
    """

    dataframe = dataframe.copy()

    dataframe[
        "rigid_dice_gain_over_affine"
    ] = (
        dataframe[
            "rigid_head_mask_dice"
        ]
        - dataframe[
            "robust_head_mask_dice"
        ]
    )

    rescued = dataframe[
        (
            dataframe[
                "robust_qc_category"
            ]
            == "fail_or_fallback"
        )
        &
        (
            dataframe[
                "rigid_qc_category"
            ]
            != "fail_or_fallback"
        )
    ].copy()

    if rescued.empty:
        raise RuntimeError(
            "No affine-failure cases were "
            "rescued by rigid registration."
        )

    qc_priority = {
        "high_confidence": 2,
        "review": 1,
        "fail_or_fallback": 0,
    }

    rescued[
        "rigid_qc_priority"
    ] = (
        rescued[
            "rigid_qc_category"
        ]
        .map(
            qc_priority
        )
    )

    rescued = rescued.sort_values(
        by=[
            "rigid_qc_priority",
            "rigid_dice_gain_over_affine",
        ],
        ascending=[
            False,
            False,
        ],
    )

    if number_of_cases > 0:
        rescued = rescued.head(
            number_of_cases
        )

    return rescued.reset_index(
        drop=True
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate visual QC comparing affine and "
            "rigid registrations for scans rescued "
            "by rigid registration."
        )
    )

    parser.add_argument(
        "--results-csv",
        required=True,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
    )

    parser.add_argument(
        "--template",
        required=True,
    )

    parser.add_argument(
        "--occipital-mask",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--number-of-cases",
        type=int,
        default=10,
        help=(
            "Number of strongest rescued scans "
            "to visualize. "
            "Use 0 to visualize all rescued scans."
        ),
    )

    args = parser.parse_args()

    results_csv = Path(
        args.results_csv
    )

    input_dir = Path(
        args.input_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        results_csv
    )

    required_columns = [
        "file_name",

        "robust_transform_path",
        "robust_qc_category",
        "robust_head_mask_dice",
        "robust_occipital_volume_ratio",

        "rigid_transform_path",
        "rigid_qc_category",
        "rigid_head_mask_dice",
        "rigid_occipital_volume_ratio",
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    selected = select_rescued_cases(
        dataframe=dataframe,
        number_of_cases=(
            args.number_of_cases
        ),
    )

    selected.to_csv(
        output_dir
        / "rigid_rescue_qc_selection.csv",
        index=False,
    )

    template = sitk.ReadImage(
        args.template,
        sitk.sitkFloat32,
    )

    occipital_mask = sitk.ReadImage(
        args.occipital_mask,
        sitk.sitkUInt8,
    )

    validate_same_geometry(
        template,
        occipital_mask,
        "FP-CIT template",
        "occipital mask",
    )

    print("=" * 72)
    print("RIGID RESCUE VISUAL QC")
    print("=" * 72)

    print(
        f"Selected rescued cases: "
        f"{len(selected)}"
    )

    print()

    failures = []
    successful = 0

    for index, row in selected.iterrows():

        file_name = str(
            row[
                "file_name"
            ]
        )

        uid = strip_nifti_suffix(
            file_name
        )

        print(
            f"[{index + 1}/{len(selected)}] "
            f"{file_name}"
        )

        print(
            "  Affine: "
            f"{row['robust_qc_category']} "
            f"Dice={row['robust_head_mask_dice']:.3f}"
        )

        print(
            "  Rigid:  "
            f"{row['rigid_qc_category']} "
            f"Dice={row['rigid_head_mask_dice']:.3f}"
        )

        print(
            "  Gain:   "
            f"{row['rigid_dice_gain_over_affine']:.3f}"
        )

        try:

            subject_path = (
                input_dir
                / file_name
            )

            if not subject_path.exists():
                raise FileNotFoundError(
                    f"Subject not found: "
                    f"{subject_path}"
                )

            affine_transform_path = Path(
                str(
                    row[
                        "robust_transform_path"
                    ]
                )
            )

            rigid_transform_path = Path(
                str(
                    row[
                        "rigid_transform_path"
                    ]
                )
            )

            if not affine_transform_path.exists():
                raise FileNotFoundError(
                    "Affine transform not found: "
                    f"{affine_transform_path}"
                )

            if not rigid_transform_path.exists():
                raise FileNotFoundError(
                    "Rigid transform not found: "
                    f"{rigid_transform_path}"
                )

            subject = sitk.ReadImage(
                str(subject_path),
                sitk.sitkFloat32,
            )

            affine_transform = (
                sitk.ReadTransform(
                    str(
                        affine_transform_path
                    )
                )
            )

            rigid_transform = (
                sitk.ReadTransform(
                    str(
                        rigid_transform_path
                    )
                )
            )

            output_path = (
                output_dir
                / (
                    f"{uid}_"
                    "affine_vs_rigid_qc.png"
                )
            )

            save_rigid_rescue_qc(
                subject=subject,
                template=template,
                occipital_mask=(
                    occipital_mask
                ),
                affine_transform=(
                    affine_transform
                ),
                rigid_transform=(
                    rigid_transform
                ),
                output_path=(
                    output_path
                ),
                uid=uid,
                affine_dice=float(
                    row[
                        "robust_head_mask_dice"
                    ]
                ),
                rigid_dice=float(
                    row[
                        "rigid_head_mask_dice"
                    ]
                ),
                affine_qc_category=str(
                    row[
                        "robust_qc_category"
                    ]
                ),
                rigid_qc_category=str(
                    row[
                        "rigid_qc_category"
                    ]
                ),
                affine_volume_ratio=float(
                    row[
                        "robust_occipital_volume_ratio"
                    ]
                ),
                rigid_volume_ratio=float(
                    row[
                        "rigid_occipital_volume_ratio"
                    ]
                ),
            )

            print(
                f"  Saved: "
                f"{output_path}"
            )

            successful += 1

        except Exception as exc:

            print(
                f"  FAILED: {exc}"
            )

            failures.append(
                {
                    "file_name": (
                        file_name
                    ),
                    "error": str(
                        exc
                    ),
                }
            )

    summary = {
        "analysis_type": (
            "rigid_rescue_visual_qc"
        ),

        "selection_definition": (
            "Existing dual-start affine "
            "fail_or_fallback cases that became "
            "review or high_confidence with rigid "
            "registration."
        ),

        "number_selected": int(
            len(selected)
        ),

        "successful": int(
            successful
        ),

        "failed": int(
            len(failures)
        ),

        "selection_priority": [
            (
                "Rigid high-confidence before "
                "rigid review."
            ),
            (
                "Within category, larger Dice gain "
                "over affine first."
            ),
        ],

        "important_note": (
            "This visual QC does not modify or "
            "normalize any NIfTI image."
        ),

        "failures": failures,
    }

    summary_path = (
        output_dir
        / "rigid_rescue_qc_statistics.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print()
    print("=" * 72)
    print("RIGID RESCUE QC COMPLETED")
    print("=" * 72)

    print(
        f"Successful: "
        f"{successful}"
    )

    print(
        f"Failed: "
        f"{len(failures)}"
    )

    print(
        f"Output: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
