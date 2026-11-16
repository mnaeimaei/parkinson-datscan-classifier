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
    Robust intensity scaling for visualization only.

    This does NOT modify preprocessing or normalization values.
    """
    array = np.asarray(
        array,
        dtype=np.float32,
    )

    finite_mask = np.isfinite(array)

    positive_values = array[
        finite_mask & (array > 0)
    ]

    if positive_values.size == 0:
        return np.zeros_like(
            array,
            dtype=np.float32,
        )

    low = float(
        np.percentile(
            positive_values,
            1,
        )
    )

    high = float(
        np.percentile(
            positive_values,
            99,
        )
    )

    if high <= low:
        high = float(
            positive_values.max()
        )

    output = np.zeros_like(
        array,
        dtype=np.float32,
    )

    if high > low:
        output[finite_mask] = (
            array[finite_mask] - low
        ) / (
            high - low
        )

    return np.clip(
        output,
        0.0,
        1.0,
    )


def validate_same_geometry(
    reference: sitk.Image,
    other: sitk.Image,
    reference_name: str,
    other_name: str,
) -> None:
    """
    Ensure the mask occupies exactly the same physical grid
    as the FP-CIT template.
    """
    if reference.GetSize() != other.GetSize():
        raise ValueError(
            f"{reference_name} and {other_name} "
            f"have different sizes: "
            f"{reference.GetSize()} vs "
            f"{other.GetSize()}"
        )

    if not np.allclose(
        reference.GetSpacing(),
        other.GetSpacing(),
        atol=1e-5,
    ):
        raise ValueError(
            f"{reference_name} and {other_name} "
            "have different voxel spacing."
        )

    if not np.allclose(
        reference.GetOrigin(),
        other.GetOrigin(),
        atol=1e-4,
    ):
        raise ValueError(
            f"{reference_name} and {other_name} "
            "have different origins."
        )

    if not np.allclose(
        reference.GetDirection(),
        other.GetDirection(),
        atol=1e-5,
    ):
        raise ValueError(
            f"{reference_name} and {other_name} "
            "have different directions."
        )


def resample_subject_to_template(
    subject: sitk.Image,
    template: sitk.Image,
    template_to_subject_transform: sitk.Transform,
) -> sitk.Image:
    """
    Resample the subject scan onto the FP-CIT template grid.

    In SimpleITK Resample(), the transform maps points from
    output space (template) to input space (subject).
    """
    return sitk.Resample(
        subject,
        template,
        template_to_subject_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def mask_center(
    mask_array: np.ndarray,
) -> tuple[int, int, int]:
    """
    Calculate the center of the bounding box of a binary mask.

    SimpleITK arrays are ordered:
        z, y, x
    """
    coordinates = np.where(
        mask_array > 0
    )

    if coordinates[0].size == 0:
        return (
            mask_array.shape[0] // 2,
            mask_array.shape[1] // 2,
            mask_array.shape[2] // 2,
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
    """
    Extract axial, coronal and sagittal slices.

    Array convention:
        [z, y, x]
    """
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


def _plot_mask_overlay(
    axis,
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    title: str,
) -> None:
    """
    Plot an image with mask contour.
    """
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

    axis.set_title(title)
    axis.axis("off")


def _plot_region_rows(
    axes,
    template_display: np.ndarray,
    registered_display: np.ndarray,
    mask_array: np.ndarray,
    start_row: int,
    region_name: str,
) -> None:
    """
    Plot two rows for a reference region:

    row N:
        FP-CIT template + mask

    row N+1:
        registered subject + same mask
    """
    center = mask_center(
        mask_array
    )

    template_views = extract_views(
        template_display,
        center,
    )

    registered_views = extract_views(
        registered_display,
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

    for column, orientation in enumerate(
        orientations
    ):
        _plot_mask_overlay(
            axis=axes[
                start_row,
                column,
            ],
            image_slice=template_views[
                orientation
            ],
            mask_slice=mask_views[
                orientation
            ],
            title=(
                f"Template + {region_name}"
                f" — {orientation}"
            ),
        )

        _plot_mask_overlay(
            axis=axes[
                start_row + 1,
                column,
            ],
            image_slice=registered_views[
                orientation
            ],
            mask_slice=mask_views[
                orientation
            ],
            title=(
                f"Registered subject + "
                f"{region_name}"
                f" — {orientation}"
            ),
        )


def save_template_reference_masks_qc(
    template: sitk.Image,
    occipital_mask: sitk.Image,
    cerebellar_mask: sitk.Image,
    output_path: str | Path,
) -> None:
    """
    Validate the anatomical reference masks directly on the
    FP-CIT template itself.

    This figure answers:

        Are the masks in sensible anatomical locations
        relative to the FP-CIT template?
    """
    output_path = Path(
        output_path
    )

    template_array = (
        sitk.GetArrayFromImage(
            template
        ).astype(np.float32)
    )

    occipital_array = (
        sitk.GetArrayFromImage(
            occipital_mask
        ) > 0
    )

    cerebellar_array = (
        sitk.GetArrayFromImage(
            cerebellar_mask
        ) > 0
    )

    template_display = (
        normalize_for_display(
            template_array
        )
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8),
    )

    regions = [
        (
            "Occipital",
            occipital_array,
            0,
        ),
        (
            "Cerebellar",
            cerebellar_array,
            1,
        ),
    ]

    for (
        region_name,
        mask_array,
        row,
    ) in regions:

        center = mask_center(
            mask_array
        )

        template_views = extract_views(
            template_display,
            center,
        )

        mask_views = extract_views(
            mask_array.astype(
                np.uint8
            ),
            center,
        )

        for column, orientation in enumerate(
            [
                "Axial",
                "Coronal",
                "Sagittal",
            ]
        ):
            _plot_mask_overlay(
                axis=axes[row, column],
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
                    f"{region_name} on FP-CIT "
                    f"template — {orientation}"
                ),
            )

    figure.suptitle(
        "FP-CIT Template Reference-Mask QC",
        fontsize=14,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_subject_template_space_qc(
    subject: sitk.Image,
    template: sitk.Image,
    occipital_mask: sitk.Image,
    cerebellar_mask: sitk.Image,
    template_to_subject_transform: sitk.Transform,
    output_path: str | Path,
    uid: str,
    head_mask_dice: float | None = None,
    occipital_positive_fraction: float | None = None,
    cerebellar_positive_fraction: float | None = None,
) -> None:
    """
    Main template-space QC.

    Produces four rows:

        Row 1:
            FP-CIT template + occipital mask

        Row 2:
            Registered subject + same occipital mask

        Row 3:
            FP-CIT template + cerebellar mask

        Row 4:
            Registered subject + same cerebellar mask

    All masks remain in template space.
    """
    output_path = Path(
        output_path
    )

    registered_subject = (
        resample_subject_to_template(
            subject=subject,
            template=template,
            template_to_subject_transform=(
                template_to_subject_transform
            ),
        )
    )

    template_array = (
        sitk.GetArrayFromImage(
            template
        ).astype(np.float32)
    )

    registered_array = (
        sitk.GetArrayFromImage(
            registered_subject
        ).astype(np.float32)
    )

    occipital_array = (
        sitk.GetArrayFromImage(
            occipital_mask
        ) > 0
    )

    cerebellar_array = (
        sitk.GetArrayFromImage(
            cerebellar_mask
        ) > 0
    )

    template_display = (
        normalize_for_display(
            template_array
        )
    )

    registered_display = (
        normalize_for_display(
            registered_array
        )
    )

    figure, axes = plt.subplots(
        4,
        3,
        figsize=(12, 15),
    )

    _plot_region_rows(
        axes=axes,
        template_display=template_display,
        registered_display=registered_display,
        mask_array=occipital_array,
        start_row=0,
        region_name="Occipital",
    )

    _plot_region_rows(
        axes=axes,
        template_display=template_display,
        registered_display=registered_display,
        mask_array=cerebellar_array,
        start_row=2,
        region_name="Cerebellar",
    )

    title = (
        f"{uid} — Template-space "
        f"Anatomical Reference QC"
    )

    details = []

    if head_mask_dice is not None:
        details.append(
            f"Dice={head_mask_dice:.3f}"
        )

    if (
        occipital_positive_fraction
        is not None
    ):
        details.append(
            "Occipital positive="
            f"{occipital_positive_fraction:.3f}"
        )

    if (
        cerebellar_positive_fraction
        is not None
    ):
        details.append(
            "Cerebellar positive="
            f"{cerebellar_positive_fraction:.3f}"
        )

    if details:
        title += (
            "\n"
            + " | ".join(details)
        )

    figure.suptitle(
        title,
        fontsize=14,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

def optional_float(
    row: pd.Series,
    column: str,
) -> float | None:
    """
    Read optional numerical value from DataFrame row.
    """
    if column not in row:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create template-space QC figures for "
            "FP-CIT registration and anatomical "
            "reference regions."
        )
    )

    parser.add_argument(
        "--selection-csv",
        required=True,
        help=(
            "registration_qc_selection.csv generated "
            "during previous visual-QC step."
        ),
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Directory containing resampled subject NIfTI scans."
        ),
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
        "--cerebellar-mask",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    selection_path = Path(
        args.selection_csv
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

    if not selection_path.exists():
        raise FileNotFoundError(
            f"Selection CSV not found: "
            f"{selection_path}"
        )

    dataframe = pd.read_csv(
        selection_path
    )

    if dataframe.empty:
        raise RuntimeError(
            "QC selection CSV is empty."
        )

    required_columns = [
        "file_name",
        "transform_path",
        "head_mask_dice",
        "qc_group",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------
    # Load template + masks
    # --------------------------------------------------

    template = sitk.ReadImage(
        args.template,
        sitk.sitkFloat32,
    )

    occipital_mask = sitk.ReadImage(
        args.occipital_mask,
        sitk.sitkUInt8,
    )

    cerebellar_mask = sitk.ReadImage(
        args.cerebellar_mask,
        sitk.sitkUInt8,
    )

    # --------------------------------------------------
    # Validate mask geometry
    # --------------------------------------------------

    validate_same_geometry(
        reference=template,
        other=occipital_mask,
        reference_name="FP-CIT template",
        other_name="occipital mask",
    )

    validate_same_geometry(
        reference=template,
        other=cerebellar_mask,
        reference_name="FP-CIT template",
        other_name="cerebellar mask",
    )

    print("=" * 72)
    print("TEMPLATE-SPACE ANATOMICAL REFERENCE QC")
    print("=" * 72)

    print(
        f"Selected scans: "
        f"{len(dataframe)}"
    )

    # --------------------------------------------------
    # First create template-only mask QC
    # --------------------------------------------------

    template_qc_path = (
        output_dir
        / "template_reference_masks_qc.png"
    )

    save_template_reference_masks_qc(
        template=template,
        occipital_mask=occipital_mask,
        cerebellar_mask=cerebellar_mask,
        output_path=template_qc_path,
    )

    print()
    print(
        "Template reference-mask QC:"
    )

    print(
        f"  {template_qc_path}"
    )

    # --------------------------------------------------
    # Subject QC
    # --------------------------------------------------

    failures: list[dict] = []

    successful = 0

    for index, row in dataframe.iterrows():

        file_name = str(
            row["file_name"]
        )

        qc_group = str(
            row["qc_group"]
        )

        uid = (
            str(row["uid"])
            if (
                "uid" in dataframe.columns
                and pd.notna(row["uid"])
            )
            else file_name
            .replace(".nii.gz", "")
            .replace(".nii", "")
        )

        print()
        print(
            f"[{index + 1}/{len(dataframe)}] "
            f"{qc_group}: {file_name}"
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

            transform_path = Path(
                str(
                    row[
                        "transform_path"
                    ]
                )
            )

            if not transform_path.exists():
                raise FileNotFoundError(
                    f"Transform not found: "
                    f"{transform_path}"
                )

            subject = sitk.ReadImage(
                str(subject_path),
                sitk.sitkFloat32,
            )

            transform = (
                sitk.ReadTransform(
                    str(transform_path)
                )
            )

            group_output_dir = (
                output_dir
                / qc_group
            )

            group_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                group_output_dir
                / (
                    f"{uid}_"
                    "template_space_reference_qc.png"
                )
            )

            save_subject_template_space_qc(
                subject=subject,
                template=template,
                occipital_mask=(
                    occipital_mask
                ),
                cerebellar_mask=(
                    cerebellar_mask
                ),
                template_to_subject_transform=(
                    transform
                ),
                output_path=output_path,
                uid=uid,
                head_mask_dice=(
                    optional_float(
                        row,
                        "head_mask_dice",
                    )
                ),
                occipital_positive_fraction=(
                    optional_float(
                        row,
                        "occipital_positive_fraction",
                    )
                ),
                cerebellar_positive_fraction=(
                    optional_float(
                        row,
                        "cerebellar_positive_fraction",
                    )
                ),
            )

            successful += 1

            print(
                f"  Saved: "
                f"{output_path}"
            )

        except Exception as exc:

            print(
                f"  FAILED: {exc}"
            )

            failures.append(
                {
                    "file_name": file_name,
                    "qc_group": qc_group,
                    "error": str(exc),
                }
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = {
        "analysis_type": (
            "template_space_anatomical_reference_qc"
        ),

        "number_selected": int(
            len(dataframe)
        ),

        "successful": int(
            successful
        ),

        "failed": int(
            len(failures)
        ),

        "template_reference_mask_qc": (
            str(template_qc_path)
        ),

        "purpose": [
            (
                "Validate occipital and cerebellar "
                "masks directly on the FP-CIT template."
            ),
            (
                "Validate registered subjects against "
                "the same fixed template-space masks."
            ),
            (
                "Do not use positive voxel fraction "
                "alone as evidence of anatomical "
                "registration correctness."
            ),
        ],

        "failures": failures,
    }

    summary_path = (
        output_dir
        / "template_space_reference_qc_summary.json"
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
    print("TEMPLATE-SPACE QC COMPLETED")
    print("=" * 72)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {len(failures)}"
    )

    print(
        f"Output: {output_dir}"
    )


if __name__ == "__main__":
    main()