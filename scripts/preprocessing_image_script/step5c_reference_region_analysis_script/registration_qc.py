from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from src.registration.reference_region_mapper import (
    map_template_mask_to_subject,
)


def strip_nifti_suffix(file_name: str) -> str:
    if file_name.endswith(".nii.gz"):
        return file_name[:-7]

    if file_name.endswith(".nii"):
        return file_name[:-4]

    return file_name


def normalize_for_display(
    array: np.ndarray,
) -> np.ndarray:
    """
    Robust 0-1 scaling for visualization ONLY.

    This has no effect on preprocessing or normalization.
    """
    array = np.asarray(
        array,
        dtype=np.float32,
    )

    finite = np.isfinite(array)

    values = array[
        finite & (array > 0)
    ]

    if values.size == 0:
        return np.zeros_like(
            array,
            dtype=np.float32,
        )

    low = float(
        np.percentile(values, 1)
    )

    high = float(
        np.percentile(values, 99)
    )

    if high <= low:
        high = float(values.max())

    output = np.zeros_like(
        array,
        dtype=np.float32,
    )

    valid = finite

    if high > low:
        output[valid] = (
            array[valid] - low
        ) / (
            high - low
        )

    output = np.clip(
        output,
        0.0,
        1.0,
    )

    return output


def validate_same_geometry(
    image_a: sitk.Image,
    image_b: sitk.Image,
    name_a: str,
    name_b: str,
) -> None:
    """
    Confirm that template-space masks use the same
    physical grid as the FP-CIT template.
    """
    if image_a.GetSize() != image_b.GetSize():
        raise ValueError(
            f"{name_a} and {name_b} have different sizes: "
            f"{image_a.GetSize()} vs {image_b.GetSize()}"
        )

    if not np.allclose(
        image_a.GetSpacing(),
        image_b.GetSpacing(),
        atol=1e-5,
    ):
        raise ValueError(
            f"{name_a} and {name_b} have different spacing."
        )

    if not np.allclose(
        image_a.GetOrigin(),
        image_b.GetOrigin(),
        atol=1e-4,
    ):
        raise ValueError(
            f"{name_a} and {name_b} have different origins."
        )

    if not np.allclose(
        image_a.GetDirection(),
        image_b.GetDirection(),
        atol=1e-5,
    ):
        raise ValueError(
            f"{name_a} and {name_b} have different directions."
        )


def resample_subject_to_template(
    subject: sitk.Image,
    template: sitk.Image,
    template_to_subject_transform: sitk.Transform,
) -> sitk.Image:
    """
    Resample the subject image onto the template grid.

    SimpleITK's Resample transform maps points in the
    output/template space into the moving/subject space.
    """
    return sitk.Resample(
        subject,
        template,
        template_to_subject_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def foreground_center(
    array: np.ndarray,
) -> tuple[int, int, int]:
    """
    Return center of the bounding box of positive voxels.

    Input array follows SimpleITK ordering:
        z, y, x
    """
    mask = (
        np.isfinite(array)
        & (array > 0)
    )

    coordinates = np.where(mask)

    if coordinates[0].size == 0:
        return (
            array.shape[0] // 2,
            array.shape[1] // 2,
            array.shape[2] // 2,
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


def mask_center(
    mask: np.ndarray,
) -> tuple[int, int, int]:
    """
    Center of bounding box of a binary mask.

    Array ordering:
        z, y, x
    """
    coordinates = np.where(
        mask > 0
    )

    if coordinates[0].size == 0:
        return (
            mask.shape[0] // 2,
            mask.shape[1] // 2,
            mask.shape[2] // 2,
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

    SimpleITK arrays are z, y, x.
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


def save_registration_qc(
    subject: sitk.Image,
    template: sitk.Image,
    transform: sitk.Transform,
    output_path: str | Path,
    uid: str,
    dice: float | None = None,
) -> None:
    """
    Create a 3x3 figure:

        Template
        Registered subject
        Absolute difference

    for axial, coronal and sagittal views.
    """
    output_path = Path(
        output_path
    )

    registered = (
        resample_subject_to_template(
            subject=subject,
            template=template,
            template_to_subject_transform=transform,
        )
    )

    template_array = (
        sitk.GetArrayFromImage(
            template
        ).astype(np.float32)
    )

    registered_array = (
        sitk.GetArrayFromImage(
            registered
        ).astype(np.float32)
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

    difference = np.abs(
        template_display
        - registered_display
    )

    center = foreground_center(
        template_display
    )

    template_views = extract_views(
        template_display,
        center,
    )

    registered_views = extract_views(
        registered_display,
        center,
    )

    difference_views = extract_views(
        difference,
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
        figsize=(12, 12),
    )

    for column, orientation in enumerate(
        orientations
    ):
        axes[0, column].imshow(
            template_views[orientation],
            cmap="gray",
        )

        axes[0, column].set_title(
            f"Template — {orientation}"
        )

        axes[1, column].imshow(
            registered_views[orientation],
            cmap="gray",
        )

        axes[1, column].set_title(
            f"Registered subject — {orientation}"
        )

        axes[2, column].imshow(
            difference_views[orientation],
            cmap="gray",
        )

        axes[2, column].set_title(
            f"Absolute difference — {orientation}"
        )

    for axis in axes.flat:
        axis.axis("off")

    title = (
        f"{uid} — Registration QC"
    )

    if dice is not None:
        title += (
            f" — head-mask Dice={dice:.3f}"
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


def _plot_mask_views(
    axes,
    subject_display: np.ndarray,
    mask_array: np.ndarray,
    row: int,
    region_name: str,
) -> None:
    center = mask_center(
        mask_array
    )

    image_views = extract_views(
        subject_display,
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
        axis = axes[
            row,
            column
        ]

        axis.imshow(
            image_views[orientation],
            cmap="gray",
        )

        mask_slice = (
            mask_views[orientation]
        )

        if np.any(mask_slice):
            axis.contour(
                mask_slice,
                levels=[0.5],
                linewidths=1.5,
            )

        axis.set_title(
            f"{region_name} — {orientation}"
        )

        axis.axis("off")


def save_reference_mask_qc(
    subject: sitk.Image,
    template_occipital_mask: sitk.Image,
    template_cerebellar_mask: sitk.Image,
    transform: sitk.Transform,
    output_path: str | Path,
    uid: str,
    occipital_positive_fraction: float | None = None,
    cerebellar_positive_fraction: float | None = None,
) -> None:
    """
    Map template reference masks back into subject space
    and display them over the ORIGINAL resampled subject.

    Row 1 = occipital
    Row 2 = cerebellar
    """
    output_path = Path(
        output_path
    )

    occipital_subject = (
        map_template_mask_to_subject(
            template_mask=(
                template_occipital_mask
            ),
            subject_image=subject,
            subject_to_template_transform=(
                transform
            ),
        )
    )

    cerebellar_subject = (
        map_template_mask_to_subject(
            template_mask=(
                template_cerebellar_mask
            ),
            subject_image=subject,
            subject_to_template_transform=(
                transform
            ),
        )
    )

    subject_array = (
        sitk.GetArrayFromImage(
            subject
        ).astype(np.float32)
    )

    subject_display = (
        normalize_for_display(
            subject_array
        )
    )

    occipital_array = (
        sitk.GetArrayFromImage(
            occipital_subject
        ) > 0
    )

    cerebellar_array = (
        sitk.GetArrayFromImage(
            cerebellar_subject
        ) > 0
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8),
    )

    _plot_mask_views(
        axes=axes,
        subject_display=subject_display,
        mask_array=occipital_array,
        row=0,
        region_name="Occipital",
    )

    _plot_mask_views(
        axes=axes,
        subject_display=subject_display,
        mask_array=cerebellar_array,
        row=1,
        region_name="Cerebellar",
    )

    title = (
        f"{uid} — Reference-region QC"
    )

    if (
        occipital_positive_fraction
        is not None
    ):
        title += (
            f"\nOccipital positive="
            f"{occipital_positive_fraction:.3f}"
        )

    if (
        cerebellar_positive_fraction
        is not None
    ):
        title += (
            " | Cerebellar positive="
            f"{cerebellar_positive_fraction:.3f}"
        )

    figure.suptitle(
        title,
        fontsize=14,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.93]
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

def select_qc_scans(
    dataframe: pd.DataFrame,
    per_group: int,
) -> pd.DataFrame:
    """
    Automatically select worst, middle and best
    registrations according to head-mask Dice.
    """
    dataframe = dataframe.copy()

    dataframe["head_mask_dice"] = (
        pd.to_numeric(
            dataframe["head_mask_dice"],
            errors="coerce",
        )
    )

    dataframe = dataframe.dropna(
        subset=["head_mask_dice"]
    )

    if dataframe.empty:
        raise ValueError(
            "No valid head_mask_dice values found."
        )

    sorted_df = dataframe.sort_values(
        "head_mask_dice"
    ).reset_index(drop=True)

    worst = (
        sorted_df
        .head(per_group)
        .copy()
    )

    worst["qc_group"] = "worst"

    best = (
        sorted_df
        .tail(per_group)
        .copy()
    )

    best["qc_group"] = "best"

    median_dice = float(
        sorted_df[
            "head_mask_dice"
        ].median()
    )

    middle = (
        sorted_df.assign(
            distance_to_median=(
                sorted_df[
                    "head_mask_dice"
                ]
                - median_dice
            ).abs()
        )
        .sort_values(
            "distance_to_median"
        )
        .head(per_group)
        .drop(
            columns=[
                "distance_to_median"
            ]
        )
        .copy()
    )

    middle["qc_group"] = "middle"

    selected = pd.concat(
        [
            worst,
            middle,
            best,
        ],
        ignore_index=True,
    )

    # Avoid accidental duplicates for a very small pilot.
    selected = (
        selected
        .drop_duplicates(
            subset=["file_name"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate visual QC figures for "
            "DaT-SPECT affine registration and "
            "mapped reference regions."
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
        "--cerebellar-mask",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--per-group",
        type=int,
        default=3,
        help=(
            "Number of worst/middle/best scans "
            "to visualize. Default: 3."
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

    selected = select_qc_scans(
        dataframe=dataframe,
        per_group=args.per_group,
    )

    print("=" * 70)
    print("REGISTRATION / REFERENCE MASK VISUAL QC")
    print("=" * 70)

    print()
    print("Selected scans:")

    for _, row in selected.iterrows():
        print(
            f"  {row['qc_group']:6s} | "
            f"{row['file_name']} | "
            f"Dice={row['head_mask_dice']:.3f}"
        )

    selected.to_csv(
        output_dir
        / "registration_qc_selection.csv",
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

    cerebellar_mask = sitk.ReadImage(
        args.cerebellar_mask,
        sitk.sitkUInt8,
    )

    validate_same_geometry(
        template,
        occipital_mask,
        "FP-CIT template",
        "occipital mask",
    )

    validate_same_geometry(
        template,
        cerebellar_mask,
        "FP-CIT template",
        "cerebellar mask",
    )

    failures: list[dict] = []

    for index, row in selected.iterrows():
        file_name = str(
            row["file_name"]
        )

        uid = (
            str(row["uid"])
            if "uid" in row
            and pd.notna(row["uid"])
            else strip_nifti_suffix(
                file_name
            )
        )

        group = str(
            row["qc_group"]
        )

        print()
        print(
            f"[{index + 1}/{len(selected)}] "
            f"{group}: {file_name}"
        )

        try:
            subject_path = (
                input_dir
                / file_name
            )

            if not subject_path.exists():
                raise FileNotFoundError(
                    f"Subject image not found: "
                    f"{subject_path}"
                )

            transform_path = Path(
                str(
                    row["transform_path"]
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

            group_dir = (
                output_dir
                / group
            )

            group_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            registration_output = (
                group_dir
                / f"{uid}_registration_qc.png"
            )

            reference_output = (
                group_dir
                / f"{uid}_reference_masks_qc.png"
            )

            save_registration_qc(
                subject=subject,
                template=template,
                transform=transform,
                output_path=(
                    registration_output
                ),
                uid=uid,
                dice=float(
                    row["head_mask_dice"]
                ),
            )

            occipital_positive = None

            if (
                "occipital_positive_fraction"
                in row
                and pd.notna(
                    row[
                        "occipital_positive_fraction"
                    ]
                )
            ):
                occipital_positive = float(
                    row[
                        "occipital_positive_fraction"
                    ]
                )

            cerebellar_positive = None

            if (
                "cerebellar_positive_fraction"
                in row
                and pd.notna(
                    row[
                        "cerebellar_positive_fraction"
                    ]
                )
            ):
                cerebellar_positive = float(
                    row[
                        "cerebellar_positive_fraction"
                    ]
                )

            save_reference_mask_qc(
                subject=subject,
                template_occipital_mask=(
                    occipital_mask
                ),
                template_cerebellar_mask=(
                    cerebellar_mask
                ),
                transform=transform,
                output_path=(
                    reference_output
                ),
                uid=uid,
                occipital_positive_fraction=(
                    occipital_positive
                ),
                cerebellar_positive_fraction=(
                    cerebellar_positive
                ),
            )

            print(
                f"  Registration QC: "
                f"{registration_output}"
            )

            print(
                f"  Reference-mask QC: "
                f"{reference_output}"
            )

        except Exception as exc:
            print(
                f"  FAILED: {exc}"
            )

            failures.append(
                {
                    "file_name": (
                        file_name
                    ),
                    "qc_group": group,
                    "error": str(exc),
                }
            )

    summary = {
        "analysis_type": (
            "visual_registration_reference_qc"
        ),

        "number_selected": int(
            len(selected)
        ),

        "successful": int(
            len(selected)
            - len(failures)
        ),

        "failed": int(
            len(failures)
        ),

        "selection": {
            "criterion": (
                "head_mask_dice"
            ),

            "groups": [
                "worst",
                "middle",
                "best",
            ],

            "per_group_requested": int(
                args.per_group
            ),
        },

        "failures": failures,
    }

    with (
        output_dir
        / "registration_qc_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print()
    print("=" * 70)
    print("VISUAL QC COMPLETED")
    print("=" * 70)

    print(
        f"Successful: "
        f"{summary['successful']}"
    )

    print(
        f"Failed: "
        f"{summary['failed']}"
    )

    print(
        f"Output: {output_dir}"
    )


if __name__ == "__main__":
    main()