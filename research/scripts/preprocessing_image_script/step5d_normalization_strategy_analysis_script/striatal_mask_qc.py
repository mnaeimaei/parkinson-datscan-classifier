from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

def normalize_for_display(
    array: np.ndarray,
) -> np.ndarray:
    """
    Robust display-only normalization.

    This does NOT modify any preprocessing data.
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


def validate_geometry(
    template_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    mask_name: str,
) -> None:
    """
    Confirm that mask and template share exactly the same grid.
    """
    if template_img.shape != mask_img.shape:
        raise ValueError(
            f"{mask_name} shape differs from template: "
            f"{mask_img.shape} vs {template_img.shape}"
        )

    if not np.allclose(
        template_img.affine,
        mask_img.affine,
        atol=1e-5,
    ):
        raise ValueError(
            f"{mask_name} affine differs from template."
        )


def bounding_box(
    mask: np.ndarray,
) -> dict[str, int]:
    """
    Bounding box for a 3D binary mask.

    Nibabel array convention:
        x, y, z
    """
    coordinates = np.where(
        mask > 0
    )

    if coordinates[0].size == 0:
        raise ValueError(
            "Mask is empty."
        )

    return {
        "x_min": int(
            coordinates[0].min()
        ),
        "x_max": int(
            coordinates[0].max()
        ),
        "y_min": int(
            coordinates[1].min()
        ),
        "y_max": int(
            coordinates[1].max()
        ),
        "z_min": int(
            coordinates[2].min()
        ),
        "z_max": int(
            coordinates[2].max()
        ),
    }


def voxel_centroid(
    mask: np.ndarray,
) -> np.ndarray:
    coordinates = np.column_stack(
        np.where(mask > 0)
    )

    if len(coordinates) == 0:
        raise ValueError(
            "Cannot calculate centroid of empty mask."
        )

    return coordinates.mean(
        axis=0
    )


def world_centroid(
    mask: np.ndarray,
    affine: np.ndarray,
) -> list[float]:
    """
    Convert mask centroid from voxel coordinates to
    physical/world coordinates.
    """
    centroid = voxel_centroid(
        mask
    )

    homogeneous = np.array(
        [
            centroid[0],
            centroid[1],
            centroid[2],
            1.0,
        ]
    )

    world = affine @ homogeneous

    return [
        float(world[0]),
        float(world[1]),
        float(world[2]),
    ]


def select_axial_slices(
    bilateral_mask: np.ndarray,
    number_of_slices: int = 5,
) -> list[int]:
    """
    Choose representative axial slices spanning the striatum.
    """
    bbox = bounding_box(
        bilateral_mask
    )

    z_min = bbox["z_min"]
    z_max = bbox["z_max"]

    if z_min == z_max:
        return [z_min]

    slices = np.linspace(
        z_min,
        z_max,
        number_of_slices,
    )

    return sorted(
        set(
            int(round(value))
            for value in slices
        )
    )


def plot_contours(
    axis,
    image_slice: np.ndarray,
    left_slice: np.ndarray,
    right_slice: np.ndarray,
    title: str,
) -> None:
    axis.imshow(
        image_slice,
        cmap="gray",
    )

    if np.any(left_slice):
        axis.contour(
            left_slice,
            levels=[0.5],
            colors=["red"],
            linewidths=1.5,
        )

    if np.any(right_slice):
        axis.contour(
            right_slice,
            levels=[0.5],
            colors=["cyan"],
            linewidths=1.5,
        )

    axis.set_title(
        title
    )

    axis.axis(
        "off"
    )


def save_striatal_mask_qc(
    template_path: str | Path,
    left_mask_path: str | Path,
    right_mask_path: str | Path,
    bilateral_mask_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:

    template_path = Path(
        template_path
    )

    left_mask_path = Path(
        left_mask_path
    )

    right_mask_path = Path(
        right_mask_path
    )

    bilateral_mask_path = Path(
        bilateral_mask_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    template_img = nib.load(
        template_path
    )

    left_img = nib.load(
        left_mask_path
    )

    right_img = nib.load(
        right_mask_path
    )

    bilateral_img = nib.load(
        bilateral_mask_path
    )

    # --------------------------------------------------
    # Geometry validation
    # --------------------------------------------------

    validate_geometry(
        template_img,
        left_img,
        "left striatal mask",
    )

    validate_geometry(
        template_img,
        right_img,
        "right striatal mask",
    )

    validate_geometry(
        template_img,
        bilateral_img,
        "bilateral striatal mask",
    )

    # --------------------------------------------------
    # Arrays
    # --------------------------------------------------

    template = template_img.get_fdata(
        dtype=np.float32
    )

    left = (
        left_img.get_fdata() > 0.5
    )

    right = (
        right_img.get_fdata() > 0.5
    )

    bilateral = (
        bilateral_img.get_fdata() > 0.5
    )

    expected_bilateral = np.logical_or(
        left,
        right,
    )

    left_count = int(
        left.sum()
    )

    right_count = int(
        right.sum()
    )

    bilateral_count = int(
        bilateral.sum()
    )

    overlap_count = int(
        np.logical_and(
            left,
            right,
        ).sum()
    )

    bilateral_difference = int(
        np.logical_xor(
            bilateral,
            expected_bilateral,
        ).sum()
    )

    if left_count == 0:
        raise RuntimeError(
            "Left striatal mask is empty."
        )

    if right_count == 0:
        raise RuntimeError(
            "Right striatal mask is empty."
        )

    if overlap_count != 0:
        raise RuntimeError(
            "Left/right striatal masks overlap: "
            f"{overlap_count} voxels."
        )

    if bilateral_difference != 0:
        raise RuntimeError(
            "Bilateral mask is not exactly equal to "
            "left OR right masks."
        )

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    display = normalize_for_display(
        template
    )

    axial_slices = select_axial_slices(
        bilateral,
        number_of_slices=5,
    )

    # Bilateral centroid for coronal view.
    bilateral_center = voxel_centroid(
        bilateral
    )

    center_y = int(
        round(
            bilateral_center[1]
        )
    )

    # Individual hemisphere centroids for sagittal views.
    left_center = voxel_centroid(
        left
    )

    right_center = voxel_centroid(
        right
    )

    left_x = int(
        round(
            left_center[0]
        )
    )

    right_x = int(
        round(
            right_center[0]
        )
    )

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    figure = plt.figure(
        figsize=(16, 8)
    )

    # Five axial slices.
    for position, z_index in enumerate(
        axial_slices,
        start=1,
    ):
        axis = figure.add_subplot(
            2,
            5,
            position,
        )

        image_slice = np.rot90(
            display[:, :, z_index]
        )

        left_slice = np.rot90(
            left[:, :, z_index]
        )

        right_slice = np.rot90(
            right[:, :, z_index]
        )

        plot_contours(
            axis=axis,
            image_slice=image_slice,
            left_slice=left_slice,
            right_slice=right_slice,
            title=f"Axial z={z_index}",
        )

    # --------------------------------------------------
    # Coronal center
    # --------------------------------------------------

    axis = figure.add_subplot(
        2,
        5,
        6,
    )

    plot_contours(
        axis=axis,
        image_slice=np.rot90(
            display[:, center_y, :]
        ),
        left_slice=np.rot90(
            left[:, center_y, :]
        ),
        right_slice=np.rot90(
            right[:, center_y, :]
        ),
        title=(
            f"Coronal y={center_y}"
        ),
    )

    # --------------------------------------------------
    # Left sagittal
    # --------------------------------------------------

    axis = figure.add_subplot(
        2,
        5,
        7,
    )

    plot_contours(
        axis=axis,
        image_slice=np.rot90(
            display[left_x, :, :]
        ),
        left_slice=np.rot90(
            left[left_x, :, :]
        ),
        right_slice=np.rot90(
            right[left_x, :, :]
        ),
        title=(
            f"Sagittal through Left x={left_x}"
        ),
    )

    # --------------------------------------------------
    # Right sagittal
    # --------------------------------------------------

    axis = figure.add_subplot(
        2,
        5,
        8,
    )

    plot_contours(
        axis=axis,
        image_slice=np.rot90(
            display[right_x, :, :]
        ),
        left_slice=np.rot90(
            left[right_x, :, :]
        ),
        right_slice=np.rot90(
            right[right_x, :, :]
        ),
        title=(
            f"Sagittal through Right x={right_x}"
        ),
    )

    # --------------------------------------------------
    # Maximum-intensity projection
    # --------------------------------------------------

    axis = figure.add_subplot(
        2,
        5,
        9,
    )

    template_mip = np.rot90(
        display.max(axis=2)
    )

    left_mip = np.rot90(
        left.max(axis=2)
    )

    right_mip = np.rot90(
        right.max(axis=2)
    )

    plot_contours(
        axis=axis,
        image_slice=template_mip,
        left_slice=left_mip,
        right_slice=right_mip,
        title="Axial MIP",
    )

    # --------------------------------------------------
    # Legend / text panel
    # --------------------------------------------------

    axis = figure.add_subplot(
        2,
        5,
        10,
    )

    axis.axis(
        "off"
    )

    information = (
        "Contour legend\n"
        "Red = Left striatum\n"
        "Cyan = Right striatum\n\n"
        f"Left voxels: {left_count}\n"
        f"Right voxels: {right_count}\n"
        f"Bilateral: {bilateral_count}\n"
        f"Overlap: {overlap_count}\n\n"
        f"Orientation: "
        f"{nib.aff2axcodes(template_img.affine)}"
    )

    axis.text(
        0.05,
        0.95,
        information,
        va="top",
        ha="left",
        fontsize=11,
    )

    figure.suptitle(
        "Striatal Masks on FP-CIT Template",
        fontsize=15,
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    figure_path = (
        output_dir
        / "striatal_masks_on_template.png"
    )

    figure.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    statistics = {
        "analysis_type": (
            "striatal_mask_template_visual_qc"
        ),

        "template": str(
            template_path
        ),

        "template_shape": [
            int(value)
            for value in template_img.shape
        ],

        "template_orientation": [
            str(value)
            for value in nib.aff2axcodes(
                template_img.affine
            )
        ],

        "left": {
            "voxel_count": (
                left_count
            ),

            "voxel_centroid": [
                float(value)
                for value in left_center
            ],

            "world_centroid_mm": (
                world_centroid(
                    left,
                    template_img.affine,
                )
            ),

            "bounding_box": (
                bounding_box(left)
            ),
        },

        "right": {
            "voxel_count": (
                right_count
            ),

            "voxel_centroid": [
                float(value)
                for value in right_center
            ],

            "world_centroid_mm": (
                world_centroid(
                    right,
                    template_img.affine,
                )
            ),

            "bounding_box": (
                bounding_box(right)
            ),
        },

        "bilateral": {
            "voxel_count": (
                bilateral_count
            ),

            "bounding_box": (
                bounding_box(
                    bilateral
                )
            ),

            "left_right_overlap_voxels": (
                overlap_count
            ),

            "difference_from_left_or_right_voxels": (
                bilateral_difference
            ),
        },

        "axial_slices_visualized": (
            axial_slices
        ),

        "figure": str(
            figure_path
        ),

        "interpretation": (
            "This step performs visual/template-space QC "
            "only. No subject images are transformed or "
            "normalized."
        ),
    }

    statistics_path = (
        output_dir
        / "striatal_mask_qc_statistics.json"
    )

    with statistics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            statistics,
            file,
            indent=4,
        )

    return statistics

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visually validate left/right striatal masks "
            "against the FP-CIT template."
        )
    )

    parser.add_argument(
        "--template",
        required=True,
    )

    parser.add_argument(
        "--left-mask",
        required=True,
    )

    parser.add_argument(
        "--right-mask",
        required=True,
    )

    parser.add_argument(
        "--bilateral-mask",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    print("=" * 72)
    print("STRIATAL MASK TEMPLATE QC")
    print("=" * 72)

    statistics = save_striatal_mask_qc(
        template_path=args.template,
        left_mask_path=args.left_mask,
        right_mask_path=args.right_mask,
        bilateral_mask_path=args.bilateral_mask,
        output_dir=args.output_dir,
    )

    print()
    print(
        "Template orientation: "
        f"{statistics['template_orientation']}"
    )

    print()
    print("Left striatum:")
    print(
        f"  Voxels: "
        f"{statistics['left']['voxel_count']}"
    )
    print(
        f"  World centroid: "
        f"{statistics['left']['world_centroid_mm']}"
    )

    print()
    print("Right striatum:")
    print(
        f"  Voxels: "
        f"{statistics['right']['voxel_count']}"
    )
    print(
        f"  World centroid: "
        f"{statistics['right']['world_centroid_mm']}"
    )

    print()
    print(
        "Left/right overlap: "
        f"{statistics['bilateral']['left_right_overlap_voxels']}"
    )

    print(
        "Bilateral-mask difference: "
        f"{statistics['bilateral']['difference_from_left_or_right_voxels']}"
    )

    print()
    print("=" * 72)
    print("STRIATAL MASK QC COMPLETED")
    print("=" * 72)

    print(
        f"Figure: "
        f"{statistics['figure']}"
    )


if __name__ == "__main__":
    main()
