from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Visualization only.

    Uses positive voxels to avoid background zeros dominating
    the intensity window. Gamma > 1 slightly suppresses
    mid-level uptake and makes high-uptake structures easier
    to distinguish.

    This does NOT modify any saved image.
    """
    image = np.asarray(image, dtype=np.float32)

    valid = (
        np.isfinite(image)
        & (image > 0)
    )

    values = image[valid]

    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    low = np.percentile(values, 5.0)
    high = np.percentile(values, 99.7)

    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    out = (image - low) / (high - low)
    out = np.clip(out, 0.0, 1.0)

    # Emphasize relatively high uptake for QC only.
    out = out ** 1.4

    # Keep non-positive/background voxels black.
    out[~valid] = 0.0

    return out


def clip_index(value: float, size: int) -> int:
    return int(
        np.clip(
            round(value),
            0,
            size - 1,
        )
    )


def draw_crosshair(
    ax,
    x: float,
    y: float,
) -> None:
    ax.axvline(
        x=x,
        linewidth=0.9,
    )

    ax.axhline(
        y=y,
        linewidth=0.9,
    )


def show_overlay(
    ax,
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    cross_x: float,
    cross_y: float,
    title: str,
) -> None:

    ax.imshow(
        image_slice,
        cmap="gray",
        origin="lower",
        interpolation="nearest",
    )

    ax.imshow(
        np.ma.masked_where(
            ~mask_slice,
            mask_slice,
        ),
        alpha=0.32,
        origin="lower",
        interpolation="nearest",
    )

    draw_crosshair(
        ax,
        cross_x,
        cross_y,
    )

    ax.set_title(title)
    ax.axis("off")


def find_left_right_sagittal_centers(
    mask: np.ndarray,
    bilateral_center_x: float,
) -> tuple[float, float]:

    coords = np.argwhere(mask)

    if coords.size == 0:
        raise RuntimeError(
            "Mapped striatal mask is empty."
        )

    # All normalized images were previously validated as RAS.
    #
    # RAS:
    # lower X  -> anatomical LEFT
    # higher X -> anatomical RIGHT
    #
    # Split around the bilateral mapped-mask center.

    left_coords = coords[
        coords[:, 0] < bilateral_center_x
    ]

    right_coords = coords[
        coords[:, 0] > bilateral_center_x
    ]

    if len(left_coords) == 0:
        raise RuntimeError(
            "Could not identify left part of mapped mask."
        )

    if len(right_coords) == 0:
        raise RuntimeError(
            "Could not identify right part of mapped mask."
        )

    left_x = float(
        left_coords[:, 0].mean()
    )

    right_x = float(
        right_coords[:, 0].mean()
    )

    return left_x, right_x


def make_enhanced_qc(
    uid: str,
    normalized_path: Path,
    mask_path: Path,
    center: np.ndarray,
    reason: str,
    output_path: Path,
) -> None:

    image_img = nib.load(
        str(normalized_path)
    )

    mask_img = nib.load(
        str(mask_path)
    )

    if image_img.shape != mask_img.shape:
        raise RuntimeError(
            f"{uid}: image/mask shape mismatch."
        )

    if not np.allclose(
        image_img.affine,
        mask_img.affine,
        atol=1e-4,
    ):
        raise RuntimeError(
            f"{uid}: image/mask affine mismatch."
        )

    orientation = nib.aff2axcodes(
        image_img.affine
    )

    if orientation != ("R", "A", "S"):
        raise RuntimeError(
            f"{uid}: expected RAS, got {orientation}"
        )

    image = np.asarray(
        image_img.dataobj,
        dtype=np.float32,
    )

    mask = (
        np.asarray(mask_img.dataobj) > 0
    )

    image_display = normalize_for_display(
        image
    )

    cx, cy, cz = (
        float(center[0]),
        float(center[1]),
        float(center[2]),
    )

    # ---------------------------------------------------------
    # Main center indices
    # ---------------------------------------------------------

    ix = clip_index(
        cx,
        image.shape[0],
    )

    iy = clip_index(
        cy,
        image.shape[1],
    )

    iz = clip_index(
        cz,
        image.shape[2],
    )

    # ---------------------------------------------------------
    # Left/right sagittal locations
    # ---------------------------------------------------------

    left_x, right_x = (
        find_left_right_sagittal_centers(
            mask,
            cx,
        )
    )

    left_ix = clip_index(
        left_x,
        image.shape[0],
    )

    right_ix = clip_index(
        right_x,
        image.shape[0],
    )

    # ---------------------------------------------------------
    # Nearby axial slices
    # ---------------------------------------------------------

    axial_indices = [
        max(0, iz - 2),
        iz,
        min(image.shape[2] - 1, iz + 2),
    ]

    # ---------------------------------------------------------
    # Plot
    #
    # Row 1:
    #   axial z-2
    #   axial center
    #   axial z+2
    #
    # Row 2:
    #   coronal center
    #   left sagittal
    #   right sagittal
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10),
    )

    # ---------------------------------------------------------
    # Axial -2
    # ---------------------------------------------------------

    z0 = axial_indices[0]

    show_overlay(
        axes[0, 0],
        image_display[:, :, z0].T,
        mask[:, :, z0].T,
        cx,
        cy,
        f"Axial z={z0}  (center − 2)",
    )

    # ---------------------------------------------------------
    # Axial center
    # ---------------------------------------------------------

    z1 = axial_indices[1]

    show_overlay(
        axes[0, 1],
        image_display[:, :, z1].T,
        mask[:, :, z1].T,
        cx,
        cy,
        f"Axial z={z1}  (center)",
    )

    # ---------------------------------------------------------
    # Axial +2
    # ---------------------------------------------------------

    z2 = axial_indices[2]

    show_overlay(
        axes[0, 2],
        image_display[:, :, z2].T,
        mask[:, :, z2].T,
        cx,
        cy,
        f"Axial z={z2}  (center + 2)",
    )

    # ---------------------------------------------------------
    # Coronal
    #
    # Display:
    # horizontal = X
    # vertical   = Z
    # ---------------------------------------------------------

    show_overlay(
        axes[1, 0],
        image_display[:, iy, :].T,
        mask[:, iy, :].T,
        cx,
        cz,
        f"Coronal y={iy}",
    )

    # ---------------------------------------------------------
    # LEFT sagittal
    #
    # Display:
    # horizontal = Y
    # vertical   = Z
    # ---------------------------------------------------------

    show_overlay(
        axes[1, 1],
        image_display[left_ix, :, :].T,
        mask[left_ix, :, :].T,
        cy,
        cz,
        (
            f"LEFT sagittal x={left_ix}\n"
            f"mask-center x={left_x:.2f}"
        ),
    )

    # ---------------------------------------------------------
    # RIGHT sagittal
    # ---------------------------------------------------------

    show_overlay(
        axes[1, 2],
        image_display[right_ix, :, :].T,
        mask[right_ix, :, :].T,
        cy,
        cz,
        (
            f"RIGHT sagittal x={right_ix}\n"
            f"mask-center x={right_x:.2f}"
        ),
    )

    fig.suptitle(
        (
            f"Step 7E — Enhanced L1 Visual QC — {uid}\n"
            f"Bilateral center voxel = "
            f"({cx:.2f}, {cy:.2f}, {cz:.2f})\n"
            f"{reason}"
        ),
        fontsize=14,
    )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Step 7E enhanced L1 visual QC."
        )
    )

    parser.add_argument(
        "--localization-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7c_l1_template_registration_data/"
            "step7c1_roi_l1_template_transform/l1_localization.csv"
        ),
    )

    parser.add_argument(
        "--original-manifest",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/"
            "step7c_l1_template_registration_data/"
            "step7c2_l1_visual_qc/"
            "l1_visual_qc_manifest.csv"
        ),
    )

    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step6e_intensity_normalization_data/step6e_normalize_occipital"
        ),
    )

    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/"
            "step7c_l1_template_registration_data/step7c1_roi_l1_template_transform"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/"
            "step7c_l1_template_registration_data/"
            "step7c3_l1_visual_qc_enhanced"
        ),
    )

    args = parser.parse_args()

    localization_csv = resolve_path(args.localization_csv)
    original_manifest = resolve_path(args.original_manifest)
    normalized_dir = resolve_path(args.normalized_dir)
    mask_dir = resolve_path(args.mask_dir)
    output_dir = resolve_path(args.output_dir)

    # ---------------------------------------------------------
    # Read Step 7C localization
    # ---------------------------------------------------------

    df = pd.read_csv(
        localization_csv
    )

    if df.empty:
        raise RuntimeError(
            "Localization CSV is empty."
        )

    # ---------------------------------------------------------
    # Use EXACT SAME 41 cases selected previously
    # ---------------------------------------------------------

    manifest = pd.read_csv(
        original_manifest
    )

    if len(manifest) == 0:
        raise RuntimeError(
            "Original Step-7E manifest is empty."
        )

    selected_uids = manifest[
        "uid"
    ].tolist()

    print()
    print("=" * 80)
    print(
        "STEP 7E — ENHANCED L1 VISUAL QC"
    )
    print("=" * 80)

    print(
        f"Cases from original manifest: "
        f"{len(selected_uids)}"
    )

    print()

    output_rows = []
    failures = []

    # ---------------------------------------------------------
    # Generate enhanced images
    # ---------------------------------------------------------

    for index, manifest_row in (
        manifest.iterrows()
    ):

        uid = str(
            manifest_row["uid"]
        )

        reason = str(
            manifest_row["reason"]
        )

        localization = df[
            df["uid"] == uid
        ]

        if len(localization) != 1:
            failures.append(
                {
                    "uid": uid,
                    "error": (
                        "UID missing or duplicated "
                        "in localization CSV."
                    ),
                }
            )
            continue

        row = localization.iloc[0]

        normalized_path = (
            normalized_dir
            / f"{uid}.nii.gz"
        )

        mask_path = (
            mask_dir
            / f"{uid}.nii.gz"
        )

        output_path = (
            output_dir
            / f"{uid}_l1_qc_enhanced.png"
        )

        try:

            if not normalized_path.exists():
                raise FileNotFoundError(
                    normalized_path
                )

            if not mask_path.exists():
                raise FileNotFoundError(
                    mask_path
                )

            center = np.asarray(
                [
                    row["l1_center_x"],
                    row["l1_center_y"],
                    row["l1_center_z"],
                ],
                dtype=np.float64,
            )

            make_enhanced_qc(
                uid=uid,
                normalized_path=normalized_path,
                mask_path=mask_path,
                center=center,
                reason=reason,
                output_path=output_path,
            )

            output_rows.append(
                {
                    "uid": uid,
                    "reason": reason,
                    "mapped_mask_voxels": int(
                        row[
                            "mapped_mask_voxels"
                        ]
                    ),
                    "bbox_size_x": int(
                        row["bbox_size_x"]
                    ),
                    "bbox_size_y": int(
                        row["bbox_size_y"]
                    ),
                    "bbox_size_z": int(
                        row["bbox_size_z"]
                    ),
                    "minimum_border_distance_voxels": int(
                        row[
                            "minimum_border_distance_voxels"
                        ]
                    ),
                    "qc_image": str(
                        output_path
                    ),
                    "manual_qc": "",
                    "manual_notes": "",
                }
            )

            print(
                f"[{index + 1:2d}/"
                f"{len(manifest)}] "
                f"{uid}"
            )

        except Exception as exc:

            failures.append(
                {
                    "uid": uid,
                    "error": str(exc),
                }
            )

    # ---------------------------------------------------------
    # Save enhanced manifest
    # ---------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_manifest = (
        output_dir
        / "l1_visual_qc_enhanced_manifest.csv"
    )

    pd.DataFrame(
        output_rows
    ).to_csv(
        output_manifest,
        index=False,
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "STEP 7E ENHANCED QC GENERATION RESULTS"
    )
    print("=" * 80)

    print(
        f"QC images generated: "
        f"{len(output_rows)}"
    )

    print(
        f"Generation failures: "
        f"{len(failures)}"
    )

    print()
    print("Output directory:")
    print(
        f"  {output_dir.resolve()}"
    )

    print()
    print("Enhanced manifest:")
    print(
        f"  {output_manifest.resolve()}"
    )

    if failures:

        print()
        print("FAILURES")
        print("-" * 80)

        for failure in failures:
            print(
                failure["uid"],
                "->",
                failure["error"],
            )

        raise RuntimeError(
            "Enhanced QC generation had failures."
        )

    print()
    print(
        "STEP 7E ENHANCED QC GENERATION: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
