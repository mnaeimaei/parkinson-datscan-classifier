from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEMPLATE_DIR = (
    PROJECT_ROOT
    / "data"
    / "template"
    / "dat_spect"
)

MASK_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data"
    / "step5d_normalization_strategy_analysis_data"
    / "build_striatal_masks"
)

TEMPLATE_PATH = (
    TEMPLATE_DIR
    / "fpcit_template_mni.nii"
)

LEFT_PATH = (
    MASK_DIR
    / "left_striatum_mask.nii.gz"
)

RIGHT_PATH = (
    MASK_DIR
    / "right_striatum_mask.nii.gz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "preprocessing_image_data"
    / "step7a_striatal_target_validation_data"
    / "step7a2_striatal_target_visual_qc"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "striatal_target_overlay.png"
)


# ============================================================
# HELPERS
# ============================================================

def load_canonical(path: Path):
    """
    Load a NIfTI and create an in-memory RAS-oriented version.

    IMPORTANT:
    This is ONLY for visualization.
    The original frozen template/masks are never modified.
    """
    img = nib.load(str(path))
    return nib.as_closest_canonical(img)


def mask_center(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        raise ValueError("Mask is empty.")

    return coords.mean(axis=0)


# ============================================================
# MAIN
# ============================================================

def main():

    for required in [
        TEMPLATE_PATH,
        LEFT_PATH,
        RIGHT_PATH,
    ]:

        if not required.exists():

            raise FileNotFoundError(
                f"Required path missing: {required}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load RAS-oriented copies ONLY for visualization
    # -----------------------------------------------------

    template_img = load_canonical(TEMPLATE_PATH)
    left_img = load_canonical(LEFT_PATH)
    right_img = load_canonical(RIGHT_PATH)

    # -----------------------------------------------------
    # Geometry validation after canonicalization
    # -----------------------------------------------------

    if template_img.shape != left_img.shape:
        raise RuntimeError("Left mask shape mismatch.")

    if template_img.shape != right_img.shape:
        raise RuntimeError("Right mask shape mismatch.")

    if not np.allclose(
        template_img.affine,
        left_img.affine,
        atol=1e-5,
    ):
        raise RuntimeError("Left mask affine mismatch.")

    if not np.allclose(
        template_img.affine,
        right_img.affine,
        atol=1e-5,
    ):
        raise RuntimeError("Right mask affine mismatch.")

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------

    template = np.asarray(
        template_img.dataobj,
        dtype=np.float32,
    )

    left = np.asarray(left_img.dataobj) > 0
    right = np.asarray(right_img.dataobj) > 0

    bilateral = left | right

    # -----------------------------------------------------
    # Centers
    # -----------------------------------------------------

    left_center = mask_center(left)
    right_center = mask_center(right)
    bilateral_center = mask_center(bilateral)

    axial_z = int(round(bilateral_center[2]))
    coronal_y = int(round(bilateral_center[1]))

    left_sagittal_x = int(round(left_center[0]))
    right_sagittal_x = int(round(right_center[0]))

    print()
    print("=" * 72)
    print("STEP 7A-2 — STRIATAL TARGET VISUAL QC")
    print("=" * 72)

    print(f"Canonical orientation: {nib.aff2axcodes(template_img.affine)}")
    print(f"Shape:                 {template_img.shape}")

    print()
    print(f"Left center:            {np.round(left_center, 2)}")
    print(f"Right center:           {np.round(right_center, 2)}")
    print(f"Bilateral center:       {np.round(bilateral_center, 2)}")

    print()
    print(f"Axial slice Z:          {axial_z}")
    print(f"Coronal slice Y:        {coronal_y}")
    print(f"Left sagittal X:        {left_sagittal_x}")
    print(f"Right sagittal X:       {right_sagittal_x}")

    # -----------------------------------------------------
    # Slice extraction
    # -----------------------------------------------------

    views = [
        {
            "name": f"Axial\nz={axial_z}",
            "image": template[:, :, axial_z].T,
            "left": left[:, :, axial_z].T,
            "right": right[:, :, axial_z].T,
        },
        {
            "name": f"Coronal\ny={coronal_y}",
            "image": template[:, coronal_y, :].T,
            "left": left[:, coronal_y, :].T,
            "right": right[:, coronal_y, :].T,
        },
        {
            "name": f"Left sagittal\nx={left_sagittal_x}",
            "image": template[left_sagittal_x, :, :].T,
            "left": left[left_sagittal_x, :, :].T,
            "right": right[left_sagittal_x, :, :].T,
        },
        {
            "name": f"Right sagittal\nx={right_sagittal_x}",
            "image": template[right_sagittal_x, :, :].T,
            "left": left[right_sagittal_x, :, :].T,
            "right": right[right_sagittal_x, :, :].T,
        },
    ]

    # -----------------------------------------------------
    # Plot
    #
    # Row 1 = template
    # Row 2 = masks
    # Row 3 = overlay
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(16, 12),
    )

    for col, view in enumerate(views):

        image = view["image"]
        left_slice = view["left"]
        right_slice = view["right"]

        # -------------------------
        # Row 1 — template
        # -------------------------

        ax = axes[0, col]

        ax.imshow(
            image,
            cmap="gray",
            origin="lower",
        )

        ax.set_title(view["name"])
        ax.axis("off")

        # -------------------------
        # Row 2 — masks only
        # -------------------------

        ax = axes[1, col]

        ax.imshow(
            np.zeros_like(image),
            cmap="gray",
            origin="lower",
        )

        ax.imshow(
            np.ma.masked_where(~left_slice, left_slice),
            cmap="Reds",
            alpha=0.9,
            origin="lower",
        )

        ax.imshow(
            np.ma.masked_where(~right_slice, right_slice),
            cmap="Blues",
            alpha=0.9,
            origin="lower",
        )

        ax.axis("off")

        # -------------------------
        # Row 3 — overlay
        # -------------------------

        ax = axes[2, col]

        ax.imshow(
            image,
            cmap="gray",
            origin="lower",
        )

        ax.imshow(
            np.ma.masked_where(~left_slice, left_slice),
            cmap="Reds",
            alpha=0.40,
            origin="lower",
        )

        ax.imshow(
            np.ma.masked_where(~right_slice, right_slice),
            cmap="Blues",
            alpha=0.40,
            origin="lower",
        )

        ax.axis("off")

    axes[0, 0].set_ylabel("Template")
    axes[1, 0].set_ylabel("Masks")
    axes[2, 0].set_ylabel("Overlay")

    fig.suptitle(
        "Step 7A — FP-CIT Template Striatal Target QC\n"
        "Red = left striatum | Blue = right striatum",
        fontsize=16,
    )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Saved QC image:")
    print(f"  {OUTPUT_PATH}")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
