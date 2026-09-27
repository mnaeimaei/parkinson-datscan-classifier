from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEMPLATE_DIR = (
    PROJECT_ROOT
    / "data/template/dat_spect"
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

BILATERAL_PATH = (
    MASK_DIR
    / "striatum_mask.nii.gz"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/preprocessing_image_data"
    / "step7a_striatal_target_validation_data"
    / "step7a1_validate_striatal_target"
    / "striatal_target_validation.json"
)


# ============================================================
# HELPERS
# ============================================================

def geometry_matches(
    reference_img: nib.Nifti1Image,
    other_img: nib.Nifti1Image,
    atol: float = 1e-5,
) -> bool:
    return (
        reference_img.shape == other_img.shape
        and np.allclose(reference_img.affine, other_img.affine, atol=atol)
    )


def load_binary_mask(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)

    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite values found in {path}")

    unique_values = np.unique(data)

    # Accept either {0, 1} or a single non-zero foreground value.
    foreground = data > 0
    mask = foreground.astype(np.uint8)

    if mask.sum() == 0:
        raise ValueError(f"Mask is empty: {path}")

    print(f"{path.name}: original unique values = {unique_values}")

    return img, mask


def voxel_center(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask > 0)
    return coords.mean(axis=0)


def voxel_bbox(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask > 0)

    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0)

    return minimum, maximum


def voxel_to_world(
    affine: np.ndarray,
    voxel_coord: np.ndarray,
) -> np.ndarray:
    homogeneous = np.append(voxel_coord, 1.0)
    world = affine @ homogeneous
    return world[:3]


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    for required in [
        TEMPLATE_PATH,
        LEFT_PATH,
        RIGHT_PATH,
        BILATERAL_PATH,
    ]:

        if not required.exists():

            raise FileNotFoundError(
                f"Required path missing: {required}"
            )

    template_img = nib.load(str(TEMPLATE_PATH))

    left_img, left = load_binary_mask(LEFT_PATH)
    right_img, right = load_binary_mask(RIGHT_PATH)
    bilateral_img, bilateral = load_binary_mask(BILATERAL_PATH)

    # ---------------------------------------------------------
    # Geometry checks
    # ---------------------------------------------------------

    geometry = {
        "left_matches_template": geometry_matches(template_img, left_img),
        "right_matches_template": geometry_matches(template_img, right_img),
        "bilateral_matches_template": geometry_matches(
            template_img,
            bilateral_img,
        ),
    }

    if not all(geometry.values()):
        raise RuntimeError(
            f"Mask/template geometry mismatch detected: {geometry}"
        )

    # ---------------------------------------------------------
    # Logical checks
    # ---------------------------------------------------------

    expected_bilateral = np.logical_or(left, right).astype(np.uint8)

    bilateral_matches_union = np.array_equal(
        bilateral,
        expected_bilateral,
    )

    overlap_voxels = int(
        np.logical_and(left > 0, right > 0).sum()
    )

    if not bilateral_matches_union:
        raise RuntimeError(
            "striatum_mask.nii.gz is not exactly equal to "
            "left_striatum_mask OR right_striatum_mask."
        )

    # ---------------------------------------------------------
    # Measurements
    # ---------------------------------------------------------

    left_center_voxel = voxel_center(left)
    right_center_voxel = voxel_center(right)
    bilateral_center_voxel = voxel_center(bilateral)

    left_center_mm = voxel_to_world(
        template_img.affine,
        left_center_voxel,
    )

    right_center_mm = voxel_to_world(
        template_img.affine,
        right_center_voxel,
    )

    bilateral_center_mm = voxel_to_world(
        template_img.affine,
        bilateral_center_voxel,
    )

    bbox_min, bbox_max = voxel_bbox(bilateral)

    zooms = template_img.header.get_zooms()[:3]

    summary = {
        "status": "PASS",
        "template": {
            "path": str(
                TEMPLATE_PATH.relative_to(PROJECT_ROOT)
            ),
            "shape": list(template_img.shape),
            "spacing_mm": [float(v) for v in zooms],
            "affine": template_img.affine.tolist(),
            "orientation": list(
                nib.aff2axcodes(template_img.affine)
            ),
        },
        "geometry_checks": geometry,
        "mask_checks": {
            "left_voxels": int(left.sum()),
            "right_voxels": int(right.sum()),
            "bilateral_voxels": int(bilateral.sum()),
            "left_right_overlap_voxels": overlap_voxels,
            "bilateral_equals_union": bilateral_matches_union,
        },
        "centers": {
            "left_voxel": [
                float(v) for v in left_center_voxel
            ],
            "right_voxel": [
                float(v) for v in right_center_voxel
            ],
            "bilateral_voxel": [
                float(v) for v in bilateral_center_voxel
            ],
            "left_physical_mm": [
                float(v) for v in left_center_mm
            ],
            "right_physical_mm": [
                float(v) for v in right_center_mm
            ],
            "bilateral_physical_mm": [
                float(v) for v in bilateral_center_mm
            ],
        },
        "bilateral_bbox": {
            "min_voxel": [
                int(v) for v in bbox_min
            ],
            "max_voxel": [
                int(v) for v in bbox_max
            ],
            "size_voxels": [
                int(v)
                for v in (bbox_max - bbox_min + 1)
            ],
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print("STEP 7A — STRIATAL TARGET VALIDATION")
    print("=" * 70)

    print(f"Template shape:       {template_img.shape}")
    print(f"Template spacing:     {zooms}")
    print(
        f"Template orientation: "
        f"{nib.aff2axcodes(template_img.affine)}"
    )

    print()
    print(f"Left voxels:          {left.sum()}")
    print(f"Right voxels:         {right.sum()}")
    print(f"Bilateral voxels:     {bilateral.sum()}")
    print(f"L/R overlap voxels:   {overlap_voxels}")

    print()
    print(
        "Left center voxel:    ",
        np.round(left_center_voxel, 2),
    )
    print(
        "Right center voxel:   ",
        np.round(right_center_voxel, 2),
    )
    print(
        "Bilateral center:     ",
        np.round(bilateral_center_voxel, 2),
    )

    print()
    print(
        "Bilateral center mm:  ",
        np.round(bilateral_center_mm, 2),
    )

    print()
    print(f"BBox min:             {bbox_min}")
    print(f"BBox max:             {bbox_max}")
    print(
        "BBox size:            ",
        bbox_max - bbox_min + 1,
    )

    print()
    print("Geometry:             PASS")
    print("Bilateral union:      PASS")
    print()
    print(f"Saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
