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

INPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step4_voxel_resampler_data"
)

TEMPLATE_PATH = (
    PROJECT_ROOT
    / "data/template/dat_spect/fpcit_template_mni.nii"
)

STEP6B1_MANIFEST = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6b_registration_rescue_data/step6b1_finalize_registration_transforms"
    / "final_transform_manifest.csv"
)

STEP6A4_AUDIT = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6a_initial_registration_data/step6a4_full_registration_audit"
    / "registration_audit.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6b_registration_rescue_data/step6b3_rescue_pnsm_similarity"
)

TRANSFORM_DIR = OUTPUT_DIR / "transforms"
QC_DIR = OUTPUT_DIR / "qc_images"

RESULT_CSV = (
    OUTPUT_DIR
    / "similarity_rescue_result.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "similarity_rescue_summary.json"
)


# ============================================================
# QC THRESHOLDS
# ============================================================

HIGH_CONFIDENCE_DICE = 0.70
REVIEW_DICE = 0.50


# ============================================================
# BASIC HELPERS
# ============================================================

def foreground_mask(
    image: sitk.Image,
) -> sitk.Image:
    """
    Keep the same foreground definition used in
    Step 6A/6B so Dice values remain comparable.
    """

    return sitk.Cast(
        image > 0,
        sitk.sitkUInt8,
    )


def qc_category(
    dice: float,
) -> str:

    if dice >= HIGH_CONFIDENCE_DICE:
        return "high_confidence"

    if dice >= REVIEW_DICE:
        return "review"

    return "failed"


def dice_score(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    transform: sitk.Transform,
) -> float:
    """
    Transform moving foreground mask into template space
    and calculate Dice.
    """

    moved_mask = sitk.Resample(
        moving_mask,
        fixed_mask,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    fixed_array = (
        sitk.GetArrayViewFromImage(
            fixed_mask
        ) > 0
    )

    moved_array = (
        sitk.GetArrayViewFromImage(
            moved_mask
        ) > 0
    )

    intersection = np.count_nonzero(
        fixed_array & moved_array
    )

    denominator = (
        np.count_nonzero(fixed_array)
        +
        np.count_nonzero(moved_array)
    )

    if denominator == 0:
        return 0.0

    return float(
        2.0
        * intersection
        / denominator
    )


# ============================================================
# PHYSICAL FOREGROUND EXTENT
# ============================================================

def foreground_physical_extent(
    image: sitk.Image,
) -> np.ndarray:
    """
    Estimate foreground physical extent in z, y, x.

    NumPy array order:
        z, y, x

    SimpleITK spacing order:
        x, y, z
    """

    array = sitk.GetArrayViewFromImage(
        image
    )

    mask = (
        np.isfinite(array)
        &
        (array > 0)
    )

    coords = np.argwhere(mask)

    if len(coords) == 0:
        raise ValueError(
            "Image has no positive foreground voxels."
        )

    minimum = coords.min(
        axis=0
    )

    maximum = coords.max(
        axis=0
    )

    voxel_extent = (
        maximum
        -
        minimum
        +
        1
    ).astype(np.float64)

    spacing_xyz = np.asarray(
        image.GetSpacing(),
        dtype=np.float64,
    )

    spacing_zyx = spacing_xyz[::-1]

    physical_extent = (
        voxel_extent
        *
        spacing_zyx
    )

    return physical_extent


def estimate_similarity_scale(
    fixed: sitk.Image,
    moving: sitk.Image,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
]:
    """
    Estimate isotropic scale.

    IMPORTANT:
    The transform passed to sitk.Resample maps:

        fixed/output physical coordinates
                    ↓
        moving/input physical coordinates

    Therefore, if the moving brain occupies a smaller
    physical extent than the fixed/template brain,
    the useful scale is generally < 1.

    We use the median extent ratio for robustness.
    """

    fixed_extent = (
        foreground_physical_extent(
            fixed
        )
    )

    moving_extent = (
        foreground_physical_extent(
            moving
        )
    )

    ratios = (
        moving_extent
        /
        fixed_extent
    )

    estimated_scale = float(
        np.median(ratios)
    )

    return (
        estimated_scale,
        fixed_extent,
        moving_extent,
    )


# ============================================================
# CENTERED INITIALIZATION
# ============================================================

def create_rigid_initial(
    fixed: sitk.Image,
    moving: sitk.Image,
    initialization: str,
) -> sitk.Euler3DTransform:

    if initialization == "moments":

        mode = (
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )

    elif initialization == "geometry":

        mode = (
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

    else:

        raise ValueError(
            f"Unknown initialization: "
            f"{initialization}"
        )

    transform = (
        sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            mode,
        )
    )

    return sitk.Euler3DTransform(
        transform
    )


# ============================================================
# BUILD SIMILARITY TRANSFORM
# ============================================================

def rigid_to_similarity(
    rigid: sitk.Euler3DTransform,
    scale: float,
) -> sitk.Similarity3DTransform:
    """
    Convert rigid initialization to a Similarity3DTransform
    and add isotropic scaling.

    Rotation, center and translation are preserved.
    """

    transform = sitk.Similarity3DTransform()

    transform.SetCenter(
        rigid.GetCenter()
    )

    # Euler rigid rotation matrix.
    transform.SetMatrix(
        rigid.GetMatrix()
    )

    transform.SetTranslation(
        rigid.GetTranslation()
    )

    transform.SetScale(
        float(scale)
    )

    return transform


# ============================================================
# SCALE SEARCH
# ============================================================

def unique_sorted_scales(
    values: np.ndarray | list[float],
) -> list[float]:

    cleaned = []

    for value in values:

        value = float(value)

        if not np.isfinite(value):
            continue

        # Extremely broad range only as a safety guard.
        if value < 0.20:
            continue

        if value > 2.50:
            continue

        cleaned.append(value)

    return sorted(
        set(
            round(
                value,
                8,
            )
            for value in cleaned
        )
    )


def search_scale(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    rigid_initial: sitk.Euler3DTransform,
    estimated_scale: float,
) -> dict:
    """
    Two-stage deterministic scale search.

    No intensity optimizer is used.

    Stage 1:
        broad search around physical-size estimate

    Stage 2:
        fine search around best coarse scale
    """

    # --------------------------------------------------------
    # Coarse scale search
    # --------------------------------------------------------

    lower = (
        estimated_scale
        *
        0.65
    )

    upper = (
        estimated_scale
        *
        1.35
    )

    coarse_scales = np.linspace(
        lower,
        upper,
        141,
    )

    # Always include:
    # - estimated scale
    # - scale 1.0 = original rigid transform
    coarse_scales = np.concatenate(
        [
            coarse_scales,
            np.asarray(
                [
                    estimated_scale,
                    1.0,
                ]
            ),
        ]
    )

    coarse_scales = (
        unique_sorted_scales(
            coarse_scales
        )
    )

    coarse_results = []

    for scale in coarse_scales:

        transform = (
            rigid_to_similarity(
                rigid_initial,
                scale,
            )
        )

        dice = dice_score(
            fixed_mask,
            moving_mask,
            transform,
        )

        coarse_results.append(
            {
                "scale": scale,
                "dice": dice,
            }
        )

    coarse_best = max(
        coarse_results,
        key=lambda item:
            item["dice"],
    )

    # --------------------------------------------------------
    # Fine scale search
    # --------------------------------------------------------

    best_scale = float(
        coarse_best["scale"]
    )

    fine_lower = (
        best_scale
        *
        0.95
    )

    fine_upper = (
        best_scale
        *
        1.05
    )

    fine_scales = np.linspace(
        fine_lower,
        fine_upper,
        201,
    )

    fine_scales = (
        unique_sorted_scales(
            fine_scales
        )
    )

    fine_results = []

    for scale in fine_scales:

        transform = (
            rigid_to_similarity(
                rigid_initial,
                scale,
            )
        )

        dice = dice_score(
            fixed_mask,
            moving_mask,
            transform,
        )

        fine_results.append(
            {
                "scale": scale,
                "dice": dice,
            }
        )

    fine_best = max(
        fine_results,
        key=lambda item:
            item["dice"],
    )

    final_scale = float(
        fine_best["scale"]
    )

    final_transform = (
        rigid_to_similarity(
            rigid_initial,
            final_scale,
        )
    )

    return {
        "estimated_scale":
            estimated_scale,

        "coarse_best_scale":
            float(
                coarse_best["scale"]
            ),

        "coarse_best_dice":
            float(
                coarse_best["dice"]
            ),

        "best_scale":
            final_scale,

        "best_dice":
            float(
                fine_best["dice"]
            ),

        "transform":
            final_transform,

        "coarse_results":
            coarse_results,

        "fine_results":
            fine_results,
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================

def robust_normalize(
    array: np.ndarray,
) -> np.ndarray:

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
        or
        not np.isfinite(hi)
        or
        hi <= lo
    ):

        return np.zeros_like(
            array,
            dtype=np.float32,
        )

    result = (
        array - lo
    ) / (
        hi - lo
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def foreground_center(
    array: np.ndarray,
) -> tuple[int, int, int]:

    mask = (
        np.isfinite(array)
        &
        (array > 0)
    )

    if np.count_nonzero(mask) > 0:

        coords = np.argwhere(
            mask
        )

        z, y, x = np.round(
            coords.mean(axis=0)
        ).astype(int)

    else:

        z, y, x = (
            np.asarray(
                array.shape
            )
            // 2
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
    center: tuple[int, int, int],
) -> dict[str, np.ndarray]:

    z, y, x = center

    return {
        "Axial":
            np.rot90(
                array[z, :, :]
            ),

        "Coronal":
            np.rot90(
                array[:, y, :]
            ),

        "Sagittal":
            np.rot90(
                array[:, :, x]
            ),
    }


# ============================================================
# QC FIGURE
# ============================================================

def create_qc_figure(
    uid: str,
    moving: sitk.Image,
    fixed: sitk.Image,
    rigid_registered: sitk.Image,
    similarity_registered: sitk.Image,
    rigid_dice: float,
    similarity_dice: float,
    scale: float,
    initialization: str,
    output_path: Path,
) -> None:

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

    rigid_array = (
        sitk.GetArrayFromImage(
            rigid_registered
        )
    )

    similarity_array = (
        sitk.GetArrayFromImage(
            similarity_registered
        )
    )

    moving_display = (
        robust_normalize(
            moving_array
        )
    )

    fixed_display = (
        robust_normalize(
            fixed_array
        )
    )

    rigid_display = (
        robust_normalize(
            rigid_array
        )
    )

    similarity_display = (
        robust_normalize(
            similarity_array
        )
    )

    moving_center = (
        foreground_center(
            moving_array
        )
    )

    template_center = (
        foreground_center(
            fixed_array
        )
    )

    moving_views = extract_views(
        moving_display,
        moving_center,
    )

    fixed_views = extract_views(
        fixed_display,
        template_center,
    )

    rigid_views = extract_views(
        rigid_display,
        template_center,
    )

    similarity_views = extract_views(
        similarity_display,
        template_center,
    )

    planes = [
        "Axial",
        "Coronal",
        "Sagittal",
    ]

    # Columns:
    # Original
    # Template
    # Rigid
    # Rigid overlay
    # Similarity
    # Similarity overlay

    fig, axes = plt.subplots(
        nrows=3,
        ncols=6,
        figsize=(22, 12),
    )

    for row, plane in enumerate(
        planes
    ):

        axes[row, 0].imshow(
            moving_views[plane],
            cmap="gray",
        )

        axes[row, 0].set_title(
            f"Original\n{plane}"
        )

        axes[row, 1].imshow(
            fixed_views[plane],
            cmap="gray",
        )

        axes[row, 1].set_title(
            f"Template\n{plane}"
        )

        axes[row, 2].imshow(
            rigid_views[plane],
            cmap="gray",
        )

        axes[row, 2].set_title(
            f"Rigid initial\n{plane}"
        )

        axes[row, 3].imshow(
            fixed_views[plane],
            cmap="gray",
        )

        axes[row, 3].imshow(
            rigid_views[plane],
            cmap="magma",
            alpha=0.45,
        )

        axes[row, 3].set_title(
            f"Rigid overlay\n{plane}"
        )

        axes[row, 4].imshow(
            similarity_views[plane],
            cmap="gray",
        )

        axes[row, 4].set_title(
            f"Similarity rescue\n{plane}"
        )

        axes[row, 5].imshow(
            fixed_views[plane],
            cmap="gray",
        )

        axes[row, 5].imshow(
            similarity_views[plane],
            cmap="magma",
            alpha=0.45,
        )

        axes[row, 5].set_title(
            f"Similarity overlay\n{plane}"
        )

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        (
            f"{uid}\n"
            f"Rigid Dice={rigid_dice:.4f}  |  "
            f"Similarity Dice={similarity_dice:.4f}  |  "
            f"Scale={scale:.4f}  |  "
            f"start={initialization}"
        ),
        fontsize=15,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.95)
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
# FAILED-UID SELECTION
# ============================================================

def collect_failed_uids(
    explicit_uids: list[str] | None,
) -> list[str]:
    """
    Resolve which scans need similarity rescue.

    --uid, if given, wins.

    Otherwise:
      1. Step 6b1 unresolved rows
      2. Step 6a4 audit_qc == failed
    """

    if explicit_uids:

        uids: list[str] = []

        for item in explicit_uids:

            for part in str(item).split(","):

                part = part.strip()

                if part:
                    uids.append(part)

        return list(dict.fromkeys(uids))

    if STEP6B1_MANIFEST.exists():

        df = pd.read_csv(
            STEP6B1_MANIFEST,
            dtype={"uid": str},
        )

        if "selection_status" in df.columns:

            unresolved = df[
                df["selection_status"].astype(str)
                == "unresolved"
            ]

            return (
                unresolved["uid"]
                .astype(str)
                .tolist()
            )

    if STEP6A4_AUDIT.exists():

        df = pd.read_csv(
            STEP6A4_AUDIT,
            dtype={"uid": str},
        )

        if "audit_qc" in df.columns:

            failed = df[
                df["audit_qc"].astype(str)
                == "failed"
            ]

            return (
                failed["uid"]
                .astype(str)
                .tolist()
            )

    raise FileNotFoundError(
        "No UIDs given and no failed/unresolved "
        "list was found. Pass --uid or run "
        "Step 6a4 / 6b1 first."
    )


def rescue_one_scan(
    uid: str,
    fixed: sitk.Image,
) -> dict:

    moving_path = (
        INPUT_DIR
        / f"{uid}.nii.gz"
    )

    if not moving_path.exists():

        raise FileNotFoundError(
            f"Required path missing: {moving_path}"
        )

    moving = sitk.ReadImage(
        str(moving_path),
        sitk.sitkFloat32,
    )

    fixed_mask = foreground_mask(
        fixed
    )

    moving_mask = foreground_mask(
        moving
    )

    (
        estimated_scale,
        fixed_extent,
        moving_extent,
    ) = estimate_similarity_scale(
        fixed,
        moving,
    )

    print("=" * 86)
    print(
        "STEP 6B3 — ISOTROPIC SIMILARITY RESCUE"
    )
    print("=" * 86)

    print(
        f"UID: {uid}"
    )

    print(
        "Template foreground extent "
        f"(z,y,x mm): "
        f"{fixed_extent}"
    )

    print(
        "Moving foreground extent "
        f"(z,y,x mm): "
        f"{moving_extent}"
    )

    print(
        f"Estimated scale: "
        f"{estimated_scale:.6f}"
    )

    print("=" * 86)

    all_results = []

    for initialization in [
        "moments",
        "geometry",
    ]:

        rigid = create_rigid_initial(
            fixed=fixed,
            moving=moving,
            initialization=initialization,
        )

        rigid_dice = dice_score(
            fixed_mask,
            moving_mask,
            rigid,
        )

        search = search_scale(
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            rigid_initial=rigid,
            estimated_scale=estimated_scale,
        )

        all_results.append(
            {
                "initialization":
                    initialization,

                "rigid_transform":
                    rigid,

                "rigid_dice":
                    rigid_dice,

                "estimated_scale":
                    estimated_scale,

                "best_scale":
                    search[
                        "best_scale"
                    ],

                "similarity_dice":
                    search[
                        "best_dice"
                    ],

                "similarity_transform":
                    search[
                        "transform"
                    ],
            }
        )

        print(
            f"{initialization.upper()}:"
        )

        print(
            f"    rigid Dice:      "
            f"{rigid_dice:.4f}"
        )

        print(
            f"    best scale:      "
            f"{search['best_scale']:.6f}"
        )

        print(
            f"    similarity Dice: "
            f"{search['best_dice']:.4f}"
        )

    best = max(
        all_results,
        key=lambda item:
            item["similarity_dice"],
    )

    best_initialization = str(
        best["initialization"]
    )

    best_scale = float(
        best["best_scale"]
    )

    best_similarity_dice = float(
        best["similarity_dice"]
    )

    best_rigid_dice = float(
        best["rigid_dice"]
    )

    best_similarity_transform = (
        best["similarity_transform"]
    )

    best_rigid_transform = (
        best["rigid_transform"]
    )

    result_qc = qc_category(
        best_similarity_dice
    )

    transform_path = (
        TRANSFORM_DIR
        / f"{uid}_similarity_rescue.h5"
    )

    sitk.WriteTransform(
        best_similarity_transform,
        str(transform_path),
    )

    rigid_registered = sitk.Resample(
        moving,
        fixed,
        best_rigid_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    similarity_registered = sitk.Resample(
        moving,
        fixed,
        best_similarity_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    qc_path = (
        QC_DIR
        / f"{uid}_similarity_rescue_qc.png"
    )

    create_qc_figure(
        uid=uid,

        moving=moving,
        fixed=fixed,

        rigid_registered=
            rigid_registered,

        similarity_registered=
            similarity_registered,

        rigid_dice=
            best_rigid_dice,

        similarity_dice=
            best_similarity_dice,

        scale=
            best_scale,

        initialization=
            best_initialization,

        output_path=
            qc_path,
    )

    return {

        "uid":
            uid,

        "estimated_scale":
            estimated_scale,

        "template_extent_z_mm":
            fixed_extent[0],

        "template_extent_y_mm":
            fixed_extent[1],

        "template_extent_x_mm":
            fixed_extent[2],

        "moving_extent_z_mm":
            moving_extent[0],

        "moving_extent_y_mm":
            moving_extent[1],

        "moving_extent_x_mm":
            moving_extent[2],

        "selected_initialization":
            best_initialization,

        "rigid_dice":
            best_rigid_dice,

        "similarity_scale":
            best_scale,

        "similarity_dice":
            best_similarity_dice,

        "dice_gain":
            (
                best_similarity_dice
                -
                best_rigid_dice
            ),

        "qc_category":
            result_qc,

        "transform_path":
            str(
                transform_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "qc_image":
            str(
                qc_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "visual_decision":
            "",

        "notes":
            "",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Isotropic similarity rescue for every "
            "scan whose rigid registration is still "
            "failed / unresolved."
        )
    )

    parser.add_argument(
        "--uid",
        action="append",
        default=None,
        help=(
            "Optional UID to rescue. Repeat or use "
            "comma-separated values. If omitted, "
            "all 6b1 unresolved (or 6a4 failed) "
            "UIDs are processed."
        ),
    )

    args = parser.parse_args()

    if not TEMPLATE_PATH.exists():

        raise FileNotFoundError(
            f"Required path missing: {TEMPLATE_PATH}"
        )

    uids = collect_failed_uids(
        args.uid
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRANSFORM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    QC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 86)
    print(
        "STEP 6B3 — SIMILARITY RESCUE"
    )
    print("=" * 86)
    print(
        f"UIDs to process: {len(uids)}"
    )

    if uids:
        print(
            "  " + ", ".join(uids)
        )

    print("=" * 86)

    if not uids:

        pd.DataFrame(
            columns=[
                "uid",
                "estimated_scale",
                "template_extent_z_mm",
                "template_extent_y_mm",
                "template_extent_x_mm",
                "moving_extent_z_mm",
                "moving_extent_y_mm",
                "moving_extent_x_mm",
                "selected_initialization",
                "rigid_dice",
                "similarity_scale",
                "similarity_dice",
                "dice_gain",
                "qc_category",
                "transform_path",
                "qc_image",
                "visual_decision",
                "notes",
            ]
        ).to_csv(
            RESULT_CSV,
            index=False,
        )

        summary = {
            "analysis": (
                "Isotropic similarity rescue for "
                "unresolved rigid registrations"
            ),
            "number_of_scans": 0,
            "uids": [],
            "qc_counts": {},
            "important_note": (
                "No failed/unresolved scans were found."
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

        print(
            "No failed/unresolved scans. Nothing to rescue."
        )

        return

    fixed = sitk.ReadImage(
        str(TEMPLATE_PATH),
        sitk.sitkFloat32,
    )

    result_rows = []

    for uid in uids:

        result_rows.append(
            rescue_one_scan(
                uid=uid,
                fixed=fixed,
            )
        )

    pd.DataFrame(
        result_rows
    ).to_csv(
        RESULT_CSV,
        index=False,
    )

    qc_counts: dict[str, int] = {}

    for row in result_rows:

        key = str(row["qc_category"])
        qc_counts[key] = qc_counts.get(key, 0) + 1

    summary = {
        "analysis": (
            "Isotropic similarity rescue for "
            "unresolved rigid registrations"
        ),
        "number_of_scans": len(result_rows),
        "uids": [
            str(row["uid"])
            for row in result_rows
        ],
        "qc_counts": qc_counts,
        "results_csv": str(
            RESULT_CSV.relative_to(PROJECT_ROOT)
        ),
        "important_note": (
            "Similarity transforms are candidate "
            "rescues. Step 6c copies them for every "
            "unresolved UID that has a rescue file."
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
    print("=" * 86)
    print("SIMILARITY RESCUE COMPLETED")
    print("=" * 86)
    print(
        f"Scans rescued: {len(result_rows)}"
    )
    print(
        f"CSV: {RESULT_CSV}"
    )
    print(
        f"Summary: {SUMMARY_JSON}"
    )
    print("=" * 86)


if __name__ == "__main__":
    main()
