from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = PROJECT_ROOT / "data/preprocessing_image_data/step4_voxel_resampler_data"

TEMPLATE_PATH = (
    PROJECT_ROOT
    / "data/template/dat_spect/fpcit_template_mni.nii"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/preprocessing_image_data/step6a_initial_registration_data/step6a1_full_rigid_registration"
)

TRANSFORM_DIR = OUTPUT_DIR / "transforms"
REGISTERED_QC_DIR = OUTPUT_DIR / "registered_qc"

CSV_PATH = OUTPUT_DIR / "registration_qc.csv"
SUMMARY_PATH = OUTPUT_DIR / "registration_summary.json"


# ============================================================
# QC THRESHOLDS
# ============================================================

HIGH_CONFIDENCE_DICE = 0.70
REVIEW_DICE = 0.50


# ============================================================
# HELPERS
# ============================================================

def get_uid(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def foreground_mask(image: sitk.Image) -> sitk.Image:
    """
    Binary foreground mask.

    Positive voxels = foreground
    zero/negative = background

    This mask is used only for registration QC.
    """
    return sitk.Cast(image > 0, sitk.sitkUInt8)


def dice_score(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    transform: sitk.Transform,
) -> float:
    """
    Transform moving mask into template space and calculate Dice.
    """

    moved_mask = sitk.Resample(
        moving_mask,
        fixed_mask,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    fixed_array = sitk.GetArrayViewFromImage(fixed_mask) > 0
    moved_array = sitk.GetArrayViewFromImage(moved_mask) > 0

    intersection = np.count_nonzero(fixed_array & moved_array)

    denominator = (
        np.count_nonzero(fixed_array)
        + np.count_nonzero(moved_array)
    )

    if denominator == 0:
        return 0.0

    return float(2.0 * intersection / denominator)


def qc_category(dice: float) -> str:
    if dice >= HIGH_CONFIDENCE_DICE:
        return "high_confidence"

    if dice >= REVIEW_DICE:
        return "review"

    return "failed"


# ============================================================
# REGISTRATION
# ============================================================

def build_registration_method() -> sitk.ImageRegistrationMethod:
    """
    Rigid registration configuration.

    IMPORTANT:
    If the optimizer/metric parameters in your final Step-5 rigid
    experiment differ from these, copy the Step-5 values here
    verbatim.

    The full-dataset logic should stay unchanged.
    """

    registration = sitk.ImageRegistrationMethod()

    # Similarity metric
    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50
    )

    registration.SetMetricSamplingStrategy(
        registration.RANDOM
    )

    registration.SetMetricSamplingPercentage(
        0.20,
        seed=42,
    )

    # Image interpolation during optimization
    registration.SetInterpolator(
        sitk.sitkLinear
    )

    # Optimizer
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=0.001,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8,
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution registration
    registration.SetShrinkFactorsPerLevel(
        shrinkFactors=[4, 2, 1]
    )

    registration.SetSmoothingSigmasPerLevel(
        smoothingSigmas=[2, 1, 0]
    )

    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return registration


def register_rigid(
    fixed: sitk.Image,
    moving: sitk.Image,
    initialization: str,
) -> dict:
    """
    Run one rigid registration.

    initialization:
        "moments"
        "geometry"
    """

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
            f"Unknown initialization: {initialization}"
        )

    # Rigid transform:
    # rotation + translation only
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        mode,
    )

    registration = build_registration_method()

    registration.SetInitialTransform(
        initial_transform,
        inPlace=False,
    )

    final_transform = registration.Execute(
        fixed,
        moving,
    )

    return {
        "transform": final_transform,
        "metric": float(
            registration.GetMetricValue()
        ),
        "iterations": int(
            registration.GetOptimizerIteration()
        ),
        "stop_condition": str(
            registration.GetOptimizerStopConditionDescription()
        ),
    }


# ============================================================
# PROCESS ONE SCAN
# ============================================================

def process_scan(
    scan_path: Path,
    fixed: sitk.Image,
    fixed_mask: sitk.Image,
    save_registered: bool,
) -> dict:

    uid = get_uid(scan_path)

    moving = sitk.ReadImage(
        str(scan_path),
        sitk.sitkFloat32,
    )

    moving_mask = foreground_mask(moving)

    # --------------------------------------------------------
    # Start 1: MOMENTS
    # --------------------------------------------------------

    moments = register_rigid(
        fixed=fixed,
        moving=moving,
        initialization="moments",
    )

    moments_dice = dice_score(
        fixed_mask,
        moving_mask,
        moments["transform"],
    )

    # --------------------------------------------------------
    # Start 2: GEOMETRY
    # --------------------------------------------------------

    geometry = register_rigid(
        fixed=fixed,
        moving=moving,
        initialization="geometry",
    )

    geometry_dice = dice_score(
        fixed_mask,
        moving_mask,
        geometry["transform"],
    )

    # --------------------------------------------------------
    # Select best rigid result
    # --------------------------------------------------------

    if moments_dice >= geometry_dice:

        selected_name = "moments"
        selected = moments
        selected_dice = moments_dice

    else:

        selected_name = "geometry"
        selected = geometry
        selected_dice = geometry_dice

    qc = qc_category(selected_dice)

    # --------------------------------------------------------
    # Save transforms
    # --------------------------------------------------------

    scan_transform_dir = TRANSFORM_DIR / uid
    scan_transform_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    moments_path = (
        scan_transform_dir / "moments.h5"
    )

    geometry_path = (
        scan_transform_dir / "geometry.h5"
    )

    selected_path = (
        scan_transform_dir / "selected.h5"
    )

    sitk.WriteTransform(
        moments["transform"],
        str(moments_path),
    )

    sitk.WriteTransform(
        geometry["transform"],
        str(geometry_path),
    )

    sitk.WriteTransform(
        selected["transform"],
        str(selected_path),
    )

    # --------------------------------------------------------
    # Optional registered image
    #
    # ONLY for visual QC.
    # Do NOT use this interpolated image for normalization.
    # --------------------------------------------------------

    registered_path = ""

    if save_registered:

        REGISTERED_QC_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        registered = sitk.Resample(
            moving,
            fixed,
            selected["transform"],
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )

        registered_path = (
            REGISTERED_QC_DIR
            / f"{uid}.nii.gz"
        )

        sitk.WriteImage(
            registered,
            str(registered_path),
            useCompression=True,
        )

    return {
        "uid": uid,
        "file_name": scan_path.name,

        "moments_dice": moments_dice,
        "geometry_dice": geometry_dice,

        "selected_initialization": selected_name,
        "selected_dice": selected_dice,

        "qc_category": qc,

        "moments_metric": moments["metric"],
        "geometry_metric": geometry["metric"],

        "moments_iterations": moments["iterations"],
        "geometry_iterations": geometry["iterations"],

        "dice_difference": float(
            abs(moments_dice - geometry_dice)
        ),

        "selected_transform": str(
            selected_path.relative_to(PROJECT_ROOT)
        ),

        "registered_qc_image": (
            str(
                registered_path.relative_to(
                    PROJECT_ROOT
                )
            )
            if registered_path
            else ""
        ),

        "status": "success",
        "error": "",
    }


# ============================================================
# CSV
# ============================================================

FIELDNAMES = [
    "uid",
    "file_name",

    "moments_dice",
    "geometry_dice",

    "selected_initialization",
    "selected_dice",
    "qc_category",

    "moments_metric",
    "geometry_metric",

    "moments_iterations",
    "geometry_iterations",

    "dice_difference",

    "selected_transform",
    "registered_qc_image",

    "status",
    "error",
]


def load_completed_uids() -> set[str]:

    if not CSV_PATH.exists():
        return set()

    completed = set()

    with CSV_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            if row["status"] == "success":
                completed.add(row["uid"])

    return completed


def append_csv(row: dict) -> None:

    file_exists = CSV_PATH.exists()

    with CSV_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# SUMMARY
# ============================================================

def create_summary() -> None:

    rows = []

    with CSV_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)
        rows = list(reader)

    successful = [
        row
        for row in rows
        if row["status"] == "success"
    ]

    failed = [
        row
        for row in rows
        if row["status"] != "success"
    ]

    qc_counts = Counter(
        row["qc_category"]
        for row in successful
    )

    initialization_counts = Counter(
        row["selected_initialization"]
        for row in successful
    )

    dice_values = np.asarray(
        [
            float(row["selected_dice"])
            for row in successful
        ],
        dtype=np.float64,
    )

    summary = {
        "analysis": (
            "Step 6A full-dataset "
            "dual-start rigid registration"
        ),

        "input_directory": str(
            INPUT_DIR.relative_to(PROJECT_ROOT)
        ),

        "template": str(
            TEMPLATE_PATH.relative_to(
                PROJECT_ROOT
            )
        ),

        "number_of_rows": len(rows),
        "successful": len(successful),
        "failed": len(failed),

        "qc_counts": dict(qc_counts),

        "selected_initialization_counts": dict(
            initialization_counts
        ),

        "selected_dice": {
            "min": (
                float(np.min(dice_values))
                if len(dice_values)
                else None
            ),

            "p05": (
                float(
                    np.percentile(
                        dice_values,
                        5,
                    )
                )
                if len(dice_values)
                else None
            ),

            "median": (
                float(np.median(dice_values))
                if len(dice_values)
                else None
            ),

            "p95": (
                float(
                    np.percentile(
                        dice_values,
                        95,
                    )
                )
                if len(dice_values)
                else None
            ),

            "max": (
                float(np.max(dice_values))
                if len(dice_values)
                else None
            ),
        },

        "qc_thresholds": {
            "high_confidence": (
                f"Dice >= {HIGH_CONFIDENCE_DICE}"
            ),

            "review": (
                f"{REVIEW_DICE} <= Dice "
                f"< {HIGH_CONFIDENCE_DICE}"
            ),

            "failed": (
                f"Dice < {REVIEW_DICE}"
            ),
        },

        "important_note": (
            "Registered images, when saved, "
            "are for QC only. "
            "Normalization must use the "
            "original resampled scan."
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N scans. "
            "Useful for testing."
        ),
    )

    parser.add_argument(
        "--save-registered",
        action="store_true",
        help=(
            "Save registered template-space "
            "images for visual QC."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Restart from the beginning.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: "
            f"{INPUT_DIR}"
        )

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Template not found: "
            f"{TEMPLATE_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRANSFORM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.overwrite and CSV_PATH.exists():
        CSV_PATH.unlink()

    # --------------------------------------------------------
    # Find scans
    # --------------------------------------------------------

    scan_paths = sorted(
        INPUT_DIR.glob("*.nii.gz")
    )

    if args.limit is not None:
        scan_paths = scan_paths[: args.limit]

    print("=" * 72)
    print(
        "STEP 6A — FULL DATASET "
        "DUAL-START RIGID REGISTRATION"
    )
    print("=" * 72)

    print(f"Input:    {INPUT_DIR}")
    print(f"Template: {TEMPLATE_PATH}")
    print(f"Output:   {OUTPUT_DIR}")
    print(f"Scans:    {len(scan_paths)}")
    print("=" * 72)

    # --------------------------------------------------------
    # Template
    # --------------------------------------------------------

    fixed = sitk.ReadImage(
        str(TEMPLATE_PATH),
        sitk.sitkFloat32,
    )

    fixed_mask = foreground_mask(fixed)

    completed = load_completed_uids()

    # --------------------------------------------------------
    # Process scans
    # --------------------------------------------------------

    total = len(scan_paths)

    for index, scan_path in enumerate(
        scan_paths,
        start=1,
    ):

        uid = get_uid(scan_path)

        if uid in completed:
            print(
                f"[{index}/{total}] "
                f"{scan_path.name} — SKIP"
            )
            continue

        print(
            f"[{index}/{total}] "
            f"{scan_path.name}"
        )

        try:

            row = process_scan(
                scan_path=scan_path,
                fixed=fixed,
                fixed_mask=fixed_mask,
                save_registered=args.save_registered,
            )

            print(
                f"    MOMENTS Dice:  "
                f"{row['moments_dice']:.3f}"
            )

            print(
                f"    GEOMETRY Dice: "
                f"{row['geometry_dice']:.3f}"
            )

            print(
                f"    selected: "
                f"{row['selected_initialization']}"
            )

            print(
                f"    QC: "
                f"{row['qc_category']} "
                f"(Dice={row['selected_dice']:.3f})"
            )

        except Exception as exc:

            print(
                f"    ERROR: {exc}"
            )

            row = {
                "uid": uid,
                "file_name": scan_path.name,

                "moments_dice": "",
                "geometry_dice": "",

                "selected_initialization": "",
                "selected_dice": "",
                "qc_category": "failed",

                "moments_metric": "",
                "geometry_metric": "",

                "moments_iterations": "",
                "geometry_iterations": "",

                "dice_difference": "",

                "selected_transform": "",
                "registered_qc_image": "",

                "status": "failed",
                "error": str(exc),
            }

        # Save immediately.
        # If processing is interrupted, completed scans
        # do not need to be repeated.
        append_csv(row)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    create_summary()

    print()
    print("=" * 72)
    print("STEP 6A COMPLETED")
    print("=" * 72)
    print(f"QC CSV:   {CSV_PATH}")
    print(f"Summary:  {SUMMARY_PATH}")
    print(f"Transforms: {TRANSFORM_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
