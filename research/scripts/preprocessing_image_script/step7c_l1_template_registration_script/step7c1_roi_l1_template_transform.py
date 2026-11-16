from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def uid_from_path(path: Path) -> str:
    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return path.stem


def find_nifti_files(directory: Path) -> list[Path]:
    files = list(directory.glob("*.nii"))
    files += list(directory.glob("*.nii.gz"))

    return sorted(set(files))


def describe(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)

    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 7C — ROI-L1 frozen Step-6 transform localizer."
        )
    )

    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path("data/preprocessing_image_data/step6e_intensity_normalization_data/step6e_normalize_occipital"),
    )

    parser.add_argument(
        "--template-mask",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step5d_normalization_strategy_analysis_data/build_striatal_masks/striatum_mask.nii.gz"
        ),
    )

    parser.add_argument(
        "--transform-root",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step6c_finalize_registration_data/step6c_finalize_all_registration_transforms/transforms"
        ),
    )

    parser.add_argument(
        "--output-mask-dir",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7c_l1_template_registration_data/step7c1_roi_l1_template_transform"
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7c_l1_template_registration_data/"
            "step7c1_roi_l1_template_transform/l1_localization.csv"
        ),
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7c_l1_template_registration_data/step7c1_roi_l1_template_transform/l1_summary.json"
        ),
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help=(
            "Optional expected scan count. "
            "If omitted, the number of input NIfTIs is used."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional pilot limit.",
    )

    args = parser.parse_args()

    normalized_dir = resolve_path(args.normalized_dir)
    template_mask_path = resolve_path(args.template_mask)
    transform_root = resolve_path(args.transform_root)
    output_mask_dir = resolve_path(args.output_mask_dir)
    output_csv = resolve_path(args.output_csv)
    summary_json = resolve_path(args.summary_json)

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------

    if not normalized_dir.exists():
        raise FileNotFoundError(
            f"Normalized directory missing: {normalized_dir}"
        )

    if not template_mask_path.exists():
        raise FileNotFoundError(
            f"Template mask missing: {template_mask_path}"
        )

    if not transform_root.exists():
        raise FileNotFoundError(
            f"Transform root missing: {transform_root}"
        )

    files = find_nifti_files(normalized_dir)

    if args.limit is None:
        if (
            args.expected_count is not None
            and len(files) != args.expected_count
        ):
            raise RuntimeError(
                f"Expected {args.expected_count} scans, "
                f"found {len(files)}."
            )

    if args.limit is not None:
        files = files[: args.limit]

    uids = [uid_from_path(p) for p in files]

    if len(uids) != len(set(uids)):
        raise RuntimeError(
            "Duplicate UIDs detected in normalized images."
        )

    # ---------------------------------------------------------
    # Load frozen template striatal mask
    # ---------------------------------------------------------

    template_mask = sitk.ReadImage(
        str(template_mask_path)
    )

    template_mask = sitk.Cast(
        template_mask > 0,
        sitk.sitkUInt8,
    )

    template_mask_array = sitk.GetArrayViewFromImage(
        template_mask
    )

    template_voxel_count = int(
        np.count_nonzero(template_mask_array)
    )

    if template_voxel_count == 0:
        raise RuntimeError(
            "Template striatal mask is empty."
        )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    output_mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    failures = []

    print()
    print("=" * 80)
    print(
        "STEP 7C — ROI-L1 FROZEN TEMPLATE-TRANSFORM LOCALIZER"
    )
    print("=" * 80)

    print(f"Normalized scans:     {len(files)}")
    print(
        f"Template mask voxels: {template_voxel_count}"
    )
    print()

    # ---------------------------------------------------------
    # Process
    # ---------------------------------------------------------

    for index, subject_path in enumerate(
        files,
        start=1,
    ):
        uid = uid_from_path(subject_path)

        transform_path = (
            transform_root
            / uid
            / "final_transform.h5"
        )

        output_mask_path = (
            output_mask_dir
            / f"{uid}.nii.gz"
        )

        try:
            if not transform_path.exists():
                raise FileNotFoundError(
                    f"Missing final transform: "
                    f"{transform_path}"
                )

            # -------------------------------------------------
            # Subject defines the final output grid.
            # -------------------------------------------------

            subject = sitk.ReadImage(
                str(subject_path)
            )

            if subject.GetDimension() != 3:
                raise ValueError(
                    f"Subject is not 3D: "
                    f"{subject.GetDimension()}D"
                )

            # -------------------------------------------------
            # Frozen Step-6 transform
            #
            # Step 6 used:
            #
            # Resample(
            #     moving_subject,
            #     fixed_template,
            #     T
            # )
            #
            # SimpleITK therefore uses T to map:
            #
            # template output point -> subject input point
            #
            # For template mask -> subject output we need:
            #
            # subject output point -> template input point
            #
            # therefore T^-1.
            # -------------------------------------------------

            transform = sitk.ReadTransform(
                str(transform_path)
            )

            inverse_transform = (
                transform.GetInverse()
            )

            # -------------------------------------------------
            # Map TEMPLATE MASK into SUBJECT space.
            #
            # Binary mask => nearest-neighbor only.
            # -------------------------------------------------

            mapped_mask = sitk.Resample(
                template_mask,
                subject,
                inverse_transform,
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8,
            )

            mapped_mask = sitk.Cast(
                mapped_mask > 0,
                sitk.sitkUInt8,
            )

            sitk.WriteImage(
                mapped_mask,
                str(output_mask_path),
                True,
            )

            # -------------------------------------------------
            # Analyze using nibabel.
            #
            # This keeps voxel/world coordinates consistent
            # with Step 7B (NIfTI RAS convention).
            # -------------------------------------------------

            mask_img = nib.load(
                str(output_mask_path)
            )

            mask = (
                np.asarray(mask_img.dataobj) > 0
            )

            subject_nib = nib.load(
                str(subject_path)
            )

            if mask_img.shape != subject_nib.shape:
                raise RuntimeError(
                    "Mapped mask shape does not match "
                    "normalized subject."
                )

            if not np.allclose(
                mask_img.affine,
                subject_nib.affine,
                atol=1e-4,
            ):
                raise RuntimeError(
                    "Mapped-mask affine does not match "
                    "normalized subject affine."
                )

            coords = np.argwhere(mask)

            if coords.size == 0:
                raise RuntimeError(
                    "Mapped striatal mask is empty."
                )

            if not np.all(np.isfinite(coords)):
                raise RuntimeError(
                    "Mapped mask coordinates "
                    "contain non-finite values."
                )

            # -------------------------------------------------
            # Center
            # -------------------------------------------------

            center_voxel = coords.mean(
                axis=0
            )

            center_world = (
                nib.affines.apply_affine(
                    mask_img.affine,
                    center_voxel,
                )
            )

            if not np.all(
                np.isfinite(center_world)
            ):
                raise RuntimeError(
                    "Physical center is non-finite."
                )

            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            bbox_min = coords.min(axis=0)
            bbox_max = coords.max(axis=0)

            bbox_size = (
                bbox_max
                - bbox_min
                + 1
            )

            shape = np.asarray(
                mask.shape,
                dtype=np.int64,
            )

            border_low = bbox_min

            border_high = (
                shape - 1 - bbox_max
            )

            minimum_border_distance = int(
                np.min(
                    np.concatenate(
                        [
                            border_low,
                            border_high,
                        ]
                    )
                )
            )

            center_inside = bool(
                np.all(center_voxel >= 0)
                and np.all(
                    center_voxel
                    <= shape - 1
                )
            )

            mapped_voxel_count = int(
                coords.shape[0]
            )

            spacing = (
                subject_nib.header
                .get_zooms()[:3]
            )

            orientation = (
                nib.aff2axcodes(
                    subject_nib.affine
                )
            )

            row = {
                "uid": uid,

                "transform_path": str(
                    transform_path
                ),
                "transform_type": (
                    transform.GetName()
                ),
                "transform_inverted_for_mapping": True,

                "mapped_mask_path": str(
                    output_mask_path
                ),

                "shape_x": int(shape[0]),
                "shape_y": int(shape[1]),
                "shape_z": int(shape[2]),

                "spacing_x_mm": float(
                    spacing[0]
                ),
                "spacing_y_mm": float(
                    spacing[1]
                ),
                "spacing_z_mm": float(
                    spacing[2]
                ),

                "orientation": "".join(
                    orientation
                ),

                "l1_center_x": float(
                    center_voxel[0]
                ),
                "l1_center_y": float(
                    center_voxel[1]
                ),
                "l1_center_z": float(
                    center_voxel[2]
                ),

                "l1_physical_x_mm": float(
                    center_world[0]
                ),
                "l1_physical_y_mm": float(
                    center_world[1]
                ),
                "l1_physical_z_mm": float(
                    center_world[2]
                ),

                "bbox_min_x": int(
                    bbox_min[0]
                ),
                "bbox_min_y": int(
                    bbox_min[1]
                ),
                "bbox_min_z": int(
                    bbox_min[2]
                ),

                "bbox_max_x": int(
                    bbox_max[0]
                ),
                "bbox_max_y": int(
                    bbox_max[1]
                ),
                "bbox_max_z": int(
                    bbox_max[2]
                ),

                "bbox_size_x": int(
                    bbox_size[0]
                ),
                "bbox_size_y": int(
                    bbox_size[1]
                ),
                "bbox_size_z": int(
                    bbox_size[2]
                ),

                "mapped_mask_voxels": (
                    mapped_voxel_count
                ),

                "minimum_border_distance_voxels": (
                    minimum_border_distance
                ),

                "center_inside_image": (
                    center_inside
                ),

                "status": "PASS",
            }

            rows.append(row)

        except Exception as exc:
            failures.append(
                {
                    "uid": uid,
                    "subject_path": str(
                        subject_path
                    ),
                    "transform_path": str(
                        transform_path
                    ),
                    "error": str(exc),
                }
            )

        if (
            index == 1
            or index % 100 == 0
            or index == len(files)
        ):
            print(
                f"[{index:4d}/{len(files)}] "
                f"processed"
            )

    # ---------------------------------------------------------
    # Failure check
    # ---------------------------------------------------------

    if failures:
        print()
        print("FAILURES")
        print("-" * 80)

        for failure in failures[:20]:
            print(
                failure["uid"],
                "->",
                failure["error"],
            )

        raise RuntimeError(
            f"{len(failures)} L1 mappings failed."
        )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = {
        "step": "7C",
        "method": (
            "ROI-L1 frozen Step-6 transform localizer"
        ),
        "transform_policy": (
            "inverse of frozen Step-6 transform "
            "for template-mask-to-subject resampling"
        ),
        "interpolation": "nearest_neighbor",
        "template_mask": str(
            template_mask_path.resolve()
        ),
        "template_mask_voxels": (
            template_voxel_count
        ),
        "processed": len(files),
        "successful": len(rows),
        "failed": len(failures),

        "center_inside_image_failures": int(
            sum(
                not r["center_inside_image"]
                for r in rows
            )
        ),

        "orientation_counts": {
            orientation: sum(
                r["orientation"] == orientation
                for r in rows
            )
            for orientation in sorted(
                set(
                    r["orientation"]
                    for r in rows
                )
            )
        },

        "center_voxel_distribution": {
            "x": describe(
                [
                    r["l1_center_x"]
                    for r in rows
                ]
            ),
            "y": describe(
                [
                    r["l1_center_y"]
                    for r in rows
                ]
            ),
            "z": describe(
                [
                    r["l1_center_z"]
                    for r in rows
                ]
            ),
        },

        "mapped_mask_voxel_distribution": (
            describe(
                [
                    r[
                        "mapped_mask_voxels"
                    ]
                    for r in rows
                ]
            )
        ),

        "bbox_size_distribution": {
            "x": describe(
                [
                    r["bbox_size_x"]
                    for r in rows
                ]
            ),
            "y": describe(
                [
                    r["bbox_size_y"]
                    for r in rows
                ]
            ),
            "z": describe(
                [
                    r["bbox_size_z"]
                    for r in rows
                ]
            ),
        },

        "minimum_border_distance_distribution": (
            describe(
                [
                    r[
                        "minimum_border_distance_voxels"
                    ]
                    for r in rows
                ]
            )
        ),
    }

    summary_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("STEP 7C RESULTS")
    print("=" * 80)

    print(
        f"Successful:              "
        f"{len(rows)}"
    )
    print(
        f"Failed:                  "
        f"{len(failures)}"
    )
    print(
        "Centers outside image:  "
        f"{summary['center_inside_image_failures']}"
    )

    print()
    print("Orientation counts:")

    for orientation, count in (
        summary[
            "orientation_counts"
        ].items()
    ):
        print(
            f"  {orientation}: {count}"
        )

    print()
    print("Mapped-mask voxel count:")
    stats = (
        summary[
            "mapped_mask_voxel_distribution"
        ]
    )

    print(
        f"  min:    {stats['min']:.1f}"
    )
    print(
        f"  median: {stats['median']:.1f}"
    )
    print(
        f"  p95:    {stats['p95']:.1f}"
    )
    print(
        f"  max:    {stats['max']:.1f}"
    )

    print()
    print("BBox median [voxels]:")

    bbox_stats = (
        summary[
            "bbox_size_distribution"
        ]
    )

    print(
        "  X:",
        round(
            bbox_stats["x"]["median"],
            2,
        ),
    )
    print(
        "  Y:",
        round(
            bbox_stats["y"]["median"],
            2,
        ),
    )
    print(
        "  Z:",
        round(
            bbox_stats["z"]["median"],
            2,
        ),
    )

    print()
    print("CSV:")
    print(
        f"  {output_csv.resolve()}"
    )

    print()
    print("Mapped masks:")
    print(
        f"  {output_mask_dir.resolve()}"
    )

    print()
    print("Summary:")
    print(
        f"  {summary_json.resolve()}"
    )

    print()
    print("STEP 7C STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
