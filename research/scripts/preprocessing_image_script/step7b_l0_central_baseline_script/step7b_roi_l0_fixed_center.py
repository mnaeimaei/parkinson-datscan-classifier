from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def uid_from_path(path: Path) -> str:
    """Return UID from .nii or .nii.gz filename."""
    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return path.stem


def voxel_to_world(
    affine: np.ndarray,
    voxel_coord: np.ndarray,
) -> np.ndarray:
    """Convert continuous voxel coordinates to physical/world coordinates."""
    voxel_h = np.append(voxel_coord, 1.0)
    world_h = affine @ voxel_h
    return world_h[:3]


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


def find_nifti_files(directory: Path) -> list[Path]:
    files = list(directory.glob("*.nii"))
    files += list(directory.glob("*.nii.gz"))

    return sorted(set(files))


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7B — ROI-L0 fixed-center baseline."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/preprocessing_image_data/step6e_intensity_normalization_data/step6e_normalize_occipital"),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/"
            "step7b_l0_central_baseline_data/"
            "step7b_roi_l0_fixed_center/l0_centers.csv"
        ),
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/"
            "step7b_l0_central_baseline_data/"
            "step7b_roi_l0_fixed_center/l0_summary.json"
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

    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    output_csv = resolve_path(args.output_csv)
    summary_json = resolve_path(args.summary_json)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    files = find_nifti_files(input_dir)

    if not files:
        raise RuntimeError(
            f"No NIfTI files found in {input_dir}"
        )

    print()
    print("=" * 76)
    print("STEP 7B — ROI-L0 FIXED-CENTER BASELINE")
    print("=" * 76)
    print(f"Input directory: {input_dir.resolve()}")
    print(f"Found scans:     {len(files)}")
    print()

    # ---------------------------------------------------------
    # UID checks
    # ---------------------------------------------------------

    uids = [uid_from_path(path) for path in files]

    duplicate_uids = sorted(
        {
            uid
            for uid in uids
            if uids.count(uid) > 1
        }
    )

    if duplicate_uids:
        raise RuntimeError(
            f"Duplicate UIDs detected: {duplicate_uids[:20]}"
        )

    if (
        args.expected_count is not None
        and len(files) != args.expected_count
    ):
        raise RuntimeError(
            f"Expected {args.expected_count} scans, "
            f"but found {len(files)}."
        )

    if args.expected_count is None:
        args.expected_count = len(files)

    # ---------------------------------------------------------
    # Process every normalized scan
    # ---------------------------------------------------------

    rows = []
    failures = []

    for index, path in enumerate(files, start=1):

        uid = uid_from_path(path)

        try:
            img = nib.load(str(path))

            if len(img.shape) != 3:
                raise ValueError(
                    f"Expected 3D image, got shape {img.shape}"
                )

            shape = np.asarray(
                img.shape,
                dtype=np.int64,
            )

            affine = np.asarray(
                img.affine,
                dtype=np.float64,
            )

            if affine.shape != (4, 4):
                raise ValueError(
                    f"Invalid affine shape: {affine.shape}"
                )

            if not np.all(np.isfinite(affine)):
                raise ValueError(
                    "Affine contains NaN or Inf."
                )

            spacing = np.asarray(
                img.header.get_zooms()[:3],
                dtype=np.float64,
            )

            if not np.all(np.isfinite(spacing)):
                raise ValueError(
                    "Spacing contains NaN or Inf."
                )

            if np.any(spacing <= 0):
                raise ValueError(
                    f"Non-positive spacing: {spacing}"
                )

            # -------------------------------------------------
            # L0 definition
            #
            # Continuous geometric center of the voxel grid:
            #
            #      (shape - 1) / 2
            #
            # Do NOT round here.
            # -------------------------------------------------

            center_voxel = (
                shape.astype(np.float64) - 1.0
            ) / 2.0

            center_world = voxel_to_world(
                affine,
                center_voxel,
            )

            if not np.all(
                np.isfinite(center_voxel)
            ):
                raise ValueError(
                    "Voxel center is not finite."
                )

            if not np.all(
                np.isfinite(center_world)
            ):
                raise ValueError(
                    "Physical center is not finite."
                )

            center_inside = bool(
                np.all(center_voxel >= 0)
                and np.all(
                    center_voxel
                    <= shape.astype(np.float64) - 1
                )
            )

            orientation = nib.aff2axcodes(affine)

            row = {
                "uid": uid,

                "shape_x": int(shape[0]),
                "shape_y": int(shape[1]),
                "shape_z": int(shape[2]),

                "spacing_x_mm": float(spacing[0]),
                "spacing_y_mm": float(spacing[1]),
                "spacing_z_mm": float(spacing[2]),

                "orientation_x": orientation[0],
                "orientation_y": orientation[1],
                "orientation_z": orientation[2],

                "l0_center_x": float(center_voxel[0]),
                "l0_center_y": float(center_voxel[1]),
                "l0_center_z": float(center_voxel[2]),

                "l0_physical_x_mm": float(center_world[0]),
                "l0_physical_y_mm": float(center_world[1]),
                "l0_physical_z_mm": float(center_world[2]),

                "center_inside_image": center_inside,
                "finite_coordinates": True,
                "status": "PASS",
            }

            rows.append(row)

        except Exception as exc:
            failures.append(
                {
                    "uid": uid,
                    "path": str(path),
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
    # Validation
    # ---------------------------------------------------------

    if failures:
        print()
        print("FAILURES")
        print("-" * 76)

        for failure in failures[:20]:
            print(
                failure["uid"],
                failure["error"],
            )

        raise RuntimeError(
            f"{len(failures)} scans failed L0 processing."
        )

    if len(rows) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} successful rows, "
            f"but got {len(rows)}."
        )

    if any(
        not row["center_inside_image"]
        for row in rows
    ):
        raise RuntimeError(
            "At least one L0 center lies outside its image."
        )

    # ---------------------------------------------------------
    # Write CSV
    # ---------------------------------------------------------

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    orientations = {}

    for row in rows:
        orientation = (
            row["orientation_x"]
            + row["orientation_y"]
            + row["orientation_z"]
        )

        orientations[orientation] = (
            orientations.get(orientation, 0) + 1
        )

    summary = {
        "step": "7B",
        "method": "ROI-L0 fixed array center",
        "definition": "(shape - 1) / 2",
        "input_directory": str(
            input_dir.resolve()
        ),
        "expected_scans": args.expected_count,
        "found_scans": len(files),
        "successful": len(rows),
        "failed": len(failures),
        "duplicate_uids": len(duplicate_uids),
        "centers_outside_image": int(
            sum(
                not row["center_inside_image"]
                for row in rows
            )
        ),
        "orientations": orientations,

        "shape_distribution": {
            "x": describe(
                [row["shape_x"] for row in rows]
            ),
            "y": describe(
                [row["shape_y"] for row in rows]
            ),
            "z": describe(
                [row["shape_z"] for row in rows]
            ),
        },

        "l0_voxel_center_distribution": {
            "x": describe(
                [row["l0_center_x"] for row in rows]
            ),
            "y": describe(
                [row["l0_center_y"] for row in rows]
            ),
            "z": describe(
                [row["l0_center_z"] for row in rows]
            ),
        },

        "l0_physical_center_distribution_mm": {
            "x": describe(
                [
                    row["l0_physical_x_mm"]
                    for row in rows
                ]
            ),
            "y": describe(
                [
                    row["l0_physical_y_mm"]
                    for row in rows
                ]
            ),
            "z": describe(
                [
                    row["l0_physical_z_mm"]
                    for row in rows
                ]
            ),
        },
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

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print()
    print("=" * 76)
    print("STEP 7B RESULTS")
    print("=" * 76)

    print(f"Successful:              {len(rows)}")
    print(f"Failed:                  {len(failures)}")
    print(f"Duplicate UIDs:          {len(duplicate_uids)}")
    print(
        "Centers outside image: ",
        summary["centers_outside_image"],
    )

    print()
    print("Orientations:")
    for orientation, count in sorted(
        orientations.items()
    ):
        print(
            f"  {orientation}: {count}"
        )

    print()
    print("L0 physical-center median [mm]:")

    physical = (
        summary[
            "l0_physical_center_distribution_mm"
        ]
    )

    print(
        "  X:",
        round(physical["x"]["median"], 3),
    )
    print(
        "  Y:",
        round(physical["y"]["median"], 3),
    )
    print(
        "  Z:",
        round(physical["z"]["median"], 3),
    )

    print()
    print(f"CSV:")
    print(f"  {output_csv.resolve()}")

    print()
    print(f"Summary:")
    print(f"  {summary_json.resolve()}")

    print()
    print("STEP 7B STATUS: PASS")
    print("=" * 76)


if __name__ == "__main__":
    main()
