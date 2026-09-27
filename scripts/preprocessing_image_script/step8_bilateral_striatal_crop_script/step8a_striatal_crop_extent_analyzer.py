#!/usr/bin/env python3

"""
Step 8A — Analyze required bilateral striatal crop extent.

Purpose
-------
Use the frozen Step-7 localization center together with the frozen
registration transform and the bilateral striatal template mask to measure
how much spatial extent is required around the localization center.

This step DOES NOT:
    - generate final crops
    - select the final crop size
    - renormalize intensities
    - modify Step-7 localization decisions

Outputs
-------
1. striatal_extent_statistics.csv
       One row per subject.

2. crop_extent_summary.json
       Dataset-level extent statistics.

3. crop_extent_candidates.csv
       Candidate dimensions for Step 8B.
       These are analytical candidates only, NOT final decisions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 8A: analyze bilateral striatal crop extent."
    )

    parser.add_argument(
        "--localization-manifest",
        type=Path,
        required=True,
        help="Frozen Step-7 localization manifest.",
    )

    parser.add_argument(
        "--registration-manifest",
        type=Path,
        required=True,
        help="Frozen Step-6B final registration manifest.",
    )

    parser.add_argument(
        "--template-striatal-mask",
        type=Path,
        required=True,
        help="Bilateral striatal ROI mask in template space.",
    )

    parser.add_argument(
        "--resampled-dir",
        type=Path,
        required=True,
        help="Directory containing Step-4 resampled RAS scans.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for Step-8A outputs.",
    )

    parser.add_argument(
        "--transform-direction",
        choices=["template_to_subject", "subject_to_template"],
        default="template_to_subject",
        help="Direction represented by stored registration matrices.",
    )

    parser.add_argument(
        "--transform-coordinate-system",
        choices=["ras", "lps"],
        default="ras",
        help="Coordinate convention used by stored transforms.",
    )

    parser.add_argument(
        "--exclude-extent-uids",
        nargs="*",
        default=[],
        help=(
            "Extra UIDs whose localization center remains valid but "
            "transformed ROI extent must not be used for crop-size "
            "estimation. Similarity-rescue registrations from the "
            "Step-6C manifest are excluded automatically."
        ),
    )

    parser.add_argument(
        "--minimum-scale",
        type=float,
        default=0.65,
        help="Lower gross transform-scale QC bound.",
    )

    parser.add_argument(
        "--maximum-scale",
        type=float,
        default=1.55,
        help="Upper gross transform-scale QC bound.",
    )

    parser.add_argument(
        "--maximum-anisotropy",
        type=float,
        default=2.0,
        help="Maximum singular-value ratio before extent review.",
    )

    return parser.parse_args()


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    """Return first matching column, case-insensitive."""

    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise ValueError(
            "Could not find required column. Tried:\n  "
            + "\n  ".join(candidates)
            + "\nAvailable columns:\n  "
            + "\n  ".join(map(str, df.columns))
        )

    return None


def as_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y", "valid", "pass", "passed"}:
        return True

    if text in {"false", "0", "no", "n", "invalid", "fail", "failed"}:
        return False

    return None


def resolve_subject_image(resampled_dir: Path, uid: str) -> Path:
    candidates = [
        resampled_dir / f"{uid}.nii.gz",
        resampled_dir / f"{uid}.nii",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Resampled image not found for UID={uid} in {resampled_dir}"
    )


# -------------------------------------------------------------------------
# Transform loading
# -------------------------------------------------------------------------


def find_matrix_in_object(obj: Any) -> np.ndarray | None:
    """Recursively search JSON/PT-like object for a 4x4 matrix."""

    if isinstance(obj, np.ndarray):
        if obj.shape == (4, 4):
            return obj.astype(float)

    if isinstance(obj, (list, tuple)):
        try:
            arr = np.asarray(obj, dtype=float)
            if arr.shape == (4, 4):
                return arr
            if arr.size == 16:
                return arr.reshape(4, 4)
        except (ValueError, TypeError):
            pass

        for item in obj:
            found = find_matrix_in_object(item)
            if found is not None:
                return found

    if isinstance(obj, dict):
        preferred_keys = [
            "matrix",
            "transform",
            "affine",
            "transform_matrix",
            "final_matrix",
            "selected_matrix",
        ]

        for key in preferred_keys:
            if key in obj:
                found = find_matrix_in_object(obj[key])
                if found is not None:
                    return found

        for value in obj.values():
            found = find_matrix_in_object(value)
            if found is not None:
                return found

    return None


def simpleitk_transform_to_matrix(path: Path) -> np.ndarray:
    """
    Convert a linear SimpleITK transform to a homogeneous 4x4 matrix.

    Evaluating the transformed origin and basis vectors makes this work
    for rigid, affine and similarity transforms without manually decoding
    their parameterization.
    """

    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            f"{path.suffix} transform requires SimpleITK for reading: {path}"
        ) from exc

    transform = sitk.ReadTransform(str(path))

    origin = np.asarray(
        transform.TransformPoint((0.0, 0.0, 0.0)),
        dtype=float,
    )

    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = origin

    for axis in range(3):
        point = np.zeros(3, dtype=float)
        point[axis] = 1.0

        transformed = np.asarray(
            transform.TransformPoint(tuple(point)),
            dtype=float,
        )

        matrix[:3, axis] = transformed - origin

    return matrix


def load_transform_matrix(path: Path) -> np.ndarray:
    suffixes = "".join(path.suffixes).lower()

    if not path.exists():
        raise FileNotFoundError(f"Transform does not exist: {path}")

    if path.suffix.lower() == ".npy":
        matrix = np.load(path)

    elif path.suffix.lower() == ".npz":
        data = np.load(path)

        matrix = None

        for key in [
            "matrix",
            "transform",
            "affine",
            "transform_matrix",
            "final_matrix",
        ]:
            if key in data:
                matrix = data[key]
                break

        if matrix is None:
            for key in data.files:
                arr = data[key]
                if np.asarray(arr).shape == (4, 4):
                    matrix = arr
                    break

        if matrix is None:
            raise ValueError(f"No 4x4 matrix found in {path}")

    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        matrix = find_matrix_in_object(obj)

        if matrix is None:
            raise ValueError(f"No 4x4 matrix found in {path}")

    elif path.suffix.lower() in {".txt", ".csv"}:
        delimiter = "," if path.suffix.lower() == ".csv" else None
        matrix = np.loadtxt(path, delimiter=delimiter)

    elif suffixes.endswith(".tfm") or suffixes.endswith(".h5"):
        matrix = simpleitk_transform_to_matrix(path)

    elif path.suffix.lower() in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                f"PyTorch is required to read transform {path}"
            ) from exc

        obj = torch.load(path, map_location="cpu")

        matrix = find_matrix_in_object(obj)

        if matrix is None:
            raise ValueError(f"No 4x4 matrix found in {path}")

    else:
        raise ValueError(
            f"Unsupported transform format: {path}"
        )

    matrix = np.asarray(matrix, dtype=float)

    if matrix.size == 16:
        matrix = matrix.reshape(4, 4)

    if matrix.shape != (4, 4):
        raise ValueError(
            f"Transform must be 4x4. Got {matrix.shape}: {path}"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Non-finite transform matrix: {path}")

    return matrix


def convert_lps_matrix_to_ras(matrix: np.ndarray) -> np.ndarray:
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    return flip @ matrix @ flip


# -------------------------------------------------------------------------
# Template ROI
# -------------------------------------------------------------------------


def load_template_mask(mask_path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(mask_path))
    data = np.asarray(img.dataobj)

    mask_indices = np.argwhere(data > 0)

    if len(mask_indices) == 0:
        raise ValueError(
            f"Template striatal mask is empty: {mask_path}"
        )

    return img, mask_indices.astype(np.float64)


# -------------------------------------------------------------------------
# Extent calculations
# -------------------------------------------------------------------------


def transform_points(
    points: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    homogeneous = np.column_stack(
        [points, np.ones(len(points), dtype=float)]
    )

    transformed = (matrix @ homogeneous.T).T

    return transformed[:, :3]


def ceil_int(value: float) -> int:
    return int(math.ceil(float(value)))


def round_up_multiple(value: int, multiple: int = 4) -> int:
    return int(math.ceil(value / multiple) * multiple)


def calculate_transform_diagnostics(
    template_to_subject_world: np.ndarray,
) -> dict[str, float]:
    linear = template_to_subject_world[:3, :3]

    singular_values = np.linalg.svd(
        linear,
        compute_uv=False,
    )

    determinant = float(np.linalg.det(linear))

    equivalent_scale = float(
        abs(determinant) ** (1.0 / 3.0)
    )

    smallest = float(np.min(singular_values))
    largest = float(np.max(singular_values))

    anisotropy_ratio = (
        largest / smallest
        if smallest > 0
        else float("inf")
    )

    return {
        "transform_determinant": determinant,
        "transform_equivalent_scale": equivalent_scale,
        "transform_sv_min": smallest,
        "transform_sv_max": largest,
        "transform_anisotropy_ratio": anisotropy_ratio,
    }


def percentile_dict(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()

    if clean.empty:
        return {}

    return {
        "min": float(clean.min()),
        "median": float(clean.median()),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
    }


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.localization_manifest.exists():
        raise FileNotFoundError(
            f"Localization manifest not found: "
            f"{args.localization_manifest}"
        )

    if not args.registration_manifest.exists():
        raise FileNotFoundError(
            f"Registration manifest not found: "
            f"{args.registration_manifest}"
        )

    if not args.template_striatal_mask.exists():
        raise FileNotFoundError(
            f"Striatal mask not found: "
            f"{args.template_striatal_mask}"
        )

    localization_df = pd.read_csv(
        args.localization_manifest
    )

    registration_df = pd.read_csv(
        args.registration_manifest
    )

    # ------------------------------------------------------------------
    # Detect localization columns
    # ------------------------------------------------------------------

    loc_uid_col = first_existing_column(
        localization_df,
        ["uid", "subject_uid", "subject_id", "id"],
    )

    center_x_col = first_existing_column(
        localization_df,
        [
            "final_center_x_vox",
            "selected_center_x_vox",
            "center_x_vox",
            "final_center_x",
            "center_x",
        ],
    )

    center_y_col = first_existing_column(
        localization_df,
        [
            "final_center_y_vox",
            "selected_center_y_vox",
            "center_y_vox",
            "final_center_y",
            "center_y",
        ],
    )

    center_z_col = first_existing_column(
        localization_df,
        [
            "final_center_z_vox",
            "selected_center_z_vox",
            "center_z_vox",
            "final_center_z",
            "center_z",
        ],
    )

    center_source_col = first_existing_column(
        localization_df,
        [
            "center_source",
            "selected_center_source",
            "final_center_source",
            "localization_source",
            "selected_method",
        ],
        required=False,
    )

    step7_extent_valid_col = first_existing_column(
        localization_df,
        [
            "extent_valid",
            "l1_extent_valid",
            "registration_extent_valid",
        ],
        required=False,
    )

    # ------------------------------------------------------------------
    # Detect registration columns
    # ------------------------------------------------------------------

    reg_uid_col = first_existing_column(
        registration_df,
        ["uid", "subject_uid", "subject_id", "id"],
    )

    transform_col = first_existing_column(
        registration_df,
        [
            "final_transform",
            "final_transform_path",
            "selected_transform",
            "selected_transform_path",
            "transform",
            "transform_path",
            "registration_transform_path",
            "matrix_path",
        ],
    )

    source_type_col = first_existing_column(
        registration_df,
        [
            "final_source_type",
            "source_type",
        ],
        required=False,
    )

    # Standardize UIDs.
    localization_df[loc_uid_col] = (
        localization_df[loc_uid_col].astype(str)
    )

    registration_df[reg_uid_col] = (
        registration_df[reg_uid_col].astype(str)
    )

    if localization_df[loc_uid_col].duplicated().any():
        raise ValueError(
            "Duplicate UIDs found in localization manifest."
        )

    if registration_df[reg_uid_col].duplicated().any():
        raise ValueError(
            "Duplicate UIDs found in registration manifest."
        )

    registration_lookup = registration_df.set_index(
        reg_uid_col
    )

    # ------------------------------------------------------------------
    # Load template ROI once
    # ------------------------------------------------------------------

    template_img, template_mask_voxels = load_template_mask(
        args.template_striatal_mask
    )

    template_affine = np.asarray(
        template_img.affine,
        dtype=float,
    )

    print("=" * 72)
    print("STEP 8A — STRIATAL CROP EXTENT ANALYSIS")
    print("=" * 72)
    print(
        f"Localization rows:       {len(localization_df)}"
    )
    print(
        f"Registration rows:       {len(registration_df)}"
    )
    print(
        f"Template ROI voxels:     {len(template_mask_voxels)}"
    )
    print(
        f"Transform direction:     {args.transform_direction}"
    )
    print(
        f"Transform coordinates:   "
        f"{args.transform_coordinate_system.upper()}"
    )
    print(
        f"Explicit extent exclude: "
        f"{args.exclude_extent_uids}"
    )
    print(
        "Similarity-rescue scans are auto-excluded "
        "from crop-size stats."
    )
    print()

    excluded_uid_set = {
        str(uid) for uid in args.exclude_extent_uids
    }

    records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Process subjects
    # ------------------------------------------------------------------

    for index, loc_row in localization_df.iterrows():
        uid = str(loc_row[loc_uid_col])

        record: dict[str, Any] = {
            "uid": uid,
            "successful": False,
            "error": "",
        }

        try:
            if uid not in registration_lookup.index:
                raise KeyError(
                    "UID missing from registration manifest."
                )

            reg_row = registration_lookup.loc[uid]

            center = np.array(
                [
                    float(loc_row[center_x_col]),
                    float(loc_row[center_y_col]),
                    float(loc_row[center_z_col]),
                ],
                dtype=float,
            )

            if not np.all(np.isfinite(center)):
                raise ValueError(
                    "Localization center contains non-finite values."
                )

            center_source = (
                str(loc_row[center_source_col])
                if center_source_col is not None
                else "unknown"
            )

            subject_path = resolve_subject_image(
                args.resampled_dir,
                uid,
            )

            subject_img = nib.load(str(subject_path))

            subject_affine = np.asarray(
                subject_img.affine,
                dtype=float,
            )

            subject_shape = tuple(
                int(v) for v in subject_img.shape[:3]
            )

            spacing = np.asarray(
                subject_img.header.get_zooms()[:3],
                dtype=float,
            )

            transform_path = Path(
                str(reg_row[transform_col])
            )

            if not transform_path.is_absolute():
                # First interpret relative to repository working directory.
                if not transform_path.exists():
                    # Then relative to registration manifest directory.
                    candidate = (
                        args.registration_manifest.parent
                        / transform_path
                    )

                    if candidate.exists():
                        transform_path = candidate

            matrix = load_transform_matrix(
                transform_path
            )

            # Convert stored transform into template -> subject RAS world.
            if args.transform_coordinate_system == "lps":
                matrix = convert_lps_matrix_to_ras(matrix)

            if args.transform_direction == "subject_to_template":
                matrix = np.linalg.inv(matrix)

            transform_diag = calculate_transform_diagnostics(
                matrix
            )

            # ----------------------------------------------------------
            # Build direct template-voxel -> subject-voxel transform
            # ----------------------------------------------------------

            template_voxel_to_subject_voxel = (
                np.linalg.inv(subject_affine)
                @ matrix
                @ template_affine
            )

            transformed_centers = transform_points(
                template_mask_voxels,
                template_voxel_to_subject_voxel,
            )

            # Each mask index represents a voxel CENTER.
            #
            # Account for the full physical voxel support rather than
            # measuring only transformed voxel centers.
            #
            # For an affine mapping, half-width of a transformed source
            # voxel along each destination axis is:
            #
            # 0.5 * sum(abs(linear coefficients))
            #
            linear_vox = (
                template_voxel_to_subject_voxel[:3, :3]
            )

            support_half_width = (
                0.5 * np.sum(
                    np.abs(linear_vox),
                    axis=1,
                )
            )

            roi_min = (
                np.min(transformed_centers, axis=0)
                - support_half_width
            )

            roi_max = (
                np.max(transformed_centers, axis=0)
                + support_half_width
            )

            # ----------------------------------------------------------
            # Directional extent relative to frozen Step-7 center
            #
            # RAS:
            #   X: left -> right
            #   Y: posterior -> anterior
            #   Z: inferior -> superior
            # ----------------------------------------------------------

            left = float(center[0] - roi_min[0])
            right = float(roi_max[0] - center[0])

            posterior = float(center[1] - roi_min[1])
            anterior = float(roi_max[1] - center[1])

            inferior = float(center[2] - roi_min[2])
            superior = float(roi_max[2] - center[2])

            bbox_vox = (
                np.ceil(roi_max)
                - np.floor(roi_min)
                + 1
            ).astype(int)

            bbox_mm = bbox_vox * spacing

            required_x = ceil_int(
                2.0 * max(left, right) + 1.0
            )

            required_y = ceil_int(
                2.0 * max(posterior, anterior) + 1.0
            )

            required_z = ceil_int(
                2.0 * max(inferior, superior) + 1.0
            )

            center_inside_bbox = bool(
                np.all(center >= roi_min)
                and np.all(center <= roi_max)
            )

            similarity_rescue = False

            if source_type_col is not None:
                similarity_rescue = (
                    str(
                        reg_row[source_type_col]
                    ).strip().lower()
                    ==
                    "similarity_rescue"
                )

            explicit_extent_exclusion = (
                uid in excluded_uid_set
                or similarity_rescue
            )

            scale_warning = bool(
                transform_diag[
                    "transform_equivalent_scale"
                ]
                < args.minimum_scale
                or transform_diag[
                    "transform_equivalent_scale"
                ]
                > args.maximum_scale
            )

            anisotropy_warning = bool(
                transform_diag[
                    "transform_anisotropy_ratio"
                ]
                > args.maximum_anisotropy
            )

            spacing_warning = bool(
                not np.allclose(
                    spacing,
                    np.array([2.46, 2.46, 2.46]),
                    atol=0.01,
                    rtol=0.0,
                )
            )

            # If Step 7 explicitly froze an extent-valid decision,
            # preserve it.
            step7_extent_valid = None

            if step7_extent_valid_col is not None:
                step7_extent_valid = as_bool(
                    loc_row[step7_extent_valid_col]
                )

            # Only L1-based centers should contribute to estimating
            # registration-derived crop extent if center_source exists.
            if center_source_col is None:
                l1_center = True
            else:
                l1_center = (
                    "l1" in center_source.strip().lower()
                )

            extent_valid_for_size_selection = bool(
                l1_center
                and not explicit_extent_exclusion
                and not scale_warning
                and not anisotropy_warning
                and (
                    step7_extent_valid is not False
                )
            )

            review_recommended = bool(
                explicit_extent_exclusion
                or scale_warning
                or anisotropy_warning
                or spacing_warning
                or not center_inside_bbox
            )

            record.update(
                {
                    "successful": True,
                    "subject_path": str(subject_path),
                    "transform_path": str(transform_path),

                    "center_x_vox": float(center[0]),
                    "center_y_vox": float(center[1]),
                    "center_z_vox": float(center[2]),
                    "center_source": center_source,

                    "shape_x": subject_shape[0],
                    "shape_y": subject_shape[1],
                    "shape_z": subject_shape[2],

                    "spacing_x_mm": float(spacing[0]),
                    "spacing_y_mm": float(spacing[1]),
                    "spacing_z_mm": float(spacing[2]),

                    "roi_min_x_vox": float(roi_min[0]),
                    "roi_min_y_vox": float(roi_min[1]),
                    "roi_min_z_vox": float(roi_min[2]),

                    "roi_max_x_vox": float(roi_max[0]),
                    "roi_max_y_vox": float(roi_max[1]),
                    "roi_max_z_vox": float(roi_max[2]),

                    "bbox_x_vox": int(bbox_vox[0]),
                    "bbox_y_vox": int(bbox_vox[1]),
                    "bbox_z_vox": int(bbox_vox[2]),

                    "bbox_x_mm": float(bbox_mm[0]),
                    "bbox_y_mm": float(bbox_mm[1]),
                    "bbox_z_mm": float(bbox_mm[2]),

                    "half_extent_left_vox": left,
                    "half_extent_right_vox": right,
                    "half_extent_posterior_vox": posterior,
                    "half_extent_anterior_vox": anterior,
                    "half_extent_inferior_vox": inferior,
                    "half_extent_superior_vox": superior,

                    "half_extent_left_mm": left * spacing[0],
                    "half_extent_right_mm": right * spacing[0],
                    "half_extent_posterior_mm": (
                        posterior * spacing[1]
                    ),
                    "half_extent_anterior_mm": (
                        anterior * spacing[1]
                    ),
                    "half_extent_inferior_mm": (
                        inferior * spacing[2]
                    ),
                    "half_extent_superior_mm": (
                        superior * spacing[2]
                    ),

                    "required_crop_x_vox": required_x,
                    "required_crop_y_vox": required_y,
                    "required_crop_z_vox": required_z,

                    "required_crop_x_mm": (
                        required_x * spacing[0]
                    ),
                    "required_crop_y_mm": (
                        required_y * spacing[1]
                    ),
                    "required_crop_z_mm": (
                        required_z * spacing[2]
                    ),

                    "center_inside_transformed_roi_bbox": (
                        center_inside_bbox
                    ),

                    **transform_diag,

                    "similarity_rescue": similarity_rescue,
                    "explicit_extent_exclusion": (
                        explicit_extent_exclusion
                    ),
                    "transform_scale_warning": scale_warning,
                    "transform_anisotropy_warning": (
                        anisotropy_warning
                    ),
                    "spacing_warning": spacing_warning,

                    "step7_extent_valid": (
                        step7_extent_valid
                    ),
                    "l1_center": l1_center,

                    "extent_valid_for_size_selection": (
                        extent_valid_for_size_selection
                    ),
                    "review_recommended": review_recommended,
                }
            )

        except Exception as exc:
            record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        records.append(record)

        if (index + 1) % 100 == 0:
            print(
                f"Processed {index + 1}/"
                f"{len(localization_df)}"
            )

    # ------------------------------------------------------------------
    # Save per-subject statistics
    # ------------------------------------------------------------------

    result_df = pd.DataFrame(records)

    statistics_path = (
        args.output_dir
        / "striatal_extent_statistics.csv"
    )

    result_df.to_csv(
        statistics_path,
        index=False,
    )

    successful_df = result_df[
        result_df["successful"] == True  # noqa: E712
    ].copy()

    eligible_df = successful_df[
        successful_df[
            "extent_valid_for_size_selection"
        ]
        == True  # noqa: E712
    ].copy()

    similarity_rescue_uids: list[str] = []

    if (
        not successful_df.empty
        and "similarity_rescue" in successful_df.columns
    ):
        similarity_rescue_uids = sorted(
            successful_df.loc[
                successful_df["similarity_rescue"]
                .fillna(False)
                .astype(bool),
                "uid",
            ]
            .astype(str)
            .tolist()
        )

    # ------------------------------------------------------------------
    # Dataset summary
    # ------------------------------------------------------------------

    metrics = [
        "bbox_x_vox",
        "bbox_y_vox",
        "bbox_z_vox",

        "bbox_x_mm",
        "bbox_y_mm",
        "bbox_z_mm",

        "half_extent_left_vox",
        "half_extent_right_vox",
        "half_extent_posterior_vox",
        "half_extent_anterior_vox",
        "half_extent_inferior_vox",
        "half_extent_superior_vox",

        "required_crop_x_vox",
        "required_crop_y_vox",
        "required_crop_z_vox",

        "required_crop_x_mm",
        "required_crop_y_mm",
        "required_crop_z_mm",

        "transform_equivalent_scale",
        "transform_anisotropy_ratio",
    ]

    extent_statistics = {}

    for metric in metrics:
        if metric in eligible_df.columns:
            extent_statistics[metric] = percentile_dict(
                eligible_df[metric]
            )

    summary = {
        "step": "8A",
        "description": (
            "Analyze required bilateral striatal crop extent"
        ),

        "number_of_localization_rows": int(
            len(localization_df)
        ),

        "number_successful": int(
            len(successful_df)
        ),

        "number_failed": int(
            len(result_df) - len(successful_df)
        ),

        "number_extent_valid_for_size_selection": int(
            len(eligible_df)
        ),

        "number_extent_excluded": int(
            len(successful_df) - len(eligible_df)
        ),

        "number_review_recommended": int(
            successful_df[
                "review_recommended"
            ].sum()
        ),

        "explicit_extent_exclusion_uids": sorted(
            excluded_uid_set
        ),

        "similarity_rescue_extent_exclusion_uids": (
            similarity_rescue_uids
        ),

        "transform_scale_bounds": {
            "minimum": args.minimum_scale,
            "maximum": args.maximum_scale,
        },

        "maximum_anisotropy": (
            args.maximum_anisotropy
        ),

        "extent_statistics": extent_statistics,

        "final_crop_size_selected": False,

        "note": (
            "Step 8A only measures crop requirements. "
            "Final dimensions must be selected in Step 8B."
        ),
    }

    summary_path = (
        args.output_dir
        / "crop_extent_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Generate analytical candidates for Step 8B.
    #
    # These are deliberately NOT declared as the selected crop.
    # ------------------------------------------------------------------

    candidates = []

    if not eligible_df.empty:
        for percentile_name, quantile in [
            ("p90", 0.90),
            ("p95", 0.95),
            ("p99", 0.99),
            ("max", 1.00),
        ]:
            base_x = ceil_int(
                eligible_df[
                    "required_crop_x_vox"
                ].quantile(quantile)
            )

            base_y = ceil_int(
                eligible_df[
                    "required_crop_y_vox"
                ].quantile(quantile)
            )

            base_z = ceil_int(
                eligible_df[
                    "required_crop_z_vox"
                ].quantile(quantile)
            )

            for margin_per_side in [0, 2, 4]:
                candidate_x = round_up_multiple(
                    base_x + 2 * margin_per_side,
                    4,
                )

                candidate_y = round_up_multiple(
                    base_y + 2 * margin_per_side,
                    4,
                )

                candidate_z = round_up_multiple(
                    base_z + 2 * margin_per_side,
                    4,
                )

                candidates.append(
                    {
                        "basis": percentile_name,
                        "quantile": quantile,
                        "safety_margin_per_side_vox": (
                            margin_per_side
                        ),
                        "candidate_x_vox": candidate_x,
                        "candidate_y_vox": candidate_y,
                        "candidate_z_vox": candidate_z,
                        "candidate_x_mm": (
                            candidate_x * 2.46
                        ),
                        "candidate_y_mm": (
                            candidate_y * 2.46
                        ),
                        "candidate_z_mm": (
                            candidate_z * 2.46
                        ),
                        "selected": False,
                    }
                )

    candidate_df = pd.DataFrame(candidates)

    candidate_path = (
        args.output_dir
        / "crop_extent_candidates.csv"
    )

    candidate_df.to_csv(
        candidate_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 8A COMPLETE")
    print("=" * 72)

    print(
        f"Successful:                  "
        f"{len(successful_df)}"
    )

    print(
        f"Failed:                      "
        f"{len(result_df) - len(successful_df)}"
    )

    print(
        f"Eligible for size analysis:  "
        f"{len(eligible_df)}"
    )

    print(
        f"Excluded from size analysis: "
        f"{len(successful_df) - len(eligible_df)}"
    )

    print(
        f"Review recommended:          "
        f"{successful_df['review_recommended'].sum()}"
    )

    print()

    if not eligible_df.empty:
        print("Required crop dimensions [voxels]")
        print("-" * 72)

        for axis in ["x", "y", "z"]:
            stats = percentile_dict(
                eligible_df[
                    f"required_crop_{axis}_vox"
                ]
            )

            print(
                f"{axis.upper()}: "
                f"median={stats['median']:.1f}  "
                f"p90={stats['p90']:.1f}  "
                f"p95={stats['p95']:.1f}  "
                f"p99={stats['p99']:.1f}  "
                f"max={stats['max']:.1f}"
            )

    print()
    print(f"Statistics: {statistics_path}")
    print(f"Summary:    {summary_path}")
    print(f"Candidates: {candidate_path}")
    print()

    print(
        "IMPORTANT: no final crop size has been selected. "
        "Use these results in Step 8B."
    )


if __name__ == "__main__":
    main()
