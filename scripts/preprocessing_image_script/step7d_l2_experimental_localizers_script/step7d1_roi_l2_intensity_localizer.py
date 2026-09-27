from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


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

    if arr.size == 0:
        return {}

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


def save_binary_mask(
    mask: np.ndarray,
    reference_img: nib.Nifti1Image,
    output_path: Path,
) -> None:
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)

    out = nib.Nifti1Image(
        mask.astype(np.uint8),
        reference_img.affine,
        header,
    )

    nib.save(
        out,
        str(output_path),
    )


def component_information(
    component_mask: np.ndarray,
    smooth: np.ndarray,
    threshold: float,
    l0_center: np.ndarray,
    shape: np.ndarray,
) -> dict:

    coords = np.argwhere(component_mask)

    nvox = int(coords.shape[0])

    centroid = coords.mean(axis=0)

    intensities = smooth[component_mask]

    excess = np.clip(
        intensities - threshold,
        0.0,
        None,
    )

    total_excess = float(excess.sum())

    if total_excess <= 0:
        total_excess = float(
            intensities.sum()
        )

    # Very broad centrality prior.
    #
    # This does NOT define the ROI location.
    # It only downweights isolated hot components
    # far away from the central brain region.
    scale = np.maximum(
        shape.astype(np.float64) * 0.35,
        1.0,
    )

    normalized_distance = (
        centroid - l0_center
    ) / scale

    centrality = float(
        np.exp(
            -0.5
            * np.sum(
                normalized_distance ** 2
            )
        )
    )

    # Reward:
    #   spatially extended component
    #   high excess uptake
    #   broadly central position
    score = float(
        total_excess
        * np.sqrt(max(nvox, 1))
        * (
            0.25
            + 0.75 * centrality
        )
    )

    return {
        "nvox": nvox,
        "centroid": centroid,
        "centrality": centrality,
        "score": score,
        "mean_intensity": float(
            np.mean(intensities)
        ),
        "max_intensity": float(
            np.max(intensities)
        ),
    }


def localize_l2(
    image: np.ndarray,
    spacing: np.ndarray,
    threshold_percentile: float = 88.0,
    sigma_voxels: float = 1.0,
    min_component_voxels: int = 8,
    max_components: int = 6,
) -> dict:

    shape = np.asarray(
        image.shape,
        dtype=np.int64,
    )

    l0_center = (
        shape.astype(np.float64) - 1.0
    ) / 2.0

    # ---------------------------------------------------------
    # Finite / positive image
    # ---------------------------------------------------------

    work = np.asarray(
        image,
        dtype=np.float32,
    ).copy()

    finite = np.isfinite(work)

    work[~finite] = 0.0
    work[work < 0] = 0.0

    # ---------------------------------------------------------
    # Mild smoothing
    #
    # sigma = 1 voxel ≈ 2.46 mm
    # ---------------------------------------------------------

    smooth = ndimage.gaussian_filter(
        work,
        sigma=sigma_voxels,
        mode="nearest",
    )

    # ---------------------------------------------------------
    # Broad central search box
    #
    # X/Y are constrained moderately.
    #
    # Z is deliberately very broad because Step-7 L1 showed
    # large Z/FOV variation, including valid striatal centers
    # close to inferior/superior image limits.
    # ---------------------------------------------------------

    fractions = {
        "x": (0.15, 0.85),
        "y": (0.15, 0.85),
        "z": (0.02, 0.98),
    }

    x0 = int(
        np.floor(
            shape[0] * fractions["x"][0]
        )
    )
    x1 = int(
        np.ceil(
            shape[0] * fractions["x"][1]
        )
    )

    y0 = int(
        np.floor(
            shape[1] * fractions["y"][0]
        )
    )
    y1 = int(
        np.ceil(
            shape[1] * fractions["y"][1]
        )
    )

    z0 = int(
        np.floor(
            shape[2] * fractions["z"][0]
        )
    )
    z1 = int(
        np.ceil(
            shape[2] * fractions["z"][1]
        )
    )

    x0 = max(0, x0)
    y0 = max(0, y0)
    z0 = max(0, z0)

    x1 = min(shape[0], x1)
    y1 = min(shape[1], y1)
    z1 = min(shape[2], z1)

    search_mask = np.zeros(
        image.shape,
        dtype=bool,
    )

    search_mask[
        x0:x1,
        y0:y1,
        z0:z1,
    ] = True

    search_values = smooth[
        search_mask
        & np.isfinite(smooth)
        & (smooth > 0)
    ]

    failure_reasons = []

    if search_values.size < 100:
        return {
            "valid": False,
            "failure_reason": (
                "insufficient_positive_search_voxels"
            ),
            "candidate_mask": np.zeros(
                image.shape,
                dtype=bool,
            ),
        }

    # ---------------------------------------------------------
    # High-uptake threshold
    # ---------------------------------------------------------

    threshold = float(
        np.percentile(
            search_values,
            threshold_percentile,
        )
    )

    raw_candidate = (
        search_mask
        & (smooth >= threshold)
    )

    # ---------------------------------------------------------
    # Connected components
    #
    # 26-connectivity in 3-D
    # ---------------------------------------------------------

    structure = (
        ndimage.generate_binary_structure(
            rank=3,
            connectivity=3,
        )
    )

    labels, number_components = (
        ndimage.label(
            raw_candidate,
            structure=structure,
        )
    )

    components = []

    for label_id in range(
        1,
        number_components + 1,
    ):

        component_mask = (
            labels == label_id
        )

        nvox = int(
            component_mask.sum()
        )

        # Main defense against an isolated hot voxel.
        if nvox < min_component_voxels:
            continue

        info = component_information(
            component_mask=component_mask,
            smooth=smooth,
            threshold=threshold,
            l0_center=l0_center,
            shape=shape,
        )

        info["label"] = label_id

        components.append(info)

    if not components:
        return {
            "valid": False,
            "failure_reason": (
                "no_spatially_extended_component"
            ),
            "candidate_mask": np.zeros(
                image.shape,
                dtype=bool,
            ),
            "threshold": threshold,
            "raw_component_count": int(
                number_components
            ),
            "retained_component_count": 0,
        }

    # ---------------------------------------------------------
    # Rank components
    # ---------------------------------------------------------

    components.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    top_score = components[0]["score"]

    retained = []

    for component in components:

        if len(retained) >= max_components:
            break

        # Keep weaker plausible components as well.
        # This matters for asymmetric/pathological uptake.
        relative_score = (
            component["score"]
            / max(top_score, 1e-12)
        )

        if (
            len(retained) == 0
            or relative_score >= 0.08
        ):
            retained.append(component)

    candidate_mask = np.zeros(
        image.shape,
        dtype=bool,
    )

    for component in retained:
        candidate_mask[
            labels == component["label"]
        ] = True

    coords = np.argwhere(
        candidate_mask
    )

    if coords.size == 0:
        return {
            "valid": False,
            "failure_reason": (
                "retained_candidate_empty"
            ),
            "candidate_mask": candidate_mask,
            "threshold": threshold,
        }

    # ---------------------------------------------------------
    # Weighted intensity center
    #
    # Only intensity ABOVE the high-uptake threshold
    # contributes strongly.
    # ---------------------------------------------------------

    weights_volume = np.clip(
        smooth - threshold,
        0.0,
        None,
    )

    weights = weights_volume[
        candidate_mask
    ]

    if (
        not np.all(np.isfinite(weights))
        or weights.sum() <= 0
    ):
        weights = smooth[
            candidate_mask
        ]

    if (
        not np.all(np.isfinite(weights))
        or weights.sum() <= 0
    ):
        weights = np.ones(
            len(coords),
            dtype=np.float64,
        )

    center = np.average(
        coords.astype(np.float64),
        axis=0,
        weights=weights,
    )

    # ---------------------------------------------------------
    # Candidate extent
    # ---------------------------------------------------------

    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)

    bbox_size = (
        bbox_max
        - bbox_min
        + 1
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

    candidate_voxels = int(
        coords.shape[0]
    )

    largest_component_voxels = int(
        max(
            component["nvox"]
            for component in retained
        )
    )

    # ---------------------------------------------------------
    # Bilateral plausibility
    #
    # Use the candidate bounding-box midpoint rather than the
    # image center, so this check is not overly dependent on FOV.
    # ---------------------------------------------------------

    split_x = float(
        (
            bbox_min[0]
            + bbox_max[0]
        )
        / 2.0
    )

    coord_x = coords[:, 0]

    left_side = coord_x < split_x
    right_side = coord_x > split_x

    left_mass = float(
        weights[left_side].sum()
    )

    right_mass = float(
        weights[right_side].sum()
    )

    if (
        left_mass > 0
        and right_mass > 0
    ):
        bilateral_balance = float(
            min(
                left_mass,
                right_mass,
            )
            / max(
                left_mass,
                right_mass,
            )
        )
    else:
        bilateral_balance = 0.0

    # ---------------------------------------------------------
    # Confidence features
    # ---------------------------------------------------------

    # Enough spatial support.
    size_score = float(
        np.clip(
            candidate_voxels / 120.0,
            0.0,
            1.0,
        )
    )

    # Minimum useful 3-D extent.
    extent_score = float(
        np.mean(
            [
                np.clip(
                    bbox_size[0] / 8.0,
                    0.0,
                    1.0,
                ),
                np.clip(
                    bbox_size[1] / 6.0,
                    0.0,
                    1.0,
                ),
                np.clip(
                    bbox_size[2] / 4.0,
                    0.0,
                    1.0,
                ),
            ]
        )
    )

    # Broad centrality only.
    central_scale = np.maximum(
        shape.astype(np.float64) * 0.35,
        1.0,
    )

    centrality = float(
        np.exp(
            -0.5
            * np.sum(
                (
                    (
                        center - l0_center
                    )
                    / central_scale
                )
                ** 2
            )
        )
    )

    confidence = float(
        0.35 * bilateral_balance
        + 0.25 * extent_score
        + 0.20 * size_score
        + 0.20 * centrality
    )

    # ---------------------------------------------------------
    # Plausibility checks
    #
    # IMPORTANT:
    # No fallback here.
    # An invalid L2 remains invalid for later comparison.
    # ---------------------------------------------------------

    center_inside = bool(
        np.all(center >= 0)
        and np.all(
            center
            <= shape.astype(
                np.float64
            ) - 1
        )
    )

    if not center_inside:
        failure_reasons.append(
            "center_outside_image"
        )

    if candidate_voxels < 20:
        failure_reasons.append(
            "candidate_too_small"
        )

    if bbox_size[0] < 6:
        failure_reasons.append(
            "insufficient_x_extent"
        )

    if bbox_size[1] < 4:
        failure_reasons.append(
            "insufficient_y_extent"
        )

    if bbox_size[2] < 3:
        failure_reasons.append(
            "insufficient_z_extent"
        )

    if bilateral_balance < 0.08:
        failure_reasons.append(
            "poor_bilateral_balance"
        )

    if minimum_border_distance < 1:
        failure_reasons.append(
            "candidate_touches_fov_border"
        )

    if not np.all(
        np.isfinite(center)
    ):
        failure_reasons.append(
            "nonfinite_center"
        )

    valid = (
        len(failure_reasons) == 0
    )

    distance_from_l0_voxels = float(
        np.linalg.norm(
            center - l0_center
        )
    )

    distance_from_l0_mm = float(
        np.linalg.norm(
            (
                center - l0_center
            )
            * spacing
        )
    )

    return {
        "valid": valid,
        "failure_reason": (
            ";".join(failure_reasons)
            if failure_reasons
            else ""
        ),

        "center": center,
        "candidate_mask": candidate_mask,

        "threshold": threshold,
        "raw_component_count": int(
            number_components
        ),
        "retained_component_count": int(
            len(retained)
        ),

        "candidate_voxels": (
            candidate_voxels
        ),
        "largest_component_voxels": (
            largest_component_voxels
        ),

        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_size": bbox_size,

        "minimum_border_distance_voxels": (
            minimum_border_distance
        ),

        "bilateral_balance": (
            bilateral_balance
        ),
        "centrality": centrality,
        "size_score": size_score,
        "extent_score": extent_score,
        "confidence": confidence,

        "distance_from_l0_voxels": (
            distance_from_l0_voxels
        ),
        "distance_from_l0_mm": (
            distance_from_l0_mm
        ),

        "l0_center": l0_center,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 7F — ROI-L2 independent "
            "intensity-based localizer."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step6e_intensity_normalization_data/step6e_normalize_occipital"
        ),
    )

    parser.add_argument(
        "--output-mask-dir",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/"
            "step7d_l2_experimental_localizers_data/step7d1_roi_l2_intensity_localizer/masks"
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7d_l2_experimental_localizers_data/"
            "step7d1_roi_l2_intensity_localizer/l2_localization.csv"
        ),
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7d_l2_experimental_localizers_data/"
            "step7d1_roi_l2_intensity_localizer/l2_summary.json"
        ),
    )

    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=88.0,
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--min-component-voxels",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    output_mask_dir = resolve_path(args.output_mask_dir)
    output_csv = resolve_path(args.output_csv)
    summary_json = resolve_path(args.summary_json)

    files = find_nifti_files(
        input_dir
    )

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

    uids = [
        uid_from_path(path)
        for path in files
    ]

    if len(uids) != len(set(uids)):
        raise RuntimeError(
            "Duplicate UIDs found."
        )

    output_mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    execution_failures = []

    print()
    print("=" * 80)
    print(
        "STEP 7F — ROI-L2 INTENSITY LOCALIZER"
    )
    print("=" * 80)

    print(f"Scans:                 {len(files)}")
    print(
        f"Threshold percentile:  "
        f"{args.threshold_percentile}"
    )
    print(
        f"Gaussian sigma:        "
        f"{args.sigma} voxel"
    )
    print(
        f"Min component voxels:  "
        f"{args.min_component_voxels}"
    )
    print()
    print(
        "IMPORTANT: L2 is evaluated RAW. "
        "No L0/L1 fallback is applied."
    )
    print()

    for index, path in enumerate(
        files,
        start=1,
    ):

        uid = uid_from_path(path)

        mask_path = (
            output_mask_dir
            / f"{uid}.nii.gz"
        )

        try:
            img = nib.load(
                str(path)
            )

            if len(img.shape) != 3:
                raise RuntimeError(
                    f"Expected 3-D image, "
                    f"got {img.shape}."
                )

            orientation = (
                nib.aff2axcodes(
                    img.affine
                )
            )

            if orientation != (
                "R",
                "A",
                "S",
            ):
                raise RuntimeError(
                    f"Expected RAS, "
                    f"got {orientation}."
                )

            image = np.asarray(
                img.dataobj,
                dtype=np.float32,
            )

            spacing = np.asarray(
                img.header.get_zooms()[:3],
                dtype=np.float64,
            )

            result = localize_l2(
                image=image,
                spacing=spacing,
                threshold_percentile=(
                    args.threshold_percentile
                ),
                sigma_voxels=args.sigma,
                min_component_voxels=(
                    args.min_component_voxels
                ),
            )

            candidate_mask = result[
                "candidate_mask"
            ]

            save_binary_mask(
                candidate_mask,
                img,
                mask_path,
            )

            if "center" in result:
                center = np.asarray(
                    result["center"],
                    dtype=np.float64,
                )

                center_world = (
                    nib.affines.apply_affine(
                        img.affine,
                        center,
                    )
                )
            else:
                center = np.asarray(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                )

                center_world = np.asarray(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                )

            bbox_min = result.get(
                "bbox_min",
                np.asarray(
                    [-1, -1, -1]
                ),
            )

            bbox_max = result.get(
                "bbox_max",
                np.asarray(
                    [-1, -1, -1]
                ),
            )

            bbox_size = result.get(
                "bbox_size",
                np.asarray(
                    [0, 0, 0]
                ),
            )

            row = {
                "uid": uid,

                "l2_center_x": float(
                    center[0]
                ),
                "l2_center_y": float(
                    center[1]
                ),
                "l2_center_z": float(
                    center[2]
                ),

                "l2_physical_x_mm": float(
                    center_world[0]
                ),
                "l2_physical_y_mm": float(
                    center_world[1]
                ),
                "l2_physical_z_mm": float(
                    center_world[2]
                ),

                "l2_valid": bool(
                    result["valid"]
                ),

                "l2_confidence": float(
                    result.get(
                        "confidence",
                        0.0,
                    )
                ),

                "fallback_used": False,

                "failure_reason": (
                    result.get(
                        "failure_reason",
                        "",
                    )
                ),

                "threshold": float(
                    result.get(
                        "threshold",
                        np.nan,
                    )
                ),

                "raw_component_count": int(
                    result.get(
                        "raw_component_count",
                        0,
                    )
                ),

                "retained_component_count": int(
                    result.get(
                        "retained_component_count",
                        0,
                    )
                ),

                "candidate_voxels": int(
                    result.get(
                        "candidate_voxels",
                        0,
                    )
                ),

                "largest_component_voxels": int(
                    result.get(
                        "largest_component_voxels",
                        0,
                    )
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

                "minimum_border_distance_voxels": int(
                    result.get(
                        "minimum_border_distance_voxels",
                        -1,
                    )
                ),

                "bilateral_balance": float(
                    result.get(
                        "bilateral_balance",
                        0.0,
                    )
                ),

                "centrality": float(
                    result.get(
                        "centrality",
                        0.0,
                    )
                ),

                "distance_from_l0_voxels": float(
                    result.get(
                        "distance_from_l0_voxels",
                        np.nan,
                    )
                ),

                "distance_from_l0_mm": float(
                    result.get(
                        "distance_from_l0_mm",
                        np.nan,
                    )
                ),

                "candidate_mask_path": str(
                    mask_path
                ),
            }

            rows.append(row)

        except Exception as exc:

            execution_failures.append(
                {
                    "uid": uid,
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

    if execution_failures:

        print()
        print("EXECUTION FAILURES")
        print("-" * 80)

        for failure in (
            execution_failures[:20]
        ):
            print(
                failure["uid"],
                "->",
                failure["error"],
            )

        raise RuntimeError(
            f"{len(execution_failures)} "
            f"execution failures."
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

    valid_rows = [
        row
        for row in rows
        if row["l2_valid"]
    ]

    invalid_rows = [
        row
        for row in rows
        if not row["l2_valid"]
    ]

    reasons = Counter()

    for row in invalid_rows:
        reason_string = (
            row["failure_reason"]
        )

        if not reason_string:
            reasons["unspecified"] += 1
            continue

        for reason in (
            reason_string.split(";")
        ):
            reasons[reason] += 1

    summary = {
        "step": "7F",
        "method": (
            "ROI-L2 independent intensity localizer"
        ),
        "uses_l1": False,
        "fallback_applied": False,

        "processed": len(rows),
        "execution_failures": len(
            execution_failures
        ),

        "l2_valid": len(valid_rows),
        "l2_invalid": len(invalid_rows),

        "parameters": {
            "threshold_percentile": (
                args.threshold_percentile
            ),
            "gaussian_sigma_voxels": (
                args.sigma
            ),
            "min_component_voxels": (
                args.min_component_voxels
            ),
        },

        "invalid_reason_counts": dict(
            reasons
        ),

        "confidence_distribution": (
            describe(
                [
                    row[
                        "l2_confidence"
                    ]
                    for row in valid_rows
                ]
            )
        ),

        "candidate_voxel_distribution": (
            describe(
                [
                    row[
                        "candidate_voxels"
                    ]
                    for row in valid_rows
                ]
            )
        ),

        "bilateral_balance_distribution": (
            describe(
                [
                    row[
                        "bilateral_balance"
                    ]
                    for row in valid_rows
                ]
            )
        ),

        "distance_from_l0_mm_distribution": (
            describe(
                [
                    row[
                        "distance_from_l0_mm"
                    ]
                    for row in valid_rows
                    if np.isfinite(
                        row[
                            "distance_from_l0_mm"
                        ]
                    )
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

    # ---------------------------------------------------------
    # Console
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 7F RESULTS")
    print("=" * 80)

    print(
        f"Processed:               "
        f"{len(rows)}"
    )

    print(
        f"Execution failures:      "
        f"{len(execution_failures)}"
    )

    print(
        f"L2 valid:                "
        f"{len(valid_rows)}"
    )

    print(
        f"L2 invalid:              "
        f"{len(invalid_rows)}"
    )

    print()

    if reasons:
        print("Invalid reasons:")

        for reason, count in (
            reasons.most_common()
        ):
            print(
                f"  {reason}: {count}"
            )

        print()

    if valid_rows:
        conf = summary[
            "confidence_distribution"
        ]

        print("L2 confidence:")
        print(
            f"  median: "
            f"{conf['median']:.4f}"
        )
        print(
            f"  p05:    "
            f"{conf['p05']:.4f}"
        )
        print(
            f"  p95:    "
            f"{conf['p95']:.4f}"
        )

        print()

        dist = summary[
            "distance_from_l0_mm_distribution"
        ]

        print("L2 distance from L0 [mm]:")
        print(
            f"  median: "
            f"{dist['median']:.2f}"
        )
        print(
            f"  p95:    "
            f"{dist['p95']:.2f}"
        )
        print(
            f"  max:    "
            f"{dist['max']:.2f}"
        )

    print()
    print("CSV:")
    print(
        f"  {output_csv.resolve()}"
    )

    print()
    print("Candidate masks:")
    print(
        f"  {output_mask_dir.resolve()}"
    )

    print()
    print("Summary:")
    print(
        f"  {summary_json.resolve()}"
    )

    print()
    print(
        "STEP 7F RAW L2 LOCALIZATION: COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
