from __future__ import annotations

import argparse
import csv
import json
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
    return sorted(
        set(
            list(directory.glob("*.nii"))
            + list(directory.glob("*.nii.gz"))
        )
    )


def describe(values) -> dict:
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return {}

    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def local_mass(
    image: np.ndarray,
    point: np.ndarray,
    radius: int = 3,
) -> float:

    x, y, z = [
        int(round(v))
        for v in point
    ]

    x0 = max(0, x - radius)
    x1 = min(image.shape[0], x + radius + 1)

    y0 = max(0, y - radius)
    y1 = min(image.shape[1], y + radius + 1)

    z0 = max(0, z - radius)
    z1 = min(image.shape[2], z + radius + 1)

    patch = image[
        x0:x1,
        y0:y1,
        z0:z1,
    ]

    positive = patch[
        np.isfinite(patch)
        & (patch > 0)
    ]

    if positive.size == 0:
        return 0.0

    # Spatial extent matters more than one isolated maximum.
    return float(
        np.sum(positive)
    )


def find_peaks(
    smooth: np.ndarray,
    search_mask: np.ndarray,
    percentile: float,
    max_peaks: int = 80,
) -> list[dict]:

    values = smooth[
        search_mask
        & np.isfinite(smooth)
        & (smooth > 0)
    ]

    if values.size == 0:
        return []

    threshold = float(
        np.percentile(
            values,
            percentile,
        )
    )

    # Local maximum over roughly 12 mm.
    maximum = ndimage.maximum_filter(
        smooth,
        size=5,
        mode="nearest",
    )

    maxima = (
        search_mask
        & (smooth == maximum)
        & (smooth >= threshold)
    )

    coords = np.argwhere(maxima)

    if len(coords) == 0:
        return []

    peak_values = smooth[
        tuple(coords.T)
    ]

    order = np.argsort(
        peak_values
    )[::-1]

    peaks = []

    # Non-maximum suppression:
    # don't retain several almost identical peaks.
    minimum_separation_vox = 4.0

    for index in order:

        point = (
            coords[index]
            .astype(np.float64)
        )

        intensity = float(
            peak_values[index]
        )

        too_close = False

        for existing in peaks:

            distance = np.linalg.norm(
                point
                - existing["point"]
            )

            if (
                distance
                < minimum_separation_vox
            ):
                too_close = True
                break

        if too_close:
            continue

        mass = local_mass(
            smooth,
            point,
            radius=3,
        )

        peaks.append(
            {
                "point": point,
                "intensity": intensity,
                "local_mass": mass,
            }
        )

        if len(peaks) >= max_peaks:
            break

    return peaks


def choose_bilateral_pair(
    peaks: list[dict],
    shape: np.ndarray,
    spacing: np.ndarray,
) -> dict | None:

    if len(peaks) < 2:
        return None

    mid_x = (
        float(shape[0]) - 1.0
    ) / 2.0

    best = None

    for i in range(len(peaks)):

        p1 = peaks[i]

        for j in range(i + 1, len(peaks)):

            p2 = peaks[j]

            x1 = p1["point"][0]
            x2 = p2["point"][0]

            # Must lie on opposite sides of approximate
            # midsagittal plane.
            if (
                (x1 < mid_x and x2 < mid_x)
                or
                (x1 > mid_x and x2 > mid_x)
            ):
                continue

            # Order anatomically in RAS:
            # lower X = left.
            if x1 < x2:
                left = p1
                right = p2
            else:
                left = p2
                right = p1

            delta_vox = (
                right["point"]
                - left["point"]
            )

            delta_mm = (
                delta_vox
                * spacing
            )

            separation_x_mm = abs(
                delta_mm[0]
            )

            dy_mm = abs(
                delta_mm[1]
            )

            dz_mm = abs(
                delta_mm[2]
            )

            # Very broad anatomical constraints.
            #
            # Expected striatal L-R separation is roughly
            # several cm, but deliberately keep these ranges
            # wide to avoid encoding L1.
            if not (
                18.0
                <= separation_x_mm
                <= 65.0
            ):
                continue

            if dy_mm > 25.0:
                continue

            if dz_mm > 25.0:
                continue

            midpoint = (
                left["point"]
                + right["point"]
            ) / 2.0

            # -------------------------------------------------
            # Pair scoring
            # -------------------------------------------------

            # Prefer ~38 mm separation, but weakly.
            separation_score = float(
                np.exp(
                    -0.5
                    * (
                        (
                            separation_x_mm
                            - 38.0
                        )
                        / 14.0
                    )
                    ** 2
                )
            )

            # Bilateral structures should be similar in Y/Z.
            y_score = float(
                np.exp(
                    -0.5
                    * (
                        dy_mm / 10.0
                    )
                    ** 2
                )
            )

            z_score = float(
                np.exp(
                    -0.5
                    * (
                        dz_mm / 10.0
                    )
                    ** 2
                )
            )

            # Avoid isolated hot voxels:
            # use spatially integrated local uptake.
            mass_left = max(
                left["local_mass"],
                1e-8,
            )

            mass_right = max(
                right["local_mass"],
                1e-8,
            )

            mass_strength = float(
                np.sqrt(
                    mass_left
                    * mass_right
                )
            )

            mass_balance = float(
                min(
                    mass_left,
                    mass_right,
                )
                /
                max(
                    mass_left,
                    mass_right,
                )
            )

            # Only a weak X-midline prior.
            midpoint_x_offset = abs(
                midpoint[0]
                - mid_x
            )

            x_scale = max(
                shape[0] * 0.15,
                1.0,
            )

            midpoint_score = float(
                np.exp(
                    -0.5
                    * (
                        midpoint_x_offset
                        / x_scale
                    )
                    ** 2
                )
            )

            score = float(
                mass_strength
                * (
                    0.20
                    + 0.80
                    * separation_score
                )
                * (
                    0.20
                    + 0.80
                    * y_score
                )
                * (
                    0.20
                    + 0.80
                    * z_score
                )
                * (
                    0.30
                    + 0.70
                    * mass_balance
                )
                * (
                    0.50
                    + 0.50
                    * midpoint_score
                )
            )

            candidate = {
                "left": left,
                "right": right,
                "midpoint": midpoint,
                "score": score,

                "separation_x_mm": (
                    separation_x_mm
                ),

                "dy_mm": dy_mm,
                "dz_mm": dz_mm,

                "mass_balance": (
                    mass_balance
                ),

                "separation_score": (
                    separation_score
                ),

                "y_score": y_score,
                "z_score": z_score,

                "midpoint_score": (
                    midpoint_score
                ),
            }

            if (
                best is None
                or score > best["score"]
            ):
                best = candidate

    return best


def localize(
    image: np.ndarray,
    spacing: np.ndarray,
    peak_percentile: float = 98.5,
) -> dict:

    work = np.asarray(
        image,
        dtype=np.float32,
    ).copy()

    work[
        ~np.isfinite(work)
    ] = 0.0

    work[
        work < 0
    ] = 0.0

    shape = np.asarray(
        work.shape,
        dtype=np.int64,
    )

    # ---------------------------------------------------------
    # Mild smoothing
    # ---------------------------------------------------------

    smooth = ndimage.gaussian_filter(
        work,
        sigma=1.0,
        mode="nearest",
    )

    # ---------------------------------------------------------
    # Broad search region
    #
    # Still deliberately independent of L1.
    # ---------------------------------------------------------

    search = np.zeros(
        work.shape,
        dtype=bool,
    )

    x0 = int(
        np.floor(
            shape[0] * 0.12
        )
    )
    x1 = int(
        np.ceil(
            shape[0] * 0.88
        )
    )

    y0 = int(
        np.floor(
            shape[1] * 0.08
        )
    )
    y1 = int(
        np.ceil(
            shape[1] * 0.92
        )
    )

    z0 = int(
        np.floor(
            shape[2] * 0.03
        )
    )
    z1 = int(
        np.ceil(
            shape[2] * 0.97
        )
    )

    search[
        x0:x1,
        y0:y1,
        z0:z1,
    ] = True

    peaks = find_peaks(
        smooth=smooth,
        search_mask=search,
        percentile=peak_percentile,
        max_peaks=80,
    )

    pair = choose_bilateral_pair(
        peaks=peaks,
        shape=shape,
        spacing=spacing,
    )

    if pair is None:
        return {
            "valid": False,
            "failure_reason": (
                "no_plausible_bilateral_peak_pair"
            ),
            "peak_count": len(peaks),
        }

    center = np.asarray(
        pair["midpoint"],
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # Confidence
    #
    # Keep this interpretable rather than claiming that every
    # result is high-confidence.
    # ---------------------------------------------------------

    confidence = float(
        0.25 * pair["separation_score"]
        + 0.25 * pair["y_score"]
        + 0.20 * pair["z_score"]
        + 0.20 * pair["mass_balance"]
        + 0.10 * pair["midpoint_score"]
    )

    valid = True
    reasons = []

    if pair["mass_balance"] < 0.12:
        valid = False
        reasons.append(
            "extreme_left_right_mass_imbalance"
        )

    if pair["y_score"] < 0.10:
        valid = False
        reasons.append(
            "left_right_y_mismatch"
        )

    if pair["z_score"] < 0.10:
        valid = False
        reasons.append(
            "left_right_z_mismatch"
        )

    return {
        "valid": valid,

        "failure_reason": (
            ";".join(reasons)
        ),

        "center": center,

        "peak_count": len(peaks),

        "left_peak": (
            pair["left"]["point"]
        ),

        "right_peak": (
            pair["right"]["point"]
        ),

        "left_local_mass": float(
            pair["left"]["local_mass"]
        ),

        "right_local_mass": float(
            pair["right"]["local_mass"]
        ),

        "pair_score": float(
            pair["score"]
        ),

        "separation_x_mm": float(
            pair["separation_x_mm"]
        ),

        "left_right_dy_mm": float(
            pair["dy_mm"]
        ),

        "left_right_dz_mm": float(
            pair["dz_mm"]
        ),

        "mass_balance": float(
            pair["mass_balance"]
        ),

        "confidence": confidence,
    }


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step6e_intensity_normalization_data/step6e_normalize_occipital"
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7d_l2_experimental_localizers_data/"
            "step7d4_roi_l2_v2_bilateral_peak_local/l2_v2_localization.csv"
        ),
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7d_l2_experimental_localizers_data/"
            "step7d4_roi_l2_v2_bilateral_peak_local/l2_v2_summary.json"
        ),
    )

    parser.add_argument(
        "--peak-percentile",
        type=float,
        default=98.5,
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    output_csv = resolve_path(args.output_csv)
    summary_json = resolve_path(args.summary_json)

    files = find_nifti_files(
        input_dir
    )

    if (
        args.expected_count is not None
        and len(files) != args.expected_count
    ):
        raise RuntimeError(
            f"Expected {args.expected_count}, "
            f"found {len(files)}."
        )

    print()
    print("=" * 80)
    print(
        "STEP 7F — L2-v2 BILATERAL PEAK LOCALIZER"
    )
    print("=" * 80)

    print(
        f"Scans:             {len(files)}"
    )

    print(
        f"Peak percentile:   "
        f"{args.peak_percentile}"
    )

    print()

    rows = []
    execution_failures = []

    for index, path in enumerate(
        files,
        start=1,
    ):

        uid = uid_from_path(path)

        try:

            img = nib.load(
                str(path)
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
                    f"Expected RAS, got "
                    f"{orientation}."
                )

            image = np.asarray(
                img.dataobj,
                dtype=np.float32,
            )

            spacing = np.asarray(
                img.header.get_zooms()[:3],
                dtype=np.float64,
            )

            result = localize(
                image=image,
                spacing=spacing,
                peak_percentile=(
                    args.peak_percentile
                ),
            )

            center = np.asarray(
                result.get(
                    "center",
                    [np.nan, np.nan, np.nan],
                ),
                dtype=np.float64,
            )

            if np.all(
                np.isfinite(center)
            ):

                physical = (
                    nib.affines.apply_affine(
                        img.affine,
                        center,
                    )
                )

            else:

                physical = np.asarray(
                    [np.nan, np.nan, np.nan]
                )

            left_peak = np.asarray(
                result.get(
                    "left_peak",
                    [np.nan, np.nan, np.nan],
                ),
                dtype=np.float64,
            )

            right_peak = np.asarray(
                result.get(
                    "right_peak",
                    [np.nan, np.nan, np.nan],
                ),
                dtype=np.float64,
            )

            rows.append(
                {
                    "uid": uid,

                    "l2v2_center_x": float(
                        center[0]
                    ),
                    "l2v2_center_y": float(
                        center[1]
                    ),
                    "l2v2_center_z": float(
                        center[2]
                    ),

                    "l2v2_physical_x_mm": float(
                        physical[0]
                    ),
                    "l2v2_physical_y_mm": float(
                        physical[1]
                    ),
                    "l2v2_physical_z_mm": float(
                        physical[2]
                    ),

                    "left_peak_x": float(
                        left_peak[0]
                    ),
                    "left_peak_y": float(
                        left_peak[1]
                    ),
                    "left_peak_z": float(
                        left_peak[2]
                    ),

                    "right_peak_x": float(
                        right_peak[0]
                    ),
                    "right_peak_y": float(
                        right_peak[1]
                    ),
                    "right_peak_z": float(
                        right_peak[2]
                    ),

                    "l2v2_valid": bool(
                        result["valid"]
                    ),

                    "l2v2_confidence": float(
                        result.get(
                            "confidence",
                            0.0,
                        )
                    ),

                    "failure_reason": (
                        result.get(
                            "failure_reason",
                            "",
                        )
                    ),

                    "peak_count": int(
                        result.get(
                            "peak_count",
                            0,
                        )
                    ),

                    "pair_score": float(
                        result.get(
                            "pair_score",
                            0.0,
                        )
                    ),

                    "separation_x_mm": float(
                        result.get(
                            "separation_x_mm",
                            np.nan,
                        )
                    ),

                    "left_right_dy_mm": float(
                        result.get(
                            "left_right_dy_mm",
                            np.nan,
                        )
                    ),

                    "left_right_dz_mm": float(
                        result.get(
                            "left_right_dz_mm",
                            np.nan,
                        )
                    ),

                    "mass_balance": float(
                        result.get(
                            "mass_balance",
                            0.0,
                        )
                    ),

                    "left_local_mass": float(
                        result.get(
                            "left_local_mass",
                            0.0,
                        )
                    ),

                    "right_local_mass": float(
                        result.get(
                            "right_local_mass",
                            0.0,
                        )
                    ),

                    "fallback_used": False,
                }
            )

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
        print(
            "EXECUTION FAILURES:",
            len(execution_failures),
        )

        for f in execution_failures[:20]:
            print(
                f["uid"],
                f["error"],
            )

        raise RuntimeError(
            "L2-v2 execution failures."
        )

    # ---------------------------------------------------------
    # Save
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

    valid = [
        row
        for row in rows
        if row["l2v2_valid"]
    ]

    invalid = [
        row
        for row in rows
        if not row["l2v2_valid"]
    ]

    summary = {
        "method": (
            "L2-v2 bilateral local-peak pair"
        ),

        "uses_l1": False,
        "fallback_applied": False,

        "processed": len(rows),

        "valid": len(valid),
        "invalid": len(invalid),

        "peak_percentile": (
            args.peak_percentile
        ),

        "confidence": describe(
            [
                row["l2v2_confidence"]
                for row in valid
            ]
        ),

        "separation_x_mm": describe(
            [
                row["separation_x_mm"]
                for row in valid
            ]
        ),

        "left_right_dy_mm": describe(
            [
                row["left_right_dy_mm"]
                for row in valid
            ]
        ),

        "left_right_dz_mm": describe(
            [
                row["left_right_dz_mm"]
                for row in valid
            ]
        ),

        "mass_balance": describe(
            [
                row["mass_balance"]
                for row in valid
            ]
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
    print("L2-v2 RESULTS")
    print("=" * 80)

    print(
        f"Processed:          {len(rows)}"
    )

    print(
        f"Valid:              {len(valid)}"
    )

    print(
        f"Invalid:            {len(invalid)}"
    )

    if valid:

        conf = summary["confidence"]

        print()
        print("Confidence:")

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
    print("CSV:")
    print(
        output_csv.resolve()
    )

    print()
    print("Summary:")
    print(
        summary_json.resolve()
    )

    print()
    print(
        "STEP 7F L2-v2: COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
