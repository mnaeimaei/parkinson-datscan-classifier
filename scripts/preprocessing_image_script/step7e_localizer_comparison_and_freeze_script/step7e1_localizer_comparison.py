from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPACING_MM = 2.46


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def describe(values: pd.Series) -> dict:
    x = values.dropna().to_numpy(dtype=np.float64)

    return {
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
        "std": float(np.std(x)),
    }


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:
    for col in candidates:
        if col in df.columns:
            return col

    raise RuntimeError(
        "None of these columns were found:\n"
        + "\n".join(candidates)
        + "\n\nAvailable columns:\n"
        + "\n".join(df.columns)
    )


def add_distance(
    df: pd.DataFrame,
    prefix: str,
    ax: str,
    ay: str,
    az: str,
    bx: str,
    by: str,
    bz: str,
) -> None:

    dx = df[ax] - df[bx]
    dy = df[ay] - df[by]
    dz = df[az] - df[bz]

    df[f"{prefix}_delta_x_vox"] = dx
    df[f"{prefix}_delta_y_vox"] = dy
    df[f"{prefix}_delta_z_vox"] = dz

    distance_vox = np.sqrt(
        dx**2 + dy**2 + dz**2
    )

    df[f"{prefix}_distance_vox"] = distance_vox

    df[f"{prefix}_distance_mm"] = (
        distance_vox * SPACING_MM
    )


def threshold_counts(
    values: pd.Series,
) -> dict:

    result = {}

    for threshold in [
        5,
        10,
        15,
        20,
        30,
        40,
        50,
    ]:

        count = int(
            (values > threshold).sum()
        )

        result[f"gt_{threshold}_mm"] = {
            "count": count,
            "percent": float(
                count / len(values) * 100
            ),
        }

    return result


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--l0-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7b_l0_central_baseline_data/"
            "step7b_roi_l0_fixed_center/l0_centers.csv"
        ),
    )

    parser.add_argument(
        "--l1-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7c_l1_template_registration_data/"
            "step7c5_l1_center_consistency/l1_center_consistency.csv"
        ),
    )

    parser.add_argument(
        "--l2v1-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7d_l2_experimental_localizers_data/"
            "step7d1_roi_l2_intensity_localizer/l2_localization.csv"
        ),
    )

    parser.add_argument(
        "--l2v2-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7d_l2_experimental_localizers_data/"
            "step7d4_roi_l2_v2_bilateral_peak_local/l2_v2_localization.csv"
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7e_localizer_comparison_and_freeze_data/"
            "step7e1_localizer_comparison/localizer_comparison.csv"
        ),
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7e_localizer_comparison_and_freeze_data/"
            "step7e1_localizer_comparison/localizer_comparison_summary.json"
        ),
    )

    parser.add_argument(
        "--outlier-csv",
        type=Path,
        default=Path(
            "data/preprocessing_image_data/step7e_localizer_comparison_and_freeze_data/"
            "step7e1_localizer_comparison/l2v2_outliers.csv"
        ),
    )

    args = parser.parse_args()

    l0_csv = resolve_path(args.l0_csv)
    l1_csv = resolve_path(args.l1_csv)
    l2v1_csv = resolve_path(args.l2v1_csv)
    l2v2_csv = resolve_path(args.l2v2_csv)
    output_csv = resolve_path(args.output_csv)
    summary_json = resolve_path(args.summary_json)
    outlier_csv = resolve_path(args.outlier_csv)

    # ---------------------------------------------------------
    # Read
    # ---------------------------------------------------------

    l0 = pd.read_csv(l0_csv)
    l1 = pd.read_csv(l1_csv)
    l2v1 = pd.read_csv(l2v1_csv)
    l2v2 = pd.read_csv(l2v2_csv)

    n_subjects = len(l1)

    for name, frame in [
        ("L0", l0),
        ("L1", l1),
        ("L2-v1", l2v1),
        ("L2-v2", l2v2),
    ]:
        if len(frame) != n_subjects:
            raise RuntimeError(
                f"{name}: expected {n_subjects} rows "
                f"(L1 size), found {len(frame)}"
            )

        if frame["uid"].duplicated().any():
            raise RuntimeError(
                f"{name}: duplicate UIDs found"
            )

    # ---------------------------------------------------------
    # Locate L0 columns robustly
    # ---------------------------------------------------------

    l0_x = find_column(
        l0,
        [
            "l0_center_x",
            "center_x",
            "center_x_voxel",
        ],
    )

    l0_y = find_column(
        l0,
        [
            "l0_center_y",
            "center_y",
            "center_y_voxel",
        ],
    )

    l0_z = find_column(
        l0,
        [
            "l0_center_z",
            "center_z",
            "center_z_voxel",
        ],
    )

    l0 = l0[
        [
            "uid",
            l0_x,
            l0_y,
            l0_z,
        ]
    ].rename(
        columns={
            l0_x: "l0_x",
            l0_y: "l0_y",
            l0_z: "l0_z",
        }
    )

    # ---------------------------------------------------------
    # L1 official center:
    # directly transformed template center
    # ---------------------------------------------------------

    l1 = l1[
        [
            "uid",
            "direct_center_x",
            "direct_center_y",
            "direct_center_z",
        ]
    ].rename(
        columns={
            "direct_center_x": "l1_x",
            "direct_center_y": "l1_y",
            "direct_center_z": "l1_z",
        }
    )

    # ---------------------------------------------------------
    # L2-v1
    # ---------------------------------------------------------

    l2v1 = l2v1[
        [
            "uid",
            "l2_center_x",
            "l2_center_y",
            "l2_center_z",
            "l2_valid",
            "l2_confidence",
        ]
    ].rename(
        columns={
            "l2_center_x": "l2v1_x",
            "l2_center_y": "l2v1_y",
            "l2_center_z": "l2v1_z",
            "l2_valid": "l2v1_valid",
            "l2_confidence": "l2v1_confidence",
        }
    )

    # ---------------------------------------------------------
    # L2-v2
    # ---------------------------------------------------------

    keep_v2 = [
        "uid",
        "l2v2_center_x",
        "l2v2_center_y",
        "l2v2_center_z",
        "l2v2_valid",
        "l2v2_confidence",
        "separation_x_mm",
        "left_right_dy_mm",
        "left_right_dz_mm",
        "mass_balance",
    ]

    l2v2 = l2v2[
        keep_v2
    ].rename(
        columns={
            "l2v2_center_x": "l2v2_x",
            "l2v2_center_y": "l2v2_y",
            "l2v2_center_z": "l2v2_z",
        }
    )

    # ---------------------------------------------------------
    # Merge
    # ---------------------------------------------------------

    df = (
        l0
        .merge(
            l1,
            on="uid",
            validate="one_to_one",
        )
        .merge(
            l2v1,
            on="uid",
            validate="one_to_one",
        )
        .merge(
            l2v2,
            on="uid",
            validate="one_to_one",
        )
    )

    if len(df) != n_subjects:
        raise RuntimeError(
            f"Merged rows = {len(df)}, expected {n_subjects}"
        )

    # ---------------------------------------------------------
    # Distances
    # ---------------------------------------------------------

    add_distance(
        df,
        "l0_l1",
        "l0_x",
        "l0_y",
        "l0_z",
        "l1_x",
        "l1_y",
        "l1_z",
    )

    add_distance(
        df,
        "l0_l2v1",
        "l0_x",
        "l0_y",
        "l0_z",
        "l2v1_x",
        "l2v1_y",
        "l2v1_z",
    )

    add_distance(
        df,
        "l0_l2v2",
        "l0_x",
        "l0_y",
        "l0_z",
        "l2v2_x",
        "l2v2_y",
        "l2v2_z",
    )

    add_distance(
        df,
        "l1_l2v1",
        "l1_x",
        "l1_y",
        "l1_z",
        "l2v1_x",
        "l2v1_y",
        "l2v1_z",
    )

    add_distance(
        df,
        "l1_l2v2",
        "l1_x",
        "l1_y",
        "l1_z",
        "l2v2_x",
        "l2v2_y",
        "l2v2_z",
    )

    # ---------------------------------------------------------
    # Did v2 improve over v1?
    # ---------------------------------------------------------

    df["l2v2_improved_over_v1"] = (
        df["l1_l2v2_distance_mm"]
        <
        df["l1_l2v1_distance_mm"]
    )

    df["l2v2_improvement_mm"] = (
        df["l1_l2v1_distance_mm"]
        -
        df["l1_l2v2_distance_mm"]
    )

    # ---------------------------------------------------------
    # Save full comparison
    # ---------------------------------------------------------

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    # ---------------------------------------------------------
    # L2-v2 outliers
    # ---------------------------------------------------------

    outliers = (
        df.sort_values(
            "l1_l2v2_distance_mm",
            ascending=False,
        )
        .head(100)
        .copy()
    )

    outlier_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    outliers.to_csv(
        outlier_csv,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    comparisons = {}

    for prefix in [
        "l0_l1",
        "l0_l2v1",
        "l0_l2v2",
        "l1_l2v1",
        "l1_l2v2",
    ]:

        values = df[
            f"{prefix}_distance_mm"
        ]

        comparisons[prefix] = {
            "distance_mm": describe(
                values
            ),
            "threshold_counts": (
                threshold_counts(values)
            ),
        }

    improved_count = int(
        df[
            "l2v2_improved_over_v1"
        ].sum()
    )

    confidence_corr = float(
        df[
            [
                "l2v2_confidence",
                "l1_l2v2_distance_mm",
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    summary = {
        "step": "7G",

        "number_of_scans": int(
            len(df)
        ),

        "spacing_mm": SPACING_MM,

        "reference_for_comparison": (
            "Validated L1 directly transformed "
            "template center"
        ),

        "comparisons": comparisons,

        "l2v2_vs_l2v1": {
            "v2_improved_count": (
                improved_count
            ),
            "v2_improved_percent": float(
                improved_count
                / len(df)
                * 100
            ),
            "median_improvement_mm": float(
                df[
                    "l2v2_improvement_mm"
                ].median()
            ),
        },

        "l2v2_confidence_vs_error": {
            "pearson_correlation": (
                confidence_corr
            )
        },

        "method_status": {
            "L0": (
                "deterministic baseline/fallback"
            ),
            "L1": (
                "validated primary candidate"
            ),
            "L2_v1": (
                "rejected: diffuse candidate and "
                "poor center accuracy"
            ),
            "L2_v2": (
                "rejected as automatic localizer: "
                "good median accuracy but "
                "catastrophic high-confidence tail"
            ),
        },

        "recommended_policy_for_step7h": {
            "primary": "L1",
            "fallback": "L0",
            "use_L2_in_final_policy": False,
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
    # Console
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "STEP 7G — LOCALIZER COMPARISON"
    )
    print("=" * 80)

    print(
        f"Cases: {len(df)}"
    )

    print()

    for prefix, label in [
        ("l0_l1", "L0 ↔ L1"),
        ("l1_l2v1", "L1 ↔ L2-v1"),
        ("l1_l2v2", "L1 ↔ L2-v2"),
    ]:

        d = comparisons[
            prefix
        ]["distance_mm"]

        print(label)

        print(
            f"  median: {d['median']:.2f} mm"
        )

        print(
            f"  p95:    {d['p95']:.2f} mm"
        )

        print(
            f"  max:    {d['max']:.2f} mm"
        )

        print()

    print(
        "L2-v2 improved over L2-v1:"
    )

    print(
        f"  {improved_count}/{len(df)} "
        f"({improved_count / len(df) * 100:.2f}%)"
    )

    print()

    print(
        "L2-v2 confidence/error correlation:"
    )

    print(
        f"  {confidence_corr:.4f}"
    )

    print()

    print("Recommended Step-7H policy:")

    print(
        "  PRIMARY  = L1"
    )

    print(
        "  FALLBACK = L0"
    )

    print(
        "  L2       = experimental only"
    )

    print()

    print("Comparison CSV:")
    print(
        output_csv.resolve()
    )

    print()

    print("Top-100 L2-v2 outliers:")
    print(
        outlier_csv.resolve()
    )

    print()

    print("Summary:")
    print(
        summary_json.resolve()
    )

    print()

    print(
        "STEP 7G: COMPLETE"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
