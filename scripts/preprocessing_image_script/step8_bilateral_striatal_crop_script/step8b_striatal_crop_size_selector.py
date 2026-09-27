#!/usr/bin/env python3

"""
Step 8B — Select and freeze the fixed bilateral striatal crop size.

Inputs
------
1. Step-8A per-subject extent statistics.
2. Bilateral striatal template mask.

Purpose
-------
Evaluate predefined fixed crop candidates around the frozen Step-7 center.

For each candidate:
    - simulate the exact fixed crop location
    - measure transformed-template ROI coverage
    - measure remaining safety margin
    - estimate boundary-padding requirements
    - calculate crop volume
    - determine whether the candidate passes

Selection policy
----------------
Choose the SMALLEST candidate that satisfies:

    1. Required eligible ROI coverage
    2. Minimum safety margin on every side

Subjects excluded from Step-8A extent estimation
(similarity-rescue registrations and scale/anisotropy outliers)
are reported but DO NOT determine crop size.

This step DOES NOT generate cropped NIfTI files.
That happens in Step 8C.
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
# Argument parsing
# -------------------------------------------------------------------------


def parse_candidate(text: str) -> tuple[int, int, int]:
    """
    Parse:
        48,48,40
        48x48x40
        48X48X40
    """

    cleaned = (
        text.lower()
        .replace("x", ",")
        .replace(" ", "")
    )

    parts = cleaned.split(",")

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid candidate '{text}'. "
            "Expected X,Y,Z, for example 48,48,40."
        )

    try:
        values = tuple(int(v) for v in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid candidate '{text}'."
        ) from exc

    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(
            "Crop dimensions must be positive integers."
        )

    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 8B: select fixed bilateral striatal crop size."
        )
    )

    parser.add_argument(
        "--extent-statistics",
        type=Path,
        required=True,
        help=(
            "Step-8A striatal_extent_statistics.csv"
        ),
    )

    parser.add_argument(
        "--template-striatal-mask",
        type=Path,
        required=True,
        help=(
            "Frozen bilateral striatal mask in DaT template space."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        help=(
            "Candidate crop size X,Y,Z. "
            "May be provided multiple times."
        ),
    )

    parser.add_argument(
        "--required-coverage",
        type=float,
        default=1.0,
        help=(
            "Required fraction of eligible subjects with "
            "complete transformed-template ROI coverage."
        ),
    )

    parser.add_argument(
        "--minimum-margin-per-side-vox",
        type=float,
        default=4.0,
        help=(
            "Minimum required safety margin between the "
            "transformed ROI and crop boundary on EVERY side."
        ),
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


REQUIRED_COLUMNS = [
    "uid",
    "successful",
    "extent_valid_for_size_selection",

    "center_x_vox",
    "center_y_vox",
    "center_z_vox",

    "shape_x",
    "shape_y",
    "shape_z",

    "roi_min_x_vox",
    "roi_min_y_vox",
    "roi_min_z_vox",

    "roi_max_x_vox",
    "roi_max_y_vox",
    "roi_max_z_vox",
]


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
                "y",
                "pass",
                "passed",
            ]
        )
    )


def validate_columns(df: pd.DataFrame) -> None:
    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Step-8A statistics file is missing columns:\n  "
            + "\n  ".join(missing)
        )


def calculate_crop_start(
    center: float,
    size: int,
) -> int:
    """
    Freeze the crop-index convention used later in Step 8C.

    The crop contains:
        [start, start + size)

    We center it as closely as possible around the floating-point
    localization center.

    Example:
        center = 50.2
        size   = 48
    """

    return int(
        math.floor(
            float(center)
            - (float(size) - 1.0) / 2.0
        )
    )


def evaluate_axis(
    center: float,
    roi_min: float,
    roi_max: float,
    image_size: int,
    crop_size: int,
) -> dict[str, Any]:
    """
    Evaluate one spatial axis.

    The transformed ROI coordinates from Step 8A represent physical
    voxel-space support.

    Crop [start, end) contains voxel centers:
        start ... end - 1

    Its approximate spatial support is therefore:
        start - 0.5  ...  end - 0.5
    """

    start = calculate_crop_start(
        center=center,
        size=crop_size,
    )

    end = start + crop_size

    crop_support_min = start - 0.5
    crop_support_max = end - 0.5

    low_margin = float(
        roi_min - crop_support_min
    )

    high_margin = float(
        crop_support_max - roi_max
    )

    covered = bool(
        low_margin >= 0.0
        and high_margin >= 0.0
    )

    pad_before = max(
        0,
        -start,
    )

    pad_after = max(
        0,
        end - image_size,
    )

    return {
        "start": int(start),
        "end": int(end),

        "crop_support_min": float(
            crop_support_min
        ),
        "crop_support_max": float(
            crop_support_max
        ),

        "low_margin": low_margin,
        "high_margin": high_margin,

        "covered": covered,

        "pad_before": int(pad_before),
        "pad_after": int(pad_after),
    }


def percentile_summary(
    values: pd.Series,
) -> dict[str, float]:
    clean = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
    )

    if clean.empty:
        return {}

    return {
        "min": float(clean.min()),
        "p01": float(clean.quantile(0.01)),
        "p05": float(clean.quantile(0.05)),
        "median": float(clean.median()),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }


# -------------------------------------------------------------------------
# Candidate evaluation
# -------------------------------------------------------------------------


def evaluate_candidate(
    eligible_df: pd.DataFrame,
    candidate: tuple[int, int, int],
    minimum_margin: float,
) -> tuple[dict[str, Any], pd.DataFrame]:

    crop_x, crop_y, crop_z = candidate

    records: list[dict[str, Any]] = []

    for _, row in eligible_df.iterrows():

        x = evaluate_axis(
            center=float(
                row["center_x_vox"]
            ),
            roi_min=float(
                row["roi_min_x_vox"]
            ),
            roi_max=float(
                row["roi_max_x_vox"]
            ),
            image_size=int(
                row["shape_x"]
            ),
            crop_size=crop_x,
        )

        y = evaluate_axis(
            center=float(
                row["center_y_vox"]
            ),
            roi_min=float(
                row["roi_min_y_vox"]
            ),
            roi_max=float(
                row["roi_max_y_vox"]
            ),
            image_size=int(
                row["shape_y"]
            ),
            crop_size=crop_y,
        )

        z = evaluate_axis(
            center=float(
                row["center_z_vox"]
            ),
            roi_min=float(
                row["roi_min_z_vox"]
            ),
            roi_max=float(
                row["roi_max_z_vox"]
            ),
            image_size=int(
                row["shape_z"]
            ),
            crop_size=crop_z,
        )

        full_coverage = bool(
            x["covered"]
            and y["covered"]
            and z["covered"]
        )

        minimum_subject_margin = float(
            min(
                x["low_margin"],
                x["high_margin"],

                y["low_margin"],
                y["high_margin"],

                z["low_margin"],
                z["high_margin"],
            )
        )

        margin_pass = bool(
            minimum_subject_margin
            >= minimum_margin
        )

        padding_required = bool(
            x["pad_before"] > 0
            or x["pad_after"] > 0
            or y["pad_before"] > 0
            or y["pad_after"] > 0
            or z["pad_before"] > 0
            or z["pad_after"] > 0
        )

        total_padding_vox = int(
            x["pad_before"]
            + x["pad_after"]
            + y["pad_before"]
            + y["pad_after"]
            + z["pad_before"]
            + z["pad_after"]
        )

        records.append(
            {
                "uid": str(
                    row["uid"]
                ),

                "candidate_x_vox": crop_x,
                "candidate_y_vox": crop_y,
                "candidate_z_vox": crop_z,

                "x_low_margin_vox": (
                    x["low_margin"]
                ),
                "x_high_margin_vox": (
                    x["high_margin"]
                ),

                "y_low_margin_vox": (
                    y["low_margin"]
                ),
                "y_high_margin_vox": (
                    y["high_margin"]
                ),

                "z_low_margin_vox": (
                    z["low_margin"]
                ),
                "z_high_margin_vox": (
                    z["high_margin"]
                ),

                "minimum_margin_vox": (
                    minimum_subject_margin
                ),

                "x_covered": x["covered"],
                "y_covered": y["covered"],
                "z_covered": z["covered"],

                "full_roi_covered": (
                    full_coverage
                ),

                "minimum_margin_pass": (
                    margin_pass
                ),

                "padding_required": (
                    padding_required
                ),

                "total_padding_vox": (
                    total_padding_vox
                ),
            }
        )

    details_df = pd.DataFrame(
        records
    )

    total = len(details_df)

    covered_count = int(
        details_df[
            "full_roi_covered"
        ].sum()
    )

    margin_pass_count = int(
        details_df[
            "minimum_margin_pass"
        ].sum()
    )

    padding_count = int(
        details_df[
            "padding_required"
        ].sum()
    )

    minimum_margin_global = float(
        details_df[
            "minimum_margin_vox"
        ].min()
    )

    crop_volume = int(
        crop_x
        * crop_y
        * crop_z
    )

    summary = {
        "candidate": (
            f"{crop_x}x{crop_y}x{crop_z}"
        ),

        "candidate_x_vox": crop_x,
        "candidate_y_vox": crop_y,
        "candidate_z_vox": crop_z,

        "candidate_x_mm": (
            crop_x * 2.46
        ),
        "candidate_y_mm": (
            crop_y * 2.46
        ),
        "candidate_z_mm": (
            crop_z * 2.46
        ),

        "crop_volume_vox": crop_volume,

        "eligible_subjects": total,

        "full_coverage_count": (
            covered_count
        ),

        "full_coverage_fraction": (
            covered_count / total
            if total
            else 0.0
        ),

        "margin_pass_count": (
            margin_pass_count
        ),

        "margin_pass_fraction": (
            margin_pass_count / total
            if total
            else 0.0
        ),

        "minimum_margin_global_vox": (
            minimum_margin_global
        ),

        "margin_distribution_vox": (
            percentile_summary(
                details_df[
                    "minimum_margin_vox"
                ]
            )
        ),

        "padding_subjects": (
            padding_count
        ),

        "padding_fraction": (
            padding_count / total
            if total
            else 0.0
        ),

        "max_total_padding_vox": int(
            details_df[
                "total_padding_vox"
            ].max()
        ),
    }

    return summary, details_df


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    candidates = (
        args.candidate
        if args.candidate
        else [
            (40, 40, 32),
            (44, 44, 36),
            (48, 48, 40),
        ]
    )

    if not (
        0.0
        < args.required_coverage
        <= 1.0
    ):
        raise ValueError(
            "--required-coverage must be "
            "between 0 and 1."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------

    if not args.extent_statistics.exists():
        raise FileNotFoundError(
            f"Step-8A statistics not found: "
            f"{args.extent_statistics}"
        )

    if not args.template_striatal_mask.exists():
        raise FileNotFoundError(
            f"Template striatal mask not found: "
            f"{args.template_striatal_mask}"
        )

    df = pd.read_csv(
        args.extent_statistics
    )

    validate_columns(df)

    successful_mask = bool_series(
        df["successful"]
    )

    eligible_mask = bool_series(
        df[
            "extent_valid_for_size_selection"
        ]
    )

    successful_df = df[
        successful_mask
    ].copy()

    eligible_df = df[
        successful_mask
        & eligible_mask
    ].copy()

    excluded_df = df[
        successful_mask
        & ~eligible_mask
    ].copy()

    if eligible_df.empty:
        raise RuntimeError(
            "No subjects are eligible for "
            "crop-size selection."
        )

    # ------------------------------------------------------------------
    # Template information
    # ------------------------------------------------------------------

    template_img = nib.load(
        str(
            args.template_striatal_mask
        )
    )

    template_data = np.asarray(
        template_img.dataobj
    )

    template_mask = (
        template_data > 0
    )

    template_voxels = int(
        template_mask.sum()
    )

    if template_voxels == 0:
        raise ValueError(
            "Template striatal mask is empty."
        )

    template_spacing = tuple(
        float(v)
        for v in template_img.header.get_zooms()[:3]
    )

    template_shape = tuple(
        int(v)
        for v in template_img.shape[:3]
    )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    print("=" * 72)
    print(
        "STEP 8B — STRIATAL CROP SIZE SELECTION"
    )
    print("=" * 72)

    print(
        f"Step-8A rows:                 {len(df)}"
    )

    print(
        f"Successful Step-8A rows:      "
        f"{len(successful_df)}"
    )

    print(
        f"Eligible for size selection:  "
        f"{len(eligible_df)}"
    )

    print(
        f"Excluded from size selection: "
        f"{len(excluded_df)}"
    )

    if not excluded_df.empty:
        print(
            "Excluded UIDs:                  "
            + ", ".join(
                excluded_df[
                    "uid"
                ].astype(str).tolist()
            )
        )

    print()
    print(
        f"Template mask:                "
        f"{args.template_striatal_mask}"
    )

    print(
        f"Template ROI voxels:          "
        f"{template_voxels}"
    )

    print(
        f"Template shape:               "
        f"{template_shape}"
    )

    print(
        f"Template spacing:             "
        f"{template_spacing}"
    )

    print()
    print(
        f"Required coverage:            "
        f"{args.required_coverage:.4f}"
    )

    print(
        f"Minimum safety margin/side:   "
        f"{args.minimum_margin_per_side_vox:.2f} vox"
    )

    print()

    # ------------------------------------------------------------------
    # Evaluate candidates
    # ------------------------------------------------------------------

    candidate_summaries: list[
        dict[str, Any]
    ] = []

    candidate_detail_frames = []

    for candidate in candidates:

        summary, details = evaluate_candidate(
            eligible_df=eligible_df,
            candidate=candidate,
            minimum_margin=(
                args.minimum_margin_per_side_vox
            ),
        )

        coverage_pass = bool(
            summary[
                "full_coverage_fraction"
            ]
            >= args.required_coverage
        )

        global_margin_pass = bool(
            summary[
                "minimum_margin_global_vox"
            ]
            >= args.minimum_margin_per_side_vox
        )

        summary[
            "coverage_pass"
        ] = coverage_pass

        summary[
            "global_margin_pass"
        ] = global_margin_pass

        summary[
            "candidate_pass"
        ] = bool(
            coverage_pass
            and global_margin_pass
        )

        candidate_summaries.append(
            summary
        )

        details[
            "candidate"
        ] = summary["candidate"]

        candidate_detail_frames.append(
            details
        )

    candidate_df = pd.DataFrame(
        candidate_summaries
    )

    candidate_df = candidate_df.sort_values(
        by=[
            "crop_volume_vox",
            "candidate_x_vox",
            "candidate_y_vox",
            "candidate_z_vox",
        ]
    ).reset_index(
        drop=True
    )

    passing_df = candidate_df[
        candidate_df[
            "candidate_pass"
        ]
        == True  # noqa: E712
    ].copy()

    # ------------------------------------------------------------------
    # Select smallest passing candidate
    # ------------------------------------------------------------------

    if passing_df.empty:
        selected = None

    else:
        selected = (
            passing_df.iloc[0]
            .to_dict()
        )

    candidate_df[
        "selected"
    ] = False

    if selected is not None:
        selected_name = selected[
            "candidate"
        ]

        candidate_df.loc[
            candidate_df[
                "candidate"
            ]
            == selected_name,
            "selected",
        ] = True

    # ------------------------------------------------------------------
    # Save candidate evaluation
    # ------------------------------------------------------------------

    candidate_path = (
        args.output_dir
        / "crop_size_candidate_evaluation.csv"
    )

    candidate_df.to_csv(
        candidate_path,
        index=False,
    )

    detail_df = pd.concat(
        candidate_detail_frames,
        ignore_index=True,
    )

    detail_path = (
        args.output_dir
        / "crop_size_subject_evaluation.csv"
    )

    detail_df.to_csv(
        detail_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Selected configuration
    # ------------------------------------------------------------------

    selected_config_path = (
        args.output_dir
        / "selected_crop_size.json"
    )

    if selected is None:

        selected_config = {
            "step": "8B",
            "selection_successful": False,
            "reason": (
                "No candidate satisfied both "
                "ROI coverage and minimum-margin criteria."
            ),
        }

    else:

        selected_config = {
            "step": "8B",
            "selection_successful": True,

            "crop_size_vox": {
                "x": int(
                    selected[
                        "candidate_x_vox"
                    ]
                ),
                "y": int(
                    selected[
                        "candidate_y_vox"
                    ]
                ),
                "z": int(
                    selected[
                        "candidate_z_vox"
                    ]
                ),
            },

            "crop_size_mm": {
                "x": float(
                    selected[
                        "candidate_x_mm"
                    ]
                ),
                "y": float(
                    selected[
                        "candidate_y_mm"
                    ]
                ),
                "z": float(
                    selected[
                        "candidate_z_mm"
                    ]
                ),
            },

            "crop_volume_vox": int(
                selected[
                    "crop_volume_vox"
                ]
            ),

            "eligible_roi_coverage_fraction": float(
                selected[
                    "full_coverage_fraction"
                ]
            ),

            "minimum_observed_margin_vox": float(
                selected[
                    "minimum_margin_global_vox"
                ]
            ),

            "minimum_required_margin_vox": float(
                args.minimum_margin_per_side_vox
            ),

            "padding_subjects": int(
                selected[
                    "padding_subjects"
                ]
            ),

            "localization_policy": (
                "Frozen Step-7 center: "
                "L1 primary, L0 fallback"
            ),

            "crop_index_policy": (
                "start = floor(center - "
                "(crop_size - 1) / 2)"
            ),

            "padding_policy": (
                "Zero padding when crop crosses "
                "subject image boundary"
            ),

            "intensity_policy": (
                "Preserve Step-6D normalized values; "
                "no additional normalization"
            ),

            "template_striatal_mask": str(
                args.template_striatal_mask
            ),

            "template_roi_voxels": (
                template_voxels
            ),

            "extent_source": str(
                args.extent_statistics
            ),
        }

    with open(
        selected_config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            selected_config,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Dataset summary
    # ------------------------------------------------------------------

    summary_path = (
        args.output_dir
        / "crop_size_selection_summary.json"
    )

    summary = {
        "step": "8B",

        "description": (
            "Select fixed bilateral striatal crop size"
        ),

        "number_total": int(
            len(df)
        ),

        "number_successful_step8a": int(
            len(successful_df)
        ),

        "number_eligible": int(
            len(eligible_df)
        ),

        "number_excluded": int(
            len(excluded_df)
        ),

        "excluded_uids": (
            excluded_df[
                "uid"
            ]
            .astype(str)
            .tolist()
        ),

        "template": {
            "mask_path": str(
                args.template_striatal_mask
            ),
            "roi_voxels": (
                template_voxels
            ),
            "shape": list(
                template_shape
            ),
            "spacing": list(
                template_spacing
            ),
        },

        "selection_policy": {
            "required_coverage": float(
                args.required_coverage
            ),
            "minimum_margin_per_side_vox": float(
                args.minimum_margin_per_side_vox
            ),
            "strategy": (
                "Smallest crop volume satisfying "
                "coverage and safety-margin criteria"
            ),
        },

        "candidates": (
            candidate_summaries
        ),

        "selection_successful": (
            selected is not None
        ),

        "selected_candidate": (
            selected[
                "candidate"
            ]
            if selected is not None
            else None
        ),
    }

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
    # Terminal report
    # ------------------------------------------------------------------

    print("=" * 72)
    print("CANDIDATE RESULTS")
    print("=" * 72)

    for _, row in candidate_df.iterrows():

        marker = (
            "  <-- SELECTED"
            if bool(
                row["selected"]
            )
            else ""
        )

        print()
        print(
            f"{row['candidate']}{marker}"
        )

        print(
            f"  ROI coverage:        "
            f"{int(row['full_coverage_count'])}/"
            f"{int(row['eligible_subjects'])} "
            f"({row['full_coverage_fraction']:.6f})"
        )

        print(
            f"  Minimum margin:      "
            f"{row['minimum_margin_global_vox']:.3f} vox"
        )

        print(
            f"  Padding subjects:    "
            f"{int(row['padding_subjects'])}"
        )

        print(
            f"  Coverage pass:       "
            f"{bool(row['coverage_pass'])}"
        )

        print(
            f"  Margin pass:         "
            f"{bool(row['global_margin_pass'])}"
        )

        print(
            f"  Candidate pass:      "
            f"{bool(row['candidate_pass'])}"
        )

    print()
    print("=" * 72)

    if selected is None:

        print(
            "STEP 8B FAILED — NO CANDIDATE PASSED"
        )

        print(
            "Do NOT continue to Step 8C."
        )

        raise RuntimeError(
            "No crop candidate satisfied "
            "the frozen Step-8B selection criteria."
        )

    print(
        "STEP 8B COMPLETE"
    )
    print("=" * 72)

    print(
        f"Selected fixed crop: "
        f"{selected['candidate']} voxels"
    )

    print(
        "Physical dimensions: "
        f"{selected['candidate_x_mm']:.2f} × "
        f"{selected['candidate_y_mm']:.2f} × "
        f"{selected['candidate_z_mm']:.2f} mm"
    )

    print(
        f"ROI coverage: "
        f"{selected['full_coverage_fraction']:.6f}"
    )

    print(
        f"Minimum safety margin: "
        f"{selected['minimum_margin_global_vox']:.3f} vox"
    )

    print(
        f"Subjects requiring padding: "
        f"{int(selected['padding_subjects'])}"
    )

    print()
    print(
        f"Candidate evaluation: {candidate_path}"
    )

    print(
        f"Subject evaluation:   {detail_path}"
    )

    print(
        f"Selected crop config: {selected_config_path}"
    )

    print(
        f"Summary:              {summary_path}"
    )

    print()
    print(
        "The selected dimensions are now ready "
        "to be consumed by Step 8C."
    )


if __name__ == "__main__":
    main()
