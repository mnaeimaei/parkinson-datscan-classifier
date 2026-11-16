#!/usr/bin/env python3

"""
Step 8E — Visual QC of fixed bilateral striatal crops.

This step selects difficult and representative subjects and generates
whole-volume versus cropped-volume visualizations.

Frozen pipeline decisions
-------------------------
Step 7:
    L1 localization center, with L0 fallback.

Step 8A:
    Transformed bilateral striatal-template extent.

Step 8B:
    Fixed global crop dimensions.

Step 8C:
    Deterministic crop generation.

Step 8D:
    Automatic QC passed.

Template
--------
The same frozen bilateral DaT striatal template mask is verified here
for provenance. Step-8A transformed ROI bounds are used for overlays,
so registration is NOT recomputed during visual QC.

Outputs
-------
output_dir/
    selected_cases/
        <uid>_visual_qc.png

    visual_qc_selection.csv
    visual_qc_summary.json

This is a MANUAL visual-QC stage. Generation of the figures does not
automatically mark Step 8E as passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 8E: visual QC of striatal crops."
    )

    parser.add_argument(
        "--extent-statistics",
        type=Path,
        required=True,
        help="Step-8A striatal_extent_statistics.csv",
    )

    parser.add_argument(
        "--selected-crop-config",
        type=Path,
        required=True,
        help="Step-8B selected_crop_size.json",
    )

    parser.add_argument(
        "--crop-manifest",
        type=Path,
        required=True,
        help="Step-8C crop manifest.",
    )

    parser.add_argument(
        "--automatic-qc-manifest",
        type=Path,
        required=True,
        help="Step-8D automatic-QC manifest.",
    )

    parser.add_argument(
        "--normalized-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--crop-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--template-striatal-mask",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--random-normal",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--random-padded",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--lowest-margin",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--highest-padding",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--special-uid",
        action="append",
        default=[],
        help=(
            "Extra UID that must always be included. "
            "Invalid-extent (center-valid) scans are included "
            "automatically."
        ),
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def to_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "pass",
        "passed",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def resolve_image(
    directory: Path,
    uid: str,
) -> Path:

    candidates = [
        directory / f"{uid}.nii.gz",
        directory / f"{uid}.nii",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"NIfTI not found for UID={uid} in {directory}"
    )


def intensity_limits(data: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(
        data[np.isfinite(data)],
        dtype=np.float64,
    )

    nonzero = finite[
        finite != 0
    ]

    values = (
        nonzero
        if len(nonzero) >= 100
        else finite
    )

    if len(values) == 0:
        return 0.0, 1.0

    vmin = float(
        np.percentile(
            values,
            1.0,
        )
    )

    vmax = float(
        np.percentile(
            values,
            99.5,
        )
    )

    if vmax <= vmin:
        vmax = vmin + 1.0

    return vmin, vmax


def clamp_slice(
    value: float,
    size: int,
) -> int:
    return int(
        np.clip(
            int(round(value)),
            0,
            size - 1,
        )
    )


def add_rectangle(
    ax,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    *,
    edgecolor: str,
    linewidth: float,
    linestyle: str = "-",
) -> None:

    rectangle = Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
    )

    ax.add_patch(
        rectangle
    )


def show_crosshair(
    ax,
    x: float,
    y: float,
) -> None:

    ax.axvline(
        x,
        linewidth=0.8,
        alpha=0.8,
    )

    ax.axhline(
        y,
        linewidth=0.8,
        alpha=0.8,
    )


# -------------------------------------------------------------------------
# Case selection
# -------------------------------------------------------------------------


def add_reason(
    reasons: dict[str, set[str]],
    uid: str,
    reason: str,
) -> None:

    reasons[str(uid)].add(
        reason
    )


def select_cases(
    merged: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:

    reasons: dict[
        str,
        set[str],
    ] = defaultdict(set)

    # --------------------------------------------------------------
    # Explicit special cases
    # --------------------------------------------------------------

    for uid in args.special_uid:

        if uid in set(
            merged["uid"].astype(str)
        ):
            add_reason(
                reasons,
                uid,
                "special_case",
            )

    # --------------------------------------------------------------
    # Invalid extent, valid center (e.g. similarity-rescue).
    # --------------------------------------------------------------

    if "extent_valid_for_size_selection" in merged.columns:

        invalid_extent = merged[
            ~merged[
                "extent_valid_for_size_selection"
            ].map(to_bool)
        ]

        for uid in invalid_extent[
            "uid"
        ].astype(str):

            add_reason(
                reasons,
                uid,
                "extent_invalid_but_center_valid",
            )

    # --------------------------------------------------------------
    # Lowest valid template-ROI margins
    # --------------------------------------------------------------

    valid_margin = merged[
        merged[
            "extent_valid_for_size_selection"
        ].map(to_bool)
    ].copy()

    if (
        "minimum_roi_margin_vox"
        in valid_margin.columns
    ):

        smallest_margin = (
            valid_margin
            .sort_values(
                "minimum_roi_margin_vox",
                ascending=True,
            )
            .head(
                args.lowest_margin
            )
        )

        for uid in smallest_margin[
            "uid"
        ].astype(str):

            add_reason(
                reasons,
                uid,
                "lowest_roi_margin",
            )

    # --------------------------------------------------------------
    # Highest padding
    # --------------------------------------------------------------

    if (
        "total_padding_vox"
        in merged.columns
    ):

        padded = merged[
            merged[
                "total_padding_vox"
            ].fillna(0)
            > 0
        ].copy()

        highest_padding = (
            padded
            .sort_values(
                "total_padding_vox",
                ascending=False,
            )
            .head(
                args.highest_padding
            )
        )

        for uid in highest_padding[
            "uid"
        ].astype(str):

            add_reason(
                reasons,
                uid,
                "highest_padding",
            )

    # --------------------------------------------------------------
    # Largest extent requirement on each axis
    # --------------------------------------------------------------

    for axis in [
        "x",
        "y",
        "z",
    ]:

        column = (
            f"required_crop_{axis}_vox"
        )

        if column not in valid_margin.columns:
            continue

        row = (
            valid_margin
            .sort_values(
                column,
                ascending=False,
            )
            .iloc[0]
        )

        add_reason(
            reasons,
            str(row["uid"]),
            f"largest_{axis}_extent",
        )

    # --------------------------------------------------------------
    # Random padded subjects
    # --------------------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    padded_pool = merged[
        merged[
            "total_padding_vox"
        ].fillna(0)
        > 0
    ].copy()

    already_selected = set(
        reasons.keys()
    )

    padded_pool = padded_pool[
        ~padded_pool[
            "uid"
        ].astype(str).isin(
            already_selected
        )
    ]

    if (
        len(padded_pool) > 0
        and args.random_padded > 0
    ):

        number = min(
            args.random_padded,
            len(padded_pool),
        )

        indices = rng.choice(
            padded_pool.index.to_numpy(),
            size=number,
            replace=False,
        )

        for uid in padded_pool.loc[
            indices,
            "uid",
        ].astype(str):

            add_reason(
                reasons,
                uid,
                "random_padded",
            )

    # --------------------------------------------------------------
    # Random ordinary, non-padded cases
    # --------------------------------------------------------------

    already_selected = set(
        reasons.keys()
    )

    normal_pool = merged[
        merged[
            "total_padding_vox"
        ].fillna(0)
        == 0
    ].copy()

    normal_pool = normal_pool[
        ~normal_pool[
            "uid"
        ].astype(str).isin(
            already_selected
        )
    ]

    if (
        len(normal_pool) > 0
        and args.random_normal > 0
    ):

        number = min(
            args.random_normal,
            len(normal_pool),
        )

        indices = rng.choice(
            normal_pool.index.to_numpy(),
            size=number,
            replace=False,
        )

        for uid in normal_pool.loc[
            indices,
            "uid",
        ].astype(str):

            add_reason(
                reasons,
                uid,
                "random_normal",
            )

    # --------------------------------------------------------------
    # Produce final selection
    # --------------------------------------------------------------

    selected_uids = list(
        reasons.keys()
    )

    selected = merged[
        merged[
            "uid"
        ].astype(str).isin(
            selected_uids
        )
    ].copy()

    selected[
        "selection_reason"
    ] = selected[
        "uid"
    ].astype(str).map(
        lambda uid: ";".join(
            sorted(
                reasons[uid]
            )
        )
    )

    selected = selected.sort_values(
        [
            "selection_reason",
            "uid",
        ]
    ).reset_index(
        drop=True
    )

    return selected


# -------------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------------


def create_visual_qc(
    row: pd.Series,
    normalized_dir: Path,
    crop_dir: Path,
    crop_shape: tuple[int, int, int],
    output_path: Path,
) -> None:

    uid = str(
        row["uid"]
    )

    whole_path = resolve_image(
        normalized_dir,
        uid,
    )

    crop_path = resolve_image(
        crop_dir,
        uid,
    )

    whole_img = nib.load(
        str(
            whole_path
        )
    )

    crop_img = nib.load(
        str(
            crop_path
        )
    )

    whole = np.asarray(
        whole_img.dataobj
    )

    crop = np.asarray(
        crop_img.dataobj
    )

    center = np.asarray(
        [
            float(
                row[
                    "center_x_vox"
                ]
            ),
            float(
                row[
                    "center_y_vox"
                ]
            ),
            float(
                row[
                    "center_z_vox"
                ]
            ),
        ]
    )

    starts = np.asarray(
        [
            int(
                row[
                    "crop_start_x"
                ]
            ),
            int(
                row[
                    "crop_start_y"
                ]
            ),
            int(
                row[
                    "crop_start_z"
                ]
            ),
        ]
    )

    ends = (
        starts
        + np.asarray(
            crop_shape
        )
    )

    roi_min = np.asarray(
        [
            float(
                row[
                    "roi_min_x_vox"
                ]
            ),
            float(
                row[
                    "roi_min_y_vox"
                ]
            ),
            float(
                row[
                    "roi_min_z_vox"
                ]
            ),
        ]
    )

    roi_max = np.asarray(
        [
            float(
                row[
                    "roi_max_x_vox"
                ]
            ),
            float(
                row[
                    "roi_max_y_vox"
                ]
            ),
            float(
                row[
                    "roi_max_z_vox"
                ]
            ),
        ]
    )

    x = clamp_slice(
        center[0],
        whole.shape[0],
    )

    y = clamp_slice(
        center[1],
        whole.shape[1],
    )

    z = clamp_slice(
        center[2],
        whole.shape[2],
    )

    local_center = (
        center - starts
    )

    cx = clamp_slice(
        local_center[0],
        crop.shape[0],
    )

    cy = clamp_slice(
        local_center[1],
        crop.shape[1],
    )

    cz = clamp_slice(
        local_center[2],
        crop.shape[2],
    )

    whole_vmin, whole_vmax = (
        intensity_limits(
            whole
        )
    )

    crop_vmin, crop_vmax = (
        intensity_limits(
            crop
        )
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    # ==============================================================
    # WHOLE — AXIAL
    # ==============================================================

    ax = axes[0, 0]

    ax.imshow(
        whole[:, :, z].T,
        origin="lower",
        cmap="gray",
        vmin=whole_vmin,
        vmax=whole_vmax,
    )

    add_rectangle(
        ax,
        starts[0] - 0.5,
        starts[1] - 0.5,
        ends[0] - 0.5,
        ends[1] - 0.5,
        edgecolor="cyan",
        linewidth=1.5,
    )

    add_rectangle(
        ax,
        roi_min[0],
        roi_min[1],
        roi_max[0],
        roi_max[1],
        edgecolor="yellow",
        linewidth=1.5,
        linestyle="--",
    )

    show_crosshair(
        ax,
        center[0],
        center[1],
    )

    ax.set_title(
        f"Whole axial — z={z}"
    )

    ax.set_xlabel(
        "X"
    )

    ax.set_ylabel(
        "Y"
    )

    # ==============================================================
    # WHOLE — CORONAL
    # ==============================================================

    ax = axes[0, 1]

    ax.imshow(
        whole[:, y, :].T,
        origin="lower",
        cmap="gray",
        vmin=whole_vmin,
        vmax=whole_vmax,
    )

    add_rectangle(
        ax,
        starts[0] - 0.5,
        starts[2] - 0.5,
        ends[0] - 0.5,
        ends[2] - 0.5,
        edgecolor="cyan",
        linewidth=1.5,
    )

    add_rectangle(
        ax,
        roi_min[0],
        roi_min[2],
        roi_max[0],
        roi_max[2],
        edgecolor="yellow",
        linewidth=1.5,
        linestyle="--",
    )

    show_crosshair(
        ax,
        center[0],
        center[2],
    )

    ax.set_title(
        f"Whole coronal — y={y}"
    )

    ax.set_xlabel(
        "X"
    )

    ax.set_ylabel(
        "Z"
    )

    # ==============================================================
    # WHOLE — SAGITTAL
    # ==============================================================

    ax = axes[0, 2]

    ax.imshow(
        whole[x, :, :].T,
        origin="lower",
        cmap="gray",
        vmin=whole_vmin,
        vmax=whole_vmax,
    )

    add_rectangle(
        ax,
        starts[1] - 0.5,
        starts[2] - 0.5,
        ends[1] - 0.5,
        ends[2] - 0.5,
        edgecolor="cyan",
        linewidth=1.5,
    )

    add_rectangle(
        ax,
        roi_min[1],
        roi_min[2],
        roi_max[1],
        roi_max[2],
        edgecolor="yellow",
        linewidth=1.5,
        linestyle="--",
    )

    show_crosshair(
        ax,
        center[1],
        center[2],
    )

    ax.set_title(
        f"Whole sagittal — x={x}"
    )

    ax.set_xlabel(
        "Y"
    )

    ax.set_ylabel(
        "Z"
    )

    # ==============================================================
    # CROP — AXIAL
    # ==============================================================

    ax = axes[1, 0]

    ax.imshow(
        crop[:, :, cz].T,
        origin="lower",
        cmap="gray",
        vmin=crop_vmin,
        vmax=crop_vmax,
    )

    show_crosshair(
        ax,
        local_center[0],
        local_center[1],
    )

    ax.set_title(
        f"Final crop axial — z={cz}"
    )

    ax.set_xlabel(
        "X"
    )

    ax.set_ylabel(
        "Y"
    )

    # ==============================================================
    # CROP — CORONAL
    # ==============================================================

    ax = axes[1, 1]

    ax.imshow(
        crop[:, cy, :].T,
        origin="lower",
        cmap="gray",
        vmin=crop_vmin,
        vmax=crop_vmax,
    )

    show_crosshair(
        ax,
        local_center[0],
        local_center[2],
    )

    ax.set_title(
        f"Final crop coronal — y={cy}"
    )

    ax.set_xlabel(
        "X"
    )

    ax.set_ylabel(
        "Z"
    )

    # ==============================================================
    # CROP — SAGITTAL
    # ==============================================================

    ax = axes[1, 2]

    ax.imshow(
        crop[cx, :, :].T,
        origin="lower",
        cmap="gray",
        vmin=crop_vmin,
        vmax=crop_vmax,
    )

    show_crosshair(
        ax,
        local_center[1],
        local_center[2],
    )

    ax.set_title(
        f"Final crop sagittal — x={cx}"
    )

    ax.set_xlabel(
        "Y"
    )

    ax.set_ylabel(
        "Z"
    )

    for ax in axes.ravel():
        ax.set_aspect(
            "equal"
        )

    title_lines = [
        f"Step 8E Visual QC — {uid}",
        (
            f"reason={row['selection_reason']} | "
            f"padding={int(row['total_padding_vox'])} vox"
        ),
    ]

    if (
        pd.notna(
            row.get(
                "minimum_roi_margin_vox"
            )
        )
    ):

        title_lines.append(
            "minimum ROI margin="
            f"{float(row['minimum_roi_margin_vox']):.3f} vox"
        )

    fig.suptitle(
        "\n".join(
            title_lines
        ),
        fontsize=12,
    )

    # Whole-scan legend explanation.
    fig.text(
        0.5,
        0.015,
        (
            "Whole scan: solid box = final 44×44×36 crop; "
            "dashed box = transformed template ROI extent; "
            "crosshair = frozen Step-7 center."
        ),
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(
        rect=[
            0,
            0.04,
            1,
            0.92,
        ]
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    selected_dir = (
        args.output_dir
        / "selected_cases"
    )

    selected_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Input checks
    # ------------------------------------------------------------------

    for path in [
        args.extent_statistics,
        args.selected_crop_config,
        args.crop_manifest,
        args.automatic_qc_manifest,
        args.template_striatal_mask,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    for directory in [
        args.normalized_dir,
        args.crop_dir,
    ]:

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Required directory not found: {directory}"
            )

    # ------------------------------------------------------------------
    # Frozen Step-8B configuration
    # ------------------------------------------------------------------

    with open(
        args.selected_crop_config,
        "r",
        encoding="utf-8",
    ) as f:

        crop_config = json.load(
            f
        )

    if not crop_config.get(
        "selection_successful",
        False,
    ):

        raise RuntimeError(
            "Step-8B selection was not successful."
        )

    crop_shape = (
        int(
            crop_config[
                "crop_size_vox"
            ]["x"]
        ),
        int(
            crop_config[
                "crop_size_vox"
            ]["y"]
        ),
        int(
            crop_config[
                "crop_size_vox"
            ]["z"]
        ),
    )

    # ------------------------------------------------------------------
    # Template provenance
    # ------------------------------------------------------------------

    template_img = nib.load(
        str(
            args.template_striatal_mask
        )
    )

    template_data = np.asarray(
        template_img.dataobj
    )

    template_roi_voxels = int(
        np.count_nonzero(
            template_data > 0
        )
    )

    expected_template_roi = (
        crop_config.get(
            "template_roi_voxels"
        )
    )

    if (
        expected_template_roi is not None
        and template_roi_voxels
        != int(
            expected_template_roi
        )
    ):

        raise RuntimeError(
            "Template mask differs from frozen Step-8B template."
        )

    template_hash = sha256_file(
        args.template_striatal_mask
    )

    # ------------------------------------------------------------------
    # Read manifests
    # ------------------------------------------------------------------

    extent_df = pd.read_csv(
        args.extent_statistics
    )

    crop_df = pd.read_csv(
        args.crop_manifest
    )

    qc_df = pd.read_csv(
        args.automatic_qc_manifest
    )

    for df in [
        extent_df,
        crop_df,
        qc_df,
    ]:

        df["uid"] = (
            df["uid"]
            .astype(str)
        )

        if df[
            "uid"
        ].duplicated().any():

            raise ValueError(
                "Duplicate UID detected."
            )

    # Only merge columns that are not already present.
    crop_columns = [
        "uid",
        "center_x_vox",
        "center_y_vox",
        "center_z_vox",
        "crop_start_x",
        "crop_start_y",
        "crop_start_z",
        "padding_required",
        "total_padding_vox",
    ]

    qc_columns = [
        "uid",
        "qc_pass",
        "review_required",
        "extent_valid",
        "roi_coverage_pass",
        "minimum_roi_margin_vox",
    ]

    merged = extent_df.merge(
        crop_df[
            crop_columns
        ],
        on="uid",
        how="inner",
        suffixes=(
            "",
            "_crop",
        ),
    )

    # Prefer frozen Step-8C center columns.
    for axis in [
        "x",
        "y",
        "z",
    ]:

        crop_center = (
            f"center_{axis}_vox_crop"
        )

        if crop_center in merged.columns:

            merged[
                f"center_{axis}_vox"
            ] = merged[
                crop_center
            ]

    merged = merged.merge(
        qc_df[
            qc_columns
        ],
        on="uid",
        how="inner",
    )

    if (
        len(extent_df) != len(crop_df)
        or len(extent_df) != len(qc_df)
        or len(merged) != len(extent_df)
    ):

        raise RuntimeError(
            "Step 8A/8C/8D manifests do not contain "
            "the same subjects. "
            f"8A={len(extent_df)}, "
            f"8C={len(crop_df)}, "
            f"8D={len(qc_df)}, "
            f"merged={len(merged)}"
        )

    if not merged[
        "qc_pass"
    ].map(
        to_bool
    ).all():

        raise RuntimeError(
            "Step 8D did not pass for every subject."
        )

    # ------------------------------------------------------------------
    # Select visual-QC subjects
    # ------------------------------------------------------------------

    selected = select_cases(
        merged=merged,
        args=args,
    )

    selection_path = (
        args.output_dir
        / "visual_qc_selection.csv"
    )

    selected.to_csv(
        selection_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Report selection
    # ------------------------------------------------------------------

    print("=" * 72)
    print(
        "STEP 8E — STRIATAL CROP VISUAL QC"
    )
    print("=" * 72)

    print(
        f"Total subjects:          {len(merged)}"
    )

    print(
        f"Selected for visual QC:  {len(selected)}"
    )

    print(
        f"Frozen crop:             "
        f"{crop_shape[0]} × "
        f"{crop_shape[1]} × "
        f"{crop_shape[2]}"
    )

    print(
        f"Template ROI voxels:     "
        f"{template_roi_voxels}"
    )

    print(
        f"Template SHA256:         "
        f"{template_hash}"
    )

    print()

    print(
        "Selected subjects"
    )

    print(
        "-" * 72
    )

    for _, row in selected.iterrows():

        print(
            f"{row['uid']}: "
            f"{row['selection_reason']}"
        )

    print()

    # ------------------------------------------------------------------
    # Generate images
    # ------------------------------------------------------------------

    generated = []

    failures = []

    for index, row in selected.iterrows():

        uid = str(
            row["uid"]
        )

        output_path = (
            selected_dir
            / f"{uid}_visual_qc.png"
        )

        try:

            create_visual_qc(
                row=row,
                normalized_dir=args.normalized_dir,
                crop_dir=args.crop_dir,
                crop_shape=crop_shape,
                output_path=output_path,
            )

            generated.append(
                str(
                    output_path
                )
            )

            print(
                f"[{index + 1}/{len(selected)}] "
                f"Saved: {output_path}"
            )

        except Exception as exc:

            failures.append(
                {
                    "uid": uid,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

            print(
                f"[{index + 1}/{len(selected)}] "
                f"FAILED {uid}: {exc}"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    reason_counts = defaultdict(
        int
    )

    for reasons in selected[
        "selection_reason"
    ]:

        for reason in str(
            reasons
        ).split(";"):

            reason_counts[
                reason
            ] += 1

    summary = {
        "step": "8E",

        "description": (
            "Visual QC of frozen bilateral "
            "striatal crops"
        ),

        "number_total_subjects": int(
            len(
                merged
            )
        ),

        "number_selected": int(
            len(
                selected
            )
        ),

        "number_images_generated": int(
            len(
                generated
            )
        ),

        "number_generation_failures": int(
            len(
                failures
            )
        ),

        "generation_failures": (
            failures
        ),

        "selection_reason_counts": dict(
            reason_counts
        ),

        "crop_shape_vox": {
            "x": crop_shape[0],
            "y": crop_shape[1],
            "z": crop_shape[2],
        },

        "template": {
            "path": str(
                args.template_striatal_mask
            ),
            "roi_voxels": (
                template_roi_voxels
            ),
            "sha256": (
                template_hash
            ),
        },

        "automatic_qc_passed_before_visual_qc": True,

        "visual_review_status": (
            "PENDING_MANUAL_REVIEW"
        ),

        "step8e_passed": False,

        "note": (
            "Image generation does not constitute visual-QC approval. "
            "Selected figures must be manually inspected before Step 8E "
            "can be frozen."
        ),
    }

    summary_path = (
        args.output_dir
        / "visual_qc_summary.json"
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

    print()
    print("=" * 72)
    print(
        "STEP 8E IMAGE GENERATION COMPLETE"
    )
    print("=" * 72)

    print(
        f"Selected:             "
        f"{len(selected)}"
    )

    print(
        f"Images generated:     "
        f"{len(generated)}"
    )

    print(
        f"Generation failures:  "
        f"{len(failures)}"
    )

    print()
    print(
        f"Images:    {selected_dir}"
    )

    print(
        f"Selection: {selection_path}"
    )

    print(
        f"Summary:   {summary_path}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Step 8E is NOT frozen yet."
    )

    print(
        "Inspect the generated visual-QC images before proceeding to Step 8F."
    )

    if failures:

        raise RuntimeError(
            "One or more visual-QC images failed to generate."
        )


if __name__ == "__main__":
    main()
