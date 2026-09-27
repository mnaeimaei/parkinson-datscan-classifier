from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.registration.affine_registrar import (
    register_affine,
    save_transform,
)

from src.registration.reference_region_mapper import (
    analyze_reference_region,
    map_template_mask_to_subject,
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def _add_prefixed(
    row: dict,
    prefix: str,
    values: dict,
) -> None:
    for key, value in values.items():
        row[f"{prefix}_{key}"] = value


def _strip_nifti_suffix(
    file_name: str,
) -> str:
    if file_name.endswith(".nii.gz"):
        return file_name[:-7]

    if file_name.endswith(".nii"):
        return file_name[:-4]

    return file_name


def add_fov_groups(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Divide scans into low / middle / high Z-axis FOV groups.

    The thresholds are determined from the entire analysis table
    for pilot selection only.
    """
    dataframe = dataframe.copy()

    values = pd.to_numeric(
        dataframe["fov_mm_z"],
        errors="coerce",
    )

    q33 = float(
        values.quantile(1 / 3)
    )

    q67 = float(
        values.quantile(2 / 3)
    )

    conditions = [
        values <= q33,
        (values > q33) & (values <= q67),
        values > q67,
    ]

    dataframe["fov_group"] = np.select(
        conditions,
        [
            "low",
            "middle",
            "high",
        ],
        default="unknown",
    )

    return dataframe


def balanced_sample(
    dataframe: pd.DataFrame,
    number_to_select: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select scans while balancing over:

        scale_decade × fov_group

    within one geometry family.

    Selection is deterministic for a fixed seed.
    """
    if len(dataframe) <= number_to_select:
        return dataframe.copy()

    dataframe = dataframe.copy()

    dataframe["pilot_stratum"] = (
        dataframe["scale_decade"].astype(str)
        + "__"
        + dataframe["fov_group"].astype(str)
    )

    rng = np.random.default_rng(seed)

    grouped_indices: dict[str, list[int]] = {}

    for stratum, group in dataframe.groupby(
        "pilot_stratum"
    ):
        indices = group.index.to_numpy().copy()

        rng.shuffle(indices)

        grouped_indices[str(stratum)] = (
            indices.tolist()
        )

    selected_indices: list[int] = []

    strata = sorted(
        grouped_indices.keys()
    )

    # Round-robin selection across strata.
    while (
        len(selected_indices)
        < number_to_select
    ):
        added = False

        for stratum in strata:

            available = grouped_indices[
                stratum
            ]

            if not available:
                continue

            selected_indices.append(
                available.pop(0)
            )

            added = True

            if (
                len(selected_indices)
                >= number_to_select
            ):
                break

        if not added:
            break

    return dataframe.loc[
        selected_indices
    ].copy()


def select_stratified_pilot(
    cohort_csv_path: str | Path,
    top_geometry_cohorts: int = 8,
    per_geometry: int = 10,
    other_count: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create approximately 90 representative scans:

        8 major geometry cohorts × 10 scans
        +
        10 scans from all remaining geometries

    Within each geometry group, selection is balanced over
    intensity-scale decade and Z-axis FOV group.
    """
    cohort_csv_path = Path(
        cohort_csv_path
    )

    dataframe = pd.read_csv(
        cohort_csv_path
    )

    required_columns = [
        "file_name",
        "geometry_cohort",
        "scale_decade",
        "fov_mm_z",
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns from acquisition "
            f"cohort table: {missing}"
        )

    if dataframe["file_name"].duplicated().any():
        raise ValueError(
            "Duplicate file names found in acquisition "
            "cohort table."
        )

    dataframe = add_fov_groups(
        dataframe
    )

    geometry_counts = (
        dataframe[
            "geometry_cohort"
        ]
        .value_counts()
    )

    major_geometry_names = (
        geometry_counts
        .head(top_geometry_cohorts)
        .index
        .tolist()
    )

    selections: list[pd.DataFrame] = []

    already_selected: set[str] = set()

    # --------------------------------------------------
    # Major geometry families
    # --------------------------------------------------

    for geometry_index, geometry_name in enumerate(
        major_geometry_names
    ):
        group = dataframe[
            dataframe["geometry_cohort"]
            == geometry_name
        ].copy()

        selected = balanced_sample(
            dataframe=group,
            number_to_select=per_geometry,
            seed=seed + geometry_index,
        )

        selected[
            "selection_group"
        ] = (
            f"major_geometry_{geometry_index + 1}"
        )

        selections.append(
            selected
        )

        already_selected.update(
            selected["file_name"].tolist()
        )

    # --------------------------------------------------
    # All remaining geometry cohorts
    # --------------------------------------------------

    other = dataframe[
        ~dataframe[
            "geometry_cohort"
        ].isin(
            major_geometry_names
        )
    ].copy()

    other_selected = balanced_sample(
        dataframe=other,
        number_to_select=other_count,
        seed=seed + 1000,
    )

    other_selected[
        "selection_group"
    ] = "other_geometries"

    selections.append(
        other_selected
    )

    already_selected.update(
        other_selected[
            "file_name"
        ].tolist()
    )

    selected = pd.concat(
        selections,
        ignore_index=True,
    )

    # --------------------------------------------------
    # Fill missing quota if a major geometry cohort
    # contained fewer than requested scans.
    # --------------------------------------------------

    requested_total = (
        top_geometry_cohorts
        * per_geometry
        + other_count
    )

    if len(selected) < requested_total:

        remaining = dataframe[
            ~dataframe[
                "file_name"
            ].isin(
                already_selected
            )
        ].copy()

        number_needed = (
            requested_total
            - len(selected)
        )

        additional = balanced_sample(
            dataframe=remaining,
            number_to_select=(
                number_needed
            ),
            seed=seed + 2000,
        )

        additional[
            "selection_group"
        ] = "quota_fill"

        selected = pd.concat(
            [
                selected,
                additional,
            ],
            ignore_index=True,
        )

    selected = (
        selected
        .drop_duplicates(
            subset=["file_name"]
        )
        .reset_index(drop=True)
    )

    selected[
        "selection_order"
    ] = np.arange(
        1,
        len(selected) + 1,
    )

    return selected


def find_subject_files(
    input_dir: str | Path,
) -> dict[str, Path]:
    """
    Create a file-name -> path lookup.
    """
    input_dir = Path(
        input_dir
    )

    files = (
        list(input_dir.rglob("*.nii"))
        + list(input_dir.rglob("*.nii.gz"))
    )

    lookup: dict[str, Path] = {}

    for path in files:

        if path.name in lookup:
            raise ValueError(
                "Duplicate NIfTI filename found: "
                f"{path.name}"
            )

        lookup[path.name] = path

    return lookup


def _describe_column(
    dataframe: pd.DataFrame,
    column: str,
) -> dict:
    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return {}

    return {
        "count": int(len(values)),
        "min": _safe_float(
            values.min()
        ),
        "p05": _safe_float(
            values.quantile(0.05)
        ),
        "p25": _safe_float(
            values.quantile(0.25)
        ),
        "median": _safe_float(
            values.median()
        ),
        "mean": _safe_float(
            values.mean()
        ),
        "p75": _safe_float(
            values.quantile(0.75)
        ),
        "p95": _safe_float(
            values.quantile(0.95)
        ),
        "max": _safe_float(
            values.max()
        ),
    }


def run_stratified_reference_pilot(
    cohort_csv_path: str | Path,
    input_dir: str | Path,
    template_path: str | Path,
    occipital_mask_path: str | Path,
    cerebellar_mask_path: str | Path,
    output_dir: str | Path,
    top_geometry_cohorts: int = 8,
    per_geometry: int = 10,
    other_count: int = 10,
    seed: int = 42,
) -> dict:
    """
    Run the larger anatomical-reference feasibility pilot.

    IMPORTANT:
        This still does NOT normalize any images.
    """
    output_dir = Path(
        output_dir
    )

    transform_dir = (
        output_dir
        / "transforms"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    transform_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Select representative scans
    # --------------------------------------------------

    selection = select_stratified_pilot(
        cohort_csv_path=(
            cohort_csv_path
        ),
        top_geometry_cohorts=(
            top_geometry_cohorts
        ),
        per_geometry=per_geometry,
        other_count=other_count,
        seed=seed,
    )

    selection_path = (
        output_dir
        / "stratified_pilot_selection.csv"
    )

    selection.to_csv(
        selection_path,
        index=False,
    )

    # --------------------------------------------------
    # Load common template/reference masks
    # --------------------------------------------------

    template = sitk.ReadImage(
        str(template_path),
        sitk.sitkFloat32,
    )

    occipital_mask = sitk.ReadImage(
        str(occipital_mask_path),
        sitk.sitkUInt8,
    )

    cerebellar_mask = sitk.ReadImage(
        str(cerebellar_mask_path),
        sitk.sitkUInt8,
    )

    subject_lookup = find_subject_files(
        input_dir
    )

    results: list[dict] = []
    failures: list[dict] = []

    total = len(selection)

    # --------------------------------------------------
    # Run registration
    # --------------------------------------------------

    for index, (_, source_row) in enumerate(
        selection.iterrows(),
        start=1,
    ):
        file_name = str(
            source_row["file_name"]
        )

        print(
            f"[{index}/{total}] "
            f"{file_name}"
        )

        try:
            if file_name not in subject_lookup:
                raise FileNotFoundError(
                    "Subject NIfTI not found: "
                    f"{file_name}"
                )

            subject_path = (
                subject_lookup[file_name]
            )

            subject = sitk.ReadImage(
                str(subject_path),
                sitk.sitkFloat32,
            )

            (
                transform,
                registration,
            ) = register_affine(
                fixed_template=template,
                moving_subject=subject,
            )

            uid = _strip_nifti_suffix(
                file_name
            )

            transform_path = (
                transform_dir
                / f"{uid}_affine.tfm"
            )

            save_transform(
                transform,
                transform_path,
            )

            # ------------------------------------------
            # Map reference masks back to original
            # resampled subject space
            # ------------------------------------------

            occipital_subject = (
                map_template_mask_to_subject(
                    template_mask=(
                        occipital_mask
                    ),
                    subject_image=subject,
                    subject_to_template_transform=(
                        transform
                    ),
                )
            )

            cerebellar_subject = (
                map_template_mask_to_subject(
                    template_mask=(
                        cerebellar_mask
                    ),
                    subject_image=subject,
                    subject_to_template_transform=(
                        transform
                    ),
                )
            )

            occipital_stats = (
                analyze_reference_region(
                    subject_image=subject,
                    subject_mask=(
                        occipital_subject
                    ),
                    template_mask=(
                        occipital_mask
                    ),
                )
            )

            cerebellar_stats = (
                analyze_reference_region(
                    subject_image=subject,
                    subject_mask=(
                        cerebellar_subject
                    ),
                    template_mask=(
                        cerebellar_mask
                    ),
                )
            )

            row = source_row.to_dict()

            row.update(
                {
                    "uid": uid,
                    "subject_path": str(
                        subject_path
                    ),
                    "transform_path": str(
                        transform_path
                    ),

                    "rigid_metric": (
                        registration[
                            "rigid_metric"
                        ]
                    ),

                    "affine_metric": (
                        registration[
                            "affine_metric"
                        ]
                    ),

                    "head_mask_dice": (
                        registration[
                            "head_mask_dice"
                        ]
                    ),

                    "rigid_stop_condition": (
                        registration[
                            "rigid_stop_condition"
                        ]
                    ),

                    "affine_stop_condition": (
                        registration[
                            "affine_stop_condition"
                        ]
                    ),
                }
            )

            _add_prefixed(
                row,
                "occipital",
                occipital_stats,
            )

            _add_prefixed(
                row,
                "cerebellar",
                cerebellar_stats,
            )

            results.append(
                row
            )

            print(
                "  Dice="
                f"{registration['head_mask_dice']:.3f}"
            )

            print(
                "  Occipital positive="
                f"{occipital_stats.get('positive_fraction')}"
            )

            print(
                "  Cerebellar positive="
                f"{cerebellar_stats.get('positive_fraction')}"
            )

        except Exception as exc:
            print(
                f"  FAILED: {exc}"
            )

            failures.append(
                {
                    "file_name": file_name,
                    "error": str(exc),
                }
            )

    results_df = pd.DataFrame(
        results
    )

    results_path = (
        output_dir
        / "stratified_pilot_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------
    # Cases needing visual review
    # --------------------------------------------------

    if len(results_df) > 0:

        lowest_dice = (
            results_df
            .sort_values(
                "head_mask_dice",
                ascending=True,
            )
            .head(15)
        )

        lowest_dice.to_csv(
            output_dir
            / "lowest_dice_cases.csv",
            index=False,
        )

        lowest_occipital = (
            results_df
            .sort_values(
                "occipital_positive_fraction",
                ascending=True,
            )
            .head(15)
        )

        lowest_occipital.to_csv(
            output_dir
            / "lowest_occipital_coverage_cases.csv",
            index=False,
        )

    # --------------------------------------------------
    # Summary statistics
    # --------------------------------------------------

    statistics_columns = [
        "head_mask_dice",
        "occipital_positive_fraction",
        "occipital_volume_ratio",
        "occipital_mean",
        "occipital_median",
        "cerebellar_positive_fraction",
        "cerebellar_volume_ratio",
    ]

    statistics = {}

    for column in statistics_columns:

        if column in results_df.columns:
            statistics[column] = (
                _describe_column(
                    results_df,
                    column,
                )
            )

    occipital_border_count = 0
    cerebellar_border_count = 0

    if (
        "occipital_touches_border"
        in results_df.columns
    ):
        occipital_border_count = int(
            results_df[
                "occipital_touches_border"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    if (
        "cerebellar_touches_border"
        in results_df.columns
    ):
        cerebellar_border_count = int(
            results_df[
                "cerebellar_touches_border"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    # Distribution of selected pilot
    geometry_counts = (
        selection[
            "geometry_cohort"
        ]
        .value_counts()
        .to_dict()
    )

    scale_counts = (
        selection[
            "scale_decade"
        ]
        .value_counts()
        .to_dict()
    )

    fov_counts = (
        selection[
            "fov_group"
        ]
        .value_counts()
        .to_dict()
    )

    summary = {
        "analysis_type": (
            "stratified_anatomical_reference_pilot"
        ),

        "requested_sampling": {
            "top_geometry_cohorts": int(
                top_geometry_cohorts
            ),
            "per_geometry": int(
                per_geometry
            ),
            "other_count": int(
                other_count
            ),
            "seed": int(seed),
        },

        "number_selected": int(
            len(selection)
        ),

        "successful": int(
            len(results)
        ),

        "failed": int(
            len(failures)
        ),

        "selection_distribution": {
            "geometry_cohorts": {
                str(key): int(value)
                for key, value
                in geometry_counts.items()
            },

            "scale_decades": {
                str(key): int(value)
                for key, value
                in scale_counts.items()
            },

            "fov_groups": {
                str(key): int(value)
                for key, value
                in fov_counts.items()
            },
        },

        "qc_statistics": statistics,

        "border_touch_counts": {
            "occipital": (
                occipital_border_count
            ),
            "cerebellar": (
                cerebellar_border_count
            ),
        },

        "important_note": (
            "This pilot evaluates registration and "
            "reference-region feasibility only. "
            "No images were normalized."
        ),

        "failures": failures,
    }

    summary_path = (
        output_dir
        / "stratified_pilot_statistics.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    return summary


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run a stratified anatomical-reference "
            "registration pilot."
        )
    )

    parser.add_argument(
        "--cohort-csv",
        required=True,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
    )

    parser.add_argument(
        "--template",
        required=True,
    )

    parser.add_argument(
        "--occipital-mask",
        required=True,
    )

    parser.add_argument(
        "--cerebellar-mask",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--top-geometry-cohorts",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--per-geometry",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--other-count",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    print("=" * 72)
    print("STRATIFIED ANATOMICAL REFERENCE PILOT")
    print("=" * 72)

    print(
        f"Cohort table: "
        f"{args.cohort_csv}"
    )

    print(
        f"Top geometry cohorts: "
        f"{args.top_geometry_cohorts}"
    )

    print(
        f"Scans per major geometry: "
        f"{args.per_geometry}"
    )

    print(
        f"Other geometries: "
        f"{args.other_count}"
    )

    print(
        f"Random seed: "
        f"{args.seed}"
    )

    print("-" * 72)

    summary = (
        run_stratified_reference_pilot(
            cohort_csv_path=(
                args.cohort_csv
            ),
            input_dir=(
                args.input_dir
            ),
            template_path=(
                args.template
            ),
            occipital_mask_path=(
                args.occipital_mask
            ),
            cerebellar_mask_path=(
                args.cerebellar_mask
            ),
            output_dir=(
                args.output_dir
            ),
            top_geometry_cohorts=(
                args.top_geometry_cohorts
            ),
            per_geometry=(
                args.per_geometry
            ),
            other_count=(
                args.other_count
            ),
            seed=args.seed,
        )
    )

    print()
    print("=" * 72)
    print("STRATIFIED PILOT COMPLETED")
    print("=" * 72)

    print(
        f"Selected: "
        f"{summary['number_selected']}"
    )

    print(
        f"Successful: "
        f"{summary['successful']}"
    )

    print(
        f"Failed: "
        f"{summary['failed']}"
    )

    print()

    print("QC statistics:")

    for metric, statistics in (
        summary[
            "qc_statistics"
        ].items()
    ):
        print()
        print(
            f"  {metric}"
        )

        print(
            f"    min:    "
            f"{statistics.get('min')}"
        )

        print(
            f"    p05:    "
            f"{statistics.get('p05')}"
        )

        print(
            f"    median: "
            f"{statistics.get('median')}"
        )

        print(
            f"    p95:    "
            f"{statistics.get('p95')}"
        )

        print(
            f"    max:    "
            f"{statistics.get('max')}"
        )

    print()
    print(
        "Border-touch counts:"
    )

    print(
        "  Occipital: "
        f"{summary['border_touch_counts']['occipital']}"
    )

    print(
        "  Cerebellar: "
        f"{summary['border_touch_counts']['cerebellar']}"
    )

    print()
    print(
        f"Results saved to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()