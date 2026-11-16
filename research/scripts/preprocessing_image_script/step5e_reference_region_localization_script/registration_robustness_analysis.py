from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.registration.affine_registrar import (
    save_transform,
)

from src.registration.reference_region_mapper import (
    analyze_reference_region,
    map_template_mask_to_subject,
)

from src.registration.robust_affine_registrar import (
    register_affine_dual_start,
)


def _safe_float(
    value: Any,
) -> float | None:

    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def _strip_nifti_suffix(
    file_name: str,
) -> str:

    if file_name.endswith(
        ".nii.gz"
    ):
        return file_name[:-7]

    if file_name.endswith(
        ".nii"
    ):
        return file_name[:-4]

    return file_name


def _as_bool(
    value: Any,
) -> bool:
    """
    Robustly interpret CSV boolean values.
    """

    if isinstance(
        value,
        bool,
    ):
        return value

    text = str(
        value
    ).strip().lower()

    return text in {
        "true",
        "1",
        "yes",
    }


def find_subject_files(
    input_dir: str | Path,
) -> dict[str, Path]:

    input_dir = Path(
        input_dir
    )

    files = (
        list(
            input_dir.rglob(
                "*.nii"
            )
        )
        + list(
            input_dir.rglob(
                "*.nii.gz"
            )
        )
    )

    lookup = {}

    for path in files:

        if path.name in lookup:
            raise RuntimeError(
                "Duplicate NIfTI filename: "
                f"{path.name}"
            )

        lookup[
            path.name
        ] = path

    return lookup


def provisional_qc(
    dice: float,
    positive_fraction: float,
    volume_ratio: float,
    touches_border: bool,
    high_dice: float = 0.70,
    fail_dice: float = 0.50,
    minimum_positive: float = 0.90,
    minimum_volume_ratio: float = 0.80,
    maximum_volume_ratio: float = 2.70,
) -> str:
    """
    Same provisional QC concept used in Step 5D.
    """

    fail = bool(
        dice < fail_dice
        or positive_fraction
        < minimum_positive
        or volume_ratio
        < minimum_volume_ratio
        or volume_ratio
        > maximum_volume_ratio
        or touches_border
    )

    if fail:
        return "fail_or_fallback"

    if dice >= high_dice:
        return "high_confidence"

    return "review"


def _describe(
    values: pd.Series,
) -> dict:

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            "count": 0
        }

    return {
        "count": int(
            len(values)
        ),

        "min": _safe_float(
            values.min()
        ),

        "p05": _safe_float(
            values.quantile(
                0.05
            )
        ),

        "p25": _safe_float(
            values.quantile(
                0.25
            )
        ),

        "median": _safe_float(
            values.median()
        ),

        "mean": _safe_float(
            values.mean()
        ),

        "p75": _safe_float(
            values.quantile(
                0.75
            )
        ),

        "p95": _safe_float(
            values.quantile(
                0.95
            )
        ),

        "max": _safe_float(
            values.max()
        ),
    }


def run_registration_robustness_analysis(
    pilot_csv: str | Path,
    input_dir: str | Path,
    template_path: str | Path,
    occipital_mask_path: str | Path,
    output_dir: str | Path,
) -> dict:

    pilot_csv = Path(
        pilot_csv
    )

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

    dataframe = pd.read_csv(
        pilot_csv
    )

    required_columns = [
        "file_name",
        "head_mask_dice",
        "occipital_positive_fraction",
        "occipital_volume_ratio",
        "occipital_touches_border",
        "reference_qc_category",
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    subject_lookup = find_subject_files(
        input_dir
    )

    template = sitk.ReadImage(
        str(
            template_path
        ),
        sitk.sitkFloat32,
    )

    occipital_mask = sitk.ReadImage(
        str(
            occipital_mask_path
        ),
        sitk.sitkUInt8,
    )

    results = []
    failures = []

    total = len(
        dataframe
    )

    for index, row in dataframe.iterrows():

        file_name = str(
            row[
                "file_name"
            ]
        )

        print(
            f"[{index + 1}/{total}] "
            f"{file_name}"
        )

        try:

            if (
                file_name
                not in subject_lookup
            ):
                raise FileNotFoundError(
                    file_name
                )

            subject = sitk.ReadImage(
                str(
                    subject_lookup[
                        file_name
                    ]
                ),
                sitk.sitkFloat32,
            )

            (
                robust_transform,
                robust_registration,
            ) = (
                register_affine_dual_start(
                    fixed_template=(
                        template
                    ),
                    moving_subject=(
                        subject
                    ),
                )
            )

            uid = (
                _strip_nifti_suffix(
                    file_name
                )
            )

            transform_path = (
                transform_dir
                / (
                    f"{uid}_"
                    "dual_start_affine.tfm"
                )
            )

            save_transform(
                robust_transform,
                transform_path,
            )

            # ------------------------------------------
            # Map occipital mask using selected transform
            # ------------------------------------------

            occipital_subject = (
                map_template_mask_to_subject(
                    template_mask=(
                        occipital_mask
                    ),
                    subject_image=subject,
                    subject_to_template_transform=(
                        robust_transform
                    ),
                )
            )

            occipital_stats = (
                analyze_reference_region(
                    subject_image=(
                        subject
                    ),
                    subject_mask=(
                        occipital_subject
                    ),
                    template_mask=(
                        occipital_mask
                    ),
                )
            )

            robust_dice = float(
                robust_registration[
                    "head_mask_dice"
                ]
            )

            robust_positive = float(
                occipital_stats[
                    "positive_fraction"
                ]
            )

            robust_volume_ratio = float(
                occipital_stats[
                    "volume_ratio"
                ]
            )

            robust_border = bool(
                occipital_stats[
                    "touches_border"
                ]
            )

            robust_category = (
                provisional_qc(
                    dice=robust_dice,
                    positive_fraction=(
                        robust_positive
                    ),
                    volume_ratio=(
                        robust_volume_ratio
                    ),
                    touches_border=(
                        robust_border
                    ),
                )
            )

            baseline_dice = float(
                row[
                    "head_mask_dice"
                ]
            )

            baseline_category = str(
                row[
                    "reference_qc_category"
                ]
            )

            output_row = (
                row.to_dict()
            )

            output_row.update(
                {
                    "robust_transform_path": (
                        str(
                            transform_path
                        )
                    ),

                    "robust_selected_initialization": (
                        robust_registration[
                            "selected_initialization"
                        ]
                    ),

                    "robust_head_mask_dice": (
                        robust_dice
                    ),

                    "robust_occipital_positive_fraction": (
                        robust_positive
                    ),

                    "robust_occipital_volume_ratio": (
                        robust_volume_ratio
                    ),

                    "robust_occipital_touches_border": (
                        robust_border
                    ),

                    "robust_occipital_mean": (
                        occipital_stats[
                            "mean"
                        ]
                    ),

                    "robust_occipital_median": (
                        occipital_stats[
                            "median"
                        ]
                    ),

                    "robust_qc_category": (
                        robust_category
                    ),

                    "dice_change": (
                        robust_dice
                        - baseline_dice
                    ),

                    "category_changed": (
                        robust_category
                        != baseline_category
                    ),

                    "robust_affine_determinant": (
                        robust_registration[
                            "affine_determinant"
                        ]
                    ),

                    "robust_affine_condition_number": (
                        robust_registration[
                            "affine_condition_number"
                        ]
                    ),
                }
            )

            candidates = (
                robust_registration[
                    "candidate_results"
                ]
            )

            for mode in (
                "moments",
                "geometry",
            ):

                candidate = (
                    candidates.get(
                        mode,
                        {}
                    )
                )

                output_row[
                    f"{mode}_candidate_dice"
                ] = candidate.get(
                    "head_mask_dice"
                )

                output_row[
                    f"{mode}_candidate_affine_metric"
                ] = candidate.get(
                    "affine_metric"
                )

                output_row[
                    f"{mode}_candidate_determinant"
                ] = candidate.get(
                    "affine_determinant"
                )

            results.append(
                output_row
            )

            print(
                "  baseline Dice="
                f"{baseline_dice:.3f}"
            )

            print(
                "  robust Dice="
                f"{robust_dice:.3f}"
            )

            print(
                "  selected="
                f"{robust_registration['selected_initialization']}"
            )

            print(
                "  QC: "
                f"{baseline_category}"
                " -> "
                f"{robust_category}"
            )

        except Exception as exc:

            print(
                f"  FAILED: {exc}"
            )

            failures.append(
                {
                    "file_name": (
                        file_name
                    ),

                    "error": (
                        str(exc)
                    ),
                }
            )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        output_dir
        / "registration_robustness_per_scan.csv",
        index=False,
    )

    if results_df.empty:
        raise RuntimeError(
            "No successful registrations."
        )

    # --------------------------------------------------
    # Baseline vs robust QC counts
    # --------------------------------------------------

    baseline_counts = (
        results_df[
            "reference_qc_category"
        ]
        .value_counts()
        .to_dict()
    )

    robust_counts = (
        results_df[
            "robust_qc_category"
        ]
        .value_counts()
        .to_dict()
    )

    initialization_counts = (
        results_df[
            "robust_selected_initialization"
        ]
        .value_counts()
        .to_dict()
    )

    improved_count = int(
        (
            results_df[
                "dice_change"
            ] > 1e-6
        ).sum()
    )

    worsened_count = int(
        (
            results_df[
                "dice_change"
            ] < -1e-6
        ).sum()
    )

    unchanged_count = int(
        len(results_df)
        - improved_count
        - worsened_count
    )

    # Category transitions.
    transitions = (
        results_df
        .groupby(
            [
                "reference_qc_category",
                "robust_qc_category",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    transitions.to_csv(
        output_dir
        / "registration_qc_transitions.csv",
        index=False,
    )

    # --------------------------------------------------
    # Most improved / remaining worst
    # --------------------------------------------------

    (
        results_df
        .sort_values(
            "dice_change",
            ascending=False,
        )
        .head(20)
        .to_csv(
            output_dir
            / "most_improved_registrations.csv",
            index=False,
        )
    )

    (
        results_df
        .sort_values(
            "robust_head_mask_dice",
            ascending=True,
        )
        .head(20)
        .to_csv(
            output_dir
            / "remaining_lowest_dice.csv",
            index=False,
        )
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = {
        "analysis_type": (
            "dual_start_affine_registration_robustness"
        ),

        "number_of_scans": int(
            len(dataframe)
        ),

        "successful": int(
            len(results_df)
        ),

        "failed": int(
            len(failures)
        ),

        "strategy": {
            "candidate_1": (
                "MOMENTS initialization "
                "+ rigid + affine"
            ),

            "candidate_2": (
                "GEOMETRY initialization "
                "+ rigid + affine"
            ),

            "selection_rule": (
                "Select candidate with larger "
                "head-mask Dice."
            ),

            "pathology_labels_used": (
                False
            ),
        },

        "baseline_qc_counts": {
            str(key): int(value)
            for key, value
            in baseline_counts.items()
        },

        "robust_qc_counts": {
            str(key): int(value)
            for key, value
            in robust_counts.items()
        },

        "selected_initialization_counts": {
            str(key): int(value)
            for key, value
            in initialization_counts.items()
        },

        "dice_comparison": {
            "baseline": (
                _describe(
                    results_df[
                        "head_mask_dice"
                    ]
                )
            ),

            "dual_start": (
                _describe(
                    results_df[
                        "robust_head_mask_dice"
                    ]
                )
            ),

            "dice_change": (
                _describe(
                    results_df[
                        "dice_change"
                    ]
                )
            ),

            "number_improved": (
                improved_count
            ),

            "number_worsened": (
                worsened_count
            ),

            "number_unchanged": (
                unchanged_count
            ),
        },

        "occipital_qc": {
            "positive_fraction": (
                _describe(
                    results_df[
                        "robust_occipital_positive_fraction"
                    ]
                )
            ),

            "volume_ratio": (
                _describe(
                    results_df[
                        "robust_occipital_volume_ratio"
                    ]
                )
            ),

            "border_touch_count": int(
                results_df[
                    "robust_occipital_touches_border"
                ]
                .map(_as_bool)
                .sum()
            ),
        },

        "important_note": (
            "This experiment evaluates registration "
            "robustness only. No images are normalized "
            "or otherwise modified."
        ),

        "failures": failures,
    }

    with (
        output_dir
        / "registration_robustness_statistics.json"
    ).open(
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
            "Evaluate dual-start affine registration "
            "robustness on the 90-scan reference pilot."
        )
    )

    parser.add_argument(
        "--pilot-csv",
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
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    print("=" * 72)
    print("DUAL-START REGISTRATION ROBUSTNESS ANALYSIS")
    print("=" * 72)

    summary = (
        run_registration_robustness_analysis(
            pilot_csv=(
                args.pilot_csv
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
            output_dir=(
                args.output_dir
            ),
        )
    )

    print()
    print("=" * 72)
    print("REGISTRATION ROBUSTNESS ANALYSIS COMPLETED")
    print("=" * 72)

    print(
        f"Successful: "
        f"{summary['successful']}"
    )

    print(
        f"Failed: "
        f"{summary['failed']}"
    )

    print()

    print("Baseline QC:")
    for category, count in (
        summary[
            "baseline_qc_counts"
        ].items()
    ):
        print(
            f"  {category}: {count}"
        )

    print()

    print("Dual-start QC:")
    for category, count in (
        summary[
            "robust_qc_counts"
        ].items()
    ):
        print(
            f"  {category}: {count}"
        )

    print()

    print(
        "Selected initialization:"
    )

    for mode, count in (
        summary[
            "selected_initialization_counts"
        ].items()
    ):
        print(
            f"  {mode}: {count}"
        )

    print()

    comparison = (
        summary[
            "dice_comparison"
        ]
    )

    print(
        "Dice median:"
    )

    print(
        "  Baseline: "
        f"{comparison['baseline']['median']}"
    )

    print(
        "  Dual-start: "
        f"{comparison['dual_start']['median']}"
    )

    print()

    print(
        "Improved scans: "
        f"{comparison['number_improved']}"
    )

    print(
        "Worsened scans: "
        f"{comparison['number_worsened']}"
    )

    print(
        "Unchanged scans: "
        f"{comparison['number_unchanged']}"
    )

    print()

    print(
        f"Output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
