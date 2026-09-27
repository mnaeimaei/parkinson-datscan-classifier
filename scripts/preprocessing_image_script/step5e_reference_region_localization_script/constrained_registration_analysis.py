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

from src.registration.constrained_registrar import (
    register_rigid_candidate,
    register_similarity_candidate,
)

from src.registration.reference_region_mapper import (
    analyze_reference_region,
    map_template_mask_to_subject,
)


QC_RANK = {
    "fail_or_fallback": 0,
    "review": 1,
    "high_confidence": 2,
}


def _safe_float(
    value: Any,
) -> float | None:

    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def _as_bool(
    value: Any,
) -> bool:

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


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

    lookup: dict[str, Path] = {}

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


def classify_qc(
    dice: float,
    positive_fraction: float,
    volume_ratio: float,
    touches_border: bool,
    high_confidence_dice: float = 0.70,
    fail_dice: float = 0.50,
    minimum_positive_fraction: float = 0.90,
    minimum_volume_ratio: float = 0.80,
    maximum_volume_ratio: float = 2.70,
) -> str:
    """
    Same provisional QC rule used in Step 5D/5E.

    These are analysis thresholds, not yet the frozen
    inference thresholds.
    """

    if (
        dice < fail_dice
        or positive_fraction
        < minimum_positive_fraction
        or volume_ratio
        < minimum_volume_ratio
        or volume_ratio
        > maximum_volume_ratio
        or touches_border
    ):
        return "fail_or_fallback"

    if dice >= high_confidence_dice:
        return "high_confidence"

    return "review"


def evaluate_transform(
    subject: sitk.Image,
    template_occipital_mask: sitk.Image,
    transform: sitk.Transform,
    registration_metadata: dict,
) -> dict[str, Any]:
    """
    Map the occipital mask into subject space and calculate
    the same QC metrics for every transform type.
    """

    subject_occipital_mask = (
        map_template_mask_to_subject(
            template_mask=(
                template_occipital_mask
            ),
            subject_image=subject,
            subject_to_template_transform=(
                transform
            ),
        )
    )

    region_stats = (
        analyze_reference_region(
            subject_image=subject,
            subject_mask=(
                subject_occipital_mask
            ),
            template_mask=(
                template_occipital_mask
            ),
        )
    )

    dice = float(
        registration_metadata[
            "head_mask_dice"
        ]
    )

    positive_fraction = float(
        region_stats[
            "positive_fraction"
        ]
    )

    volume_ratio = float(
        region_stats[
            "volume_ratio"
        ]
    )

    touches_border = bool(
        region_stats[
            "touches_border"
        ]
    )

    qc_category = classify_qc(
        dice=dice,
        positive_fraction=(
            positive_fraction
        ),
        volume_ratio=volume_ratio,
        touches_border=(
            touches_border
        ),
    )

    return {
        "qc_category": (
            qc_category
        ),

        "head_mask_dice": (
            dice
        ),

        "occipital_positive_fraction": (
            positive_fraction
        ),

        "occipital_volume_ratio": (
            volume_ratio
        ),

        "occipital_touches_border": (
            touches_border
        ),

        "occipital_mean": (
            region_stats["mean"]
        ),

        "occipital_median": (
            region_stats["median"]
        ),

        "registration_metadata": (
            registration_metadata
        ),
    }


def choose_best_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Selection is deliberately NOT Dice-only.

    First:
        high_confidence > review > fail

    Then, only within the same QC category:
        higher head-mask Dice wins.

    Final tie-breaker:
        mapped occipital volume ratio closer to 1.0.
    """

    if not candidates:
        raise RuntimeError(
            "No registration candidates."
        )

    def candidate_key(
        candidate: dict[str, Any],
    ) -> tuple[
        int,
        float,
        float,
    ]:

        category = candidate[
            "qc_category"
        ]

        dice = float(
            candidate[
                "head_mask_dice"
            ]
        )

        volume_ratio = float(
            candidate[
                "occipital_volume_ratio"
            ]
        )

        if volume_ratio > 0:
            volume_closeness = -abs(
                np.log(
                    volume_ratio
                )
            )
        else:
            volume_closeness = (
                -np.inf
            )

        return (
            QC_RANK[
                category
            ],
            dice,
            volume_closeness,
        )

    return max(
        candidates,
        key=candidate_key,
    )


def choose_best_initialization(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:

    return choose_best_candidate(
        candidates
    )


def _describe(
    values: pd.Series,
) -> dict[str, Any]:

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            "count": 0,
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


def run_constrained_registration_analysis(
    robustness_csv: str | Path,
    input_dir: str | Path,
    template_path: str | Path,
    occipital_mask_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:

    robustness_csv = Path(
        robustness_csv
    )

    output_dir = Path(
        output_dir
    )

    transform_root = (
        output_dir
        / "transforms"
    )

    rigid_transform_dir = (
        transform_root
        / "rigid"
    )

    similarity_transform_dir = (
        transform_root
        / "similarity"
    )

    rigid_transform_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    similarity_transform_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        robustness_csv
    )

    required_columns = [
        "file_name",

        "robust_transform_path",
        "robust_selected_initialization",
        "robust_head_mask_dice",

        "robust_occipital_positive_fraction",
        "robust_occipital_volume_ratio",
        "robust_occipital_touches_border",
        "robust_occipital_mean",
        "robust_occipital_median",

        "robust_qc_category",
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns in robustness CSV: "
            f"{missing}"
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

    results: list[dict] = []

    failures: list[dict] = []

    total = len(
        dataframe
    )

    for row_index, source_row in dataframe.iterrows():

        file_name = str(
            source_row[
                "file_name"
            ]
        )

        uid = _strip_nifti_suffix(
            file_name
        )

        print()
        print(
            f"[{row_index + 1}/{total}] "
            f"{file_name}"
        )

        try:

            if file_name not in subject_lookup:
                raise FileNotFoundError(
                    f"Subject not found: "
                    f"{file_name}"
                )

            subject = sitk.ReadImage(
                str(
                    subject_lookup[
                        file_name
                    ]
                ),
                sitk.sitkFloat32,
            )

            # ==================================================
            # RIGID
            # ==================================================

            rigid_candidates = []

            for mode in (
                "moments",
                "geometry",
            ):

                try:

                    (
                        transform,
                        metadata,
                    ) = register_rigid_candidate(
                        fixed_template=(
                            template
                        ),
                        moving_subject=(
                            subject
                        ),
                        initialization_mode=(
                            mode
                        ),
                    )

                    evaluation = evaluate_transform(
                        subject=subject,
                        template_occipital_mask=(
                            occipital_mask
                        ),
                        transform=transform,
                        registration_metadata=(
                            metadata
                        ),
                    )

                    evaluation.update(
                        {
                            "method": "rigid",
                            "initialization": mode,
                            "transform": transform,
                        }
                    )

                    rigid_candidates.append(
                        evaluation
                    )

                except Exception as exc:

                    print(
                        f"  Rigid {mode} FAILED: "
                        f"{exc}"
                    )

            if not rigid_candidates:
                raise RuntimeError(
                    "Both rigid registrations failed."
                )

            rigid_best = (
                choose_best_initialization(
                    rigid_candidates
                )
            )

            rigid_path = (
                rigid_transform_dir
                / f"{uid}_rigid.tfm"
            )

            save_transform(
                rigid_best[
                    "transform"
                ],
                rigid_path,
            )

            # ==================================================
            # SIMILARITY
            # ==================================================

            similarity_candidates = []

            for mode in (
                "moments",
                "geometry",
            ):

                try:

                    (
                        transform,
                        metadata,
                    ) = (
                        register_similarity_candidate(
                            fixed_template=(
                                template
                            ),
                            moving_subject=(
                                subject
                            ),
                            initialization_mode=(
                                mode
                            ),
                        )
                    )

                    evaluation = evaluate_transform(
                        subject=subject,
                        template_occipital_mask=(
                            occipital_mask
                        ),
                        transform=transform,
                        registration_metadata=(
                            metadata
                        ),
                    )

                    evaluation.update(
                        {
                            "method": (
                                "similarity"
                            ),

                            "initialization": (
                                mode
                            ),

                            "transform": (
                                transform
                            ),
                        }
                    )

                    similarity_candidates.append(
                        evaluation
                    )

                except Exception as exc:

                    print(
                        f"  Similarity {mode} FAILED: "
                        f"{exc}"
                    )

            if not similarity_candidates:
                raise RuntimeError(
                    "Both similarity registrations failed."
                )

            similarity_best = (
                choose_best_initialization(
                    similarity_candidates
                )
            )

            similarity_path = (
                similarity_transform_dir
                / f"{uid}_similarity.tfm"
            )

            save_transform(
                similarity_best[
                    "transform"
                ],
                similarity_path,
            )

            # ==================================================
            # EXISTING DUAL-START AFFINE
            # ==================================================

            affine_candidate = {
                "method": "affine",

                "initialization": str(
                    source_row[
                        "robust_selected_initialization"
                    ]
                ),

                "transform": None,

                "transform_path": str(
                    source_row[
                        "robust_transform_path"
                    ]
                ),

                "qc_category": str(
                    source_row[
                        "robust_qc_category"
                    ]
                ),

                "head_mask_dice": float(
                    source_row[
                        "robust_head_mask_dice"
                    ]
                ),

                "occipital_positive_fraction": float(
                    source_row[
                        "robust_occipital_positive_fraction"
                    ]
                ),

                "occipital_volume_ratio": float(
                    source_row[
                        "robust_occipital_volume_ratio"
                    ]
                ),

                "occipital_touches_border": _as_bool(
                    source_row[
                        "robust_occipital_touches_border"
                    ]
                ),

                "occipital_mean": float(
                    source_row[
                        "robust_occipital_mean"
                    ]
                ),

                "occipital_median": float(
                    source_row[
                        "robust_occipital_median"
                    ]
                ),
            }

            # ==================================================
            # SELECT BEST METHOD
            # ==================================================

            final_candidates = [
                rigid_best,
                similarity_best,
                affine_candidate,
            ]

            selected = choose_best_candidate(
                final_candidates
            )

            if selected["method"] == "rigid":
                selected_transform_path = str(
                    rigid_path
                )

            elif (
                selected["method"]
                == "similarity"
            ):
                selected_transform_path = str(
                    similarity_path
                )

            else:
                selected_transform_path = str(
                    source_row[
                        "robust_transform_path"
                    ]
                )

            # ==================================================
            # SAVE ROW
            # ==================================================

            output_row = (
                source_row.to_dict()
            )

            output_row.update(
                {
                    # -----------------------------
                    # Rigid
                    # -----------------------------

                    "rigid_selected_initialization": (
                        rigid_best[
                            "initialization"
                        ]
                    ),

                    "rigid_transform_path": (
                        str(
                            rigid_path
                        )
                    ),

                    "rigid_qc_category": (
                        rigid_best[
                            "qc_category"
                        ]
                    ),

                    "rigid_head_mask_dice": (
                        rigid_best[
                            "head_mask_dice"
                        ]
                    ),

                    "rigid_occipital_positive_fraction": (
                        rigid_best[
                            "occipital_positive_fraction"
                        ]
                    ),

                    "rigid_occipital_volume_ratio": (
                        rigid_best[
                            "occipital_volume_ratio"
                        ]
                    ),

                    "rigid_occipital_touches_border": (
                        rigid_best[
                            "occipital_touches_border"
                        ]
                    ),

                    "rigid_occipital_mean": (
                        rigid_best[
                            "occipital_mean"
                        ]
                    ),

                    "rigid_determinant": (
                        rigid_best[
                            "registration_metadata"
                        ][
                            "determinant"
                        ]
                    ),

                    "rigid_condition_number": (
                        rigid_best[
                            "registration_metadata"
                        ][
                            "condition_number"
                        ]
                    ),

                    # -----------------------------
                    # Similarity
                    # -----------------------------

                    "similarity_selected_initialization": (
                        similarity_best[
                            "initialization"
                        ]
                    ),

                    "similarity_transform_path": (
                        str(
                            similarity_path
                        )
                    ),

                    "similarity_qc_category": (
                        similarity_best[
                            "qc_category"
                        ]
                    ),

                    "similarity_head_mask_dice": (
                        similarity_best[
                            "head_mask_dice"
                        ]
                    ),

                    "similarity_occipital_positive_fraction": (
                        similarity_best[
                            "occipital_positive_fraction"
                        ]
                    ),

                    "similarity_occipital_volume_ratio": (
                        similarity_best[
                            "occipital_volume_ratio"
                        ]
                    ),

                    "similarity_occipital_touches_border": (
                        similarity_best[
                            "occipital_touches_border"
                        ]
                    ),

                    "similarity_occipital_mean": (
                        similarity_best[
                            "occipital_mean"
                        ]
                    ),

                    "similarity_scale": (
                        similarity_best[
                            "registration_metadata"
                        ][
                            "scale"
                        ]
                    ),

                    "similarity_determinant": (
                        similarity_best[
                            "registration_metadata"
                        ][
                            "determinant"
                        ]
                    ),

                    "similarity_condition_number": (
                        similarity_best[
                            "registration_metadata"
                        ][
                            "condition_number"
                        ]
                    ),

                    # -----------------------------
                    # Final selected candidate
                    # -----------------------------

                    "selected_method": (
                        selected[
                            "method"
                        ]
                    ),

                    "selected_initialization": (
                        selected[
                            "initialization"
                        ]
                    ),

                    "selected_transform_path": (
                        selected_transform_path
                    ),

                    "selected_qc_category": (
                        selected[
                            "qc_category"
                        ]
                    ),

                    "selected_head_mask_dice": (
                        selected[
                            "head_mask_dice"
                        ]
                    ),

                    "selected_occipital_positive_fraction": (
                        selected[
                            "occipital_positive_fraction"
                        ]
                    ),

                    "selected_occipital_volume_ratio": (
                        selected[
                            "occipital_volume_ratio"
                        ]
                    ),

                    "selected_occipital_touches_border": (
                        selected[
                            "occipital_touches_border"
                        ]
                    ),

                    "selected_occipital_mean": (
                        selected[
                            "occipital_mean"
                        ]
                    ),
                }
            )

            # Candidate-specific Dice for debugging.
            for candidate in rigid_candidates:

                mode = candidate[
                    "initialization"
                ]

                output_row[
                    f"rigid_{mode}_candidate_dice"
                ] = candidate[
                    "head_mask_dice"
                ]

                output_row[
                    f"rigid_{mode}_candidate_qc"
                ] = candidate[
                    "qc_category"
                ]

            for candidate in similarity_candidates:

                mode = candidate[
                    "initialization"
                ]

                output_row[
                    f"similarity_{mode}_candidate_dice"
                ] = candidate[
                    "head_mask_dice"
                ]

                output_row[
                    f"similarity_{mode}_candidate_qc"
                ] = candidate[
                    "qc_category"
                ]

            results.append(
                output_row
            )

            print(
                "  Affine:     "
                f"{affine_candidate['qc_category']:16s} "
                f"Dice={affine_candidate['head_mask_dice']:.3f}"
            )

            print(
                "  Rigid:      "
                f"{rigid_best['qc_category']:16s} "
                f"Dice={rigid_best['head_mask_dice']:.3f}"
            )

            print(
                "  Similarity: "
                f"{similarity_best['qc_category']:16s} "
                f"Dice={similarity_best['head_mask_dice']:.3f}"
            )

            print(
                "  SELECTED:   "
                f"{selected['method']} "
                f"({selected['qc_category']})"
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

    # ==================================================
    # DATAFRAME
    # ==================================================

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:
        raise RuntimeError(
            "No successful scans."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_dir
        / "constrained_registration_per_scan.csv",
        index=False,
    )

    # ==================================================
    # METHOD QC COUNTS
    # ==================================================

    method_qc_rows = []

    method_columns = {
        "affine": (
            "robust_qc_category"
        ),

        "rigid": (
            "rigid_qc_category"
        ),

        "similarity": (
            "similarity_qc_category"
        ),

        "selected": (
            "selected_qc_category"
        ),
    }

    for method, column in (
        method_columns.items()
    ):

        counts = (
            results_df[
                column
            ]
            .value_counts()
            .to_dict()
        )

        for category in (
            "high_confidence",
            "review",
            "fail_or_fallback",
        ):

            method_qc_rows.append(
                {
                    "method": (
                        method
                    ),

                    "qc_category": (
                        category
                    ),

                    "count": int(
                        counts.get(
                            category,
                            0,
                        )
                    ),
                }
            )

    pd.DataFrame(
        method_qc_rows
    ).to_csv(
        output_dir
        / "method_qc_counts.csv",
        index=False,
    )

    # ==================================================
    # AFFINE -> FINAL TRANSITIONS
    # ==================================================

    transitions = (
        results_df
        .groupby(
            [
                "robust_qc_category",
                "selected_qc_category",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    transitions.to_csv(
        output_dir
        / "affine_to_selected_qc_transitions.csv",
        index=False,
    )

    # ==================================================
    # REMAINING FAILURES
    # ==================================================

    remaining_failures = results_df[
        results_df[
            "selected_qc_category"
        ]
        == "fail_or_fallback"
    ].copy()

    remaining_failures = (
        remaining_failures
        .sort_values(
            "selected_head_mask_dice",
            ascending=True,
        )
    )

    remaining_failures.to_csv(
        output_dir
        / "remaining_fail_or_fallback.csv",
        index=False,
    )

    # ==================================================
    # METHOD COUNTS
    # ==================================================

    selected_method_counts = (
        results_df[
            "selected_method"
        ]
        .value_counts()
        .to_dict()
    )

    selected_qc_counts = (
        results_df[
            "selected_qc_category"
        ]
        .value_counts()
        .to_dict()
    )

    affine_qc_counts = (
        results_df[
            "robust_qc_category"
        ]
        .value_counts()
        .to_dict()
    )

    # ==================================================
    # RESCUE COUNTS
    # ==================================================

    affine_fail = (
        results_df[
            "robust_qc_category"
        ]
        == "fail_or_fallback"
    )

    selected_not_fail = (
        results_df[
            "selected_qc_category"
        ]
        != "fail_or_fallback"
    )

    rescued_from_affine_fail = int(
        (
            affine_fail
            & selected_not_fail
        ).sum()
    )

    affine_review = (
        results_df[
            "robust_qc_category"
        ]
        == "review"
    )

    selected_high = (
        results_df[
            "selected_qc_category"
        ]
        == "high_confidence"
    )

    review_to_high = int(
        (
            affine_review
            & selected_high
        ).sum()
    )

    # ==================================================
    # SUMMARY
    # ==================================================

    summary = {
        "analysis_type": (
            "constrained_registration_comparison"
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

        "methods": {
            "rigid": (
                "Dual-start rigid registration: "
                "rotation + translation only."
            ),

            "similarity": (
                "Dual-start rigid followed by similarity: "
                "rotation + translation + one isotropic scale."
            ),

            "affine": (
                "Existing dual-start affine registration "
                "from Step 5E-1."
            ),
        },

        "selection_rule": {
            "primary": (
                "Highest QC category: "
                "high_confidence > review > fail_or_fallback."
            ),

            "secondary": (
                "Within the same category, highest "
                "head-mask Dice."
            ),

            "tertiary": (
                "If still tied, mapped occipital volume "
                "ratio closest to 1.0."
            ),

            "pathology_labels_used": (
                False
            ),
        },

        "affine_qc_counts": {
            str(key): int(value)
            for key, value
            in affine_qc_counts.items()
        },

        "selected_qc_counts": {
            str(key): int(value)
            for key, value
            in selected_qc_counts.items()
        },

        "selected_method_counts": {
            str(key): int(value)
            for key, value
            in selected_method_counts.items()
        },

        "rescue_statistics": {
            "affine_fail_rescued_to_review_or_high": (
                rescued_from_affine_fail
            ),

            "affine_review_rescued_to_high": (
                review_to_high
            ),
        },

        "dice_statistics": {
            "affine": _describe(
                results_df[
                    "robust_head_mask_dice"
                ]
            ),

            "rigid": _describe(
                results_df[
                    "rigid_head_mask_dice"
                ]
            ),

            "similarity": _describe(
                results_df[
                    "similarity_head_mask_dice"
                ]
            ),

            "selected": _describe(
                results_df[
                    "selected_head_mask_dice"
                ]
            ),
        },

        "occipital_volume_ratio_statistics": {
            "affine": _describe(
                results_df[
                    "robust_occipital_volume_ratio"
                ]
            ),

            "rigid": _describe(
                results_df[
                    "rigid_occipital_volume_ratio"
                ]
            ),

            "similarity": _describe(
                results_df[
                    "similarity_occipital_volume_ratio"
                ]
            ),

            "selected": _describe(
                results_df[
                    "selected_occipital_volume_ratio"
                ]
            ),
        },

        "similarity_scale_statistics": (
            _describe(
                results_df[
                    "similarity_scale"
                ]
            )
        ),

        "transform_sanity": {
            "rigid_determinant": (
                _describe(
                    results_df[
                        "rigid_determinant"
                    ]
                )
            ),

            "similarity_determinant": (
                _describe(
                    results_df[
                        "similarity_determinant"
                    ]
                )
            ),

            "similarity_condition_number": (
                _describe(
                    results_df[
                        "similarity_condition_number"
                    ]
                )
            ),
        },

        "important_note": (
            "This remains a registration feasibility/QC "
            "experiment. No NIfTI images were normalized."
        ),

        "failures": failures,
    }

    with (
        output_dir
        / "constrained_registration_statistics.json"
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
            "Compare rigid, similarity and existing "
            "dual-start affine registration for "
            "occipital reference localization."
        )
    )

    parser.add_argument(
        "--robustness-csv",
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
    print("CONSTRAINED REGISTRATION COMPARISON")
    print("=" * 72)

    summary = (
        run_constrained_registration_analysis(
            robustness_csv=(
                args.robustness_csv
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
    print("CONSTRAINED REGISTRATION COMPARISON COMPLETED")
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
    print("Previous dual-start affine QC:")

    for category, count in (
        summary[
            "affine_qc_counts"
        ].items()
    ):
        print(
            f"  {category}: {count}"
        )

    print()
    print("Best-of-methods QC:")

    for category, count in (
        summary[
            "selected_qc_counts"
        ].items()
    ):
        print(
            f"  {category}: {count}"
        )

    print()
    print("Selected registration method:")

    for method, count in (
        summary[
            "selected_method_counts"
        ].items()
    ):
        print(
            f"  {method}: {count}"
        )

    print()
    print("Rescues:")

    print(
        "  Affine fail -> review/high: "
        f"{summary['rescue_statistics']['affine_fail_rescued_to_review_or_high']}"
    )

    print(
        "  Affine review -> high: "
        f"{summary['rescue_statistics']['affine_review_rescued_to_high']}"
    )

    print()
    print("Median Dice:")

    for method in (
        "rigid",
        "similarity",
        "affine",
        "selected",
    ):

        print(
            f"  {method}: "
            f"{summary['dice_statistics'][method]['median']}"
        )

    print()
    print(
        "Similarity scale median: "
        f"{summary['similarity_scale_statistics']['median']}"
    )

    print()
    print(
        f"Output: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
