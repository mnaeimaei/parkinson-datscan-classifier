"""
aggregate_cv_results.py

Aggregate K-fold cross-validation predictions.

Primary responsibilities
------------------------
1. Read validation predictions from every fold.

2. Verify:
       - expected number of folds
       - UID uniqueness within each fold
       - no subject appears in multiple validation folds
       - labels are binary
       - probabilities are valid

3. Calculate metrics for every fold.

4. Calculate:
       mean
       standard deviation
       minimum
       maximum

   across folds.

5. Concatenate all validation predictions into one OOF dataset.

6. Calculate GLOBAL OOF metrics.

7. Save:
       fold_metrics.csv
       oof_predictions.csv
       cv_summary.json

Important
---------
For proper K-fold OOF evaluation:

    every dataset subject must appear in validation exactly once.

For this project:

    expected subjects = 1362
    expected folds    = 5

These values can be checked via command-line arguments rather than being
hardcoded into the generic engine.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.evaluation.evaluate import (
    _json_safe,
    read_prediction_csv,
)

from src.evaluation.metrics import (
    PERFORMANCE_METRICS,
    compute_binary_metrics,
)


# =====================================================================
# LOAD FOLDS
# =====================================================================


def load_fold_predictions(
    prediction_paths: Sequence[
        str | Path
    ],
) -> list[pd.DataFrame]:
    """
    Read fold prediction CSV files and attach fold index.
    """

    if not prediction_paths:

        raise ValueError(
            "No fold prediction files were supplied."
        )

    fold_frames: list[
        pd.DataFrame
    ] = []

    for fold_index, path in enumerate(
        prediction_paths
    ):

        df = read_prediction_csv(
            path
        )

        if df.empty:

            raise ValueError(
                f"Fold {fold_index} prediction file is empty."
            )

        if df[
            "uid"
        ].duplicated().any():

            duplicates = (
                df.loc[
                    df[
                        "uid"
                    ].duplicated(
                        keep=False
                    ),
                    "uid",
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Fold {fold_index} contains duplicate UIDs:\n"
                + "\n".join(
                    f"  - {uid}"
                    for uid
                    in duplicates[:20]
                )
            )

        df = df.copy()

        df.insert(
            0,
            "fold",
            fold_index,
        )

        df[
            "source_file"
        ] = str(
            Path(path)
            .expanduser()
            .resolve()
        )

        fold_frames.append(
            df
        )

    return fold_frames


# =====================================================================
# CROSS-FOLD VALIDATION
# =====================================================================


def validate_oof_structure(
    fold_frames: Sequence[
        pd.DataFrame
    ],
    *,
    expected_folds: int | None = None,
    expected_subjects: int | None = None,
) -> None:
    """
    Verify that fold validation sets form a proper OOF partition.
    """

    if expected_folds is not None:

        if len(
            fold_frames
        ) != expected_folds:

            raise ValueError(
                "Unexpected number of folds.\n"
                f"Expected: {expected_folds}\n"
                f"Actual  : {len(fold_frames)}"
            )

    seen_uids: set[
        str
    ] = set()

    duplicate_cross_fold: set[
        str
    ] = set()

    for frame in fold_frames:

        current_uids = set(
            frame[
                "uid"
            ].astype(str)
        )

        overlap = (
            seen_uids
            & current_uids
        )

        duplicate_cross_fold.update(
            overlap
        )

        seen_uids.update(
            current_uids
        )

    if duplicate_cross_fold:

        raise RuntimeError(
            "Subjects appear in more than one validation fold.\n"
            "This is not a valid OOF partition.\n"
            + "\n".join(
                f"  - {uid}"
                for uid
                in sorted(
                    duplicate_cross_fold
                )[:30]
            )
        )

    if expected_subjects is not None:

        if len(
            seen_uids
        ) != expected_subjects:

            raise RuntimeError(
                "OOF subject count mismatch.\n"
                f"Expected: {expected_subjects}\n"
                f"Actual  : {len(seen_uids)}"
            )


# =====================================================================
# FOLD METRICS
# =====================================================================


def calculate_fold_metrics(
    fold_frames: Sequence[
        pd.DataFrame
    ],
    *,
    threshold: float,
    ece_bins: int = 10,
) -> pd.DataFrame:
    """
    Calculate metrics independently for every fold.
    """

    rows: list[
        dict[str, Any]
    ] = []

    for fold_index, df in enumerate(
        fold_frames
    ):

        metrics = compute_binary_metrics(
            df[
                "is_pathologic"
            ].to_numpy(),

            df[
                "probability"
            ].to_numpy(),

            threshold=threshold,

            ece_bins=ece_bins,
        )

        row: dict[
            str,
            Any,
        ] = {
            "fold": fold_index,
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# FOLD SUMMARY
# =====================================================================


def summarize_fold_metrics(
    fold_metrics: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """
    Compute mean/std/min/max for selected performance metrics.

    Standard deviation uses sample SD (ddof=1) when multiple folds exist.
    """

    summary: dict[
        str,
        dict[str, float]
    ] = {}

    for metric_name in (
        PERFORMANCE_METRICS
    ):

        if (
            metric_name
            not in fold_metrics.columns
        ):
            continue

        values = (
            pd.to_numeric(
                fold_metrics[
                    metric_name
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

        finite_values = values[
            np.isfinite(
                values
            )
        ]

        if (
            finite_values.size
            == 0
        ):

            summary[
                metric_name
            ] = {
                "mean": float(
                    "nan"
                ),
                "std": float(
                    "nan"
                ),
                "min": float(
                    "nan"
                ),
                "max": float(
                    "nan"
                ),
            }

            continue

        if (
            finite_values.size
            > 1
        ):

            std = float(
                np.std(
                    finite_values,
                    ddof=1,
                )
            )

        else:

            std = 0.0

        summary[
            metric_name
        ] = {
            "mean": float(
                np.mean(
                    finite_values
                )
            ),

            "std": std,

            "min": float(
                np.min(
                    finite_values
                )
            ),

            "max": float(
                np.max(
                    finite_values
                )
            ),
        }

    return summary


# =====================================================================
# COMPLETE OOF PREDICTIONS
# =====================================================================


def build_oof_predictions(
    fold_frames: Sequence[
        pd.DataFrame
    ],
    *,
    threshold: float,
) -> pd.DataFrame:
    """
    Combine all validation folds into one OOF prediction table.
    """

    oof = pd.concat(
        fold_frames,
        axis=0,
        ignore_index=True,
    )

    if oof[
        "uid"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate UID detected after building OOF table."
        )

    oof[
        "predicted_label"
    ] = (
        oof[
            "probability"
        ]
        >= threshold
    ).astype(
        np.int64
    )

    oof[
        "correct"
    ] = (
        oof[
            "predicted_label"
        ]
        == oof[
            "is_pathologic"
        ]
    ).astype(
        np.int64
    )

    # -------------------------------------------------------------
    # Stable ordering:
    # first by fold, then UID.
    # -------------------------------------------------------------

    oof = (
        oof
        .sort_values(
            [
                "fold",
                "uid",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    return oof


# =====================================================================
# GLOBAL OOF METRICS
# =====================================================================


def calculate_global_oof_metrics(
    oof_predictions: pd.DataFrame,
    *,
    threshold: float,
    ece_bins: int = 10,
) -> dict[str, float | int]:
    """
    Calculate metrics across ALL OOF predictions simultaneously.
    """

    return compute_binary_metrics(
        oof_predictions[
            "is_pathologic"
        ].to_numpy(),

        oof_predictions[
            "probability"
        ].to_numpy(),

        threshold=threshold,

        ece_bins=ece_bins,
    )


# =====================================================================
# MAIN AGGREGATION FUNCTION
# =====================================================================


def aggregate_cv_results(
    *,
    prediction_paths: Sequence[
        str | Path
    ],
    output_dir: str | Path,
    threshold: float = 0.5,
    expected_folds: int | None = 5,
    expected_subjects: int | None = None,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """
    Aggregate complete K-fold CV evaluation.

    Returns
    -------
    dict
        CV summary.
    """

    output_dir = (
        Path(
            output_dir
        )
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Read folds.
    # -------------------------------------------------------------

    fold_frames = (
        load_fold_predictions(
            prediction_paths
        )
    )

    # -------------------------------------------------------------
    # Verify OOF structure.
    # -------------------------------------------------------------

    validate_oof_structure(
        fold_frames,
        expected_folds=(
            expected_folds
        ),
        expected_subjects=(
            expected_subjects
        ),
    )

    # -------------------------------------------------------------
    # Fold-level metrics.
    # -------------------------------------------------------------

    fold_metrics = (
        calculate_fold_metrics(
            fold_frames,
            threshold=threshold,
            ece_bins=ece_bins,
        )
    )

    fold_summary = (
        summarize_fold_metrics(
            fold_metrics
        )
    )

    # -------------------------------------------------------------
    # Complete OOF table.
    # -------------------------------------------------------------

    oof_predictions = (
        build_oof_predictions(
            fold_frames,
            threshold=threshold,
        )
    )

    # -------------------------------------------------------------
    # Global OOF metrics.
    # -------------------------------------------------------------

    global_oof_metrics = (
        calculate_global_oof_metrics(
            oof_predictions,
            threshold=threshold,
            ece_bins=ece_bins,
        )
    )

    # -------------------------------------------------------------
    # Fold sizes.
    # -------------------------------------------------------------

    fold_sizes = {
        str(
            fold_index
        ): int(
            len(frame)
        )
        for fold_index, frame
        in enumerate(
            fold_frames
        )
    }

    summary: dict[
        str,
        Any,
    ] = {
        "n_folds": len(
            fold_frames
        ),

        "n_oof_subjects": int(
            len(
                oof_predictions
            )
        ),

        "threshold": float(
            threshold
        ),

        "ece_bins": int(
            ece_bins
        ),

        "fold_sizes": (
            fold_sizes
        ),

        "fold_metric_summary": (
            fold_summary
        ),

        "global_oof_metrics": (
            global_oof_metrics
        ),
    }

    # -------------------------------------------------------------
    # SAVE fold_metrics.csv
    # -------------------------------------------------------------

    fold_metrics_path = (
        output_dir
        / "fold_metrics.csv"
    )

    fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
    )

    # -------------------------------------------------------------
    # SAVE oof_predictions.csv
    # -------------------------------------------------------------

    oof_path = (
        output_dir
        / "oof_predictions.csv"
    )

    oof_predictions.to_csv(
        oof_path,
        index=False,
    )

    # -------------------------------------------------------------
    # SAVE cv_summary.json
    # -------------------------------------------------------------

    summary_path = (
        output_dir
        / "cv_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            _json_safe(
                summary
            ),
            file,
            indent=2,
            sort_keys=True,
        )

    return {
        "summary": summary,

        "fold_metrics": (
            fold_metrics
        ),

        "oof_predictions": (
            oof_predictions
        ),

        "paths": {
            "fold_metrics": (
                fold_metrics_path
            ),

            "oof_predictions": (
                oof_path
            ),

            "cv_summary": (
                summary_path
            ),
        },
    }


# =====================================================================
# PRINT SUMMARY
# =====================================================================


def print_cv_summary(
    result: dict[
        str,
        Any,
    ],
) -> None:
    """
    Human-readable CV summary.
    """

    summary = result[
        "summary"
    ]

    print(
        "=" * 78
    )

    print(
        "CROSS-VALIDATION AGGREGATION"
    )

    print(
        "=" * 78
    )

    print()

    print(
        f"Folds        : "
        f"{summary['n_folds']}"
    )

    print(
        f"OOF subjects : "
        f"{summary['n_oof_subjects']}"
    )

    print(
        f"Threshold    : "
        f"{summary['threshold']}"
    )

    print()
    print(
        "FOLD MEAN ± SD"
    )

    print(
        "-" * 78
    )

    fold_summary = (
        summary[
            "fold_metric_summary"
        ]
    )

    for metric in (
        PERFORMANCE_METRICS
    ):

        if metric not in fold_summary:
            continue

        info = fold_summary[
            metric
        ]

        print(
            f"{metric:20s}: "
            f"{info['mean']:.6f} "
            f"± "
            f"{info['std']:.6f}"
        )

    print()
    print(
        "GLOBAL OOF PERFORMANCE"
    )

    print(
        "-" * 78
    )

    global_metrics = (
        summary[
            "global_oof_metrics"
        ]
    )

    for metric in (
        PERFORMANCE_METRICS
    ):

        if metric not in global_metrics:
            continue

        value = global_metrics[
            metric
        ]

        print(
            f"{metric:20s}: "
            f"{value:.6f}"
        )


# =====================================================================
# CLI
# =====================================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate K-fold CV predictions into "
            "fold-level and global OOF results."
        )
    )

    parser.add_argument(
        "--fold-predictions",
        required=True,
        nargs="+",
        type=Path,
        help=(
            "Validation prediction CSV for each fold, "
            "in fold order."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
    )

    parser.add_argument(
        "--expected-folds",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--expected-subjects",
        default=None,
        type=int,
        help=(
            "For this project's complete OOF evaluation "
            "this should normally be 1362."
        ),
    )

    parser.add_argument(
        "--ece-bins",
        default=10,
        type=int,
    )

    args = parser.parse_args()

    result = aggregate_cv_results(
        prediction_paths=(
            args.fold_predictions
        ),

        output_dir=(
            args.output_dir
        ),

        threshold=(
            args.threshold
        ),

        expected_folds=(
            args.expected_folds
        ),

        expected_subjects=(
            args.expected_subjects
        ),

        ece_bins=(
            args.ece_bins
        ),
    )

    print_cv_summary(
        result
    )


if __name__ == "__main__":
    main()