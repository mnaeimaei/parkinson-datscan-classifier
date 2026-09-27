"""
evaluate.py

Evaluation of one binary-classification prediction set.

Typical uses
------------
1. Evaluate one CV validation fold.
2. Evaluate complete OOF predictions.
3. Evaluate calibrated predictions.
4. Evaluate an inference prediction CSV when labels are available.

Outputs
-------
predictions.csv
metrics.json
confusion_matrix.csv

No model training occurs here.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_binary_metrics,
    sigmoid_numpy,
    validate_labels,
    validate_probabilities,
)


# =====================================================================
# RESULT
# =====================================================================


@dataclass
class EvaluationResult:
    """
    Complete evaluation result.
    """

    metrics: dict[
        str,
        float | int,
    ]

    predictions: pd.DataFrame

    threshold: float


# =====================================================================
# HELPERS
# =====================================================================


def _json_safe(
    value: Any,
) -> Any:
    """
    Convert NaN/Inf/NumPy values into JSON-safe forms.
    """

    if isinstance(
        value,
        dict,
    ):

        return {
            key: _json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):

        return [
            _json_safe(
                item
            )
            for item
            in value
        ]

    if isinstance(
        value,
        np.integer,
    ):

        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):

        value = float(
            value
        )

    if isinstance(
        value,
        float,
    ):

        if not math.isfinite(
            value
        ):
            return None

    return value


def _as_uid_list(
    uids: Sequence[Any],
) -> list[str]:
    """
    Normalize subject identifiers.
    """

    normalized = [
        str(uid)
        for uid in uids
    ]

    if not normalized:

        raise ValueError(
            "UID list is empty."
        )

    if len(
        normalized
    ) != len(
        set(normalized)
    ):

        duplicates = sorted(
            {
                uid
                for uid in normalized
                if normalized.count(uid) > 1
            }
        )

        raise ValueError(
            "Duplicate UIDs detected:\n"
            + "\n".join(
                f"  - {uid}"
                for uid
                in duplicates[:20]
            )
        )

    return normalized


# =====================================================================
# CORE EVALUATION
# =====================================================================


def evaluate_predictions(
    *,
    uids: Sequence[Any],
    labels: Any,
    probabilities: Optional[
        Any
    ] = None,
    logits: Optional[
        Any
    ] = None,
    threshold: float = 0.5,
    fold: Optional[
        int | str
    ] = None,
    ece_bins: int = 10,
) -> EvaluationResult:
    """
    Evaluate one prediction set.

    At least one of:
        probabilities
        logits

    must be supplied.

    If only logits are supplied:
        probabilities = sigmoid(logits)
    """

    uid_list = _as_uid_list(
        uids
    )

    y_true = validate_labels(
        labels
    )

    if (
        probabilities is None
        and logits is None
    ):

        raise ValueError(
            "Either probabilities or logits must be provided."
        )

    # -------------------------------------------------------------
    # Probability
    # -------------------------------------------------------------

    if probabilities is not None:

        y_probability = (
            validate_probabilities(
                probabilities
            )
        )

    else:

        y_probability = (
            sigmoid_numpy(
                logits
            )
        )

    # -------------------------------------------------------------
    # Logits
    # -------------------------------------------------------------

    if logits is not None:

        y_logits = np.asarray(
            logits
        )

        if (
            y_logits.ndim == 2
            and y_logits.shape[1] == 1
        ):

            y_logits = (
                y_logits[:, 0]
            )

        if y_logits.ndim != 1:

            raise ValueError(
                "logits must have shape [N] or [N,1]."
            )

        y_logits = y_logits.astype(
            np.float64
        )

        if not np.isfinite(
            y_logits
        ).all():

            raise ValueError(
                "Logits contain NaN or infinite values."
            )

    else:

        # Recover logit only for output convenience.
        #
        # Clip to avoid +/-inf at probability 0/1.
        clipped = np.clip(
            y_probability,
            1e-12,
            1.0 - 1e-12,
        )

        y_logits = np.log(
            clipped
            / (
                1.0
                - clipped
            )
        )

    n_samples = len(
        uid_list
    )

    if (
        len(y_true) != n_samples
        or len(y_probability)
        != n_samples
        or len(y_logits)
        != n_samples
    ):

        raise ValueError(
            "UID, label, logit, and probability counts "
            "must all match."
        )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    metrics = compute_binary_metrics(
        y_true,
        y_probability,
        threshold=threshold,
        ece_bins=ece_bins,
    )

    # -------------------------------------------------------------
    # Per-subject result
    # -------------------------------------------------------------

    predicted_labels = (
        y_probability
        >= threshold
    ).astype(
        np.int64
    )

    predictions_df = pd.DataFrame(
        {
            "uid": uid_list,

            "is_pathologic": (
                y_true.astype(
                    np.int64
                )
            ),

            "logit": (
                y_logits.astype(
                    np.float64
                )
            ),

            "probability": (
                y_probability.astype(
                    np.float64
                )
            ),

            "predicted_label": (
                predicted_labels
            ),

            "correct": (
                predicted_labels
                == y_true
            ).astype(
                np.int64
            ),
        }
    )

    if fold is not None:

        predictions_df.insert(
            1,
            "fold",
            fold,
        )

    return EvaluationResult(
        metrics=metrics,
        predictions=predictions_df,
        threshold=float(
            threshold
        ),
    )


# =====================================================================
# SAVE RESULT
# =====================================================================


def save_evaluation_result(
    result: EvaluationResult,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Save evaluation artifacts.

    Files
    -----
    predictions.csv
    metrics.json
    confusion_matrix.csv
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

    predictions_path = (
        output_dir
        / "predictions.csv"
    )

    metrics_path = (
        output_dir
        / "metrics.json"
    )

    confusion_path = (
        output_dir
        / "confusion_matrix.csv"
    )

    # -------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------

    result.predictions.to_csv(
        predictions_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            _json_safe(
                result.metrics
            ),
            file,
            indent=2,
            sort_keys=True,
        )

    # -------------------------------------------------------------
    # Confusion matrix
    #
    # Rows = true
    # Columns = predicted
    # -------------------------------------------------------------

    confusion_df = pd.DataFrame(
        [
            [
                result.metrics[
                    "tn"
                ],
                result.metrics[
                    "fp"
                ],
            ],
            [
                result.metrics[
                    "fn"
                ],
                result.metrics[
                    "tp"
                ],
            ],
        ],
        index=[
            "true_normal_0",
            "true_pathologic_1",
        ],
        columns=[
            "pred_normal_0",
            "pred_pathologic_1",
        ],
    )

    confusion_df.to_csv(
        confusion_path
    )

    return {
        "predictions": (
            predictions_path
        ),
        "metrics": (
            metrics_path
        ),
        "confusion_matrix": (
            confusion_path
        ),
    }


# =====================================================================
# READ EXISTING PREDICTION CSV
# =====================================================================


def read_prediction_csv(
    path: str | Path,
) -> pd.DataFrame:
    """
    Read prediction CSV and normalize columns into:

        uid
        is_pathologic
        logit
        probability

    Supported probability columns include:

        probability
        final_probability
        mean_probability

    Supported logit columns:

        logit
        mean_logit

    Supported label columns:

        is_pathologic
        label
        target
    """

    path = (
        Path(
            path
        )
        .expanduser()
        .resolve()
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    df = pd.read_csv(
        path
    )

    if "uid" not in df.columns:

        raise ValueError(
            f"{path} does not contain 'uid'."
        )

    # -------------------------------------------------------------
    # Label
    # -------------------------------------------------------------

    label_column = None

    for candidate in (
        "is_pathologic",
        "label",
        "target",
    ):

        if candidate in df.columns:

            label_column = (
                candidate
            )

            break

    if label_column is None:

        raise ValueError(
            f"{path} does not contain ground-truth labels."
        )

    # -------------------------------------------------------------
    # Probability
    # -------------------------------------------------------------

    probability_column = None

    for candidate in (
        "probability",
        "final_probability",
        "mean_probability",
    ):

        if candidate in df.columns:

            probability_column = (
                candidate
            )

            break

    # -------------------------------------------------------------
    # Logit
    # -------------------------------------------------------------

    logit_column = None

    for candidate in (
        "logit",
        "mean_logit",
    ):

        if candidate in df.columns:

            logit_column = (
                candidate
            )

            break

    if (
        probability_column
        is None
        and logit_column
        is None
    ):

        raise ValueError(
            f"{path} contains neither probability nor logit."
        )

    canonical = pd.DataFrame(
        {
            "uid": (
                df["uid"]
                .astype(str)
            ),

            "is_pathologic": (
                df[
                    label_column
                ]
            ),
        }
    )

    if logit_column is not None:

        canonical[
            "logit"
        ] = df[
            logit_column
        ]

    if probability_column is not None:

        canonical[
            "probability"
        ] = df[
            probability_column
        ]

    if (
        "probability"
        not in canonical.columns
    ):

        canonical[
            "probability"
        ] = sigmoid_numpy(
            canonical[
                "logit"
            ].to_numpy()
        )

    if (
        "logit"
        not in canonical.columns
    ):

        probability = np.clip(
            canonical[
                "probability"
            ].to_numpy(
                dtype=np.float64
            ),
            1e-12,
            1.0 - 1e-12,
        )

        canonical[
            "logit"
        ] = np.log(
            probability
            / (
                1.0
                - probability
            )
        )

    return canonical


# =====================================================================
# EVALUATE CSV
# =====================================================================


def evaluate_prediction_csv(
    input_csv: str | Path,
    *,
    output_dir: Optional[
        str | Path
    ] = None,
    threshold: float = 0.5,
    fold: Optional[
        int | str
    ] = None,
) -> EvaluationResult:
    """
    Evaluate an existing prediction CSV.
    """

    df = read_prediction_csv(
        input_csv
    )

    result = evaluate_predictions(
        uids=df[
            "uid"
        ].tolist(),

        labels=df[
            "is_pathologic"
        ].to_numpy(),

        probabilities=df[
            "probability"
        ].to_numpy(),

        logits=df[
            "logit"
        ].to_numpy(),

        threshold=threshold,

        fold=fold,
    )

    if output_dir is not None:

        save_evaluation_result(
            result,
            output_dir=output_dir,
        )

    return result


# =====================================================================
# CLI
# =====================================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate binary-classification predictions."
        )
    )

    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Prediction CSV.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Evaluation output directory.",
    )

    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
    )

    parser.add_argument(
        "--fold",
        default=None,
    )

    args = parser.parse_args()

    result = (
        evaluate_prediction_csv(
            args.predictions,
            output_dir=(
                args.output_dir
            ),
            threshold=(
                args.threshold
            ),
            fold=args.fold,
        )
    )

    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)

    for key, value in (
        result.metrics.items()
    ):

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key:22s}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"{key:22s}: "
                f"{value}"
            )


if __name__ == "__main__":
    main()