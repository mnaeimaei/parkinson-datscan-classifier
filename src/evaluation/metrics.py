"""
metrics.py

Binary-classification metrics for the DaT-SPECT project.

Ground truth
------------
0 = normal
1 = pathologic

Inputs
------
y_true:
    binary ground-truth labels

y_probability:
    probability of class 1 (pathologic)

threshold:
    probability threshold used for hard classification

Default:
    threshold = 0.5

Metrics
-------
Threshold independent:
    AUROC
    AUPRC
    Average Precision
    Brier score
    Log loss
    ECE

Threshold dependent:
    Accuracy
    Balanced accuracy
    Sensitivity / Recall
    Specificity
    Precision / PPV
    NPV
    F1
    MCC
    TP / TN / FP / FN

Important
---------
A threshold must NOT be independently optimized on each CV validation
fold when reporting unbiased CV results.

Use either:
    - frozen threshold = 0.5

or later:
    - one threshold selected according to a separately frozen policy.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =====================================================================
# METRIC NAMES USED FOR CV AGGREGATION
# =====================================================================

PERFORMANCE_METRICS = (
    "auroc",
    "auprc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "npv",
    "f1",
    "mcc",
    "brier_score",
    "log_loss",
    "ece",
)


# =====================================================================
# INPUT CONVERSION
# =====================================================================


def _to_numpy_1d(
    values: Any,
    *,
    name: str,
) -> np.ndarray:
    """
    Convert Tensor/list/ndarray into a 1-D NumPy array.
    """

    if isinstance(
        values,
        Tensor,
    ):

        values = (
            values
            .detach()
            .cpu()
            .numpy()
        )

    array = np.asarray(
        values
    )

    if array.ndim == 2:

        if array.shape[1] != 1:

            raise ValueError(
                f"{name} must have shape [N] or [N,1], "
                f"received {array.shape}."
            )

        array = array[:, 0]

    elif array.ndim != 1:

        raise ValueError(
            f"{name} must have shape [N] or [N,1], "
            f"received {array.shape}."
        )

    return array


# =====================================================================
# VALIDATION
# =====================================================================


def validate_labels(
    y_true: Any,
) -> np.ndarray:
    """
    Validate binary labels and return int64 [N].
    """

    labels = _to_numpy_1d(
        y_true,
        name="y_true",
    )

    if labels.size == 0:

        raise ValueError(
            "y_true is empty."
        )

    if not np.isfinite(
        labels
    ).all():

        raise ValueError(
            "y_true contains NaN or infinite values."
        )

    unique_values = set(
        np.unique(
            labels
        ).tolist()
    )

    if not unique_values.issubset(
        {0, 1, 0.0, 1.0}
    ):

        raise ValueError(
            "Binary labels must contain only 0 and 1. "
            f"Received: {sorted(unique_values)}"
        )

    return labels.astype(
        np.int64,
        copy=False,
    )


def validate_probabilities(
    y_probability: Any,
) -> np.ndarray:
    """
    Validate class-1 probabilities and return float64 [N].
    """

    probabilities = _to_numpy_1d(
        y_probability,
        name="y_probability",
    ).astype(
        np.float64,
        copy=False,
    )

    if probabilities.size == 0:

        raise ValueError(
            "y_probability is empty."
        )

    if not np.isfinite(
        probabilities
    ).all():

        raise ValueError(
            "Probabilities contain NaN or infinite values."
        )

    if (
        probabilities.min() < 0.0
        or probabilities.max() > 1.0
    ):

        raise ValueError(
            "Probabilities must be within [0,1]. "
            f"Observed range: "
            f"[{probabilities.min()}, "
            f"{probabilities.max()}]"
        )

    return probabilities


# =====================================================================
# LOGITS -> PROBABILITY
# =====================================================================


def sigmoid_numpy(
    logits: Any,
) -> np.ndarray:
    """
    Numerically stable sigmoid.

    Input:
        [N] or [N,1] logits

    Output:
        [N] probabilities
    """

    x = _to_numpy_1d(
        logits,
        name="logits",
    ).astype(
        np.float64,
        copy=False,
    )

    if not np.isfinite(
        x
    ).all():

        raise ValueError(
            "Logits contain NaN or infinite values."
        )

    output = np.empty_like(
        x,
        dtype=np.float64,
    )

    positive = x >= 0

    output[positive] = (
        1.0
        / (
            1.0
            + np.exp(
                -x[positive]
            )
        )
    )

    exp_x = np.exp(
        x[~positive]
    )

    output[~positive] = (
        exp_x
        / (
            1.0
            + exp_x
        )
    )

    return output


# =====================================================================
# CALIBRATION ERROR
# =====================================================================


def expected_calibration_error(
    y_true: Any,
    y_probability: Any,
    *,
    n_bins: int = 10,
) -> float:
    """
    Fixed-width Expected Calibration Error (ECE).

    Bins probability predictions into equal-width intervals.

    For each bin:

        |mean confidence - observed positive fraction|

    weighted by the proportion of samples in that bin.

    Lower is better.
    """

    if n_bins < 2:

        raise ValueError(
            "n_bins must be >= 2."
        )

    labels = validate_labels(
        y_true
    )

    probabilities = (
        validate_probabilities(
            y_probability
        )
    )

    if (
        labels.shape[0]
        != probabilities.shape[0]
    ):

        raise ValueError(
            "Label/probability sample counts do not match."
        )

    # -------------------------------------------------------------
    # Produce bin IDs:
    #
    # [0, 1/n_bins)
    # ...
    # final bin includes 1.0
    # -------------------------------------------------------------

    boundaries = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_ids = np.digitize(
        probabilities,
        boundaries[1:-1],
        right=False,
    )

    n_samples = len(
        labels
    )

    ece = 0.0

    for bin_index in range(
        n_bins
    ):

        mask = (
            bin_ids
            == bin_index
        )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        mean_probability = float(
            probabilities[
                mask
            ].mean()
        )

        observed_frequency = float(
            labels[
                mask
            ].mean()
        )

        bin_weight = (
            count
            / n_samples
        )

        ece += (
            bin_weight
            * abs(
                mean_probability
                - observed_frequency
            )
        )

    return float(
        ece
    )


# =====================================================================
# SAFE DIVISION
# =====================================================================


def _safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Return NaN if denominator is zero.
    """

    if denominator == 0:
        return float("nan")

    return float(
        numerator
        / denominator
    )


# =====================================================================
# MAIN METRIC FUNCTION
# =====================================================================


def compute_binary_metrics(
    y_true: Any,
    y_probability: Any,
    *,
    threshold: float = 0.5,
    ece_bins: int = 10,
) -> dict[str, float | int]:
    """
    Calculate complete binary classification metrics.

    Parameters
    ----------
    y_true:
        Ground truth labels.

    y_probability:
        Probability of class 1.

    threshold:
        Frozen hard-classification threshold.

    ece_bins:
        Number of ECE bins.

    Returns
    -------
    dict
        Complete metric dictionary.
    """

    if not 0.0 <= threshold <= 1.0:

        raise ValueError(
            "threshold must satisfy 0 <= threshold <= 1."
        )

    labels = validate_labels(
        y_true
    )

    probabilities = (
        validate_probabilities(
            y_probability
        )
    )

    if (
        labels.shape[0]
        != probabilities.shape[0]
    ):

        raise ValueError(
            "y_true and y_probability must contain "
            "the same number of samples."
        )

    n_samples = int(
        labels.shape[0]
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    # -------------------------------------------------------------
    # Confusion matrix.
    #
    # Explicit labels=[0,1] guarantees a 2x2 matrix.
    # -------------------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        int(value)
        for value
        in cm.ravel()
    )

    n_negative = int(
        (labels == 0).sum()
    )

    n_positive = int(
        (labels == 1).sum()
    )

    prevalence = (
        n_positive
        / n_samples
    )

    # -------------------------------------------------------------
    # Threshold-independent discrimination metrics.
    #
    # AUROC/AUPRC require both classes to be present.
    # -------------------------------------------------------------

    has_both_classes = (
        n_negative > 0
        and n_positive > 0
    )

    if has_both_classes:

        auroc = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )

        average_precision = float(
            average_precision_score(
                labels,
                probabilities,
            )
        )

        precision_curve, recall_curve, _ = (
            precision_recall_curve(
                labels,
                probabilities,
            )
        )

        auprc = float(
            auc(
                recall_curve,
                precision_curve,
            )
        )

    else:

        auroc = float("nan")
        auprc = float("nan")
        average_precision = float(
            "nan"
        )

    # -------------------------------------------------------------
    # Threshold-based metrics.
    # -------------------------------------------------------------

    accuracy = float(
        accuracy_score(
            labels,
            predictions,
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            labels,
            predictions,
        )
    )

    sensitivity = float(
        recall_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        )
    )

    precision = float(
        precision_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        )
    )

    specificity = _safe_divide(
        tn,
        tn + fp,
    )

    npv = _safe_divide(
        tn,
        tn + fn,
    )

    # MCC may be zero for degenerate predictions.
    mcc = float(
        matthews_corrcoef(
            labels,
            predictions,
        )
    )

    # -------------------------------------------------------------
    # Probability/calibration metrics.
    # -------------------------------------------------------------

    brier = float(
        brier_score_loss(
            labels,
            probabilities,
        )
    )

    binary_log_loss = float(
        log_loss(
            labels,
            probabilities,
            labels=[0, 1],
        )
    )

    ece = float(
        expected_calibration_error(
            labels,
            probabilities,
            n_bins=ece_bins,
        )
    )

    return {
        # ---------------------------------------------------------
        # Dataset
        # ---------------------------------------------------------
        "n_samples": n_samples,
        "n_negative": n_negative,
        "n_positive": n_positive,
        "prevalence": float(
            prevalence
        ),

        # ---------------------------------------------------------
        # Frozen threshold
        # ---------------------------------------------------------
        "threshold": float(
            threshold
        ),

        # ---------------------------------------------------------
        # Confusion matrix
        # ---------------------------------------------------------
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,

        # ---------------------------------------------------------
        # Ranking/discrimination
        # ---------------------------------------------------------
        "auroc": auroc,
        "auprc": auprc,
        "average_precision": (
            average_precision
        ),

        # ---------------------------------------------------------
        # Threshold-dependent
        # ---------------------------------------------------------
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "ppv": precision,
        "npv": npv,
        "f1": f1,
        "mcc": mcc,

        # ---------------------------------------------------------
        # Probability/calibration
        # ---------------------------------------------------------
        "brier_score": brier,
        "log_loss": binary_log_loss,
        "ece": ece,
    }


# =====================================================================
# METRICS DIRECTLY FROM LOGITS
# =====================================================================


def compute_binary_metrics_from_logits(
    y_true: Any,
    logits: Any,
    *,
    threshold: float = 0.5,
    ece_bins: int = 10,
) -> dict[str, float | int]:
    """
    Convenience wrapper for raw model logits.
    """

    probabilities = sigmoid_numpy(
        logits
    )

    return compute_binary_metrics(
        y_true,
        probabilities,
        threshold=threshold,
        ece_bins=ece_bins,
    )


# =====================================================================
# TRAINER-COMPATIBLE METRIC FUNCTION
# =====================================================================


def make_metric_function(
    metric_name: str,
    *,
    threshold: float = 0.5,
) -> Callable[
    [Tensor, Tensor],
    float,
]:
    """
    Create metric function compatible with:

        Trainer(selection_metric_fn=...)

    Example
    -------
    metric_fn = make_metric_function(
        "auroc"
    )

    trainer = Trainer(
        ...,
        selection_metric_fn=metric_fn,
        selection_metric_name="auroc",
        selection_mode="max",
    )
    """

    metric_name = (
        metric_name
        .lower()
        .strip()
    )

    available = set(
        PERFORMANCE_METRICS
    )

    if metric_name not in available:

        raise ValueError(
            f"Unknown metric {metric_name!r}. "
            f"Available: {sorted(available)}"
        )

    def metric_fn(
        targets: Tensor,
        probabilities: Tensor,
    ) -> float:

        metrics = compute_binary_metrics(
            targets,
            probabilities,
            threshold=threshold,
        )

        value = metrics[
            metric_name
        ]

        return float(
            value
        )

    return metric_fn


# =====================================================================
# METRIC DIRECTION
# =====================================================================


def metric_direction(
    metric_name: str,
) -> str:
    """
    Return whether higher or lower is better.
    """

    metric_name = (
        metric_name
        .lower()
        .strip()
    )

    lower_is_better = {
        "brier_score",
        "log_loss",
        "ece",
    }

    if metric_name in lower_is_better:
        return "min"

    if metric_name in PERFORMANCE_METRICS:
        return "max"

    raise ValueError(
        f"Unknown metric: {metric_name}"
    )