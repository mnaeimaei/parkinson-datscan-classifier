"""
losses.py

Loss construction for binary DaT-SPECT classification.

All project models return:
    logits: [B, 1]

Labels:
    0 = normal
    1 = pathologic

The default loss is BCEWithLogitsLoss.

IMPORTANT:
    Models must NOT apply sigmoid internally.

BCEWithLogitsLoss internally combines:
    sigmoid + binary cross entropy
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def build_loss(
    *,
    pos_weight: Optional[float] = None,
    reduction: str = "mean",
) -> nn.Module:
    """
    Build binary classification loss.

    Parameters
    ----------
    pos_weight:
        Optional positive-class weight.

        None:
            standard BCEWithLogitsLoss

        float:
            weighted positive class

        Example:
            pos_weight = n_negative / n_positive

        For this project the class imbalance is modest, so
        pos_weight=None is a reasonable default baseline.

    reduction:
        "mean", "sum", or "none".

    Returns
    -------
    nn.BCEWithLogitsLoss
    """

    if reduction not in {
        "mean",
        "sum",
        "none",
    }:
        raise ValueError(
            "reduction must be one of "
            "{'mean', 'sum', 'none'}, "
            f"received {reduction!r}."
        )

    if pos_weight is None:

        return nn.BCEWithLogitsLoss(
            reduction=reduction,
        )

    if pos_weight <= 0:
        raise ValueError(
            "pos_weight must be > 0, "
            f"received {pos_weight}."
        )

    weight_tensor = torch.tensor(
        [float(pos_weight)],
        dtype=torch.float32,
    )

    return nn.BCEWithLogitsLoss(
        pos_weight=weight_tensor,
        reduction=reduction,
    )


def validate_binary_targets(
    targets: torch.Tensor,
) -> None:
    """
    Verify that a target tensor contains only binary labels.

    Accepted values:
        0
        1
    """

    if targets.numel() == 0:
        raise ValueError(
            "Target tensor is empty."
        )

    if not torch.isfinite(targets).all():
        raise ValueError(
            "Target tensor contains NaN or infinite values."
        )

    unique_values = torch.unique(
        targets.detach()
    )

    valid = torch.logical_or(
        unique_values == 0,
        unique_values == 1,
    )

    if not bool(valid.all()):
        raise ValueError(
            "Binary classification targets must contain "
            "only 0 and 1. "
            f"Received values: {unique_values.tolist()}"
        )


def prepare_binary_targets(
    targets: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert labels into the exact format expected by
    BCEWithLogitsLoss.

    Output:
        shape = [B, 1]
        dtype = float32
    """

    targets = targets.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    if targets.ndim == 1:
        targets = targets.unsqueeze(1)

    elif (
        targets.ndim == 2
        and targets.shape[1] == 1
    ):
        pass

    else:
        raise ValueError(
            "Binary targets must have shape [B] or [B,1], "
            f"received {tuple(targets.shape)}."
        )

    validate_binary_targets(
        targets
    )

    return targets