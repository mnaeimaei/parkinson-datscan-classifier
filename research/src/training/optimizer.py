"""
optimizer.py

Optimizer construction for model training.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import Optimizer


SUPPORTED_OPTIMIZERS = {
    "adamw",
    "adam",
    "sgd",
}


def _get_trainable_parameters(
    model: nn.Module,
) -> list[nn.Parameter]:
    """
    Return trainable model parameters only.
    """

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not parameters:
        raise RuntimeError(
            "Model contains no trainable parameters."
        )

    return parameters


def build_optimizer(
    model: nn.Module,
    *,
    name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    betas: tuple[float, float] = (
        0.9,
        0.999,
    ),
    eps: float = 1e-8,
    momentum: float = 0.9,
) -> Optimizer:
    """
    Build optimizer.

    Default
    -------
    AdamW:
        learning_rate = 1e-4
        weight_decay  = 1e-4

    Parameters
    ----------
    model:
        Model containing parameters to optimize.

    name:
        adamw
        adam
        sgd

    learning_rate:
        Base learning rate.

    weight_decay:
        L2/decoupled weight decay depending on optimizer.

    betas:
        Adam / AdamW beta parameters.

    eps:
        Adam / AdamW epsilon.

    momentum:
        SGD momentum.
    """

    name = name.lower().strip()

    if name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer: {name}. "
            f"Supported: {sorted(SUPPORTED_OPTIMIZERS)}"
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be > 0."
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay must be >= 0."
        )

    parameters = (
        _get_trainable_parameters(
            model
        )
    )

    if name == "adamw":

        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    if name == "adam":

        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    if name == "sgd":

        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=momentum > 0,
        )

    raise RuntimeError(
        "Unreachable optimizer branch."
    )


def get_learning_rates(
    optimizer: Optimizer,
) -> list[float]:
    """
    Return current LR for every optimizer parameter group.
    """

    return [
        float(group["lr"])
        for group in optimizer.param_groups
    ]