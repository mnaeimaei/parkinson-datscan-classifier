"""
dual_input_fusion.py

Reusable Scenario-C late-fusion classifier.

Purpose
-------
Scenario C uses two separate inputs from the same subject:

    whole volume
    striatal ROI

Each stream is processed by its own independent copy of the selected
experiment model.  The two subject-level feature vectors are then
concatenated and classified by one binary head.

This module keeps Scenario-C fusion out of experiment scripts so all
models share exactly the same fusion policy.

Branch contract
---------------
Each branch model must expose:

    feature_dim: int
    forward_features(x) -> Tensor [B, feature_dim]

All current project model families expose this interface directly or
through inheritance.

Output
------
    [B, 1] raw logit

No sigmoid is applied internally.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class DualInputFeatureFusionClassifier(nn.Module):
    """
    Generic two-branch late-fusion binary classifier.

    Parameters
    ----------
    whole_branch:
        Model instance used only for the whole-volume stream.

    roi_branch:
        Independent model instance used only for the ROI stream.

    dropout:
        Optional dropout after feature concatenation and before the final
        binary classifier.  Default 0.0 keeps the fusion head minimal.

    Notes
    -----
    The two branches must be distinct Python objects.  Weight sharing is
    intentionally not used because whole and ROI represent different input
    scales and the project design treats them as separate streams.
    """

    def __init__(
        self,
        *,
        whole_branch: nn.Module,
        roi_branch: nn.Module,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if whole_branch is roi_branch:
            raise ValueError(
                "Scenario C requires two independent branch instances. "
                "whole_branch and roi_branch cannot be the same object."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1, "
                f"received {dropout}."
            )

        self.whole_branch = whole_branch
        self.roi_branch = roi_branch

        self.whole_feature_dim = self._feature_dim(
            whole_branch,
            branch_name="whole_branch",
        )
        self.roi_feature_dim = self._feature_dim(
            roi_branch,
            branch_name="roi_branch",
        )

        self.feature_dim = (
            self.whole_feature_dim
            + self.roi_feature_dim
        )

        self.dropout = nn.Dropout(
            p=float(dropout)
        )

        self.classifier = nn.Linear(
            in_features=self.feature_dim,
            out_features=1,
            bias=True,
        )

        nn.init.kaiming_normal_(
            self.classifier.weight,
            mode="fan_in",
            nonlinearity="linear",
        )
        nn.init.zeros_(
            self.classifier.bias
        )

    @staticmethod
    def _feature_dim(
        branch: nn.Module,
        *,
        branch_name: str,
    ) -> int:
        if not hasattr(branch, "forward_features"):
            raise TypeError(
                f"{branch_name} must expose forward_features(x)."
            )

        value = getattr(
            branch,
            "feature_dim",
            None,
        )

        if value is None:
            raise TypeError(
                f"{branch_name} must expose integer feature_dim."
            )

        try:
            feature_dim = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{branch_name}.feature_dim must be an integer."
            ) from exc

        if feature_dim < 1:
            raise ValueError(
                f"{branch_name}.feature_dim must be >= 1."
            )

        return feature_dim

    @staticmethod
    def _validate_features(
        features: Tensor,
        *,
        expected_batch: int,
        expected_dim: int,
        branch_name: str,
    ) -> Tensor:
        if not isinstance(features, Tensor):
            raise TypeError(
                f"{branch_name}.forward_features() must return Tensor."
            )

        if features.ndim != 2:
            raise RuntimeError(
                f"{branch_name} feature output must have shape [B,F], "
                f"received {tuple(features.shape)}."
            )

        if features.shape[0] != expected_batch:
            raise RuntimeError(
                f"{branch_name} feature batch size mismatch."
            )

        if features.shape[1] != expected_dim:
            raise RuntimeError(
                f"{branch_name} feature dimension mismatch. "
                f"Expected {expected_dim}, received {features.shape[1]}."
            )

        if not torch.isfinite(features).all():
            raise FloatingPointError(
                f"{branch_name} produced NaN/Inf features."
            )

        return features

    def forward_features(
        self,
        whole: Tensor,
        roi: Tensor,
    ) -> Tensor:
        """
        Return concatenated whole/ROI subject features.

        Output:
            [B, whole_feature_dim + roi_feature_dim]
        """

        if not isinstance(whole, Tensor):
            raise TypeError("whole must be torch.Tensor.")

        if not isinstance(roi, Tensor):
            raise TypeError("roi must be torch.Tensor.")

        if whole.ndim < 1 or roi.ndim < 1:
            raise ValueError("whole and roi must include a batch dimension.")

        if whole.shape[0] != roi.shape[0]:
            raise RuntimeError(
                "Whole/ROI batch sizes differ: "
                f"whole={whole.shape[0]}, roi={roi.shape[0]}."
            )

        batch_size = int(
            whole.shape[0]
        )

        whole_features = (
            self.whole_branch.forward_features(
                whole
            )
        )

        roi_features = (
            self.roi_branch.forward_features(
                roi
            )
        )

        whole_features = self._validate_features(
            whole_features,
            expected_batch=batch_size,
            expected_dim=self.whole_feature_dim,
            branch_name="whole_branch",
        )

        roi_features = self._validate_features(
            roi_features,
            expected_batch=batch_size,
            expected_dim=self.roi_feature_dim,
            branch_name="roi_branch",
        )

        return torch.cat(
            (
                whole_features,
                roi_features,
            ),
            dim=1,
        )

    def forward(
        self,
        whole: Tensor,
        roi: Tensor,
    ) -> Tensor:
        features = self.forward_features(
            whole,
            roi,
        )

        features = self.dropout(
            features
        )

        logits = self.classifier(
            features
        )

        if (
            logits.ndim != 2
            or logits.shape[1] != 1
        ):
            raise RuntimeError(
                "Scenario-C classifier must return [B,1] logits."
            )

        return logits


def build_dual_input_fusion(
    *,
    whole_branch: nn.Module,
    roi_branch: Optional[nn.Module] = None,
    dropout: float = 0.0,
    deepcopy_roi_from_whole: bool = False,
) -> DualInputFeatureFusionClassifier:
    """
    Factory for Scenario-C late fusion.

    Preferred usage is to build two branches independently and pass both.
    ``deepcopy_roi_from_whole=True`` is provided only as a convenience for
    architectures whose initialization has already been fully resolved.
    It must not be combined with an explicit ``roi_branch``.
    """

    if deepcopy_roi_from_whole:
        if roi_branch is not None:
            raise ValueError(
                "Do not pass roi_branch when deepcopy_roi_from_whole=True."
            )
        roi_branch = copy.deepcopy(
            whole_branch
        )

    if roi_branch is None:
        raise ValueError(
            "roi_branch is required unless deepcopy_roi_from_whole=True."
        )

    return DualInputFeatureFusionClassifier(
        whole_branch=whole_branch,
        roi_branch=roi_branch,
        dropout=dropout,
    )
