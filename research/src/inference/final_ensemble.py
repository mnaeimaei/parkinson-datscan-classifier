"""
final_ensemble.py

Frozen competition-ensemble aggregation layer.

Deployment contract
-------------------
Input:
    one RAW, uncalibrated probability vector per frozen member model.

Pipeline:
    raw member probability
        -> member-specific final calibrator
        -> selected member probability
        -> equal-weight logit mean across members
        -> final ensemble calibrator
        -> final probability in [epsilon, 1-epsilon]

For the current frozen predictor:
    ENS328 = P1 + P2 + P3 + P7 + P8
    aggregation = logit_mean
    final ensemble calibration = temperature

The module is intentionally model-agnostic. The model/checkpoint inference
layer only has to provide one raw probability per member and subject.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.calibration.probability_calibration import ProbabilityCalibrator


def _as_probability_vector(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)

    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]

    if array.ndim != 1:
        raise ValueError(
            f"{name} must have shape [N] or [N,1], received {array.shape}."
        )

    if array.size == 0:
        raise ValueError(f"{name} is empty.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf.")

    if array.min() < 0.0 or array.max() > 1.0:
        raise ValueError(
            f"{name} must be within [0,1], observed "
            f"[{array.min()}, {array.max()}]."
        )

    return array


def probability_to_logit(
    probability: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    p = np.clip(
        np.asarray(probability, dtype=np.float64),
        epsilon,
        1.0 - epsilon,
    )
    return np.log(p) - np.log1p(-p)


def sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.empty_like(x)

    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))

    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)

    return out


class FinalEnsemblePredictor:
    """
    Apply the frozen member-calibration -> ensemble -> final-calibration chain.

    Parameters are loaded only from the frozen JSON configuration and the
    serialized ProbabilityCalibrator JSON files referenced by it.
    """

    def __init__(
        self,
        *,
        config_path: str | Path,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()

        with self.config_path.open("r", encoding="utf-8") as file:
            self.config = json.load(file)

        self.config_dir = self.config_path.parent

        self.members = tuple(self.config["ensemble"]["members"])
        self.aggregation_rule = str(
            self.config["ensemble"]["aggregation_rule"]
        )
        self.epsilon = float(
            self.config["numerics"].get("probability_epsilon", 1e-6)
        )

        if self.aggregation_rule not in {
            "probability_mean",
            "logit_mean",
        }:
            raise ValueError(
                f"Unsupported aggregation rule: {self.aggregation_rule}"
            )

        member_meta = self.config["members"]
        if set(member_meta) != set(self.members):
            raise ValueError(
                "Member metadata keys do not match frozen ensemble members."
            )

        self.member_calibrators: dict[str, ProbabilityCalibrator] = {}
        for member_id in self.members:
            relative = Path(
                member_meta[member_id]["final_calibrator_path"]
            )
            path = (
                relative
                if relative.is_absolute()
                else self.config_dir / relative
            ).resolve()

            calibrator = ProbabilityCalibrator.load(path)
            if not calibrator.fitted:
                raise RuntimeError(
                    f"{member_id} calibrator is not marked fitted: {path}"
                )
            self.member_calibrators[member_id] = calibrator

        final_relative = Path(
            self.config["ensemble"]["final_calibrator_path"]
        )
        final_path = (
            final_relative
            if final_relative.is_absolute()
            else self.config_dir / final_relative
        ).resolve()

        self.final_calibrator = ProbabilityCalibrator.load(final_path)
        if not self.final_calibrator.fitted:
            raise RuntimeError(
                f"Final ensemble calibrator is not marked fitted: {final_path}"
            )

    def apply_member_calibration(
        self,
        raw_member_probabilities: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Apply each frozen member-specific calibrator.

        Input probabilities must be the RAW outputs produced by the member's
        five-fold inference ensemble, before any post-hoc calibration.
        """
        supplied = set(raw_member_probabilities)
        expected = set(self.members)

        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)

        if missing:
            raise ValueError(
                f"Missing frozen ensemble members: {missing}"
            )
        if extra:
            raise ValueError(
                f"Unexpected ensemble members supplied: {extra}"
            )

        selected: dict[str, np.ndarray] = {}
        expected_length = None

        for member_id in self.members:
            raw = _as_probability_vector(
                raw_member_probabilities[member_id],
                name=f"{member_id} raw probability",
            )

            if expected_length is None:
                expected_length = len(raw)
            elif len(raw) != expected_length:
                raise ValueError(
                    "All member probability vectors must have equal length."
                )

            selected[member_id] = self.member_calibrators[
                member_id
            ].predict_proba(raw)

        return selected

    def aggregate_selected_member_probabilities(
        self,
        selected_member_probabilities: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """
        Aggregate probabilities after member-specific calibration.
        """
        rows = []
        expected_length = None

        for member_id in self.members:
            if member_id not in selected_member_probabilities:
                raise ValueError(
                    f"Missing selected probability for {member_id}."
                )

            p = _as_probability_vector(
                selected_member_probabilities[member_id],
                name=f"{member_id} selected probability",
            )

            if expected_length is None:
                expected_length = len(p)
            elif len(p) != expected_length:
                raise ValueError(
                    "All selected member probability vectors must "
                    "have equal length."
                )

            rows.append(p)

        matrix = np.column_stack(rows)

        if self.aggregation_rule == "probability_mean":
            raw_ensemble = matrix.mean(axis=1)
        else:
            logits = probability_to_logit(
                matrix,
                epsilon=self.epsilon,
            )
            raw_ensemble = sigmoid(logits.mean(axis=1))

        return np.clip(
            raw_ensemble,
            self.epsilon,
            1.0 - self.epsilon,
        )

    def predict_from_selected_member_probabilities(
        self,
        selected_member_probabilities: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """
        Use when member-specific calibration has already been applied.
        """
        raw_ensemble = self.aggregate_selected_member_probabilities(
            selected_member_probabilities
        )
        return self.final_calibrator.predict_proba(raw_ensemble)

    def predict_from_raw_member_probabilities(
        self,
        raw_member_probabilities: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """
        Main deployment entry point.

        raw member probabilities
            -> final member calibrators
            -> frozen aggregation
            -> final ensemble calibrator
        """
        selected = self.apply_member_calibration(
            raw_member_probabilities
        )
        return self.predict_from_selected_member_probabilities(selected)

    def predict_with_intermediates(
        self,
        raw_member_probabilities: Mapping[str, np.ndarray],
    ) -> dict:
        """
        Return final probability plus intermediate values for debugging/QC.
        """
        selected = self.apply_member_calibration(
            raw_member_probabilities
        )
        raw_ensemble = self.aggregate_selected_member_probabilities(
            selected
        )
        final_probability = self.final_calibrator.predict_proba(
            raw_ensemble
        )

        return {
            "selected_member_probabilities": selected,
            "raw_ensemble_probability": raw_ensemble,
            "final_probability": final_probability,
        }
