"""
probability_calibration.py

Authoritative post-hoc probability calibration for the DaT-SPECT project.

Competition objective
---------------------
The DrivenData challenge is ranked by Log Loss, so final predictions must be
well-calibrated probabilities in [0, 1]. This module is intentionally separate
from neural-network training and is used only after OOF predictions exist.

Canonical calibration input
---------------------------
The canonical input to every calibrator is an uncalibrated probability.

For parametric methods we internally convert probability -> logit:

    temperature:
        sigmoid(logit(p) / T)

    platt:
        sigmoid(a * logit(p) + b), with a > 0

For a single sigmoid model, logit(sigmoid(raw_logit)) == raw_logit (up to
numerical precision), so this is equivalent to calibrating the raw logit.
Using probability as the public interface is also compatible with later
probability averaging / ensembling.

Supported methods
-----------------
none
    Identity mapping.

temperature
    One-parameter monotonic temperature scaling.

platt
    Two-parameter monotonic logistic calibration with positive slope.

isotonic
    Monotonic non-parametric calibration.

Evaluation vs deployment
------------------------
For unbiased method selection, calibration must be evaluated by cross-fitting:
fit the calibrator without the held-out subjects, then apply it to them.

After the final predictor + calibration method is selected, a final calibrator
may be fitted on all available OOF predictions and saved for inference.

NEVER fit calibration using competition/test labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from torch import Tensor


CalibrationMethod = Literal[
    "none",
    "temperature",
    "platt",
    "isotonic",
]


# =====================================================================
# INPUT HELPERS
# =====================================================================


def _as_numpy_1d(values, *, name: str) -> np.ndarray:
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()

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

    return array


def _validate_labels(labels) -> np.ndarray:
    labels = _as_numpy_1d(labels, name="labels")
    unique = set(np.unique(labels).tolist())

    if not unique.issubset({0, 1, 0.0, 1.0}):
        raise ValueError("Labels must contain only 0/1.")

    if len(unique) < 2:
        raise ValueError("Calibration requires both classes.")

    return labels.astype(np.float64, copy=False)


def _validate_probabilities(probabilities) -> np.ndarray:
    probabilities = _as_numpy_1d(probabilities, name="probabilities")

    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise ValueError(
            "Probabilities must be within [0,1]. "
            f"Observed range [{probabilities.min()}, {probabilities.max()}]."
        )

    return probabilities


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def probability_to_logit(probabilities, *, epsilon: float = 1e-6) -> np.ndarray:
    """Numerically stable logit transform."""
    if not (0.0 < epsilon < 0.5):
        raise ValueError("epsilon must satisfy 0 < epsilon < 0.5.")

    p = _validate_probabilities(probabilities)
    p = np.clip(p, epsilon, 1.0 - epsilon)
    return np.log(p) - np.log1p(-p)


def _resolve_probabilities(*, probabilities=None, logits=None) -> np.ndarray:
    """
    Resolve the canonical probability input.

    Probabilities are preferred when both are supplied. Logits are accepted as
    a convenience and converted with sigmoid.
    """
    if probabilities is not None:
        return _validate_probabilities(probabilities)

    if logits is None:
        raise ValueError("Provide probabilities or logits.")

    logits_np = _as_numpy_1d(logits, name="logits")
    return _sigmoid(logits_np)


# =====================================================================
# CALIBRATOR
# =====================================================================


@dataclass
class ProbabilityCalibrator:
    """Fitted post-hoc probability calibrator."""

    method: CalibrationMethod

    # Fixed clipping used for numerical safety and to avoid exact 0/1 outputs.
    output_epsilon: float = 1e-6

    # Temperature scaling parameter.
    temperature: float = 1.0

    # Platt parameters:
    # calibrated = sigmoid(platt_a * logit(p) + platt_b)
    # platt_a is constrained positive during fitting.
    platt_a: float = 1.0
    platt_b: float = 0.0

    # Isotonic thresholds.
    isotonic_x: tuple[float, ...] = ()
    isotonic_y: tuple[float, ...] = ()

    fitted: bool = False

    def __post_init__(self) -> None:
        if self.method not in {"none", "temperature", "platt", "isotonic"}:
            raise ValueError(f"Unknown calibration method: {self.method}")

        if not (0.0 < float(self.output_epsilon) < 0.5):
            raise ValueError("output_epsilon must satisfy 0 < epsilon < 0.5.")

    @property
    def calibration_input(self) -> str:
        """
        Input expected by inference.py.

        All methods use the uncalibrated probability as their public input.
        """
        return "probability"

    # =================================================================
    # FIT
    # =================================================================

    def fit(
        self,
        *,
        labels,
        probabilities=None,
        logits=None,
        max_iter: int = 150,
    ) -> "ProbabilityCalibrator":
        y = _validate_labels(labels)
        p = _resolve_probabilities(probabilities=probabilities, logits=logits)

        if len(p) != len(y):
            raise ValueError("Prediction/label lengths differ.")

        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        if self.method == "none":
            self.fitted = True
            return self

        if self.method == "temperature":
            z = probability_to_logit(p, epsilon=self.output_epsilon)
            self.temperature = self._fit_temperature(
                logits=z,
                labels=y,
                max_iter=max_iter,
            )
            self.fitted = True
            return self

        if self.method == "platt":
            z = probability_to_logit(p, epsilon=self.output_epsilon)
            self.platt_a, self.platt_b = self._fit_platt(
                logits=z,
                labels=y,
                max_iter=max_iter,
            )
            self.fitted = True
            return self

        if self.method == "isotonic":
            model = IsotonicRegression(
                y_min=float(self.output_epsilon),
                y_max=float(1.0 - self.output_epsilon),
                increasing=True,
                out_of_bounds="clip",
            )
            model.fit(p, y)
            self.isotonic_x = tuple(float(v) for v in model.X_thresholds_)
            self.isotonic_y = tuple(float(v) for v in model.y_thresholds_)
            self.fitted = True
            return self

        raise RuntimeError(f"Unhandled calibration method: {self.method}")

    @staticmethod
    def _fit_temperature(
        *,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int,
    ) -> float:
        """Fit positive T for sigmoid(logit / T)."""
        x = torch.tensor(logits, dtype=torch.float64).reshape(-1, 1)
        y = torch.tensor(labels, dtype=torch.float64).reshape(-1, 1)

        log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [log_temperature],
            lr=0.1,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature)
            loss = F.binary_cross_entropy_with_logits(x / temperature, y)
            loss.backward()
            return loss

        optimizer.step(closure)

        temperature = float(torch.exp(log_temperature.detach()).item())
        if not np.isfinite(temperature):
            raise RuntimeError("Temperature optimization produced a non-finite value.")

        # Defensive range: avoids pathological numerical solutions.
        return float(np.clip(temperature, 0.05, 20.0))

    @staticmethod
    def _fit_platt(
        *,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int,
    ) -> tuple[float, float]:
        """
        Fit monotonic Platt scaling:

            sigmoid(a * logit + b), with a > 0.

        Positive slope preserves ranking, keeping calibration conceptually
        separate from discrimination/model selection.
        """
        x = torch.tensor(logits, dtype=torch.float64).reshape(-1, 1)
        y = torch.tensor(labels, dtype=torch.float64).reshape(-1, 1)

        log_a = torch.zeros((), dtype=torch.float64, requires_grad=True)
        b = torch.zeros((), dtype=torch.float64, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [log_a, b],
            lr=0.1,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            a = torch.exp(log_a)
            calibrated_logits = a * x + b
            loss = F.binary_cross_entropy_with_logits(calibrated_logits, y)
            loss.backward()
            return loss

        optimizer.step(closure)

        a_value = float(torch.exp(log_a.detach()).item())
        b_value = float(b.detach().item())

        if not np.isfinite(a_value) or not np.isfinite(b_value):
            raise RuntimeError("Platt optimization produced non-finite parameters.")

        # Very wide defensive bounds; normal fitted values should be far inside.
        a_value = float(np.clip(a_value, 1e-4, 1e4))
        b_value = float(np.clip(b_value, -50.0, 50.0))
        return a_value, b_value

    # =================================================================
    # APPLY
    # =================================================================

    def predict_proba(self, probabilities) -> np.ndarray:
        """Apply the fitted calibrator to uncalibrated probabilities."""
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted.")

        p = _validate_probabilities(probabilities)

        if self.method == "none":
            calibrated = p

        elif self.method == "temperature":
            z = probability_to_logit(p, epsilon=self.output_epsilon)
            calibrated = _sigmoid(z / self.temperature)

        elif self.method == "platt":
            z = probability_to_logit(p, epsilon=self.output_epsilon)
            calibrated = _sigmoid(self.platt_a * z + self.platt_b)

        elif self.method == "isotonic":
            if not self.isotonic_x or not self.isotonic_y:
                raise RuntimeError("Isotonic calibrator has no fitted thresholds.")

            calibrated = np.interp(
                p,
                np.asarray(self.isotonic_x, dtype=np.float64),
                np.asarray(self.isotonic_y, dtype=np.float64),
                left=float(self.isotonic_y[0]),
                right=float(self.isotonic_y[-1]),
            )

        else:
            raise RuntimeError(f"Unknown method: {self.method}")

        calibrated = np.asarray(calibrated, dtype=np.float64)
        return np.clip(
            calibrated,
            float(self.output_epsilon),
            float(1.0 - self.output_epsilon),
        )

    def predict_from(self, *, probabilities=None, logits=None) -> np.ndarray:
        """Convenience wrapper accepting probabilities or logits."""
        p = _resolve_probabilities(probabilities=probabilities, logits=logits)
        return self.predict_proba(p)

    def __call__(self, values: Tensor) -> Tensor:
        """
        Callable interface compatible with InferenceEngine.

        `values` must be the uncalibrated probability tensor because
        calibration_input == "probability".
        """
        probability = self.predict_proba(values)
        return torch.tensor(probability, dtype=torch.float32).reshape(-1, 1)

    # =================================================================
    # SERIALIZATION
    # =================================================================

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "output_epsilon": float(self.output_epsilon),
            "temperature": float(self.temperature),
            "platt_a": float(self.platt_a),
            "platt_b": float(self.platt_b),
            "isotonic_x": list(self.isotonic_x),
            "isotonic_y": list(self.isotonic_y),
            "fitted": bool(self.fitted),
            "calibration_input": self.calibration_input,
            "parametric_definition": (
                "temperature/platt operate on logit(uncalibrated_probability)"
            ),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, sort_keys=True)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ProbabilityCalibrator":
        path = Path(path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)

        return cls(
            method=state["method"],
            output_epsilon=float(state.get("output_epsilon", 1e-6)),
            temperature=float(state.get("temperature", 1.0)),
            platt_a=float(state.get("platt_a", 1.0)),
            platt_b=float(state.get("platt_b", 0.0)),
            isotonic_x=tuple(float(v) for v in state.get("isotonic_x", [])),
            isotonic_y=tuple(float(v) for v in state.get("isotonic_y", [])),
            fitted=bool(state.get("fitted", False)),
        )


# =====================================================================
# FINAL-DEPLOYMENT HELPER
# =====================================================================


def fit_calibrator_from_oof_csv(
    oof_csv: str | Path,
    *,
    method: CalibrationMethod = "temperature",
    output_epsilon: float = 1e-6,
    max_iter: int = 150,
) -> ProbabilityCalibrator:
    """
    Fit one final calibrator using ALL OOF predictions.

    Use this only AFTER the calibration method has been selected by an
    unbiased cross-fitted evaluation.

    Expected columns:
        uid
        is_pathologic
        probability

    A `logit` column may be present but is not required because probability is
    the canonical calibration input.
    """
    path = Path(oof_csv).expanduser().resolve()
    df = pd.read_csv(path)

    required = {"uid", "is_pathologic", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OOF file is missing columns: {sorted(missing)}")

    if df["uid"].astype(str).duplicated().any():
        raise ValueError("OOF predictions contain duplicate UIDs.")

    calibrator = ProbabilityCalibrator(
        method=method,
        output_epsilon=output_epsilon,
    )
    calibrator.fit(
        labels=df["is_pathologic"].to_numpy(),
        probabilities=df["probability"].to_numpy(),
        max_iter=max_iter,
    )
    return calibrator
