"""
inference.py

Generic deterministic inference engine for the DaT-SPECT classifier project.

Responsibilities
----------------
Supports:

    Scenario A:
        whole-volume input

    Scenario B:
        striatal ROI input

    Scenario C:
        whole-volume + ROI fusion input

Also supports:

    - single-checkpoint inference
    - 5-fold CV checkpoint ensembling
    - CUDA AMP inference
    - raw-logit collection
    - probability collection
    - optional probability calibration
    - optional binary thresholding
    - per-subject UID tracking
    - optional ground-truth labels
    - CSV output
    - strict consistency checks across fold models

Important
---------
This module performs INFERENCE ONLY.

It does NOT:
    - train
    - calculate training loss
    - call backward()
    - update model parameters
    - apply training augmentation
    - apply test-time augmentation automatically
    - apply ImageNet/Kinetics preprocessing
    - perform medical-image preprocessing

Inputs must already follow the frozen project preprocessing pipeline.

Expected model output:
    [B, 1] raw logits

Expected model inputs
---------------------
Single-input models:

    whole:
        [B, 1, 160, 192, 192]

    ROI:
        [B, 1, 36, 44, 44]

2.5D models:
    [B, 1, S, H, W]

Scenario C:
    model(whole, roi)

    where both tensors belong to the same subject.

Fold ensemble
-------------
For one experiment:

    fold_0 best_model.pt
    fold_1 best_model.pt
    fold_2 best_model.pt
    fold_3 best_model.pt
    fold_4 best_model.pt
                 |
                 v
       inference on same subjects
                 |
                 v
       average fold predictions

Both are retained:

    mean_logit
        mean of raw fold logits

    mean_probability
        mean of sigmoid(fold_logit)

By default the final uncalibrated ensemble probability is:

    mean(sigmoid(logit_fold))

Probability calibration
-----------------------
An optional calibration function may be supplied.

The calibration function must accept a CPU float32 Tensor with shape:

    [N, 1]

and return calibrated probabilities with shape:

    [N] or [N, 1]

Calibration may operate on:

    "logit"
or
    "probability"

depending on how the project's calibration model was fitted.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    Optional,
    Sequence,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.training.checkpointing import load_model_weights


# =====================================================================
# TYPES
# =====================================================================

EnsembleMethod = Literal[
    "mean_probability",
    "mean_logit",
]

CalibrationInput = Literal[
    "logit",
    "probability",
]


# =====================================================================
# SINGLE-CHECKPOINT RESULT
# =====================================================================


@dataclass(frozen=True)
class ModelPrediction:
    """
    Predictions produced by one model/checkpoint.
    """

    uids: tuple[str, ...]

    logits: Tensor
    probabilities: Tensor

    labels: Optional[Tensor]

    checkpoint_path: Optional[str]

    n_samples: int


# =====================================================================
# FINAL INFERENCE RESULT
# =====================================================================


@dataclass(frozen=True)
class InferenceResult:
    """
    Final inference result.

    Shapes
    ------
    mean_logits:
        [N, 1]

    mean_probabilities:
        [N, 1]

    final_probabilities:
        [N, 1]

    predicted_labels:
        [N, 1]

    fold_logits:
        [K, N, 1]

    fold_probabilities:
        [K, N, 1]

    where:
        N = number of subjects
        K = number of checkpoints/folds
    """

    uids: tuple[str, ...]

    mean_logits: Tensor

    mean_probabilities: Tensor

    final_probabilities: Tensor

    predicted_labels: Tensor

    labels: Optional[Tensor]

    fold_logits: Tensor

    fold_probabilities: Tensor

    checkpoint_paths: tuple[str, ...]

    ensemble_method: str

    threshold: float

    calibrated: bool

    n_samples: int

    n_models: int


# =====================================================================
# GENERIC INFERENCE ENGINE
# =====================================================================


class InferenceEngine:
    """
    Generic deterministic binary-classification inference engine.

    Parameters
    ----------
    device:
        torch.device("cuda")
        or
        torch.device("cpu")

    use_amp:
        If True and CUDA is used, inference uses FP16 autocast.

    threshold:
        Binary decision threshold.

        Default:
            0.5

        IMPORTANT:
            If a competition or validation procedure later freezes
            another threshold, pass that frozen threshold explicitly.

    ensemble_method:
        "mean_probability"

            final uncalibrated probability:
                mean(sigmoid(fold logits))

        "mean_logit"

            final uncalibrated probability:
                sigmoid(mean(fold logits))
    """

    def __init__(
        self,
        *,
        device: torch.device,
        use_amp: bool = True,
        threshold: float = 0.5,
        ensemble_method: EnsembleMethod = "mean_probability",
    ) -> None:

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must satisfy 0 <= threshold <= 1, "
                f"received {threshold}."
            )

        if ensemble_method not in {
            "mean_probability",
            "mean_logit",
        }:
            raise ValueError(
                "ensemble_method must be one of "
                "{'mean_probability', 'mean_logit'}, "
                f"received {ensemble_method!r}."
            )

        self.device = device

        self.threshold = float(
            threshold
        )

        self.ensemble_method = (
            ensemble_method
        )

        # -------------------------------------------------------------
        # AMP is enabled only for CUDA.
        # -------------------------------------------------------------

        self.amp_enabled = bool(
            use_amp
            and device.type == "cuda"
        )

    # =================================================================
    # BATCH INPUT HELPERS
    # =================================================================

    def _move_tensor(
        self,
        tensor: Tensor,
    ) -> Tensor:
        """
        Move model input to inference device.
        """

        return tensor.to(
            device=self.device,
            non_blocking=True,
        )

    @staticmethod
    def _extract_uids(
        batch: Mapping[str, Any],
        batch_size: int,
    ) -> list[str]:
        """
        Extract subject UIDs.

        UID is required for inference because fold predictions must be
        matched by subject rather than relying solely on DataLoader order.
        """

        if "uid" not in batch:
            raise KeyError(
                "Inference batch must contain 'uid'."
            )

        raw_uids = batch[
            "uid"
        ]

        if isinstance(
            raw_uids,
            str,
        ):

            uids = [
                raw_uids
            ]

        elif isinstance(
            raw_uids,
            Tensor,
        ):

            raw_list = raw_uids.tolist()

            if not isinstance(
                raw_list,
                list,
            ):
                raw_list = [
                    raw_list
                ]

            uids = [
                str(item)
                for item in raw_list
            ]

        else:

            try:

                uids = [
                    str(item)
                    for item in raw_uids
                ]

            except TypeError as exc:

                raise TypeError(
                    "Could not interpret batch['uid']."
                ) from exc

        if len(uids) != batch_size:

            raise RuntimeError(
                "Number of UIDs does not match batch size.\n"
                f"UIDs      : {len(uids)}\n"
                f"Batch size: {batch_size}"
            )

        return uids

    @staticmethod
    def _extract_optional_labels(
        batch: Mapping[str, Any],
        batch_size: int,
    ) -> Optional[Tensor]:
        """
        Extract labels if present.

        In real unseen/test inference labels may not exist.

        Accepted keys:
            label
            target
            is_pathologic
        """

        labels = None

        for key in (
            "label",
            "target",
            "is_pathologic",
        ):

            if key in batch:

                labels = batch[
                    key
                ]

                break

        if labels is None:
            return None

        if not isinstance(
            labels,
            Tensor,
        ):
            labels = torch.as_tensor(
                labels
            )

        labels = labels.detach().to(
            dtype=torch.float32,
            device="cpu",
        )

        if labels.ndim == 1:

            labels = labels.unsqueeze(
                1
            )

        elif (
            labels.ndim == 2
            and labels.shape[1] == 1
        ):
            pass

        else:

            raise ValueError(
                "Labels must have shape [B] or [B,1], "
                f"received {tuple(labels.shape)}."
            )

        if labels.shape[0] != batch_size:

            raise RuntimeError(
                "Label count does not match batch size."
            )

        if not torch.isfinite(
            labels
        ).all():

            raise ValueError(
                "Labels contain NaN or infinite values."
            )

        unique_labels = torch.unique(
            labels
        )

        valid = torch.logical_or(
            unique_labels == 0,
            unique_labels == 1,
        )

        if not bool(
            valid.all()
        ):

            raise ValueError(
                "Labels must contain only 0 and 1. "
                f"Received: {unique_labels.tolist()}"
            )

        return labels

    # =================================================================
    # BATCH PARSING
    # =================================================================

    def _parse_batch(
        self,
        batch: Any,
    ) -> tuple[
        Any,
        list[str],
        Optional[Tensor],
    ]:
        """
        Parse one DataLoader batch.

        Supported forms
        ---------------

        Scenario A:

            {
                "uid": ...,
                "whole": tensor,
            }

        Scenario B:

            {
                "uid": ...,
                "roi": tensor,
            }

        Generic single input:

            {
                "uid": ...,
                "input": tensor,
            }

        Scenario C:

            {
                "uid": ...,
                "whole": tensor,
                "roi": tensor,
            }

        Optional labels may also be present.
        """

        if not isinstance(
            batch,
            Mapping,
        ):

            raise TypeError(
                "Inference DataLoader must return a "
                "dictionary/mapping."
            )

        # -------------------------------------------------------------
        # Scenario C:
        # whole + ROI
        # -------------------------------------------------------------

        if (
            "whole" in batch
            and "roi" in batch
        ):

            whole = batch[
                "whole"
            ]

            roi = batch[
                "roi"
            ]

            if not isinstance(
                whole,
                Tensor,
            ):
                raise TypeError(
                    "'whole' must be torch.Tensor."
                )

            if not isinstance(
                roi,
                Tensor,
            ):
                raise TypeError(
                    "'roi' must be torch.Tensor."
                )

            if (
                whole.shape[0]
                != roi.shape[0]
            ):

                raise RuntimeError(
                    "Scenario C batch-size mismatch.\n"
                    f"Whole: {whole.shape[0]}\n"
                    f"ROI  : {roi.shape[0]}"
                )

            batch_size = int(
                whole.shape[0]
            )

            uids = self._extract_uids(
                batch,
                batch_size,
            )

            labels = (
                self._extract_optional_labels(
                    batch,
                    batch_size,
                )
            )

            inputs = (
                self._move_tensor(
                    whole
                ),
                self._move_tensor(
                    roi
                ),
            )

            return (
                inputs,
                uids,
                labels,
            )

        # -------------------------------------------------------------
        # Single-input scenario
        # -------------------------------------------------------------

        if "input" in batch:

            x = batch[
                "input"
            ]

        elif "image" in batch:

            x = batch[
                "image"
            ]

        elif "whole" in batch:

            x = batch[
                "whole"
            ]

        elif "roi" in batch:

            x = batch[
                "roi"
            ]

        else:

            raise KeyError(
                "Could not determine inference input. "
                "Expected one of: "
                "'input', 'image', 'whole', 'roi', "
                "or both 'whole' and 'roi'."
            )

        if not isinstance(
            x,
            Tensor,
        ):

            raise TypeError(
                "Inference input must be torch.Tensor."
            )

        if x.ndim < 1:

            raise ValueError(
                "Inference tensor has invalid shape."
            )

        batch_size = int(
            x.shape[0]
        )

        uids = self._extract_uids(
            batch,
            batch_size,
        )

        labels = (
            self._extract_optional_labels(
                batch,
                batch_size,
            )
        )

        x = self._move_tensor(
            x
        )

        return (
            x,
            uids,
            labels,
        )

    # =================================================================
    # MODEL FORWARD
    # =================================================================

    @staticmethod
    def _validate_logits(
        logits: Tensor,
        expected_batch_size: int,
    ) -> Tensor:
        """
        Enforce project output contract:

            [B,1] raw logits
        """

        if not isinstance(
            logits,
            Tensor,
        ):

            raise TypeError(
                "Model must return torch.Tensor logits."
            )

        if logits.ndim == 1:

            logits = logits.unsqueeze(
                1
            )

        if (
            logits.ndim != 2
            or logits.shape[1] != 1
        ):

            raise RuntimeError(
                "Binary classifier must return "
                "[B,1] raw logits. "
                f"Received {tuple(logits.shape)}."
            )

        if (
            logits.shape[0]
            != expected_batch_size
        ):

            raise RuntimeError(
                "Model output batch size does not match input batch."
            )

        if not torch.isfinite(
            logits
        ).all():

            raise FloatingPointError(
                "Model produced NaN or infinite logits."
            )

        return logits

    def _forward_model(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Tensor:
        """
        Forward either:

            model(x)

        or:

            model(whole, roi)
        """

        if isinstance(
            inputs,
            tuple,
        ):

            expected_batch_size = (
                inputs[0].shape[0]
            )

            logits = model(
                *inputs
            )

        else:

            expected_batch_size = (
                inputs.shape[0]
            )

            logits = model(
                inputs
            )

        return self._validate_logits(
            logits,
            expected_batch_size,
        )

    # =================================================================
    # ONE MODEL / ONE CHECKPOINT
    # =================================================================

    @torch.inference_mode()
    def predict_model(
        self,
        *,
        model: nn.Module,
        loader: DataLoader,
        checkpoint_path: Optional[
            str | Path
        ] = None,
    ) -> ModelPrediction:
        """
        Run deterministic inference for one model/checkpoint.

        If checkpoint_path is supplied, model weights are loaded before
        inference.

        The supplied model object must already have the correct
        architecture.
        """

        # -------------------------------------------------------------
        # Load trained weights.
        # -------------------------------------------------------------

        resolved_checkpoint: Optional[
            str
        ] = None

        if checkpoint_path is not None:

            checkpoint_path = (
                Path(
                    checkpoint_path
                )
                .expanduser()
                .resolve()
            )

            load_model_weights(
                checkpoint_path,
                model=model,
                map_location="cpu",
                strict=True,
            )

            resolved_checkpoint = str(
                checkpoint_path
            )

        # -------------------------------------------------------------
        # Inference mode.
        # -------------------------------------------------------------

        model = model.to(
            self.device
        )

        model.eval()

        all_uids: list[str] = []

        all_logits: list[
            Tensor
        ] = []

        all_labels: list[
            Tensor
        ] = []

        labels_available: Optional[
            bool
        ] = None

        for batch in loader:

            (
                inputs,
                uids,
                labels,
            ) = self._parse_batch(
                batch
            )

            # ---------------------------------------------------------
            # CUDA AMP inference.
            #
            # No GradScaler is required during inference because there
            # is no backward pass.
            # ---------------------------------------------------------

            with torch.autocast(
                device_type=self.device.type,
                dtype=(
                    torch.float16
                    if self.device.type == "cuda"
                    else torch.bfloat16
                ),
                enabled=self.amp_enabled,
            ):

                logits = (
                    self._forward_model(
                        model,
                        inputs,
                    )
                )

            logits = (
                logits
                .detach()
                .float()
                .cpu()
            )

            all_uids.extend(
                uids
            )

            all_logits.append(
                logits
            )

            # ---------------------------------------------------------
            # Labels must either be present for every batch or absent
            # for every batch.
            # ---------------------------------------------------------

            current_has_labels = (
                labels is not None
            )

            if labels_available is None:

                labels_available = (
                    current_has_labels
                )

            elif (
                labels_available
                != current_has_labels
            ):

                raise RuntimeError(
                    "Labels are present for some inference batches "
                    "but absent for others."
                )

            if labels is not None:

                all_labels.append(
                    labels
                )

        if not all_logits:

            raise RuntimeError(
                "Inference DataLoader produced zero samples."
            )

        logits_tensor = torch.cat(
            all_logits,
            dim=0,
        )

        probabilities = torch.sigmoid(
            logits_tensor
        )

        # -------------------------------------------------------------
        # UID uniqueness is mandatory.
        # -------------------------------------------------------------

        if len(
            set(all_uids)
        ) != len(
            all_uids
        ):

            duplicate_uids = sorted(
                {
                    uid
                    for uid in all_uids
                    if all_uids.count(uid) > 1
                }
            )

            raise RuntimeError(
                "Duplicate UIDs encountered during inference:\n"
                + "\n".join(
                    f"  - {uid}"
                    for uid in duplicate_uids
                )
            )

        if (
            len(all_uids)
            != logits_tensor.shape[0]
        ):

            raise RuntimeError(
                "UID/prediction count mismatch."
            )

        if labels_available:

            labels_tensor = torch.cat(
                all_labels,
                dim=0,
            )

            if (
                labels_tensor.shape[0]
                != logits_tensor.shape[0]
            ):

                raise RuntimeError(
                    "Label/prediction count mismatch."
                )

        else:

            labels_tensor = None

        return ModelPrediction(
            uids=tuple(
                all_uids
            ),
            logits=logits_tensor,
            probabilities=probabilities,
            labels=labels_tensor,
            checkpoint_path=(
                resolved_checkpoint
            ),
            n_samples=(
                logits_tensor.shape[0]
            ),
        )

    # =================================================================
    # ALIGN FOLD PREDICTIONS BY UID
    # =================================================================

    @staticmethod
    def _align_prediction_to_reference(
        prediction: ModelPrediction,
        reference_uids: Sequence[str],
    ) -> ModelPrediction:
        """
        Reorder one checkpoint's predictions to exactly match the
        reference UID ordering.

        This makes fold ensembling robust even if DataLoader ordering
        accidentally differs between passes.
        """

        index_by_uid = {
            uid: index
            for index, uid
            in enumerate(
                prediction.uids
            )
        }

        reference_set = set(
            reference_uids
        )

        prediction_set = set(
            prediction.uids
        )

        if reference_set != prediction_set:

            missing = sorted(
                reference_set
                - prediction_set
            )

            extra = sorted(
                prediction_set
                - reference_set
            )

            lines = [
                "Fold inference subjects do not match.",
                "",
                f"Missing subjects: {len(missing)}",
            ]

            lines.extend(
                f"  - {uid}"
                for uid in missing[:20]
            )

            lines.append(
                f"Extra subjects: {len(extra)}"
            )

            lines.extend(
                f"  - {uid}"
                for uid in extra[:20]
            )

            raise RuntimeError(
                "\n".join(lines)
            )

        indices = torch.tensor(
            [
                index_by_uid[uid]
                for uid
                in reference_uids
            ],
            dtype=torch.long,
        )

        aligned_logits = (
            prediction.logits[
                indices
            ]
        )

        aligned_probabilities = (
            prediction.probabilities[
                indices
            ]
        )

        if prediction.labels is not None:

            aligned_labels = (
                prediction.labels[
                    indices
                ]
            )

        else:

            aligned_labels = None

        return ModelPrediction(
            uids=tuple(
                reference_uids
            ),
            logits=aligned_logits,
            probabilities=(
                aligned_probabilities
            ),
            labels=aligned_labels,
            checkpoint_path=(
                prediction.checkpoint_path
            ),
            n_samples=(
                len(reference_uids)
            ),
        )

    # =================================================================
    # CALIBRATION
    # =================================================================

    @staticmethod
    def _apply_calibration(
        *,
        mean_logits: Tensor,
        uncalibrated_probabilities: Tensor,
        calibration_fn: Callable[
            [Tensor],
            Tensor,
        ],
        calibration_input: CalibrationInput,
    ) -> Tensor:
        """
        Apply previously fitted probability calibration.

        The calibration function MUST already be fitted.

        It must NOT learn anything from inference/test labels here.
        """

        if calibration_input == "logit":

            calibration_values = (
                mean_logits
            )

        elif calibration_input == "probability":

            calibration_values = (
                uncalibrated_probabilities
            )

        else:

            raise ValueError(
                "calibration_input must be "
                "'logit' or 'probability'."
            )

        calibrated = calibration_fn(
            calibration_values
        )

        if not isinstance(
            calibrated,
            Tensor,
        ):

            calibrated = torch.as_tensor(
                calibrated,
                dtype=torch.float32,
            )

        calibrated = (
            calibrated
            .detach()
            .float()
            .cpu()
        )

        if calibrated.ndim == 1:

            calibrated = (
                calibrated.unsqueeze(
                    1
                )
            )

        if (
            calibrated.ndim != 2
            or calibrated.shape[1] != 1
        ):

            raise RuntimeError(
                "Calibration function must return "
                "[N] or [N,1] probabilities. "
                f"Received {tuple(calibrated.shape)}."
            )

        if (
            calibrated.shape[0]
            != mean_logits.shape[0]
        ):

            raise RuntimeError(
                "Calibration output sample count mismatch."
            )

        if not torch.isfinite(
            calibrated
        ).all():

            raise FloatingPointError(
                "Calibration produced NaN or infinite values."
            )

        # -------------------------------------------------------------
        # Calibrator output must actually represent probabilities.
        # -------------------------------------------------------------

        if (
            calibrated.min().item() < 0.0
            or calibrated.max().item() > 1.0
        ):

            raise ValueError(
                "Calibration output must contain probabilities "
                "between 0 and 1."
            )

        return calibrated

    # =================================================================
    # CHECKPOINT ENSEMBLE
    # =================================================================

    def predict_checkpoint_ensemble(
        self,
        *,
        model: nn.Module,
        checkpoint_paths: Sequence[
            str | Path
        ],
        loader: DataLoader,
        calibration_fn: Optional[
            Callable[
                [Tensor],
                Tensor,
            ]
        ] = None,
        calibration_input: CalibrationInput = "logit",
    ) -> InferenceResult:
        """
        Run inference from one or more trained checkpoints.

        Typical usage
        -------------
        Five-fold CV experiment:

            checkpoint_paths = [
                fold_0/best_model.pt,
                fold_1/best_model.pt,
                fold_2/best_model.pt,
                fold_3/best_model.pt,
                fold_4/best_model.pt,
            ]

        Memory behavior
        ---------------
        Only ONE model copy is kept on the GPU.

        Each checkpoint is loaded sequentially:

            load fold 0
                -> predict
            load fold 1
                -> predict
            ...
            load fold 4
                -> predict

        Predictions are stored on CPU.

        This is especially important for large 3D models.
        """

        if not checkpoint_paths:

            raise ValueError(
                "At least one checkpoint path is required."
            )

        resolved_paths = tuple(
            str(
                Path(path)
                .expanduser()
                .resolve()
            )
            for path in checkpoint_paths
        )

        predictions: list[
            ModelPrediction
        ] = []

        reference_uids: Optional[
            tuple[str, ...]
        ] = None

        reference_labels: Optional[
            Tensor
        ] = None

        # -------------------------------------------------------------
        # Sequential fold/checkpoint inference.
        # -------------------------------------------------------------

        for index, checkpoint_path in enumerate(
            resolved_paths
        ):

            prediction = self.predict_model(
                model=model,
                loader=loader,
                checkpoint_path=(
                    checkpoint_path
                ),
            )

            # ---------------------------------------------------------
            # First checkpoint defines canonical UID ordering.
            # ---------------------------------------------------------

            if index == 0:

                reference_uids = (
                    prediction.uids
                )

                reference_labels = (
                    prediction.labels
                )

                aligned_prediction = (
                    prediction
                )

            else:

                assert reference_uids is not None

                aligned_prediction = (
                    self._align_prediction_to_reference(
                        prediction,
                        reference_uids,
                    )
                )

                # -----------------------------------------------------
                # Ground-truth labels, if present, must match exactly
                # across checkpoint passes.
                # -----------------------------------------------------

                if (
                    reference_labels is None
                    and aligned_prediction.labels
                    is not None
                ):

                    raise RuntimeError(
                        "Label availability differs between "
                        "checkpoint inference passes."
                    )

                if (
                    reference_labels is not None
                    and aligned_prediction.labels
                    is None
                ):

                    raise RuntimeError(
                        "Label availability differs between "
                        "checkpoint inference passes."
                    )

                if (
                    reference_labels is not None
                    and aligned_prediction.labels
                    is not None
                ):

                    if not torch.equal(
                        reference_labels,
                        aligned_prediction.labels,
                    ):

                        raise RuntimeError(
                            "Ground-truth labels changed between "
                            "checkpoint inference passes."
                        )

            predictions.append(
                aligned_prediction
            )

        assert reference_uids is not None

        # -------------------------------------------------------------
        # Stack fold predictions:
        #
        # K individual [N,1]
        #
        # becomes:
        #
        # [K,N,1]
        # -------------------------------------------------------------

        fold_logits = torch.stack(
            [
                prediction.logits
                for prediction
                in predictions
            ],
            dim=0,
        )

        fold_probabilities = torch.stack(
            [
                prediction.probabilities
                for prediction
                in predictions
            ],
            dim=0,
        )

        # -------------------------------------------------------------
        # Keep BOTH aggregate forms for auditability.
        # -------------------------------------------------------------

        mean_logits = (
            fold_logits.mean(
                dim=0
            )
        )

        mean_probabilities = (
            fold_probabilities.mean(
                dim=0
            )
        )

        # -------------------------------------------------------------
        # Select uncalibrated ensemble output.
        # -------------------------------------------------------------

        if (
            self.ensemble_method
            == "mean_probability"
        ):

            uncalibrated_final = (
                mean_probabilities
            )

        elif (
            self.ensemble_method
            == "mean_logit"
        ):

            uncalibrated_final = (
                torch.sigmoid(
                    mean_logits
                )
            )

        else:

            raise RuntimeError(
                "Unexpected ensemble method."
            )

        # -------------------------------------------------------------
        # Optional calibration.
        #
        # Calibration function MUST already have been fitted using the
        # appropriate training/OOF procedure.
        # -------------------------------------------------------------

        if calibration_fn is not None:

            final_probabilities = (
                self._apply_calibration(
                    mean_logits=mean_logits,
                    uncalibrated_probabilities=(
                        uncalibrated_final
                    ),
                    calibration_fn=(
                        calibration_fn
                    ),
                    calibration_input=(
                        calibration_input
                    ),
                )
            )

            calibrated = True

        else:

            final_probabilities = (
                uncalibrated_final
            )

            calibrated = False

        # -------------------------------------------------------------
        # Binary decisions.
        # -------------------------------------------------------------

        predicted_labels = (
            final_probabilities
            >= self.threshold
        ).to(
            dtype=torch.int64
        )

        return InferenceResult(
            uids=reference_uids,

            mean_logits=(
                mean_logits
            ),

            mean_probabilities=(
                mean_probabilities
            ),

            final_probabilities=(
                final_probabilities
            ),

            predicted_labels=(
                predicted_labels
            ),

            labels=reference_labels,

            fold_logits=(
                fold_logits
            ),

            fold_probabilities=(
                fold_probabilities
            ),

            checkpoint_paths=(
                resolved_paths
            ),

            ensemble_method=(
                self.ensemble_method
            ),

            threshold=(
                self.threshold
            ),

            calibrated=calibrated,

            n_samples=len(
                reference_uids
            ),

            n_models=len(
                predictions
            ),
        )


# =====================================================================
# SAVE PREDICTIONS
# =====================================================================


def save_inference_csv(
    result: InferenceResult,
    output_path: str | Path,
    *,
    include_fold_predictions: bool = False,
) -> Path:
    """
    Save per-subject predictions to CSV.

    Basic columns
    -------------
    uid
    mean_logit
    mean_probability
    final_probability
    predicted_label

    Optional:
        is_pathologic

    If include_fold_predictions=True:
        fold_0_logit
        fold_0_probability
        ...
    """

    output_path = (
        Path(
            output_path
        )
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "uid",
        "mean_logit",
        "mean_probability",
        "final_probability",
        "predicted_label",
    ]

    if result.labels is not None:

        fieldnames.append(
            "is_pathologic"
        )

    if include_fold_predictions:

        for fold_index in range(
            result.n_models
        ):

            fieldnames.extend(
                [
                    (
                        f"model_{fold_index}_logit"
                    ),
                    (
                        f"model_{fold_index}_probability"
                    ),
                ]
            )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index, uid in enumerate(
            result.uids
        ):

            row: dict[
                str,
                Any,
            ] = {
                "uid": uid,

                "mean_logit": float(
                    result.mean_logits[
                        index,
                        0,
                    ].item()
                ),

                "mean_probability": float(
                    result.mean_probabilities[
                        index,
                        0,
                    ].item()
                ),

                "final_probability": float(
                    result.final_probabilities[
                        index,
                        0,
                    ].item()
                ),

                "predicted_label": int(
                    result.predicted_labels[
                        index,
                        0,
                    ].item()
                ),
            }

            if result.labels is not None:

                row[
                    "is_pathologic"
                ] = int(
                    result.labels[
                        index,
                        0,
                    ].item()
                )

            if include_fold_predictions:

                for fold_index in range(
                    result.n_models
                ):

                    row[
                        f"model_{fold_index}_logit"
                    ] = float(
                        result.fold_logits[
                            fold_index,
                            index,
                            0,
                        ].item()
                    )

                    row[
                        (
                            f"model_{fold_index}_probability"
                        )
                    ] = float(
                        result.fold_probabilities[
                            fold_index,
                            index,
                            0,
                        ].item()
                    )

            writer.writerow(
                row
            )

    return output_path


# =====================================================================
# SIMPLE CONVENIENCE FUNCTION
# =====================================================================


def run_inference(
    *,
    model: nn.Module,
    checkpoint_paths: Sequence[
        str | Path
    ],
    loader: DataLoader,
    device: torch.device,
    output_csv: Optional[
        str | Path
    ] = None,
    use_amp: bool = True,
    threshold: float = 0.5,
    ensemble_method: EnsembleMethod = "mean_probability",
    calibration_fn: Optional[
        Callable[
            [Tensor],
            Tensor,
        ]
    ] = None,
    calibration_input: CalibrationInput = "logit",
    include_fold_predictions: bool = False,
) -> InferenceResult:
    """
    High-level convenience interface.

    Example
    -------
    result = run_inference(
        model=model,
        checkpoint_paths=[
            ".../fold_0/best_model.pt",
            ".../fold_1/best_model.pt",
            ".../fold_2/best_model.pt",
            ".../fold_3/best_model.pt",
            ".../fold_4/best_model.pt",
        ],
        loader=test_loader,
        device=torch.device("cuda"),
        output_csv="predictions.csv",
    )
    """

    engine = InferenceEngine(
        device=device,
        use_amp=use_amp,
        threshold=threshold,
        ensemble_method=(
            ensemble_method
        ),
    )

    result = (
        engine.predict_checkpoint_ensemble(
            model=model,
            checkpoint_paths=(
                checkpoint_paths
            ),
            loader=loader,
            calibration_fn=(
                calibration_fn
            ),
            calibration_input=(
                calibration_input
            ),
        )
    )

    if output_csv is not None:

        save_inference_csv(
            result,
            output_csv,
            include_fold_predictions=(
                include_fold_predictions
            ),
        )

    return result