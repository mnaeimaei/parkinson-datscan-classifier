"""
model03_resnet18_3d_medicalnet_pretrained.py

MedicalNet-compatible MONAI 3D ResNet-18 with MedicalNet / Med3D
pretrained encoder initialization.

E2 vs E3
--------
E2:
    Identical architecture
    Random/Kaiming initialization

E3:
    Identical architecture
    Encoder initialized from:
        resnet_18_23dataset.pth

The ONLY intended difference between E2 and E3 is encoder initialization.

Architecture
------------
Input:
    [B, 1, D, H, W]

Encoder:
    Exactly the same encoder as model02_resnet18_3d_scratch.py

    ResNet-18:
        BasicBlock [2, 2, 2, 2]

    Channels:
        64 -> 128 -> 256 -> 512

    MedicalNet-compatible settings:
        spatial_dims=3
        n_input_channels=1
        shortcut_type="A"
        bias_downsample=True
        feed_forward=False

Head:
    Global average pooling is already performed by the MONAI encoder.
    Encoder output:
        [B, 512]

    New binary classifier:
        Linear(512 -> 1)

Output:
    [B, 1] raw logit

Important
---------
The pretrained MedicalNet task/segmentation head is NOT loaded.

The checkpoint loader:

    1. Loads resnet_18_23dataset.pth safely.
    2. Extracts its state_dict.
    3. Removes prefixes such as "module.".
    4. Ignores task-specific checkpoint layers such as conv_seg.
    5. Checks tensor names.
    6. Checks tensor shapes.
    7. Requires essentially the complete intended encoder to transfer.
    8. Fails loudly if important encoder tensors are missing.

No sigmoid is applied inside the network.
Use BCEWithLogitsLoss during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor

from src.models.model02_resnet18_3d_scratch import (
    BIAS_DOWNSAMPLE,
    BLOCK_INPLANES,
    BLOCK_STRUCTURE,
    FEATURE_DIM,
    SHORTCUT_TYPE,
    ResNet18_3D_Scratch,
)


# ---------------------------------------------------------------------
# MedicalNet checkpoint configuration
# ---------------------------------------------------------------------

EXPECTED_CHECKPOINT_NAME = "resnet_18_23dataset.pth"

# Prefixes commonly introduced when saving wrapped models.
CHECKPOINT_PREFIXES = (
    "module.",
    "model.",
    "network.",
    "backbone.",
    "encoder.",
)

# Task-specific tensors that should NOT be transferred into our encoder.
#
# Original MedicalNet checkpoints may contain segmentation/task heads.
IGNORED_HEAD_PREFIXES = (
    "conv_seg.",
    "fc.",
    "classifier.",
    "head.",
)


@dataclass(frozen=True)
class PretrainedLoadReport:
    """
    Summary of MedicalNet encoder checkpoint loading.
    """

    checkpoint_path: str

    checkpoint_tensor_count: int
    target_encoder_tensor_count: int

    loaded_tensor_count: int
    loaded_parameter_count: int

    ignored_checkpoint_keys: tuple[str, ...]
    unexpected_checkpoint_keys: tuple[str, ...]
    missing_allowed_keys: tuple[str, ...]

    tensor_coverage: float
    parameter_coverage: float


class ResNet18_3D_MedicalNet_Pretrained(ResNet18_3D_Scratch):
    """
    E3 — MedicalNet-pretrained 3D ResNet-18.

    This class inherits directly from E2 so that the architecture is
    guaranteed to remain identical.

    The only difference is that the encoder is initialized from
    MedicalNet pretrained weights.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        # -------------------------------------------------------------
        # Build EXACTLY the same architecture as E2.
        # -------------------------------------------------------------
        super().__init__()

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        # -------------------------------------------------------------
        # Replace the random E2 encoder initialization with MedicalNet
        # pretrained encoder tensors.
        # -------------------------------------------------------------
        self.pretrained_load_report = self._load_medicalnet_encoder(
            self.checkpoint_path
        )

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # self.classifier remains newly initialized.
        #
        # We deliberately load weights ONLY into self.encoder.
        # Therefore the MedicalNet segmentation/task head can never
        # overwrite our binary classifier.
        # -------------------------------------------------------------

    def _load_medicalnet_encoder(
        self,
        checkpoint_path: Path,
    ) -> PretrainedLoadReport:
        """
        Load and validate MedicalNet pretrained encoder weights.

        The loading policy is intentionally strict.

        All meaningful encoder tensors must be found and have exactly
        matching shapes.

        The only missing keys tolerated are BatchNorm bookkeeping tensors:

            *.num_batches_tracked

        because older PyTorch checkpoints may not contain them.
        """

        # -------------------------------------------------------------
        # 1. Validate checkpoint path.
        # -------------------------------------------------------------

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "MedicalNet checkpoint was not found:\n"
                f"    {checkpoint_path}"
            )

        if not checkpoint_path.is_file():
            raise ValueError(
                "MedicalNet checkpoint path is not a file:\n"
                f"    {checkpoint_path}"
            )

        # Optional filename warning/error.
        #
        # For this experiment we specifically expect the 23-dataset
        # ResNet-18 checkpoint.
        if checkpoint_path.name != EXPECTED_CHECKPOINT_NAME:
            raise ValueError(
                "Unexpected MedicalNet checkpoint filename.\n"
                f"Expected : {EXPECTED_CHECKPOINT_NAME}\n"
                f"Received : {checkpoint_path.name}\n\n"
                "E3 is defined specifically as MedicalNet ResNet-18 "
                "pretrained on the 23-dataset checkpoint."
            )

        # -------------------------------------------------------------
        # 2. Load checkpoint.
        #
        # weights_only=True is deliberately used. We do not need arbitrary
        # serialized Python objects from a pretrained checkpoint.
        # -------------------------------------------------------------

        checkpoint: Any = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        # -------------------------------------------------------------
        # 3. Extract state_dict.
        # -------------------------------------------------------------

        raw_state_dict = self._extract_state_dict(checkpoint)

        if not raw_state_dict:
            raise RuntimeError(
                "MedicalNet checkpoint contains an empty state_dict."
            )

        # -------------------------------------------------------------
        # 4. Normalize checkpoint keys.
        # -------------------------------------------------------------

        normalized_state_dict: dict[str, Tensor] = {}

        for original_key, value in raw_state_dict.items():

            if not isinstance(value, Tensor):
                continue

            normalized_key = self._normalize_checkpoint_key(original_key)

            if normalized_key in normalized_state_dict:
                raise RuntimeError(
                    "Duplicate checkpoint key after prefix normalization:\n"
                    f"    {normalized_key}"
                )

            normalized_state_dict[normalized_key] = value

        # -------------------------------------------------------------
        # 5. Get the exact E2 encoder state dictionary.
        # -------------------------------------------------------------

        target_state_dict = self.encoder.state_dict()

        # -------------------------------------------------------------
        # 6. Match checkpoint -> target encoder.
        # -------------------------------------------------------------

        transferable_state_dict: dict[str, Tensor] = {}

        ignored_checkpoint_keys: list[str] = []
        unexpected_checkpoint_keys: list[str] = []

        shape_mismatches: list[
            tuple[str, tuple[int, ...], tuple[int, ...]]
        ] = []

        for checkpoint_key, checkpoint_tensor in normalized_state_dict.items():

            # ---------------------------------------------------------
            # Explicitly discard pretrained task heads.
            # ---------------------------------------------------------
            if self._is_ignored_head_key(checkpoint_key):
                ignored_checkpoint_keys.append(checkpoint_key)
                continue

            # ---------------------------------------------------------
            # Ignore keys that do not exist in our encoder, but record
            # them so the loading operation is fully auditable.
            # ---------------------------------------------------------
            if checkpoint_key not in target_state_dict:
                unexpected_checkpoint_keys.append(checkpoint_key)
                continue

            target_tensor = target_state_dict[checkpoint_key]

            # ---------------------------------------------------------
            # Exact shape match is mandatory.
            # ---------------------------------------------------------
            if checkpoint_tensor.shape != target_tensor.shape:
                shape_mismatches.append(
                    (
                        checkpoint_key,
                        tuple(checkpoint_tensor.shape),
                        tuple(target_tensor.shape),
                    )
                )
                continue

            transferable_state_dict[checkpoint_key] = checkpoint_tensor

        # -------------------------------------------------------------
        # 7. Shape mismatches are never silently accepted.
        # -------------------------------------------------------------

        if shape_mismatches:

            lines = [
                "MedicalNet checkpoint contains encoder tensors with "
                "incompatible shapes:",
                "",
            ]

            for key, checkpoint_shape, target_shape in shape_mismatches:
                lines.append(
                    f"  {key}: "
                    f"checkpoint={checkpoint_shape}, "
                    f"target={target_shape}"
                )

            raise RuntimeError("\n".join(lines))

        # -------------------------------------------------------------
        # 8. Find target encoder tensors that were not transferred.
        # -------------------------------------------------------------

        missing_keys = [
            key
            for key in target_state_dict
            if key not in transferable_state_dict
        ]

        allowed_missing_keys = [
            key
            for key in missing_keys
            if key.endswith("num_batches_tracked")
        ]

        critical_missing_keys = [
            key
            for key in missing_keys
            if not key.endswith("num_batches_tracked")
        ]

        # -------------------------------------------------------------
        # Every actual encoder tensor must transfer successfully.
        # -------------------------------------------------------------

        if critical_missing_keys:

            message = [
                "MedicalNet encoder loading FAILED.",
                "",
                "Important encoder tensors are missing from the checkpoint:",
                "",
            ]

            message.extend(
                f"  - {key}"
                for key in critical_missing_keys
            )

            message.extend(
                [
                    "",
                    "The checkpoint will NOT be partially accepted.",
                    "",
                    "E3 requires essentially complete MedicalNet encoder "
                    "initialization so E2-vs-E3 remains scientifically valid.",
                ]
            )

            raise RuntimeError("\n".join(message))

        # -------------------------------------------------------------
        # 9. Calculate coverage before loading.
        # -------------------------------------------------------------

        meaningful_target_keys = [
            key
            for key in target_state_dict
            if not key.endswith("num_batches_tracked")
        ]

        meaningful_loaded_keys = [
            key
            for key in transferable_state_dict
            if not key.endswith("num_batches_tracked")
        ]

        tensor_coverage = (
            len(meaningful_loaded_keys)
            / len(meaningful_target_keys)
        )

        total_target_parameters = sum(
            target_state_dict[key].numel()
            for key in meaningful_target_keys
        )

        loaded_parameters = sum(
            transferable_state_dict[key].numel()
            for key in meaningful_loaded_keys
        )

        parameter_coverage = (
            loaded_parameters
            / total_target_parameters
        )

        # -------------------------------------------------------------
        # We expect complete meaningful encoder transfer.
        #
        # The tolerance is numerical only.
        # -------------------------------------------------------------

        if tensor_coverage < 0.999999:
            raise RuntimeError(
                "Insufficient MedicalNet encoder tensor coverage:\n"
                f"    {tensor_coverage:.6%}"
            )

        if parameter_coverage < 0.999999:
            raise RuntimeError(
                "Insufficient MedicalNet encoder parameter coverage:\n"
                f"    {parameter_coverage:.6%}"
            )

        # -------------------------------------------------------------
        # 10. Actually load encoder tensors.
        # -------------------------------------------------------------

        incompatible = self.encoder.load_state_dict(
            transferable_state_dict,
            strict=False,
        )

        # Only num_batches_tracked may legitimately remain missing.
        remaining_critical_missing = [
            key
            for key in incompatible.missing_keys
            if not key.endswith("num_batches_tracked")
        ]

        if remaining_critical_missing:
            raise RuntimeError(
                "Critical encoder tensors remained missing after "
                "load_state_dict():\n"
                + "\n".join(
                    f"  - {key}"
                    for key in remaining_critical_missing
                )
            )

        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected tensors were passed to encoder.load_state_dict():\n"
                + "\n".join(
                    f"  - {key}"
                    for key in incompatible.unexpected_keys
                )
            )

        # -------------------------------------------------------------
        # 11. Return complete loading report.
        # -------------------------------------------------------------

        return PretrainedLoadReport(
            checkpoint_path=str(checkpoint_path),

            checkpoint_tensor_count=len(normalized_state_dict),
            target_encoder_tensor_count=len(target_state_dict),

            loaded_tensor_count=len(transferable_state_dict),
            loaded_parameter_count=loaded_parameters,

            ignored_checkpoint_keys=tuple(
                sorted(ignored_checkpoint_keys)
            ),

            unexpected_checkpoint_keys=tuple(
                sorted(unexpected_checkpoint_keys)
            ),

            missing_allowed_keys=tuple(
                sorted(allowed_missing_keys)
            ),

            tensor_coverage=tensor_coverage,
            parameter_coverage=parameter_coverage,
        )

    @staticmethod
    def _extract_state_dict(
        checkpoint: Any,
    ) -> Mapping[str, Tensor]:
        """
        Extract a state dictionary from common checkpoint formats.

        MedicalNet commonly stores weights under:

            checkpoint["state_dict"]

        Raw state dictionaries are also supported.
        """

        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                "MedicalNet checkpoint must contain a mapping/dictionary, "
                f"received {type(checkpoint).__name__}."
            )

        # Preferred / original MedicalNet format.
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

            if not isinstance(state_dict, Mapping):
                raise TypeError(
                    'checkpoint["state_dict"] must be a mapping.'
                )

            return state_dict

        # Additional common checkpoint formats.
        for candidate in (
            "model_state_dict",
            "model",
            "network",
            "weights",
        ):
            if candidate in checkpoint:

                state_dict = checkpoint[candidate]

                if isinstance(state_dict, Mapping):
                    return state_dict

        # Raw state_dict:
        #
        # {
        #     "conv1.weight": Tensor(...),
        #     "bn1.weight": Tensor(...),
        #     ...
        # }
        if all(
            isinstance(key, str)
            for key in checkpoint.keys()
        ):
            tensor_values = [
                value
                for value in checkpoint.values()
                if isinstance(value, Tensor)
            ]

            if tensor_values:
                return checkpoint  # type: ignore[return-value]

        raise RuntimeError(
            "Could not find a usable state_dict inside "
            "the MedicalNet checkpoint."
        )

    @staticmethod
    def _normalize_checkpoint_key(key: str) -> str:
        """
        Remove known wrapper prefixes.

        Example:

            module.layer1.0.conv1.weight

        becomes:

            layer1.0.conv1.weight
        """

        normalized = key

        # Repeat because checkpoints can occasionally have nested wrappers,
        # for example:
        #
        #     module.model.conv1.weight
        #
        changed = True

        while changed:
            changed = False

            for prefix in CHECKPOINT_PREFIXES:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    changed = True

        return normalized

    @staticmethod
    def _is_ignored_head_key(key: str) -> bool:
        """
        Return True for pretrained task-specific heads.
        """

        return any(
            key.startswith(prefix)
            for prefix in IGNORED_HEAD_PREFIXES
        )


def build_model(
    checkpoint_path: str | Path,
) -> ResNet18_3D_MedicalNet_Pretrained:
    """
    Build E3 using the frozen MedicalNet checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to:

            resnet_18_23dataset.pth
    """

    return ResNet18_3D_MedicalNet_Pretrained(
        checkpoint_path=checkpoint_path
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def verify_architecture_matches_e2(
    e3_model: ResNet18_3D_MedicalNet_Pretrained,
) -> None:
    """
    Verify that E2 and E3 have identical parameter names and shapes.

    Values are intentionally NOT compared because E2 and E3 are supposed
    to have different initialization.
    """

    e2_model = ResNet18_3D_Scratch()

    e2_state = e2_model.state_dict()
    e3_state = e3_model.state_dict()

    if e2_state.keys() != e3_state.keys():
        raise RuntimeError(
            "E2 and E3 do not have identical state_dict keys."
        )

    shape_mismatches = []

    for key in e2_state:
        if e2_state[key].shape != e3_state[key].shape:
            shape_mismatches.append(
                (
                    key,
                    tuple(e2_state[key].shape),
                    tuple(e3_state[key].shape),
                )
            )

    if shape_mismatches:

        lines = [
            "E2/E3 architecture mismatch:",
            "",
        ]

        for key, e2_shape, e3_shape in shape_mismatches:
            lines.append(
                f"  {key}: E2={e2_shape}, E3={e3_shape}"
            )

        raise RuntimeError("\n".join(lines))


if __name__ == "__main__":
    # =============================================================
    # CHECKPOINT + ARCHITECTURE SANITY CHECK ONLY
    #
    # This does NOT:
    #     - train
    #     - create an optimizer
    #     - calculate loss
    #     - call backward()
    #     - update model parameters
    # =============================================================

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Validate E3 MedicalNet pretrained ResNet-18."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help=(
            "Path to resnet_18_23dataset.pth"
        ),
    )

    args = parser.parse_args()

    model = build_model(
        checkpoint_path=args.checkpoint
    )

    model.eval()

    # -------------------------------------------------------------
    # Verify E2/E3 architectural identity.
    # -------------------------------------------------------------

    verify_architecture_matches_e2(model)

    report = model.pretrained_load_report

    print("=" * 76)
    print(
        "MODEL 03 — MEDICALNET-COMPATIBLE MONAI "
        "3D RESNET-18 — PRETRAINED"
    )
    print("=" * 76)

    print()
    print("Architecture:")
    print(f"  Blocks          : {BLOCK_STRUCTURE}")
    print(f"  Channels        : {BLOCK_INPLANES}")
    print(f"  Shortcut type   : {SHORTCUT_TYPE}")
    print(f"  Bias downsample : {BIAS_DOWNSAMPLE}")
    print(f"  Feature dim     : {FEATURE_DIM}")
    print("  Pretrained      : YES")

    print()
    print("Checkpoint:")
    print(f"  Path : {report.checkpoint_path}")

    print()
    print("Pretrained transfer:")
    print(
        f"  Checkpoint tensors : "
        f"{report.checkpoint_tensor_count}"
    )
    print(
        f"  Encoder tensors    : "
        f"{report.target_encoder_tensor_count}"
    )
    print(
        f"  Loaded tensors     : "
        f"{report.loaded_tensor_count}"
    )
    print(
        f"  Tensor coverage    : "
        f"{report.tensor_coverage:.2%}"
    )
    print(
        f"  Parameter coverage : "
        f"{report.parameter_coverage:.2%}"
    )

    print()
    print(
        f"Ignored task/head tensors : "
        f"{len(report.ignored_checkpoint_keys)}"
    )

    for key in report.ignored_checkpoint_keys:
        print(f"    {key}")

    print()
    print(
        f"Unexpected checkpoint tensors : "
        f"{len(report.unexpected_checkpoint_keys)}"
    )

    for key in report.unexpected_checkpoint_keys:
        print(f"    {key}")

    print()
    print(
        f"Allowed missing bookkeeping tensors : "
        f"{len(report.missing_allowed_keys)}"
    )

    for key in report.missing_allowed_keys:
        print(f"    {key}")

    print()
    print(
        f"Trainable parameters : "
        f"{count_trainable_parameters(model):,}"
    )

    # -------------------------------------------------------------
    # ROI-sized synthetic forward-pass validation.
    # -------------------------------------------------------------

    dummy_input = torch.zeros(
        2,
        1,
        36,
        44,
        44,
        dtype=torch.float32,
    )

    with torch.no_grad():

        features = model.forward_features(dummy_input)

        logits = model(dummy_input)

    print()
    print(f"Input shape   : {tuple(dummy_input.shape)}")
    print(f"Feature shape : {tuple(features.shape)}")
    print(f"Logit shape   : {tuple(logits.shape)}")

    assert features.shape == (2, FEATURE_DIM)
    assert logits.shape == (2, 1)

    assert torch.isfinite(features).all()
    assert torch.isfinite(logits).all()

    print()
    print("E2/E3 architecture match : PASS")
    print("MedicalNet transfer       : PASS")
    print("Forward-pass check        : PASS")