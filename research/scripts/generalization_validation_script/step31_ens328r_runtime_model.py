"""
model_bundle/runtime_model.py

Portable inference engine for the frozen ENS328R predictor.

IMPORTANT
---------
This module starts AFTER deterministic image preprocessing has produced the
same two frozen NIfTI products used by training:

    whole NIfTI: (192, 192, 160)  -> torch [1,160,192,192]
    ROI NIfTI:   (44, 44, 36)     -> torch [1,36,44,44]

It does NOT perform:
- orientation standardization
- resampling
- registration
- reference-region mapping
- intensity normalization
- ROI localization/cropping
- whole-volume crop/pad
- stochastic augmentation

Those operations remain the responsibility of the already-validated
competition preprocessing pipeline.

Frozen predictor
----------------
P1 = E6-S5 -> ROI R3D-18
P2 = E7-S4 -> whole 2.5D ResNet18 + attention
P3 = E6-S2 -> ROI R3D-18
P7 = E4-S2 -> ROI 2.5D ResNet18 + attention
P8 = E2-S2 -> ROI MONAI ResNet18

For each member:
    5 fold checkpoints -> sigmoid(logit) -> arithmetic probability mean

Then:
    member-specific final calibration
    -> equal-weight logit mean across P1/P2/P3/P7/P8
    -> final ENS328R temperature calibrator
    -> continuous is_pathologic probability

No pretrained weights are downloaded at inference.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch


BUNDLE_ROOT = Path(__file__).resolve().parent

# Let the copied internal package be imported as top-level `src`.
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from src.inference.final_ensemble import FinalEnsemblePredictor
from src.models.model02_resnet18_3d_scratch import build_model as build_e2
from src.models.model05_r3d18_scratch import build_model as build_e5
from src.models.model07_resnet18_2p5d_attention_scratch import build_model as build_e7


FROZEN_MEMBERS = ("P1", "R_E6", "P2", "P3", "P7", "P8")
N_FOLDS = 5

WHOLE_NIFTI_SHAPE = (192, 192, 160)  # [X,Y,Z]
ROI_NIFTI_SHAPE = (44, 44, 36)       # [X,Y,Z]

WHOLE_TENSOR_SHAPE = (1, 160, 192, 192)  # [C,D,H,W]
ROI_TENSOR_SHAPE = (1, 36, 44, 44)

WHOLE_2P5D_COUNT = 40
WHOLE_2P5D_SHAPE = (1, 40, 192, 192)

MEMBER_INPUT = {
    "P1": "roi",
    "R_E6": "roi",
    "P2": "whole_2p5d",
    "P3": "roi",
    "P7": "roi_2p5d",
    "P8": "roi",
}

# Runtime builders intentionally avoid pretrained constructors.
# Phase 13A proved every selected checkpoint strict-loads into these.
MEMBER_BUILDER = {
    "P1": build_e5,
    "R_E6": build_e5,  # E6 checkpoint -> identical E5 scratch architecture
    "P2": build_e7,
    "P3": build_e5,
    "P7": build_e7,  # E4 checkpoint -> identical E7 scratch architecture
    "P8": build_e2,
}


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _validate_case(case: Mapping) -> dict:
    required = {"uid", "whole_path", "roi_path"}
    missing = required - set(case)
    if missing:
        raise ValueError(f"Case is missing keys: {sorted(missing)}")

    uid = str(case["uid"])
    whole_path = Path(case["whole_path"]).expanduser().resolve()
    roi_path = Path(case["roi_path"]).expanduser().resolve()

    if not whole_path.is_file():
        raise FileNotFoundError(f"{uid}: whole NIfTI missing: {whole_path}")
    if not roi_path.is_file():
        raise FileNotFoundError(f"{uid}: ROI NIfTI missing: {roi_path}")

    return {
        "uid": uid,
        "whole_path": whole_path,
        "roi_path": roi_path,
    }


def load_frozen_nifti_tensor(
    path: str | Path,
    *,
    expected_nifti_shape: tuple[int, int, int],
    expected_tensor_shape: tuple[int, int, int, int],
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Exact Step-12 tensor convention:
        nibabel [X,Y,Z] -> float32 -> [Z,Y,X] -> [C,D,H,W].

    No scaling/normalization is introduced here.
    """
    path = Path(path).expanduser().resolve()
    image = nib.load(str(path))

    shape_xyz = tuple(int(v) for v in image.shape)
    if shape_xyz != tuple(expected_nifti_shape):
        raise ValueError(
            f"{path}: NIfTI shape {shape_xyz} != {expected_nifti_shape}"
        )

    array = np.asarray(image.dataobj, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"{path}: expected 3D array, got {array.shape}")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{path}: contains NaN/Inf")

    array = np.transpose(array, (2, 1, 0))
    array = np.ascontiguousarray(array)
    tensor = torch.from_numpy(array).unsqueeze(0)

    if tuple(tensor.shape) != tuple(expected_tensor_shape):
        raise RuntimeError(
            f"{path}: tensor shape {tuple(tensor.shape)} "
            f"!= {expected_tensor_shape}"
        )
    if tensor.dtype != torch.float32:
        raise TypeError(f"{path}: tensor dtype {tensor.dtype} != float32")

    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{path}: invalid NIfTI affine")

    return tensor.contiguous(), affine


# ---------------------------------------------------------------------
# Exact manifest-independent version of the frozen E4/E7 whole selector.
# The geometry below mirrors scripts/experiments_script/
# two_point_five_d_input.py. Unlike training, test UIDs are not present
# in Step-10, so paths/affines are passed directly.
# ---------------------------------------------------------------------

def _apply_affine(affine: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    affine = np.asarray(affine, dtype=np.float64)
    points_xyz = np.asarray(points_xyz, dtype=np.float64)

    if affine.shape != (4, 4):
        raise ValueError(f"Expected 4x4 affine, got {affine.shape}")
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected N x 3 points, got {points_xyz.shape}")

    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([points_xyz, ones], axis=1)
    mapped = homogeneous @ affine.T
    return mapped[:, :3]


def _roi_corners_xyz(
    roi_shape_xyz: tuple[int, int, int],
) -> np.ndarray:
    x_max = roi_shape_xyz[0] - 1
    y_max = roi_shape_xyz[1] - 1
    z_max = roi_shape_xyz[2] - 1

    return np.asarray(
        [
            [x, y, z]
            for x in (0.0, float(x_max))
            for y in (0.0, float(y_max))
            for z in (0.0, float(z_max))
        ],
        dtype=np.float64,
    )


def localized_whole_2p5d_indices(
    *,
    whole_affine: np.ndarray,
    roi_affine: np.ndarray,
    whole_shape_xyz: tuple[int, int, int] = WHOLE_NIFTI_SHAPE,
    roi_shape_xyz: tuple[int, int, int] = ROI_NIFTI_SHAPE,
    selected_count: int = WHOLE_2P5D_COUNT,
) -> tuple[int, ...]:
    """
    Frozen E4/E7 rule:
    ROI corners -> world -> whole voxel Z -> 40 consecutive slices.
    """
    if tuple(whole_shape_xyz) != WHOLE_NIFTI_SHAPE:
        raise ValueError(
            f"Unexpected whole shape {whole_shape_xyz} != {WHOLE_NIFTI_SHAPE}"
        )
    if tuple(roi_shape_xyz) != ROI_NIFTI_SHAPE:
        raise ValueError(
            f"Unexpected ROI shape {roi_shape_xyz} != {ROI_NIFTI_SHAPE}"
        )

    corners_roi = _roi_corners_xyz(roi_shape_xyz)
    corners_world = _apply_affine(roi_affine, corners_roi)

    try:
        whole_inverse = np.linalg.inv(
            np.asarray(whole_affine, dtype=np.float64)
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("Whole-volume NIfTI affine is singular") from exc

    corners_whole = _apply_affine(whole_inverse, corners_world)
    z_values = corners_whole[:, 2]

    if not np.isfinite(z_values).all():
        raise ValueError("Non-finite ROI-to-whole mapped Z coordinate")

    z_low = float(z_values.min())
    z_high = float(z_values.max())

    if z_low < -0.51 or z_high > (whole_shape_xyz[2] - 1 + 0.51):
        raise ValueError(
            "ROI maps outside whole-volume axial extent: "
            f"z=[{z_low:.4f},{z_high:.4f}]"
        )

    localized_span = z_high - z_low + 1.0
    if localized_span > selected_count + 0.51:
        raise ValueError(
            f"ROI axial span {localized_span:.4f} exceeds "
            f"{selected_count}-slice window"
        )

    center = 0.5 * (z_low + z_high)
    ideal_start = center - (selected_count - 1) / 2.0
    start = int(np.floor(ideal_start + 0.5))

    max_start = whole_shape_xyz[2] - selected_count
    start = max(0, min(start, max_start))
    indices = tuple(range(start, start + selected_count))

    if z_low < indices[0] - 0.51 or z_high > indices[-1] + 0.51:
        raise RuntimeError(
            "Could not build 40-slice window containing ROI extent: "
            f"roi_z=[{z_low:.4f},{z_high:.4f}], "
            f"window=[{indices[0]},{indices[-1]}]"
        )

    return indices


def select_whole_2p5d(
    whole_tensor: torch.Tensor,
    *,
    whole_affine: np.ndarray,
    roi_affine: np.ndarray,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if tuple(whole_tensor.shape) != WHOLE_TENSOR_SHAPE:
        raise ValueError(
            f"Whole tensor {tuple(whole_tensor.shape)} != {WHOLE_TENSOR_SHAPE}"
        )

    indices = localized_whole_2p5d_indices(
        whole_affine=whole_affine,
        roi_affine=roi_affine,
    )

    index = torch.as_tensor(indices, dtype=torch.long)
    selected = torch.index_select(
        whole_tensor,
        dim=1,
        index=index,
    ).contiguous()

    if tuple(selected.shape) != WHOLE_2P5D_SHAPE:
        raise RuntimeError(
            f"Selected whole tensor {tuple(selected.shape)} "
            f"!= {WHOLE_2P5D_SHAPE}"
        )

    return selected, indices


def _extract_logits(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        logits = output
    elif isinstance(output, Mapping):
        for key in ("logits", "logit", "output", "prediction"):
            value = output.get(key)
            if torch.is_tensor(value):
                logits = value
                break
        else:
            raise TypeError(
                f"Model dict output has no recognized tensor key: {output.keys()}"
            )
    elif isinstance(output, (tuple, list)):
        tensors = [v for v in output if torch.is_tensor(v)]
        if len(tensors) != 1:
            raise TypeError(
                f"Expected one tensor in model output, got {len(tensors)}"
            )
        logits = tensors[0]
    else:
        raise TypeError(f"Unsupported model output: {type(output).__name__}")

    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)

    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError(
            f"Expected logits [B,1], received {tuple(logits.shape)}"
        )

    return logits


def _portable_state_dict(path: Path) -> dict[str, torch.Tensor]:
    # Phase 13B stores a raw tensor-only state_dict.
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"{path}: portable state_dict is invalid")
    if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise RuntimeError(
            f"{path}: portable weights contain non-tensor state"
        )

    return state


class ENS328RRuntime:
    """
    Frozen six-member, five-fold runtime.

    The class keeps only one fold model on the accelerator at a time.
    This intentionally minimizes GPU memory and works offline.
    """

    def __init__(
        self,
        *,
        bundle_root: str | Path | None = None,
        device: str = "auto",
        amp: bool = True,
        verify_hashes: bool = False,
    ) -> None:
        self.bundle_root = (
            Path(bundle_root).expanduser().resolve()
            if bundle_root is not None
            else BUNDLE_ROOT
        )

        self.manifest_path = (
            self.bundle_root
            / "portable_checkpoint_manifest.json"
        )
        self.predictor_config_path = (
            self.bundle_root
            / "final_predictor"
            / "final_predictor_config.json"
        )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        if not self.predictor_config_path.is_file():
            raise FileNotFoundError(self.predictor_config_path)

        self.manifest = _load_json(self.manifest_path)
        self.final_predictor = FinalEnsemblePredictor(
            config_path=self.predictor_config_path
        )

        if tuple(self.final_predictor.members) != FROZEN_MEMBERS:
            raise RuntimeError(
                f"Frozen member mismatch: {self.final_predictor.members}"
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        self.amp = bool(amp and self.device.type == "cuda")
        self.verify_hashes = bool(verify_hashes)

        self._validate_manifest()

    def _validate_manifest(self) -> None:
        members = self.manifest.get("members", {})
        if tuple(members.keys()) != FROZEN_MEMBERS:
            raise RuntimeError(
                "Portable manifest member order/content differs from ENS328R"
            )

        for member_id in FROZEN_MEMBERS:
            entries = members[member_id]
            if len(entries) != N_FOLDS:
                raise RuntimeError(
                    f"{member_id}: expected {N_FOLDS} folds, got {len(entries)}"
                )

            observed = [int(v["fold"]) for v in entries]
            if observed != list(range(N_FOLDS)):
                raise RuntimeError(
                    f"{member_id}: fold list {observed} != 0..4"
                )

            for entry in entries:
                path = self.bundle_root / entry["relative_path"]
                if not path.is_file():
                    raise FileNotFoundError(path)

                if self.verify_hashes:
                    observed_hash = sha256_file(path)
                    expected_hash = entry["sha256"]
                    if observed_hash != expected_hash:
                        raise RuntimeError(
                            f"{member_id} fold {entry['fold']}: SHA256 mismatch"
                        )

    @staticmethod
    def _prepare_case_input(
        case: Mapping,
        member_id: str,
    ) -> tuple[torch.Tensor, dict]:
        """
        Return one unbatched [C,D,H,W] model input plus diagnostics.
        """
        case = _validate_case(case)
        uid = case["uid"]

        if MEMBER_INPUT[member_id] in {"roi", "roi_2p5d"}:
            roi, _ = load_frozen_nifti_tensor(
                case["roi_path"],
                expected_nifti_shape=ROI_NIFTI_SHAPE,
                expected_tensor_shape=ROI_TENSOR_SHAPE,
            )
            return roi, {
                "uid": uid,
                "member": member_id,
                "input": "roi",
                "shape": list(roi.shape),
            }

        if MEMBER_INPUT[member_id] == "whole_2p5d":
            whole, whole_affine = load_frozen_nifti_tensor(
                case["whole_path"],
                expected_nifti_shape=WHOLE_NIFTI_SHAPE,
                expected_tensor_shape=WHOLE_TENSOR_SHAPE,
            )

            # Need only the ROI affine/geometry, not its voxel intensities.
            roi_image = nib.load(str(case["roi_path"]))
            roi_shape = tuple(int(v) for v in roi_image.shape)
            if roi_shape != ROI_NIFTI_SHAPE:
                raise ValueError(
                    f"{uid}: ROI NIfTI shape {roi_shape} != {ROI_NIFTI_SHAPE}"
                )
            roi_affine = np.asarray(roi_image.affine, dtype=np.float64)

            selected, indices = select_whole_2p5d(
                whole,
                whole_affine=whole_affine,
                roi_affine=roi_affine,
            )
            return selected, {
                "uid": uid,
                "member": member_id,
                "input": "whole_2p5d",
                "shape": list(selected.shape),
                "selected_depth_indices": list(indices),
            }

        raise RuntimeError(
            f"Unhandled member input policy for {member_id}"
        )

    def _predict_one_fold(
        self,
        *,
        member_id: str,
        fold_entry: Mapping,
        cases: Sequence[Mapping],
    ) -> tuple[np.ndarray, list[dict]]:
        builder = MEMBER_BUILDER[member_id]
        model = builder()

        weights_path = (
            self.bundle_root
            / str(fold_entry["relative_path"])
        ).resolve()

        state_dict = _portable_state_dict(weights_path)
        incompatible = model.load_state_dict(
            state_dict,
            strict=True,
        )

        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"{member_id} fold {fold_entry['fold']}: "
                "strict state_dict load was not exact"
            )

        model.eval()
        model.to(self.device)

        probabilities = np.empty(len(cases), dtype=np.float64)
        diagnostics: list[dict] = []

        if self.amp:
            amp_context = lambda: torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
        else:
            amp_context = contextlib.nullcontext

        with torch.inference_mode():
            for index, case in enumerate(cases):
                tensor, diag = self._prepare_case_input(
                    case,
                    member_id,
                )

                batch = tensor.unsqueeze(0).to(
                    self.device,
                    non_blocking=False,
                )

                with amp_context():
                    logits = _extract_logits(model(batch))

                probability = torch.sigmoid(
                    logits.float()
                ).detach().cpu().numpy().reshape(-1)

                if probability.size != 1:
                    raise RuntimeError(
                        f"{member_id}: expected one probability per case"
                    )

                value = float(probability[0])
                if not np.isfinite(value) or not (0.0 <= value <= 1.0):
                    raise FloatingPointError(
                        f"{member_id}: invalid probability {value}"
                    )

                probabilities[index] = value
                diag["fold"] = int(fold_entry["fold"])
                diagnostics.append(diag)

                del batch, tensor, logits

        model.to("cpu")
        del model, state_dict

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return probabilities, diagnostics

    def predict_preprocessed_cases(
        self,
        cases: Sequence[Mapping],
        *,
        return_diagnostics: bool = False,
    ):
        """
        Parameters
        ----------
        cases:
            Sequence of dictionaries:
                {
                    "uid": "...",
                    "whole_path": "/.../fixed_whole.nii.gz",
                    "roi_path": "/.../striatal_roi.nii.gz",
                }

        Returns
        -------
        final_probability:
            ndarray [N], continuous is_pathologic probabilities.

        If return_diagnostics=True, also returns a dictionary containing
        raw five-fold member probabilities and final ensemble intermediates.
        """
        cases = [_validate_case(case) for case in cases]

        if not cases:
            raise ValueError("No cases were supplied")

        uids = [case["uid"] for case in cases]
        if len(set(uids)) != len(uids):
            raise ValueError("Duplicate UID in runtime cases")

        raw_member_probabilities: dict[str, np.ndarray] = {}
        fold_probability_tables: dict[str, np.ndarray] = {}
        all_input_diagnostics: list[dict] = []

        for member_id in FROZEN_MEMBERS:
            entries = self.manifest["members"][member_id]
            fold_matrix = np.empty(
                (len(cases), N_FOLDS),
                dtype=np.float64,
            )

            for fold_entry in entries:
                fold = int(fold_entry["fold"])
                probability, diagnostics = self._predict_one_fold(
                    member_id=member_id,
                    fold_entry=fold_entry,
                    cases=cases,
                )
                fold_matrix[:, fold] = probability

                if return_diagnostics:
                    all_input_diagnostics.extend(diagnostics)

            raw_member_probabilities[member_id] = fold_matrix.mean(axis=1)
            fold_probability_tables[member_id] = fold_matrix

        result = self.final_predictor.predict_with_intermediates(
            raw_member_probabilities
        )

        final_probability = np.asarray(
            result["final_probability"],
            dtype=np.float64,
        ).reshape(-1)

        if len(final_probability) != len(cases):
            raise RuntimeError(
                "Final ensemble probability count does not match cases"
            )

        if not np.isfinite(final_probability).all():
            raise FloatingPointError(
                "Final ENS328R probabilities contain NaN/Inf"
            )

        if return_diagnostics:
            return final_probability, {
                "uids": uids,
                "device": str(self.device),
                "amp": self.amp,
                "raw_member_probabilities": raw_member_probabilities,
                "fold_probabilities": fold_probability_tables,
                "selected_member_probabilities":
                    result["selected_member_probabilities"],
                "raw_ensemble_probability":
                    result["raw_ensemble_probability"],
                "final_probability": final_probability,
                "input_diagnostics": all_input_diagnostics,
            }

        return final_probability
