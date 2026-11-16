#!/usr/bin/env python3
"""
two_point_five_d_input.py

Frozen deterministic 2.5D input policy shared by E4 and E7.

WHY THIS VERSION
----------------
E4 and E7 must use exactly the same slice-selection rule so their comparison
isolates ImageNet pretraining.

For whole-volume input, the selected slices must be tied to the already-frozen
striatal localization rather than being arbitrary raw-volume central slices or
uniformly distributed slices across the complete D=160 volume.

The Step-10 supervised manifest contains BOTH:
    - whole_path
    - roi_path

The ROI NIfTI is the frozen bilateral-striatal crop carried forward from the
Step-7/Step-8 localization/crop pipeline.

Therefore this selector uses the ROI crop geometry itself as the frozen,
label-independent localization reference.

WHOLE-VOLUME RULE
-----------------
For each UID:

1. Read only the NIfTI headers/affines for the frozen whole and ROI files.
2. Map the 8 corners of the ROI crop from ROI voxel coordinates -> world
   coordinates -> whole-volume voxel coordinates.
3. Determine the ROI crop's axial extent in the whole-volume Z index.
4. Select exactly 40 CONSECUTIVE whole-volume axial slices centered on that
   localized extent.
5. Shift the 40-slice window only when needed to stay inside D=160.
6. Require that the complete mapped ROI axial extent fits inside the 40-slice
   window. If geometry is inconsistent, FAIL loudly.

This means the 2.5D whole-volume input is localized around the frozen striatal
crop and still retains a small amount of superior/inferior whole-volume context.

ROI RULE
--------
Use all 36 frozen ROI axial slices.

    raw ROI:       [1,36,44,44]
    selected ROI:  [1,36,44,44]

AUGMENTATION
------------
This selector is deterministic and runs before stochastic augmentation.

For E4/E7 augmented scenarios, use:
    spatial_mode="inplane_2d"

That applies the same sampled 2-D affine transform to every selected axial
slice without interpolating/mixing across the slice/depth axis.

NO intensity normalization, clipping, resampling, registration, or label-based
logic is performed here.

MANIFEST RESOLUTION
-------------------
Preferred environment variable:
    DATSCAN_SUPERVISED_MANIFEST

Otherwise:
    <project_root>/data/preprocessing_supervised_data/
    step10_supervised_dataset_manifest_data/
    supervised_dataset/supervised_dataset_manifest.csv

The official E4/E7 Slurm scripts use the canonical frozen Step-10 manifest.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


WHOLE_RAW_SHAPE = (1, 160, 192, 192)  # [C,D,H,W]
ROI_RAW_SHAPE = (1, 36, 44, 44)       # [C,D,H,W]

WHOLE_SELECTED_COUNT = 40
WHOLE_SELECTED_SHAPE = (1, WHOLE_SELECTED_COUNT, 192, 192)
ROI_SELECTED_SHAPE = ROI_RAW_SHAPE

MANIFEST_ENV = "DATSCAN_SUPERVISED_MANIFEST"
POLICY_NAME = "e4_e7_striatum_localized_2p5d_v2"

# Loaded once per Python/DataLoader-worker process.
_UID_TO_PATHS: dict[str, tuple[Path, Path]] | None = None

# Computed once per UID per process and then reused across epochs.
_UID_TO_WHOLE_INDICES: dict[str, tuple[int, ...]] = {}


def _project_root() -> Path:
    # .../<project>/scripts/experiments_script/two_point_five_d_input.py
    return Path(__file__).resolve().parents[2]


def _resolve_manifest_path() -> Path:
    explicit = os.environ.get(MANIFEST_ENV, "").strip()

    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = _project_root() / path
        path = path.resolve()
    else:
        path = (
            _project_root()
            / "data"
            / "preprocessing_supervised_data"
            / "step10_supervised_dataset_manifest_data"
            / "supervised_dataset"
            / "supervised_dataset_manifest.csv"
        ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            "Frozen Step-10 supervised manifest required by E4/E7 "
            f"slice selection was not found:\n    {path}"
        )

    return path


def _find_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    logical_name: str,
) -> str:
    for name in candidates:
        if name in dataframe.columns:
            return name

    raise ValueError(
        f"Could not identify {logical_name!r} in Step-10 manifest. "
        f"Tried {candidates}; columns={list(dataframe.columns)}"
    )


def _resolve_data_path(value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = _project_root() / path
    return path.resolve()


def _load_uid_path_table() -> dict[str, tuple[Path, Path]]:
    global _UID_TO_PATHS

    if _UID_TO_PATHS is not None:
        return _UID_TO_PATHS

    manifest_path = _resolve_manifest_path()
    dataframe = pd.read_csv(manifest_path)

    uid_col = _find_column(
        dataframe,
        ("uid", "UID", "subject_uid", "subject_id"),
        "UID column",
    )

    whole_col = _find_column(
        dataframe,
        ("whole_path", "whole_volume_path", "whole", "whole_file"),
        "whole-volume path column",
    )

    roi_col = _find_column(
        dataframe,
        ("roi_path", "striatal_path", "striatal_roi_path", "roi", "roi_file"),
        "ROI path column",
    )

    table: dict[str, tuple[Path, Path]] = {}

    for _, row in dataframe.iterrows():
        uid = str(row[uid_col])

        if uid in table:
            raise ValueError(
                f"Duplicate UID in frozen Step-10 manifest: {uid}"
            )

        whole_path = _resolve_data_path(row[whole_col])
        roi_path = _resolve_data_path(row[roi_col])

        table[uid] = (whole_path, roi_path)

    _UID_TO_PATHS = table
    return table


def _apply_affine(
    affine: np.ndarray,
    points_xyz: np.ndarray,
) -> np.ndarray:
    """
    Apply a 4x4 affine to N x 3 points.

    Implemented locally so the geometry logic can be unit-tested without
    depending on nibabel's helper functions.
    """
    affine = np.asarray(affine, dtype=np.float64)
    points_xyz = np.asarray(points_xyz, dtype=np.float64)

    if affine.shape != (4, 4):
        raise ValueError(f"Expected 4x4 affine, got {affine.shape}.")

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(
            f"Expected N x 3 points, got {points_xyz.shape}."
        )

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


def _localized_window_from_affines(
    *,
    whole_affine: np.ndarray,
    whole_shape_xyz: tuple[int, int, int],
    roi_affine: np.ndarray,
    roi_shape_xyz: tuple[int, int, int],
    selected_count: int = WHOLE_SELECTED_COUNT,
) -> tuple[int, ...]:
    """
    Convert the frozen ROI crop geometry into a deterministic whole-volume
    axial window.

    Returns indices in the MODEL tensor depth axis, which is NIfTI Z after:
        [X,Y,Z] -> [Z,Y,X].
    """
    if tuple(whole_shape_xyz) != (192, 192, 160):
        raise ValueError(
            "Unexpected whole NIfTI shape for E4/E7 localization: "
            f"{whole_shape_xyz} != (192, 192, 160)"
        )

    if tuple(roi_shape_xyz) != (44, 44, 36):
        raise ValueError(
            "Unexpected ROI NIfTI shape for E4/E7 localization: "
            f"{roi_shape_xyz} != (44, 44, 36)"
        )

    corners_roi = _roi_corners_xyz(roi_shape_xyz)

    corners_world = _apply_affine(
        roi_affine,
        corners_roi,
    )

    try:
        whole_inverse = np.linalg.inv(
            np.asarray(whole_affine, dtype=np.float64)
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Whole-volume NIfTI affine is singular."
        ) from exc

    corners_whole = _apply_affine(
        whole_inverse,
        corners_world,
    )

    z_values = corners_whole[:, 2]

    if not np.isfinite(z_values).all():
        raise ValueError(
            "Non-finite ROI-to-whole mapped Z coordinate."
        )

    z_low = float(z_values.min())
    z_high = float(z_values.max())

    # Small numerical tolerance is allowed because the files may contain
    # floating-point affine round-off.
    if z_low < -0.51 or z_high > (whole_shape_xyz[2] - 1 + 0.51):
        raise ValueError(
            "Frozen ROI crop maps outside the whole-volume axial extent: "
            f"z=[{z_low:.4f}, {z_high:.4f}], "
            f"whole_depth={whole_shape_xyz[2]}"
        )

    localized_span = z_high - z_low + 1.0

    if localized_span > selected_count + 0.51:
        raise ValueError(
            "The frozen ROI axial extent is wider than the configured "
            f"{selected_count}-slice E4/E7 whole-volume window: "
            f"mapped_span={localized_span:.4f} slices."
        )

    center = 0.5 * (z_low + z_high)

    # Round to the nearest integer start for a window whose center is as
    # close as possible to the localized ROI center.
    ideal_start = center - (selected_count - 1) / 2.0
    start = int(np.floor(ideal_start + 0.5))

    max_start = whole_shape_xyz[2] - selected_count
    start = max(0, min(start, max_start))

    indices = tuple(range(start, start + selected_count))

    # Final inclusion guard.
    if z_low < indices[0] - 0.51 or z_high > indices[-1] + 0.51:
        raise RuntimeError(
            "Could not construct a 40-slice whole-volume window containing "
            "the complete frozen ROI axial extent. "
            f"roi_z=[{z_low:.4f},{z_high:.4f}], "
            f"window=[{indices[0]},{indices[-1]}]"
        )

    return indices


def _whole_indices_for_uid(uid: str) -> tuple[int, ...]:
    if uid in _UID_TO_WHOLE_INDICES:
        return _UID_TO_WHOLE_INDICES[uid]

    table = _load_uid_path_table()

    if uid not in table:
        raise KeyError(
            f"UID {uid!r} is not present in the frozen Step-10 manifest."
        )

    whole_path, roi_path = table[uid]

    if not whole_path.is_file():
        raise FileNotFoundError(
            f"Whole-volume NIfTI not found for UID {uid}: {whole_path}"
        )

    if not roi_path.is_file():
        raise FileNotFoundError(
            f"ROI NIfTI not found for UID {uid}: {roi_path}"
        )

    # Lazy import: common_cv_experiment.py already depends on nibabel on HPC.
    import nibabel as nib

    whole_img = nib.load(str(whole_path))
    roi_img = nib.load(str(roi_path))

    indices = _localized_window_from_affines(
        whole_affine=np.asarray(whole_img.affine, dtype=np.float64),
        whole_shape_xyz=tuple(int(x) for x in whole_img.shape),
        roi_affine=np.asarray(roi_img.affine, dtype=np.float64),
        roi_shape_xyz=tuple(int(x) for x in roi_img.shape),
    )

    _UID_TO_WHOLE_INDICES[uid] = indices
    return indices


def _select_depth(
    tensor: torch.Tensor,
    indices: tuple[int, ...],
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(
            "Expected one unbatched [C,D,H,W] tensor, "
            f"received {tuple(tensor.shape)}."
        )

    index = torch.as_tensor(
        indices,
        dtype=torch.long,
        device=tensor.device,
    )

    return torch.index_select(
        tensor,
        dim=1,
        index=index,
    ).contiguous()


def select_2p5d_slices(
    sample: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply the frozen deterministic E4/E7 slice-selection policy.

    UID and label are preserved exactly.
    """
    output = dict(sample)
    uid = str(output["uid"])

    if "whole" in output:
        whole = output["whole"]

        if tuple(whole.shape) != WHOLE_RAW_SHAPE:
            raise ValueError(
                "Unexpected whole-volume tensor before 2.5D selection: "
                f"{tuple(whole.shape)} != {WHOLE_RAW_SHAPE}"
            )

        indices = _whole_indices_for_uid(uid)

        whole = _select_depth(
            whole,
            indices,
        )

        if tuple(whole.shape) != WHOLE_SELECTED_SHAPE:
            raise RuntimeError(
                "Whole-volume 2.5D selection produced unexpected shape: "
                f"{tuple(whole.shape)} != {WHOLE_SELECTED_SHAPE}"
            )

        output["whole"] = whole

    if "roi" in output:
        roi = output["roi"]

        if tuple(roi.shape) != ROI_RAW_SHAPE:
            raise ValueError(
                "Unexpected ROI tensor before 2.5D selection: "
                f"{tuple(roi.shape)} != {ROI_RAW_SHAPE}"
            )

        # Frozen policy: retain every ROI slice.
        output["roi"] = roi.contiguous()

    return output


def slice_selection_metadata() -> dict[str, Any]:
    return {
        "policy_name": POLICY_NAME,
        "axis": "axial_depth_D",
        "whole_raw_shape": list(WHOLE_RAW_SHAPE),
        "whole_selected_count": WHOLE_SELECTED_COUNT,
        "whole_selected_shape": list(WHOLE_SELECTED_SHAPE),
        "whole_strategy": (
            "subject_specific_40_consecutive_slices_centered_on_"
            "frozen_striatal_roi_axial_extent"
        ),
        "whole_localization_source": (
            "frozen Step-10 roi_path NIfTI affine; ROI crop carried forward "
            "from Step-7/Step-8 striatal localization"
        ),
        "whole_geometry_mapping": (
            "ROI voxel corners -> ROI affine -> world -> inverse whole affine "
            "-> whole NIfTI Z -> model depth D"
        ),
        "whole_fallback_to_raw_center": False,
        "whole_fallback_to_full_depth_uniform": False,
        "roi_raw_shape": list(ROI_RAW_SHAPE),
        "roi_selected_count": 36,
        "roi_selected_shape": list(ROI_SELECTED_SHAPE),
        "roi_strategy": "all_36_slices",
        "deterministic": True,
        "subject_specific": True,
        "label_independent": True,
        "additional_normalization": False,
        "additional_clipping": False,
        "additional_resampling": False,
        "additional_registration": False,
    }
