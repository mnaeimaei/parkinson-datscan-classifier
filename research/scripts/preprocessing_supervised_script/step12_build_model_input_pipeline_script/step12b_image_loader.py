from __future__ import annotations

from pathlib import Path
import numpy as np

WHOLE_XYZ_SHAPE = (192, 192, 160)
ROI_XYZ_SHAPE = (44, 44, 36)


def _import_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "nibabel is required for Step 12 image loading. Install/use the project's existing environment."
        ) from exc
    return nib


def load_nifti_xyz(path: str | Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {p}")

    nib = _import_nibabel()
    img = nib.load(str(p))
    if len(img.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {img.shape} for {p}")
    if tuple(img.shape) != tuple(expected_shape):
        raise ValueError(f"Unexpected source shape for {p}: got {img.shape}, expected {expected_shape}")

    # get_fdata applies any NIfTI scale/intercept and emits the requested dtype.
    arr = img.get_fdata(dtype=np.float32)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if arr.size == 0:
        raise ValueError(f"Empty image array: {p}")
    if not np.isfinite(arr).all():
        raise ValueError(f"NaN/Inf found in image: {p}")
    return arr


def load_whole_xyz(path: str | Path) -> np.ndarray:
    return load_nifti_xyz(path, WHOLE_XYZ_SHAPE)


def load_roi_xyz(path: str | Path) -> np.ndarray:
    return load_nifti_xyz(path, ROI_XYZ_SHAPE)
