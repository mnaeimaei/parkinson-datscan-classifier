from __future__ import annotations

import numpy as np
import torch

WHOLE_CDHW_SHAPE = (1, 160, 192, 192)
ROI_CDHW_SHAPE = (1, 36, 44, 44)


def xyz_to_cdhw(array_xyz: np.ndarray, expected_cdhw: tuple[int, int, int, int]) -> torch.Tensor:
    if array_xyz.ndim != 3:
        raise ValueError(f"Expected XYZ 3D array, got {array_xyz.shape}")

    # NIfTI/nibabel: [X,Y,Z] -> model spatial order [D,H,W] = [Z,Y,X].
    zyx = np.transpose(array_xyz, (2, 1, 0))
    zyx = np.ascontiguousarray(zyx, dtype=np.float32)
    tensor = torch.from_numpy(zyx).unsqueeze(0)  # [C,D,H,W]

    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    if tuple(tensor.shape) != tuple(expected_cdhw):
        raise ValueError(f"Unexpected model tensor shape: got {tuple(tensor.shape)}, expected {expected_cdhw}")
    if not torch.isfinite(tensor).all():
        raise ValueError("NaN/Inf found after tensor conversion.")
    return tensor.contiguous()


def whole_xyz_to_tensor(array_xyz: np.ndarray) -> torch.Tensor:
    return xyz_to_cdhw(array_xyz, WHOLE_CDHW_SHAPE)


def roi_xyz_to_tensor(array_xyz: np.ndarray) -> torch.Tensor:
    return xyz_to_cdhw(array_xyz, ROI_CDHW_SHAPE)
