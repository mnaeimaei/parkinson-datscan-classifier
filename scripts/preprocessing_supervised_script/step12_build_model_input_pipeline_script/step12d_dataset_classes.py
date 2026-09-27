from __future__ import annotations

from typing import Any
import pandas as pd
import torch
from torch.utils.data import Dataset

from step12b_image_loader import load_whole_xyz, load_roi_xyz
from step12c_tensor_conversion import whole_xyz_to_tensor, roi_xyz_to_tensor
from step12e_augmentation import ConservativeAffine3D


class _BaseDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augmentation: ConservativeAffine3D | None = None):
        self.frame = frame.reset_index(drop=True).copy()
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def _label(row: pd.Series) -> torch.Tensor:
        return torch.tensor(float(row["is_pathologic"]), dtype=torch.float32)


class WholeVolumeDataset(_BaseDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        whole = whole_xyz_to_tensor(load_whole_xyz(row["whole_volume_path"]))
        if self.augmentation is not None:
            whole = self.augmentation(whole)
        return {"whole": whole, "label": self._label(row), "uid": str(row["uid"])}


class StriatalDataset(_BaseDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        roi = roi_xyz_to_tensor(load_roi_xyz(row["striatal_path"]))
        if self.augmentation is not None:
            roi = self.augmentation(roi)
        return {"roi": roi, "label": self._label(row), "uid": str(row["uid"])}


class DualInputDataset(_BaseDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        whole = whole_xyz_to_tensor(load_whole_xyz(row["whole_volume_path"]))
        roi = roi_xyz_to_tensor(load_roi_xyz(row["striatal_path"]))

        # One sampled transform is reused for whole + ROI. Because both are at
        # 2.46-mm isotropic spacing, the same translation in voxels corresponds
        # to the same physical displacement; rotations are likewise shared.
        if self.augmentation is not None:
            params = self.augmentation.sample_parameters()
            whole = self.augmentation.apply(whole, params)
            roi = self.augmentation.apply(roi, params)

        return {
            "whole": whole,
            "roi": roi,
            "label": self._label(row),
            "uid": str(row["uid"]),
        }
