from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    probability: float = 0.50
    max_rotation_deg: float = 5.0
    max_translation_voxels: float = 2.0


class ConservativeAffine3D:
    """Small train-only 3D rigid perturbation. No left/right flips; no intensity normalization."""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    @staticmethod
    def _rotation_matrix(rx: float, ry: float, rz: float, device, dtype) -> torch.Tensor:
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        Rx = torch.tensor([[1,0,0],[0,cx,-sx],[0,sx,cx]], device=device, dtype=dtype)
        Ry = torch.tensor([[cy,0,sy],[0,1,0],[-sy,0,cy]], device=device, dtype=dtype)
        Rz = torch.tensor([[cz,-sz,0],[sz,cz,0],[0,0,1]], device=device, dtype=dtype)
        return Rz @ Ry @ Rx

    def sample_parameters(self, generator: torch.Generator | None = None) -> dict[str, float | bool]:
        if not self.config.enabled:
            return {"apply": False}

        rand = lambda: torch.rand((), generator=generator).item()
        if rand() >= self.config.probability:
            return {"apply": False}

        r = self.config.max_rotation_deg
        t = self.config.max_translation_voxels
        return {
            "apply": True,
            "rx_deg": (2 * rand() - 1) * r,
            "ry_deg": (2 * rand() - 1) * r,
            "rz_deg": (2 * rand() - 1) * r,
            "tx_vox": (2 * rand() - 1) * t,
            "ty_vox": (2 * rand() - 1) * t,
            "tz_vox": (2 * rand() - 1) * t,
        }

    def apply(self, tensor_cdhw: torch.Tensor, params: dict[str, float | bool]) -> torch.Tensor:
        if not bool(params.get("apply", False)):
            return tensor_cdhw
        if tensor_cdhw.ndim != 4:
            raise ValueError(f"Expected [C,D,H,W], got {tuple(tensor_cdhw.shape)}")

        x = tensor_cdhw.unsqueeze(0)  # [N,C,D,H,W]
        _, _, d, h, w = x.shape
        device, dtype = x.device, x.dtype

        rx = math.radians(float(params["rx_deg"]))
        ry = math.radians(float(params["ry_deg"]))
        rz = math.radians(float(params["rz_deg"]))
        R = self._rotation_matrix(rx, ry, rz, device, dtype)

        # affine_grid coordinates are x,y,z corresponding W,H,D.
        theta = torch.zeros((1, 3, 4), device=device, dtype=dtype)
        theta[0, :, :3] = R
        theta[0, 0, 3] = 2.0 * float(params["tx_vox"]) / max(w, 1)
        theta[0, 1, 3] = 2.0 * float(params["ty_vox"]) / max(h, 1)
        theta[0, 2, 3] = 2.0 * float(params["tz_vox"]) / max(d, 1)

        grid = F.affine_grid(theta, size=x.shape, align_corners=False)
        y = F.grid_sample(
            x,
            grid,
            mode="bilinear",      # trilinear for 5D tensors
            padding_mode="zeros",
            align_corners=False,
        )
        return y.squeeze(0).contiguous()

    def __call__(self, tensor_cdhw: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.apply(tensor_cdhw, self.sample_parameters(generator))
