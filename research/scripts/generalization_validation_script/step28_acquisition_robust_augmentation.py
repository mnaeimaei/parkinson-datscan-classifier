#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AcquisitionRobustConfig:
    resolution_probability: float = 0.55
    resolution_xy_scale_min: float = 0.58
    resolution_xy_scale_max: float = 0.92
    resolution_z_scale_min: float = 0.70
    resolution_z_scale_max: float = 0.98

    blur_probability: float = 0.35
    blur_sigma_min_vox: float = 0.25
    blur_sigma_max_vox: float = 0.85

    poisson_probability: float = 0.30
    poisson_count_scale_min: float = 70.0
    poisson_count_scale_max: float = 220.0

    gamma_probability: float = 0.30
    gamma_min: float = 0.92
    gamma_max: float = 1.08

    intensity_scale_probability: float = 0.25
    intensity_scale_min: float = 0.94
    intensity_scale_max: float = 1.06

    gaussian_noise_probability: float = 0.20
    gaussian_noise_std_fraction_min: float = 0.005
    gaussian_noise_std_fraction_max: float = 0.020

    def validate(self) -> None:
        probs = (
            self.resolution_probability,
            self.blur_probability,
            self.poisson_probability,
            self.gamma_probability,
            self.intensity_scale_probability,
            self.gaussian_noise_probability,
        )
        if any(p < 0.0 or p > 1.0 for p in probs):
            raise ValueError("All probabilities must be in [0,1].")
        if not (0 < self.resolution_xy_scale_min <= self.resolution_xy_scale_max <= 1):
            raise ValueError("Invalid XY resolution range.")
        if not (0 < self.resolution_z_scale_min <= self.resolution_z_scale_max <= 1):
            raise ValueError("Invalid Z resolution range.")
        if not (0 <= self.blur_sigma_min_vox <= self.blur_sigma_max_vox):
            raise ValueError("Invalid blur range.")
        if not (0 < self.poisson_count_scale_min <= self.poisson_count_scale_max):
            raise ValueError("Invalid Poisson count scale.")
        if not (0 < self.gamma_min <= self.gamma_max):
            raise ValueError("Invalid gamma range.")
        if not (0 < self.intensity_scale_min <= self.intensity_scale_max):
            raise ValueError("Invalid intensity scale.")
        if not (
            0 <= self.gaussian_noise_std_fraction_min
            <= self.gaussian_noise_std_fraction_max
        ):
            raise ValueError("Invalid Gaussian noise range.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _uniform(low: float, high: float, device: torch.device) -> float:
    if low == high:
        return float(low)
    return float(torch.empty((), device=device).uniform_(low, high).item())


def _event(probability: float, device: torch.device) -> bool:
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    return bool((torch.rand((), device=device) < probability).item())


def _check(x: torch.Tensor) -> None:
    if not torch.is_tensor(x):
        raise TypeError("Input must be a torch.Tensor.")
    if x.ndim != 4:
        raise ValueError(f"Expected [C,D,H,W], got {tuple(x.shape)}.")
    if x.dtype != torch.float32:
        raise TypeError(f"Expected float32, got {x.dtype}.")
    if not bool(torch.isfinite(x).all().item()):
        raise FloatingPointError("Input contains NaN/Inf.")


def _resolution_degrade(x: torch.Tensor, z_scale: float, xy_scale: float) -> torch.Tensor:
    _, d, h, w = x.shape
    low_d = max(2, min(d, int(round(d * z_scale))))
    low_h = max(2, min(h, int(round(h * xy_scale))))
    low_w = max(2, min(w, int(round(w * xy_scale))))
    if (low_d, low_h, low_w) == (d, h, w):
        return x

    y = F.interpolate(
        x.unsqueeze(0),
        size=(low_d, low_h, low_w),
        mode="trilinear",
        align_corners=False,
    )
    y = F.interpolate(
        y,
        size=(d, h, w),
        mode="trilinear",
        align_corners=False,
    )
    return y.squeeze(0)


def _kernel1d(sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if sigma <= 1e-6:
        return torch.ones(1, dtype=dtype, device=device)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def _gaussian_blur3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 1e-6:
        return x
    c = x.shape[0]
    k = _kernel1d(sigma, x.dtype, x.device)
    r = k.numel() // 2
    y = x.unsqueeze(0)

    kd = k.view(1, 1, -1, 1, 1).repeat(c, 1, 1, 1, 1)
    y = F.pad(y, (0, 0, 0, 0, r, r), mode="replicate")
    y = F.conv3d(y, kd, groups=c)

    kh = k.view(1, 1, 1, -1, 1).repeat(c, 1, 1, 1, 1)
    y = F.pad(y, (0, 0, r, r, 0, 0), mode="replicate")
    y = F.conv3d(y, kh, groups=c)

    kw = k.view(1, 1, 1, 1, -1).repeat(c, 1, 1, 1, 1)
    y = F.pad(y, (r, r, 0, 0, 0, 0), mode="replicate")
    y = F.conv3d(y, kw, groups=c)

    return y.squeeze(0)


def _poisson_noise(x: torch.Tensor, count_scale: float) -> torch.Tensor:
    mask = x >= 0.0
    positive = torch.clamp(x, min=0.0)
    sampled = torch.poisson(positive * count_scale) / count_scale
    return torch.where(mask, sampled, x)


class AcquisitionRobustAugmenter:
    """
    Wraps the existing authoritative S5 augmenter.
    FrozenDATScanDataset calls __call__ for ROI-only training.
    """

    def __init__(self, *, base_augmenter, config: AcquisitionRobustConfig | None = None):
        if base_augmenter is None:
            raise ValueError("base_augmenter cannot be None.")
        self.base_augmenter = base_augmenter
        self.config = config or AcquisitionRobustConfig()
        self.config.validate()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _check(x)
        original_shape = tuple(x.shape)

        y = self.base_augmenter(x)
        _check(y)
        if tuple(y.shape) != original_shape:
            raise RuntimeError("Base S5 augmentation changed shape.")

        cfg = self.config
        device = y.device

        if _event(cfg.resolution_probability, device):
            y = _resolution_degrade(
                y,
                z_scale=_uniform(cfg.resolution_z_scale_min, cfg.resolution_z_scale_max, device),
                xy_scale=_uniform(cfg.resolution_xy_scale_min, cfg.resolution_xy_scale_max, device),
            )

        if _event(cfg.blur_probability, device):
            y = _gaussian_blur3d(
                y,
                _uniform(cfg.blur_sigma_min_vox, cfg.blur_sigma_max_vox, device),
            )

        if _event(cfg.poisson_probability, device):
            y = _poisson_noise(
                y,
                _uniform(cfg.poisson_count_scale_min, cfg.poisson_count_scale_max, device),
            )

        if _event(cfg.gamma_probability, device):
            gamma = _uniform(cfg.gamma_min, cfg.gamma_max, device)
            y = torch.sign(y) * torch.pow(torch.abs(y), gamma)

        if _event(cfg.intensity_scale_probability, device):
            y = y * _uniform(cfg.intensity_scale_min, cfg.intensity_scale_max, device)

        if _event(cfg.gaussian_noise_probability, device):
            fraction = _uniform(
                cfg.gaussian_noise_std_fraction_min,
                cfg.gaussian_noise_std_fraction_max,
                device,
            )
            std = max(float(y.std(unbiased=False).item()), 1e-6)
            y = y + torch.randn_like(y) * (fraction * std)

        y = y.contiguous()
        _check(y)
        if tuple(y.shape) != original_shape:
            raise RuntimeError(
                f"Step-28 augmentation changed shape: {original_shape} -> {tuple(y.shape)}"
            )
        return y


def self_test() -> None:
    class Identity:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return x

    cfg = AcquisitionRobustConfig(
        resolution_probability=1.0,
        blur_probability=1.0,
        poisson_probability=1.0,
        gamma_probability=1.0,
        intensity_scale_probability=1.0,
        gaussian_noise_probability=1.0,
    )
    aug = AcquisitionRobustAugmenter(base_augmenter=Identity(), config=cfg)
    x = torch.rand((1, 36, 44, 44), dtype=torch.float32) * 4.0

    torch.manual_seed(2028)
    y1 = aug(x.clone())
    torch.manual_seed(2028)
    y2 = aug(x.clone())

    assert y1.shape == x.shape
    assert y1.dtype == torch.float32
    assert torch.isfinite(y1).all()
    assert not torch.equal(y1, x)
    assert torch.equal(y1, y2)


if __name__ == "__main__":
    self_test()
    print("STEP 28 acquisition-robust augmentation self-test: PASS")
