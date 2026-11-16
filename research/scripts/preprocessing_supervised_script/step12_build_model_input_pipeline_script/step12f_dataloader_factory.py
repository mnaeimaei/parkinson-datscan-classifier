from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import torch
from torch.utils.data import DataLoader

from step12d_dataset_classes import WholeVolumeDataset, StriatalDataset, DualInputDataset
from step12e_augmentation import AugmentationConfig, ConservativeAffine3D
from step12g_reproducibility import seed_worker, make_generator


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    seed: int = 42


def build_dataset(scenario: str, frame: pd.DataFrame, augment: bool, aug_cfg: AugmentationConfig | None = None):
    scenario = scenario.upper()
    transform = None
    if augment:
        cfg = aug_cfg or AugmentationConfig(enabled=True)
        if not cfg.enabled:
            cfg = AugmentationConfig(
                enabled=True,
                probability=cfg.probability,
                max_rotation_deg=cfg.max_rotation_deg,
                max_translation_voxels=cfg.max_translation_voxels,
            )
        transform = ConservativeAffine3D(cfg)

    if scenario == "A":
        return WholeVolumeDataset(frame, transform)
    if scenario == "B":
        return StriatalDataset(frame, transform)
    if scenario == "C":
        return DualInputDataset(frame, transform)
    raise ValueError("scenario must be one of A, B, C")


def build_loader(dataset, training: bool, config: LoaderConfig) -> DataLoader:
    # persistent_workers is invalid when num_workers == 0.
    persistent = config.persistent_workers and config.num_workers > 0
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent,
        worker_init_fn=seed_worker if config.num_workers > 0 else None,
        generator=make_generator(config.seed),
        drop_last=False,
    )
