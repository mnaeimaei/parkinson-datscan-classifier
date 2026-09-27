from __future__ import annotations

from dataclasses import asdict
from typing import Any
import pandas as pd
import torch

from step12f_dataloader_factory import LoaderConfig, build_dataset, build_loader

EXPECTED = {
    "A": {"whole": (1, 160, 192, 192)},
    "B": {"roi": (1, 36, 44, 44)},
    "C": {"whole": (1, 160, 192, 192), "roi": (1, 36, 44, 44)},
}


def _validate_batch(batch: dict[str, Any], scenario: str) -> dict[str, Any]:
    result: dict[str, Any] = {"scenario": scenario, "passed": True}
    for key, sample_shape in EXPECTED[scenario].items():
        tensor = batch[key]
        expected_batch_tail = tuple(sample_shape)
        if tensor.dtype != torch.float32:
            raise AssertionError(f"{scenario}/{key}: dtype={tensor.dtype}, expected float32")
        if tuple(tensor.shape[1:]) != expected_batch_tail:
            raise AssertionError(
                f"{scenario}/{key}: batch shape={tuple(tensor.shape)}, expected [B,{','.join(map(str, sample_shape))}]"
            )
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{scenario}/{key}: NaN/Inf detected")
        if tensor.numel() == 0:
            raise AssertionError(f"{scenario}/{key}: empty tensor")
        result[f"{key}_batch_shape"] = list(tensor.shape)
        result[f"{key}_min"] = float(tensor.min().item())
        result[f"{key}_max"] = float(tensor.max().item())

    labels = batch["label"]
    if labels.dtype != torch.float32:
        raise AssertionError(f"Labels dtype={labels.dtype}, expected float32")
    observed = set(labels.detach().cpu().numpy().tolist())
    if not observed.issubset({0.0, 1.0}):
        raise AssertionError(f"Invalid labels in batch: {observed}")
    if len(batch["uid"]) != labels.shape[0]:
        raise AssertionError("UID count does not equal batch label count")
    result["uids"] = list(batch["uid"])
    result["labels"] = [float(x) for x in labels.tolist()]
    return result


def smoke_test_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, scenario: str, cfg: LoaderConfig) -> dict[str, Any]:
    overlap = set(train_df["uid"]) & set(val_df["uid"])
    if overlap:
        raise AssertionError(f"Train/validation overlap: {list(overlap)[:10]}")

    # Step-12 baseline smoke test intentionally uses augment=False.
    train_ds = build_dataset(scenario, train_df, augment=False)
    val_ds = build_dataset(scenario, val_df, augment=False)
    train_loader = build_loader(train_ds, training=True, config=cfg)
    val_loader = build_loader(val_ds, training=False, config=cfg)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    return {
        "scenario": scenario,
        "train_subjects": len(train_ds),
        "val_subjects": len(val_ds),
        "uid_overlap": 0,
        "loader_config": asdict(cfg),
        "train_batch": _validate_batch(train_batch, scenario),
        "val_batch": _validate_batch(val_batch, scenario),
        "passed": True,
    }
