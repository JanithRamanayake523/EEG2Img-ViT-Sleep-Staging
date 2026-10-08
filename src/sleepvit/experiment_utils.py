"""Helpers shared by the experiment scripts: sampling a fixed set of epochs and keeping splits identical."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .data.build_images import manifest_path, setting_name
from .data.manifest import balanced_sample, load_manifest, select_dataset, split
from .utils import get_logger

log = get_logger()


def best_setting_name(cfg) -> str:
    b = cfg.best_setting
    return setting_name(b.transform, b.fs, b.resample)


def require_manifest(cfg, name: str) -> pd.DataFrame:
    """Load a manifest or explain exactly which command creates it."""
    if not manifest_path(cfg.paths.processed, name).exists():
        raise FileNotFoundError(
            f"Manifest for setting '{name}' not found under {cfg.paths.processed}. Create it with:\n"
            f"  python scripts/01_build_images.py --settings {name}")
    return load_manifest(cfg.paths.processed, name)


def fixed_split_keys(cfg, ref_df: pd.DataFrame, per_class: int | None, val_fraction: float,
                     ds_id: str = "DS9") -> pd.DataFrame:
    """Sample epochs once and split them; returns ``(subject, epoch, stage, split)``.

    Experiments that compare image settings (Exp. 1, 2) must classify *the same epochs*, so the sample and
    the train/validation assignment are drawn once on a reference manifest and then re-applied to every
    other manifest with :func:`apply_split_keys`.
    """
    df = select_dataset(ref_df, ds_id)
    if per_class:
        df = balanced_sample(df, per_class, cfg.seed)
    train, val = split(df, val_fraction, cfg.seed, cfg.experiments.split_by)
    keys = pd.concat([train.assign(split="train"), val.assign(split="val")])
    return keys[["subject", "epoch", "stage", "split"]].drop_duplicates(["subject", "epoch", "split"]).reset_index(drop=True)


def apply_split_keys(df: pd.DataFrame, keys: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = df.merge(keys[["subject", "epoch", "split"]], on=["subject", "epoch"], how="inner")
    return (merged[merged["split"] == "train"].reset_index(drop=True),
            merged[merged["split"] == "val"].reset_index(drop=True))


def save_keys(keys: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys.to_csv(path, index=False)
