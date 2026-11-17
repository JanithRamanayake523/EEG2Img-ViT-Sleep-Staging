"""Manifest handling: dataset subsets (DS1-DS9), balancing, up-sampling and train/validation splits."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..constants import DATASET_GROUPS


def load_manifest(processed_root: str | Path, name: str) -> pd.DataFrame:
    """Read a manifest and add an absolute ``path`` column."""
    processed_root = Path(processed_root)
    df = pd.read_csv(processed_root / "manifests" / f"{name}.csv")
    df["path"] = df["image"].map(lambda p: str(processed_root / p))
    return df


def select_dataset(df: pd.DataFrame, ds_id: str) -> pd.DataFrame:
    """Rows of dataset DS1..DS9 (DS9 = every subject)."""
    if ds_id not in DATASET_GROUPS:
        raise KeyError(f"Unknown dataset {ds_id}; expected one of {list(DATASET_GROUPS)}")
    return df[df["group"].isin(DATASET_GROUPS[ds_id])].reset_index(drop=True)


def balanced_sample(df: pd.DataFrame, per_class: int, seed: int = 42) -> pd.DataFrame:
    """Draw ``per_class`` rows per stage; classes with fewer rows are sampled with replacement."""
    parts = []
    for _stage, group in df.groupby("stage"):
        parts.append(group.sample(n=per_class, replace=len(group) < per_class, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def upsample(df: pd.DataFrame, per_class: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Balance by duplicating minority-class rows up to the size of the largest class (or ``per_class``)."""
    target = per_class or int(df["stage"].value_counts().max())
    parts = []
    for _stage, group in df.groupby("stage"):
        extra = target - len(group)
        parts.append(group if extra <= 0 else pd.concat(
            [group, group.sample(n=extra, replace=True, random_state=seed)]))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split(df: pd.DataFrame, val_fraction: float, seed: int = 42, by: str = "epoch") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/validation split.

    ``by="epoch"``  : stratified random split of epochs (the thesis protocol; neighbouring epochs of the
                      same night can land on both sides, which makes the estimate optimistic).
    ``by="subject"``: whole subjects are held out, which measures generalisation to unseen people.
    """
    if by == "epoch":
        train, val = train_test_split(df, test_size=val_fraction, stratify=df["stage"], random_state=seed)
    elif by == "subject":
        rng = np.random.RandomState(seed)
        subjects = np.array(sorted(df["subject"].unique()))
        rng.shuffle(subjects)
        n_val = max(1, int(round(len(subjects) * val_fraction)))
        val_subjects = set(subjects[:n_val])
        val = df[df["subject"].isin(val_subjects)]
        train = df[~df["subject"].isin(val_subjects)]
    else:
        raise ValueError(f"split by must be 'epoch' or 'subject', got {by}")
    return train.reset_index(drop=True), val.reset_index(drop=True)


def synthetic_manifest(root: str | Path, name: str, per_class: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Manifest of GAN images stored as ``<root>/<name>/class_<k>/*.png``."""
    root = Path(root)
    rows = []
    for class_dir in sorted((root / name).glob("class_*")):
        stage = int(class_dir.name.split("_")[1])
        files = sorted(class_dir.glob("*.png"))
        for f in files:
            rows.append({"path": str(f), "image": str(f.relative_to(root)), "subject": "synthetic",
                         "group": "synthetic", "epoch": -1, "stage": stage})
    df = pd.DataFrame(rows)
    if per_class and not df.empty:
        df = pd.concat([g.sample(n=min(per_class, len(g)), random_state=seed) for _, g in df.groupby("stage")])
    return df.reset_index(drop=True)
