"""PyTorch datasets and data loaders over manifests."""
from __future__ import annotations

from typing import Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


def classifier_transform(mean: Sequence[float], std: Sequence[float], augment: str = "none", size: int = 224):
    ops = []
    if augment == "light":
        ops.append(T.ColorJitter(brightness=0.1, contrast=0.2))
    ops += [T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


def gan_transform(size: int = 224):
    """Images scaled to [-1, 1] to match the generator's tanh output."""
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])


class ManifestDataset(Dataset):
    """Returns ``(image_tensor, stage_id)`` for each manifest row (needs ``path`` and ``stage`` columns)."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.labels = df["stage"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as img:
            image = self.transform(img.convert("RGB"))
        return image, self.labels[i]


def make_loader(df: pd.DataFrame, transform, batch_size: int, shuffle: bool, num_workers: int = 0,
                drop_last: bool = False) -> DataLoader:
    return DataLoader(
        ManifestDataset(df, transform), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=drop_last, persistent_workers=num_workers > 0,
    )
