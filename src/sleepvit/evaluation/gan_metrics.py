"""Quality metrics for synthetic images: SSIM, Frechet Inception Distance and MPED (Section 7.5)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image


def load_images(paths: List[str | Path], size: int = 224) -> np.ndarray:
    """Load RGB images as float32 arrays in [0, 1], shape (N, H, W, 3)."""
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0)
    return np.stack(out)


def ssim_to_nearest_real(real: np.ndarray, fake: np.ndarray, max_pairs: int = 200) -> float:
    """Mean SSIM between each synthetic image and its nearest real image (pixel L2 distance).

    A random real/fake pairing is meaningless because epochs are unrelated, so each synthetic image is
    compared with the most similar real one: this measures whether it is structurally plausible.
    """
    from skimage.metrics import structural_similarity

    fake = fake[:max_pairs]
    r = torch.from_numpy(real.reshape(len(real), -1))
    scores = []
    for f in fake:
        j = torch.cdist(torch.from_numpy(f.reshape(1, -1)), r).argmin().item()
        scores.append(structural_similarity(f, real[j], channel_axis=-1, data_range=1.0))
    return float(np.mean(scores))


def mped(images: np.ndarray, max_images: int = 500) -> float:
    """Mean pairwise Euclidean distance between flattened images in [0, 1]; higher means more diverse."""
    x = torch.from_numpy(images[:max_images].reshape(min(len(images), max_images), -1))
    d = torch.cdist(x, x)
    n = d.shape[0]
    return float(d.sum().item() / (n * (n - 1)))


def fid(real: np.ndarray, fake: np.ndarray, device: torch.device) -> float:
    """Frechet Inception Distance via torchmetrics (needs ``torch-fidelity``)."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def feed(arr: np.ndarray, is_real: bool) -> None:
        for i in range(0, len(arr), 32):
            batch = torch.from_numpy(arr[i:i + 32]).permute(0, 3, 1, 2).to(device)
            metric.update(batch, real=is_real)

    feed(real, True)
    feed(fake, False)
    return float(metric.compute().item())


def evaluate_synthetic(real_paths: List[str | Path], fake_paths: List[str | Path], device: torch.device,
                       compute_fid: bool = True) -> Dict[str, float]:
    real, fake = load_images(real_paths), load_images(fake_paths)
    out = {"ssim": ssim_to_nearest_real(real, fake), "mped_fake": mped(fake), "mped_real": mped(real)}
    if compute_fid:
        try:
            out["fid"] = fid(real, fake, device)
        except ImportError as exc:  # torch-fidelity not installed
            out["fid"] = float("nan")
            out["fid_error"] = str(exc)
    return out
