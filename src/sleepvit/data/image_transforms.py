"""Time series -> image transformations (thesis Section 6.1.1) and the 4x4 channel grid."""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.signal import spectrogram

from ..constants import IMAGE_SIZE, TRANSFORMS


def _colormap(name: str):
    import matplotlib

    return matplotlib.colormaps[name]


def _minmax(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Row-wise min-max scaling to [lo, hi]; constant rows map to the midpoint (no division by 0)."""
    mn = x.min(axis=-1, keepdims=True)
    mx = x.max(axis=-1, keepdims=True)
    span = mx - mn
    safe = np.where(span == 0, 1.0, span)
    out = (x - mn) / safe
    out = np.where(span == 0, 0.5, out)
    return lo + out * (hi - lo)


def _to_rgb(matrix01: np.ndarray, cmap) -> np.ndarray:
    """(H, W) floats in [0, 1] -> (H, W, 3) uint8."""
    return (cmap(np.clip(matrix01, 0, 1))[..., :3] * 255).astype(np.uint8)


def gaf_images(epoch: np.ndarray, method: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """Gramian Angular Summation/Difference Field of every channel -> (channels, size, size) in [-1, 1].

    GASF(i, j) = cos(phi_i + phi_j),  GADF(i, j) = sin(phi_i - phi_j),  phi = arccos(x_scaled).
    The series is PAA-reduced to ``size`` points by pyts before the Gram matrix is formed.
    """
    from pyts.image import GramianAngularField

    scaled = np.clip(_minmax(epoch, -1.0, 1.0), -1.0, 1.0)
    gaf = GramianAngularField(image_size=size, method="summation" if method == "gasf" else "difference",
                              sample_range=None)
    return gaf.fit_transform(scaled)


def mtf_images(epoch: np.ndarray, size: int = IMAGE_SIZE, n_bins: int = 8, strategy: str = "quantile") -> np.ndarray:
    """Markov Transition Field of every channel -> (channels, size, size) in [0, 1]."""
    from pyts.image import MarkovTransitionField

    scaled = _minmax(epoch, 0.0, 1.0)
    mtf = MarkovTransitionField(image_size=size, n_bins=n_bins, strategy=strategy)
    return mtf.fit_transform(scaled)


def spectrogram_images(epoch: np.ndarray, fs: float, size: int = IMAGE_SIZE, nperseg: int = 64,
                       noverlap: int = 32, log: bool = False) -> np.ndarray:
    """STFT power spectrogram of every channel, resized to (channels, size, size) in [0, 1]."""
    out = np.empty((epoch.shape[0], size, size), dtype=np.float32)
    for i, channel in enumerate(epoch):
        _, _, sxx = spectrogram(channel, fs=fs, nperseg=nperseg, noverlap=noverlap)
        if log:
            sxx = np.log1p(sxx)
        sxx = _minmax(sxx.reshape(1, -1), 0.0, 1.0).reshape(sxx.shape)
        out[i] = np.asarray(Image.fromarray(sxx.astype(np.float32), mode="F").resize((size, size), Image.BILINEAR))
    return out


def channel_images(epoch: np.ndarray, transform: str, fs: float, cfg) -> np.ndarray:
    """Dispatch to a transform. Returns per-channel matrices scaled to [0, 1] for colour mapping."""
    size = cfg.images.size
    if transform in ("gasf", "gadf"):
        return (gaf_images(epoch, transform, size) + 1.0) / 2.0
    if transform == "mtf":
        return mtf_images(epoch, size, cfg.images.mtf.n_bins, cfg.images.mtf.strategy)
    if transform == "spectrogram":
        s = cfg.images.spectrogram
        return spectrogram_images(epoch, fs, size, s.nperseg, s.noverlap, s.log)
    raise ValueError(f"Unknown transform '{transform}'. Choose from {TRANSFORMS}.")


def epoch_to_image(epoch: np.ndarray, transform: str, fs: float, cfg) -> Image.Image:
    """16 channels of one epoch -> one RGB image.

    Each channel becomes a ``size x size`` colour image; the 16 images are tiled on a 4x4 grid and the
    result is resized back to ``size x size`` (Figure 6.3). Min-max scaling per channel and epoch is
    part of every transformation, so absolute amplitudes are not encoded.
    """
    grid, size = cfg.images.grid, cfg.images.size
    if epoch.shape[0] != grid * grid:
        raise ValueError(f"Expected {grid * grid} channels, got {epoch.shape[0]}")
    cmap = _colormap(cfg.images.colormap)
    mats = channel_images(epoch, transform, fs, cfg)
    # normalise each channel image to [0, 1] for the colour map (as the notebooks did)
    flat = mats.reshape(mats.shape[0], -1)
    mats = _minmax(flat, 0.0, 1.0).reshape(mats.shape)
    tiles = np.stack([_to_rgb(m, cmap) for m in mats])                      # (16, size, size, 3)
    mosaic = tiles.reshape(grid, grid, size, size, 3).transpose(0, 2, 1, 3, 4).reshape(grid * size, grid * size, 3)
    return Image.fromarray(mosaic).resize((size, size), Image.BICUBIC)
