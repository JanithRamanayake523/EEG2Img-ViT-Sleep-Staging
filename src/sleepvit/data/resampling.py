"""Frequency down-sampling by FFT or by the discrete wavelet transform (thesis Section 6.1.2)."""
from __future__ import annotations

import numpy as np
import pywt
from scipy import signal


def resample_fft(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Fourier-domain resampling: drop spectral content above the new Nyquist, then inverse FFT."""
    n_out = int(round(x.shape[-1] * fs_out / fs_in))
    return signal.resample(x, n_out, axis=-1)


def resample_dwt(x: np.ndarray, fs_in: float, fs_out: float, wavelet: str = "db4") -> np.ndarray:
    """Down-sample by repeatedly keeping the DWT approximation coefficients.

    Each level halves the sampling rate, so `fs_in / fs_out` must be a power of two
    (512 -> 256 -> 128 -> 64 in the experiments). The periodisation mode makes every level exactly
    half as long as its input.
    """
    ratio = fs_in / fs_out
    levels = int(round(np.log2(ratio)))
    if levels < 0 or not np.isclose(2 ** levels, ratio):
        raise ValueError(f"DWT resampling needs fs_in/fs_out to be a power of two, got {fs_in}/{fs_out}")
    n = x.shape[-1] - (x.shape[-1] % (2 ** levels))
    approx = x[..., :n]
    for _ in range(levels):
        approx, _detail = pywt.dwt(approx, wavelet, mode="periodization", axis=-1)
    return approx


def resample(x: np.ndarray, fs_in: float, fs_out: float, method: str, wavelet: str = "db4") -> np.ndarray:
    """Dispatch on ``method``: ``orig`` (no-op), ``fft`` or ``dwt``."""
    if method == "orig" or fs_in == fs_out:
        return x
    if method == "fft":
        return resample_fft(x, fs_in, fs_out)
    if method == "dwt":
        return resample_dwt(x, fs_in, fs_out, wavelet)
    raise ValueError(f"Unknown resampling method: {method}")
