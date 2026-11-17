"""Load a CAP EDF recording and turn it into filtered, standardised 30 s epochs of 16 channels."""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import signal

from ..constants import (
    CANONICAL_CHANNELS,
    CHANNEL_ALIASES,
    CHANNEL_FALLBACKS,
    EEG_CHANNELS,
    FILTER_BANDS,
    NOTCH_HZ,
    SOURCE_CHANNELS,
)
from .resampling import resample


class ChannelResolutionError(RuntimeError):
    """Raised when a required channel (or any fallback) is missing from a recording."""


@dataclass
class Recording:
    data: np.ndarray            # (16, n_samples), microvolts, canonical channel order
    fs: float                   # sampling rate of `data`
    sources: Dict[str, str]     # canonical channel -> EDF channel it was read from
    start_time: str             # EDF header start time 'hh.mm.ss'


def _norm(name: str) -> str:
    return re.sub(r"^eeg[\s_-]*", "", name.strip().lower())


def resolve_channels(available: List[str]) -> Dict[str, str]:
    """Map each of the 15 source channels to an EDF channel name.

    Order of preference: the channel's own aliases, then its fallbacks (a neighbouring or
    contralateral derivation). This replaces the hand-written ``channel_mapping`` blocks that were
    copy-pasted into every per-subject notebook.
    """
    lookup = {_norm(n): n for n in available}
    sources: Dict[str, str] = {}
    for channel in SOURCE_CHANNELS:
        candidates = CHANNEL_ALIASES[channel] + CHANNEL_FALLBACKS[channel]
        match = next((lookup[_norm(c)] for c in candidates if _norm(c) in lookup), None)
        if match is None:
            raise ChannelResolutionError(f"No source found for channel {channel}; available: {available}")
        sources[channel] = match
    return sources


def _filter_band(channel: str):
    if channel in EEG_CHANNELS:
        return FILTER_BANDS["eeg"]
    return FILTER_BANDS[channel.lower()] if channel.lower() in FILTER_BANDS else None


def _bandpass_and_notch(x: np.ndarray, fs: float, band, notch: bool) -> np.ndarray:
    import mne

    x = x.astype(np.float64)
    nyquist = fs / 2.0
    if band is not None:
        low, high = band
        high = min(high, nyquist - 1.0)  # never ask for a low-pass above Nyquist
        x = mne.filter.filter_data(x, fs, low, high, method="fir", fir_design="firwin", verbose="ERROR")
    if notch and NOTCH_HZ < nyquist:
        x = mne.filter.notch_filter(x, fs, NOTCH_HZ, method="fir", verbose="ERROR")
    return x


def load_recording(edf_path: str | Path, base_fs: float = 512.0, avg_before_filter: bool = True) -> Recording:
    """Read, harmonise channels, add the average channel and apply the Table 6.1 filters.

    Note: the original notebooks called ``raw.filter(l_freq=35, h_freq=0.5)`` which in MNE is a
    *band-stop* between 0.5 and 35 Hz. Here ``l_freq`` is the high-pass edge, i.e. a real band-pass.
    """
    import mne

    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    sources = resolve_channels(raw.ch_names)
    needed = sorted(set(sources.values()), key=raw.ch_names.index)
    raw.pick(needed)
    raw.load_data()
    fs = float(raw.info["sfreq"])
    data_by_name = dict(zip(raw.ch_names, raw.get_data() * 1e6))  # volts -> microvolts
    data = np.stack([data_by_name[sources[ch]] for ch in SOURCE_CHANNELS])
    start = raw.info["meas_date"]
    start_time = start.strftime("%H.%M.%S") if start is not None else "00.00.00"
    del raw

    if fs != base_fs:  # a few CAP recordings are not at 512 Hz; bring them to a common base rate
        frac = Fraction(base_fs / fs).limit_denominator(1000)
        data = signal.resample_poly(data, frac.numerator, frac.denominator, axis=-1)
        fs = base_fs

    channels = list(SOURCE_CHANNELS)
    avg = data.mean(axis=0, keepdims=True)
    filtered = []
    for ch, row in zip(channels, data):
        filtered.append(_bandpass_and_notch(row, fs, _filter_band(ch), notch=True))
    if avg_before_filter:  # as in the notebooks: the average only receives the notch filter
        avg_row = _bandpass_and_notch(avg[0], fs, None, notch=True)
    else:
        avg_row = np.mean(filtered, axis=0)
    out = np.vstack([np.stack(filtered), avg_row[None]]).astype(np.float32)
    assert out.shape[0] == len(CANONICAL_CHANNELS)
    return Recording(out, fs, sources, start_time)


def epoch_signal(data: np.ndarray, fs: float, epoch_length: int = 30) -> np.ndarray:
    """Cut (channels, samples) into non-overlapping epochs -> (n_epochs, channels, samples_per_epoch)."""
    spe = int(round(fs * epoch_length))
    n_epochs = data.shape[-1] // spe
    trimmed = data[:, : n_epochs * spe]
    return trimmed.reshape(data.shape[0], n_epochs, spe).transpose(1, 0, 2)


def resample_recording(rec: Recording, fs_out: float, method: str, epoch_length: int = 30,
                       wavelet: str = "db4") -> Recording:
    """Resample the whole recording (not epoch by epoch, which would add edge artefacts)."""
    if method == "orig" or fs_out == rec.fs:
        return rec
    spe_in = int(round(rec.fs * epoch_length))
    n_epochs = rec.data.shape[-1] // spe_in
    data = rec.data[:, : n_epochs * spe_in].astype(np.float64)  # whole epochs -> FFT-friendly length
    out = resample(data, rec.fs, fs_out, method, wavelet).astype(np.float32)
    return Recording(out, fs_out, rec.sources, rec.start_time)


def standardize(epochs: np.ndarray) -> np.ndarray:
    """Z-score every channel over all given epochs (the notebooks' ``StandardScaler`` step)."""
    mean = epochs.mean(axis=(0, 2), keepdims=True)
    std = epochs.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return (epochs - mean) / std
