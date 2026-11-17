"""Parse RemLogic hypnogram exports (the ``<subject>.txt`` files of the CAP database)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..constants import STAGE_TO_ID

DAY_SECONDS = 24 * 3600


def _to_seconds(token: str) -> float:
    """'22:18:02' or '22.18.02' -> seconds since midnight."""
    h, m, s = (int(x) for x in re.split(r"[:.]", token.strip())[:3])
    return h * 3600 + m * 60 + s


def parse_hypnogram(path: str | Path) -> pd.DataFrame:
    """Return a table with columns ``stage, event, onset_s, duration_s``.

    Files are tab separated; some carry an extra ``Position`` column. Onsets are unwrapped so that
    recordings that cross midnight keep increasing (the original notebooks added 24 h by hand).
    """
    with open(path, "r", encoding="latin-1") as f:
        lines = f.read().splitlines()

    header_idx = next((i for i, line in enumerate(lines) if line.startswith("Sleep Stage")), None)
    if header_idx is None:
        raise ValueError(f"No 'Sleep Stage' header found in {path}")
    columns = [c.strip() for c in lines[header_idx].split("\t")]
    time_col = next(c for c in columns if c.lower().startswith("time"))
    event_col = next(c for c in columns if c.lower() == "event")
    dur_col = next(c for c in columns if c.lower().startswith("duration"))

    rows = []
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < len(columns):
            continue
        record = dict(zip(columns, (p.strip() for p in parts)))
        try:
            onset = _to_seconds(record[time_col])
            duration = float(record[dur_col])
        except (ValueError, IndexError):
            continue
        rows.append({
            "stage": record["Sleep Stage"],
            "event": record[event_col],
            "onset_s": onset,
            "duration_s": duration,
        })

    df = pd.DataFrame(rows, columns=["stage", "event", "onset_s", "duration_s"])
    day, prev = 0, None
    unwrapped = []
    for onset in df["onset_s"]:
        if prev is not None and onset < prev - 6 * 3600:
            day += 1
        prev = onset
        unwrapped.append(onset + day * DAY_SECONDS)
    df["onset_s"] = unwrapped
    return df


def hypnogram_to_epoch_labels(
    df: pd.DataFrame,
    epoch_length: int = 30,
    first_epoch: int = 0,
) -> np.ndarray:
    """Convert event rows into an array of stage ids indexed by epoch number.

    Only 30 s sleep-stage rows are used (``SLEEP-*`` events); arousal (MCAP) events are ignored.
    ``first_epoch`` is the signal epoch that the first scored epoch corresponds to. Unscored
    positions are ``-1``.
    """
    scored = df[df["event"].str.upper().str.startswith("SLEEP-") & ~df["event"].str.upper().str.contains("UNSCORED")]
    scored = scored[scored["duration_s"] == epoch_length]
    if scored.empty:  # fall back to the notebooks' rule: every 30 s row
        scored = df[df["duration_s"] == epoch_length]
    scored = scored[scored["stage"].isin(STAGE_TO_ID)]
    if scored.empty:
        return np.zeros(0, dtype=np.int64)

    t0 = scored["onset_s"].iloc[0]
    idx = np.rint((scored["onset_s"].to_numpy() - t0) / epoch_length).astype(int) + first_epoch
    labels = np.full(idx.max() + 1, -1, dtype=np.int64)
    labels[idx] = scored["stage"].map(STAGE_TO_ID).to_numpy()
    return labels


def first_scored_onset(df: pd.DataFrame, epoch_length: int = 30) -> Optional[float]:
    scored = df[df["event"].str.upper().str.startswith("SLEEP-") & (df["duration_s"] == epoch_length)]
    return None if scored.empty else float(scored["onset_s"].iloc[0])


def offset_from_edf_start(first_onset_s: float, edf_start: str, epoch_length: int = 30) -> int:
    """Number of epochs between the EDF start time and the first scored epoch.

    ``edf_start`` is the header time ('hh.mm.ss'). Negative or implausibly large values (EDF headers
    in the CAP database are sometimes anonymised) are rejected by the caller.
    """
    delta = (first_onset_s - _to_seconds(edf_start)) % DAY_SECONDS
    if delta > DAY_SECONDS / 2:
        delta -= DAY_SECONDS
    return int(round(delta / epoch_length))
