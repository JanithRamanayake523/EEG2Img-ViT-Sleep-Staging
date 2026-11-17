"""Find CAP subjects on disk and read cheap EDF header information."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..constants import CHANNEL_ALIASES, EEG_CHANNELS, GROUP_PREFIXES

_SUBJECT_RE = re.compile(r"^(" + "|".join(sorted(GROUP_PREFIXES, key=len, reverse=True)) + r")(\d+)$", re.I)


@dataclass
class Subject:
    sid: str            # e.g. "nfle12"
    group: str          # e.g. "nfle" -> see GROUP_PREFIXES values
    edf_path: Path
    txt_path: Optional[Path]
    fs: Optional[float] = None

    @property
    def has_hypnogram(self) -> bool:
        return self.txt_path is not None


def parse_subject_id(stem: str):
    """'nfle12' -> ('nfle12', 'nfle'); returns None for names that are not CAP subjects."""
    match = _SUBJECT_RE.match(stem.strip())
    if not match:
        return None
    return stem.lower(), GROUP_PREFIXES[match.group(1).lower()]


def read_edf_header(path: str | Path):
    """Return (start_time 'hh.mm.ss', labels, per-channel sampling rates, n_records) without MNE."""
    with open(path, "rb") as f:
        header = f.read(256)
        n_signals = int(header[252:256])
        n_records = int(header[236:244])
        record_duration = float(header[244:252])
        labels = f.read(16 * n_signals)
        f.seek(256 + (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80) * n_signals)
        samples_per_record = [int(f.read(8)) for _ in range(n_signals)]
    names = [labels[i * 16:(i + 1) * 16].decode("latin-1").strip() for i in range(n_signals)]
    rates = [s / record_duration for s in samples_per_record]
    return header[176:184].decode().strip(), names, rates, n_records


def eeg_sampling_rate(path: str | Path) -> Optional[float]:
    """Sampling rate of the EEG derivations (maximum over channels that match known aliases)."""
    _, names, rates, _ = read_edf_header(path)
    aliases = {a.lower() for ch in EEG_CHANNELS for a in CHANNEL_ALIASES[ch]}
    eeg_rates = [r for n, r in zip(names, rates) if n.lower() in aliases]
    return max(eeg_rates) if eeg_rates else None


def discover_subjects(raw_root: str | Path, required_fs: Optional[float] = None) -> List[Subject]:
    """Recursively scan `raw_root` for ``<sid>.edf`` and the matching ``<sid>.txt`` hypnogram."""
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_root}")

    txt_index = {p.stem.lower(): p for p in raw_root.rglob("*.txt")}
    subjects: List[Subject] = []
    seen = set()
    for edf in sorted(raw_root.rglob("*.edf")):
        parsed = parse_subject_id(edf.stem)
        if parsed is None or parsed[0] in seen:
            continue
        sid, group = parsed
        seen.add(sid)
        fs = eeg_sampling_rate(edf)
        if required_fs is not None and fs != required_fs:
            continue
        subjects.append(Subject(sid, group, edf, txt_index.get(sid), fs))
    return subjects
