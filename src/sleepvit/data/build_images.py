"""Subject recording -> labelled epoch images + manifest rows."""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils import get_logger
from .annotations import (
    first_scored_onset, hypnogram_to_epoch_labels, offset_from_edf_start, parse_hypnogram,
)
from .discovery import Subject
from .edf_preprocess import (
    ChannelResolutionError, epoch_signal, load_recording, resample_recording, standardize,
)
from .image_transforms import epoch_to_image

log = get_logger()

Setting = Tuple[str, int, str]  # (transform, fs, resample method)


def setting_name(transform: str, fs: int, resample: str) -> str:
    """e.g. ('gasf', 128, 'dwt') -> 'gasf_128hz_dwt'."""
    return f"{transform}_{int(fs)}hz_{resample}"


def parse_setting(name: str) -> Setting:
    transform, fs, method = name.split("_")
    return transform, int(fs.replace("hz", "")), method


def image_dir(processed_root: str | Path, name: str, sid: str) -> Path:
    return Path(processed_root) / "images" / name / sid


def manifest_path(processed_root: str | Path, name: str) -> Path:
    return Path(processed_root) / "manifests" / f"{name}.csv"


def _epoch_offset(subject: Subject, hypnogram: pd.DataFrame, cfg, start_time: str) -> int:
    offset = int(cfg.data.manual_offsets_epochs.get(subject.sid, 0)) if cfg.data.manual_offsets_epochs else 0
    if cfg.data.align == "edf_start":
        onset = first_scored_onset(hypnogram, cfg.data.epoch_length_s)
        if onset is not None:
            offset += max(offset_from_edf_start(onset, start_time, cfg.data.epoch_length_s), 0)
    return offset


def build_subject(subject: Subject, settings: Sequence[Setting], cfg, processed_root: str | Path,
                  overwrite: bool = False) -> Dict[str, List[dict]]:
    """Generate all requested image settings for one subject.

    Pre-processing is shared between settings with the same (fs, resample method); transforms that
    share them are generated from the same standardised epochs.
    Returns ``{setting_name: [manifest rows]}``.
    """
    if subject.txt_path is None:
        raise FileNotFoundError(f"{subject.sid}: no hypnogram .txt found")

    epoch_len = cfg.data.epoch_length_s
    hypnogram = parse_hypnogram(subject.txt_path)
    rec = load_recording(subject.edf_path, base_fs=cfg.data.required_fs,
                         avg_before_filter=cfg.preprocessing.avg_before_filter)
    offset = _epoch_offset(subject, hypnogram, cfg, rec.start_time)
    labels = hypnogram_to_epoch_labels(hypnogram, epoch_len, first_epoch=offset)

    rows: Dict[str, List[dict]] = {}
    by_rate: Dict[Tuple[int, str], List[str]] = {}
    for transform, fs, method in settings:
        by_rate.setdefault((fs, method), []).append(transform)

    for (fs, method), transforms in by_rate.items():
        sampled = resample_recording(rec, fs, method, epoch_len, cfg.preprocessing.dwt_wavelet)
        epochs = epoch_signal(sampled.data, sampled.fs, epoch_len)
        n = min(len(epochs), len(labels))
        keep = np.flatnonzero(labels[:n] >= 0)
        epochs = epochs[keep]
        if cfg.preprocessing.standardize:
            epochs = standardize(epochs)

        for transform in transforms:
            name = setting_name(transform, fs, method)
            out_dir = image_dir(processed_root, name, subject.sid)
            out_dir.mkdir(parents=True, exist_ok=True)
            sub_rows = []
            for epoch_idx, epoch in zip(keep, epochs):
                path = out_dir / f"{subject.sid}_epoch{epoch_idx}.png"
                if overwrite or not path.exists():
                    epoch_to_image(epoch, transform, sampled.fs, cfg).save(path)
                sub_rows.append({
                    "image": str(path.relative_to(processed_root)).replace("\\", "/"),
                    "subject": subject.sid,
                    "group": subject.group,
                    "epoch": int(epoch_idx),
                    "stage": int(labels[epoch_idx]),
                })
            rows[name] = sub_rows
    return rows


def _worker(args):
    subject, settings, cfg, processed_root, overwrite = args
    try:
        return subject.sid, build_subject(subject, settings, cfg, processed_root, overwrite), None
    except (ChannelResolutionError, FileNotFoundError, ValueError) as exc:
        return subject.sid, {}, str(exc)


def build_dataset(subjects: Iterable[Subject], settings: Sequence[Setting], cfg, processed_root: str | Path,
                  workers: int = 1, overwrite: bool = False) -> Dict[str, pd.DataFrame]:
    """Process subjects (optionally in parallel) and write one manifest CSV per setting."""
    subjects = [s for s in subjects if s.sid not in set(cfg.data.exclude_subjects or [])]
    jobs = [(s, settings, cfg, str(processed_root), overwrite) for s in subjects]
    all_rows: Dict[str, List[dict]] = {setting_name(*s): [] for s in settings}
    failures: Dict[str, str] = {}

    def collect(sid, rows, err):
        if err:
            failures[sid] = err
            log.warning("Skipped %s: %s", sid, err)
            return
        for name, sub_rows in rows.items():
            all_rows[name].extend(sub_rows)
        log.info("Finished %s (%d epochs)", sid, len(next(iter(rows.values()), [])))

    start = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed([pool.submit(_worker, j) for j in jobs]):
                collect(*fut.result())
    else:
        for job in jobs:
            collect(*_worker(job))

    manifests = {}
    for name, rows in all_rows.items():
        df = pd.DataFrame(rows).sort_values(["subject", "epoch"]).reset_index(drop=True) if rows else pd.DataFrame(
            columns=["image", "subject", "group", "epoch", "stage"])
        path = manifest_path(processed_root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        manifests[name] = df
        log.info("Wrote %s (%d rows)", path, len(df))
    log.info("Done in %.1f min; %d subjects skipped", (time.time() - start) / 60, len(failures))
    if failures:
        pd.Series(failures, name="reason").to_csv(Path(processed_root) / "manifests" / "skipped_subjects.csv")
    return manifests
