"""Inventory of the CAP recordings: sampling rates, hypnogram availability, stage counts (Tables 4.1/4.2).

    python scripts/00_inspect_dataset.py
"""
import argparse
from pathlib import Path

import pandas as pd

from sleepvit.config import add_common_args, load_config
from sleepvit.constants import STAGE_NAMES
from sleepvit.data.annotations import hypnogram_to_epoch_labels, parse_hypnogram
from sleepvit.data.discovery import discover_subjects
from sleepvit.utils import ensure_dir, get_logger

log = get_logger()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)

    subjects = discover_subjects(cfg.paths.raw_data, required_fs=None)
    rows = []
    for s in subjects:
        counts = {name: 0 for name in STAGE_NAMES}
        if s.txt_path is not None:
            labels = hypnogram_to_epoch_labels(parse_hypnogram(s.txt_path), cfg.data.epoch_length_s)
            for k, name in enumerate(STAGE_NAMES):
                counts[name] = int((labels == k).sum())
        rows.append({"subject": s.sid, "group": s.group, "fs_eeg": s.fs, "has_hypnogram": s.has_hypnogram,
                     "selected": s.fs == cfg.data.required_fs and s.has_hypnogram, **counts})
    df = pd.DataFrame(rows)
    out = ensure_dir(Path(cfg.paths.results) / "dataset")
    df.to_csv(out / "subjects.csv", index=False)

    chosen = df[df["selected"]]
    print(f"{len(df)} subjects found, {len(chosen)} recorded at {cfg.data.required_fs} Hz with a hypnogram\n")
    print("Subjects per group (selected):")
    print(chosen.groupby("group").size().to_string(), "\n")
    totals = chosen[STAGE_NAMES].sum()
    summary = pd.DataFrame({"epochs": totals, "percent": 100 * totals / totals.sum()}).round(2)
    print("Epochs per stage (selected subjects, cf. Table 4.2):")
    print(summary.to_string())
    by_group = chosen.groupby("group")[STAGE_NAMES].sum()
    by_group["total"] = by_group.sum(axis=1)
    by_group.to_csv(out / "stage_counts_by_group.csv")
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
