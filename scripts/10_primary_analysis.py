"""Primary analysis (Chapter 5): artefact detection and inter-subject variability.

    python scripts/10_primary_analysis.py --artifact-subject n1 --subjects n1 ins8 nfle10 --stage S3

Produces in ``results/primary_analysis``:
  * <sid>_psd_raw.png        power spectrum of the unfiltered EEG (60 Hz mains line, cardiac/EMG bumps)
  * <sid>_ica_scores.png     ICA component correlation with the EOG channel (eye-movement components)
  * stage_psd.png            average EEG power spectrum of one stage for several subjects (Fig. 5.3 analogue)
  * connectivity_<sid>.png   channel-by-channel absolute correlation during that stage (Fig. 5.4 analogue)

The CAP derivations are bipolar, so scalp topographies of ICA components (Fig. 5.2) cannot be drawn
without inventing electrode positions; component scores against the EOG channel are produced instead.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import welch

from sleepvit.config import add_common_args, load_config
from sleepvit.constants import EEG_CHANNELS, STAGE_NAMES, STAGE_TO_ID
from sleepvit.data.annotations import hypnogram_to_epoch_labels, parse_hypnogram
from sleepvit.data.discovery import discover_subjects
from sleepvit.data.edf_preprocess import epoch_signal, load_recording, resolve_channels
from sleepvit.utils import ensure_dir, get_logger

log = get_logger()


def artifact_plots(subject, out: Path, minutes: int = 20) -> None:
    import mne

    raw = mne.io.read_raw_edf(str(subject.edf_path), preload=False, verbose="ERROR")
    sources = resolve_channels(raw.ch_names)
    eeg = sorted({sources[c] for c in EEG_CHANNELS}, key=raw.ch_names.index)
    picks = eeg + [sources["EOG"]]
    raw.pick(picks).crop(tmax=min(minutes * 60, raw.times[-1])).load_data()

    spectrum = raw.copy().pick(eeg).compute_psd(fmax=100, verbose="ERROR")
    fig = spectrum.plot(show=False)
    fig.savefig(out / f"{subject.sid}_psd_raw.png", dpi=150)
    plt.close("all")

    filtered = raw.copy().filter(1.0, 40.0, verbose="ERROR")
    filtered.set_channel_types({sources["EOG"]: "eog"})
    ica = mne.preprocessing.ICA(n_components=min(10, len(eeg)), random_state=42, max_iter="auto")
    ica.fit(filtered.copy().pick(eeg), verbose="ERROR")
    eog_idx, scores = ica.find_bads_eog(filtered, ch_name=sources["EOG"], verbose="ERROR")
    log.info("%s: ICA components flagged as eye movement: %s", subject.sid, eog_idx)
    fig = ica.plot_scores(scores, exclude=eog_idx, show=False)
    fig.savefig(out / f"{subject.sid}_ica_scores.png", dpi=150)
    plt.close("all")


def stage_epochs(subject, cfg, stage_id: int):
    rec = load_recording(subject.edf_path, cfg.data.required_fs, cfg.preprocessing.avg_before_filter)
    epochs = epoch_signal(rec.data, rec.fs, cfg.data.epoch_length_s)
    labels = hypnogram_to_epoch_labels(parse_hypnogram(subject.txt_path), cfg.data.epoch_length_s)
    n = min(len(epochs), len(labels))
    return epochs[:n][labels[:n] == stage_id][:, : len(EEG_CHANNELS)], rec.fs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--artifact-subject", default=None)
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--stage", default="S3", choices=[s for s in STAGE_TO_ID if s != "MT"])
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    out = ensure_dir(Path(cfg.paths.results) / "primary_analysis")
    by_id = {s.sid: s for s in discover_subjects(cfg.paths.raw_data, required_fs=cfg.data.required_fs)}

    if args.artifact_subject:
        artifact_plots(by_id[args.artifact_subject], out)

    stage_id = STAGE_TO_ID[args.stage]
    fig, ax = plt.subplots(figsize=(7, 5))
    for sid in args.subjects:
        epochs, fs = stage_epochs(by_id[sid], cfg, stage_id)
        if len(epochs) == 0:
            log.warning("%s has no %s epochs", sid, args.stage)
            continue
        freqs, pxx = welch(epochs, fs=fs, nperseg=1024, axis=-1)
        mean_psd = pxx.mean(axis=(0, 1))
        band = (freqs >= 0.5) & (freqs <= 35)
        ax.semilogy(freqs[band], mean_psd[band], label=f"{sid} (n={len(epochs)})")

        corr = np.mean([np.abs(np.corrcoef(e)) for e in epochs], axis=0)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, vmin=0, vmax=1, cmap="viridis", xticklabels=EEG_CHANNELS, yticklabels=EEG_CHANNELS, ax=ax2)
        ax2.set_title(f"{sid}: channel connectivity ({args.stage})")
        fig2.tight_layout()
        fig2.savefig(out / f"connectivity_{sid}.png", dpi=150)
        plt.close(fig2)
    ax.set(xlabel="Frequency (Hz)", ylabel="Power spectral density", title=f"EEG spectrum, stage {args.stage}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "stage_psd.png", dpi=150)
    log.info("wrote figures to %s", out)


if __name__ == "__main__":
    main()
