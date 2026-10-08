"""Experiment 2 - sampling frequency and resampling method (Section 6.2.3, Table 7.3)

Compares 512 Hz (original) with 256 / 128 / 64 Hz obtained by FFT or DWT, using the transformation that
won Experiment 1 (GASF by default). All settings classify the same epochs.

Prerequisite:
    python scripts/01_build_images.py --transforms gasf --fs 512 --resample orig
    python scripts/01_build_images.py --transforms gasf --fs 256 128 64 --resample fft dwt
"""
import argparse
from pathlib import Path

from sleepvit.config import add_common_args, load_config
from sleepvit.data.build_images import setting_name
from sleepvit.experiment_utils import apply_split_keys, fixed_split_keys, require_manifest, save_keys
from sleepvit.training.pipeline import run_classification, summarise
from sleepvit.utils import ensure_dir, get_logger, seed_everything

log = get_logger()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--transform", default=None, help="default: best_setting.transform")
    parser.add_argument("--fs", nargs="*", type=int, default=[256, 128, 64])
    parser.add_argument("--methods", nargs="*", default=["fft", "dwt"])
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)
    transform = args.transform or cfg.best_setting.transform
    out = ensure_dir(Path(cfg.paths.results) / "exp2_sampling")

    settings = [setting_name(transform, cfg.data.required_fs, "orig")]
    settings += [setting_name(transform, fs, m) for fs in args.fs for m in args.methods]
    keys = fixed_split_keys(cfg, require_manifest(cfg, settings[0]), cfg.experiments.epochs_per_class,
                            cfg.experiments.val_fraction, cfg.experiments.dataset)
    save_keys(keys, out / "split_keys.csv")

    results = {}
    for name in settings:
        log.info("=== %s ===", name)
        train_df, val_df = apply_split_keys(require_manifest(cfg, name), keys)
        results[name] = run_classification(cfg, train_df, val_df, cfg.classifier.pretrained_model, True,
                                           out / name, Path(cfg.paths.checkpoints) / "exp2" / name)
    table = summarise(results, ("accuracy", "f1_macro"))
    table.to_csv(out / "table_7_3_sampling.csv")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
