"""Experiment 6 - does GAN-augmented data help? (Section 6.2.7, Figure 7.8)

1. (--generate) sample ``gan.images_per_class`` synthetic images per stage from the per-class generators.
2. Fine-tune the best pre-trained ViT on three training sets with ``augmentation.train_images_per_class``
   images per stage:  original | synthetic only | original + synthetic.
3. All three are evaluated on the *same real* validation images saved by Experiment 5.

Prerequisite: scripts/06_exp5_train_dcgan.py
"""
import argparse
from pathlib import Path

import pandas as pd

from sleepvit.config import add_common_args, load_config
from sleepvit.constants import STAGE_NAMES
from sleepvit.data.manifest import balanced_sample, synthetic_manifest
from sleepvit.evaluation.plots import bar_chart
from sleepvit.experiment_utils import apply_split_keys, best_setting_name, require_manifest
from sleepvit.training.gan_trainer import generate_images, load_generator
from sleepvit.training.pipeline import run_classification, summarise
from sleepvit.utils import ensure_dir, get_device, get_logger, seed_everything

log = get_logger()


def mixed_set(original: pd.DataFrame, synthetic: pd.DataFrame, per_class: int, original_fraction: float, seed: int):
    parts = []
    n_orig = int(round(per_class * original_fraction))
    for k in sorted(original["stage"].unique()):
        parts.append(balanced_sample(original[original["stage"] == k], n_orig, seed))
        parts.append(balanced_sample(synthetic[synthetic["stage"] == k], per_class - n_orig, seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--generate", action="store_true", help="(re)generate the synthetic images first")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    out = ensure_dir(Path(cfg.paths.results) / "exp6_gan_augmentation")
    name = best_setting_name(cfg)
    syn_root = Path(cfg.paths.processed) / "synthetic"

    if args.generate:
        for k in range(len(STAGE_NAMES)):
            g = load_generator(cfg, Path(cfg.paths.checkpoints) / "gan" / f"class_{k}" / "gan.pt", device)
            generate_images(g, cfg.gan.images_per_class, syn_root / name / f"class_{k}", cfg.gan.latent_dim, device,
                            prefix=f"c{k}")

    keys = pd.read_csv(Path(cfg.paths.results) / "exp5_dcgan" / "split_keys.csv")
    train, val = apply_split_keys(require_manifest(cfg, name), keys)
    synthetic = synthetic_manifest(syn_root, name)
    if synthetic.empty:
        raise SystemExit("No synthetic images found - run with --generate")
    n = cfg.augmentation.train_images_per_class

    sets = {
        "original": balanced_sample(train, n, cfg.seed),
        "synthetic": balanced_sample(synthetic, n, cfg.seed),
        "original_plus_synthetic": mixed_set(train, synthetic, n, cfg.augmentation.mixed_original_fraction, cfg.seed),
    }
    results = {}
    for label, tr in sets.items():
        log.info("=== %s (%d training images) ===", label, len(tr))
        results[label] = run_classification(cfg, tr, val, cfg.classifier.pretrained_model, True, out / label,
                                            Path(cfg.paths.checkpoints) / "exp6" / label)
    table = summarise(results, ("accuracy", "f1_macro"))
    table.to_csv(out / "figure_7_8_gan_augmentation.csv")
    bar_chart({k.replace("_", " "): v for k, v in table["accuracy"].items()}, "Accuracy (%)", out / "figure_7_8.png")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
