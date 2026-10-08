"""Experiment 5 - train and evaluate the DCGAN (Section 6.2.6, Tables 7.6/7.7, Figure 7.7)

1. 70/30 train/validation split of DS9 images (GASF / DWT / 128 Hz by default).
2. Train one shared DCGAN on all training images of every stage.
3. Fine-tune it separately on each sleep stage (one generator per class).
4. Score generated images against real validation images of the same stage: SSIM, FID, MPED.

The GAN only ever sees *training* images; the validation images are kept for Experiment 6.

Prerequisite: python scripts/01_build_images.py --settings gasf_128hz_dwt
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from sleepvit.config import add_common_args, load_config
from sleepvit.constants import STAGE_NAMES
from sleepvit.data.manifest import select_dataset, split
from sleepvit.evaluation.gan_metrics import evaluate_synthetic, load_images
from sleepvit.evaluation.plots import image_grid, pixel_histograms
from sleepvit.experiment_utils import best_setting_name, require_manifest
from sleepvit.training.gan_trainer import generate_images, load_generator, train_gan
from sleepvit.utils import ensure_dir, get_device, get_logger, save_json, seed_everything

log = get_logger()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--skip-pretrain", action="store_true", help="reuse the shared checkpoint")
    parser.add_argument("--classes", nargs="*", type=int, default=list(range(len(STAGE_NAMES))))
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    out = ensure_dir(Path(cfg.paths.results) / "exp5_dcgan")
    ckpt = Path(cfg.paths.checkpoints) / "gan"

    df = select_dataset(require_manifest(cfg, best_setting_name(cfg)), cfg.experiments.dataset)
    train, val = split(df, cfg.gan.val_fraction, cfg.seed, cfg.experiments.split_by)
    pd.concat([train.assign(split="train"), val.assign(split="val")])[["subject", "epoch", "stage", "split"]] \
        .to_csv(out / "split_keys.csv", index=False)  # re-used by Experiment 6

    shared = ckpt / "shared" / "gan.pt"
    if not args.skip_pretrain:
        log.info("--- shared DCGAN on all classes ---")
        train_gan(cfg, train, device, ckpt / "shared", cfg.gan.pretrain_epochs, tag="shared")

    scores = {}
    for k in args.classes:
        log.info("--- fine-tuning class %d (%s) ---", k, STAGE_NAMES[k])
        class_dir = ckpt / f"class_{k}"
        _, _, history = train_gan(cfg, train[train["stage"] == k], device, class_dir, cfg.gan.epochs,
                                  init_from=shared if shared.exists() else None, tag=STAGE_NAMES[k])
        save_json(history, out / f"class_{k}_history.json")

        g = load_generator(cfg, class_dir / "gan.pt", device)
        eval_dir = out / f"class_{k}_samples"
        generate_images(g, cfg.gan.eval_samples, eval_dir, cfg.gan.latent_dim, device)
        real_paths = val[val["stage"] == k]["path"].head(cfg.gan.eval_samples).tolist()
        fake_paths = sorted(eval_dir.glob("*.png"))
        scores[STAGE_NAMES[k]] = evaluate_synthetic(real_paths, fake_paths, device)
        log.info("%s: %s", STAGE_NAMES[k], scores[STAGE_NAMES[k]])

        image_grid([real_paths[0], str(fake_paths[0])], ["Original", "Synthetic"], out / f"class_{k}_real_vs_fake.png", 2)
        pixel_histograms(load_images(real_paths[:100]), load_images([str(p) for p in fake_paths[:100]]),
                         out / f"class_{k}_pixel_hist.png")

    table = pd.DataFrame(scores).T
    table.loc["mean"] = table.mean(numeric_only=True)
    table.to_csv(out / "gan_quality.csv")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
