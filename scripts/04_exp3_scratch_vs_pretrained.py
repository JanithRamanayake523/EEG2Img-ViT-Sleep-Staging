"""Experiment 3 - from-scratch ViT vs. fine-tuned pre-trained ViT, on balanced and imbalanced data
(Section 6.2.4, Figure 7.5).

* imbalanced: a random sample of DS9 that keeps the natural stage proportions.
* balanced  : the same sample after the split, with the *training* part up-sampled to the size of its
              largest class (validation data is never duplicated, so there is no leakage).
Both variants are evaluated on the natural-distribution validation set and on a class-balanced version of it.

Prerequisite: python scripts/01_build_images.py --settings gasf_128hz_dwt
"""
import argparse
from pathlib import Path

from sleepvit.config import add_common_args, load_config
from sleepvit.data.manifest import balanced_sample, split, select_dataset, upsample
from sleepvit.experiment_utils import best_setting_name, require_manifest
from sleepvit.training.pipeline import run_classification, summarise
from sleepvit.utils import ensure_dir, get_logger, seed_everything

log = get_logger()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--n-total", type=int, default=None,
                        help="epochs sampled from DS9 (default: 6 x experiments.epochs_per_class)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)
    out = ensure_dir(Path(cfg.paths.results) / "exp3_scratch_vs_pretrained")

    df = select_dataset(require_manifest(cfg, best_setting_name(cfg)), cfg.experiments.dataset)
    n_total = args.n_total or 6 * cfg.experiments.epochs_per_class
    df = df.sample(n=min(n_total, len(df)), random_state=cfg.seed)
    train, val = split(df, cfg.experiments.val_fraction, cfg.seed, cfg.experiments.split_by)
    val_balanced = balanced_sample(val, int(val["stage"].value_counts().min()), cfg.seed)
    train_balanced = upsample(train, seed=cfg.seed)
    log.info("train natural=%d balanced=%d | val natural=%d balanced=%d", len(train), len(train_balanced),
             len(val), len(val_balanced))

    variants = {"balanced": (train_balanced, val_balanced, {"val_natural": val}),
                "imbalanced": (train, val, {"val_balanced": val_balanced})}
    results = {}
    for data_name, (tr, va, extra) in variants.items():
        for model_key, pretrained in (("scratch", False), (cfg.classifier.pretrained_model, True)):
            label = f"{'pretrained' if pretrained else 'scratch'}_{data_name}"
            log.info("=== %s ===", label)
            results[label] = run_classification(cfg, tr, va, model_key, pretrained, out / label,
                                                Path(cfg.paths.checkpoints) / "exp3" / label, extra_eval=extra)
    table = summarise(results, ("accuracy", "f1_macro"))
    table.to_csv(out / "figure_7_5_scratch_vs_pretrained.csv")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
