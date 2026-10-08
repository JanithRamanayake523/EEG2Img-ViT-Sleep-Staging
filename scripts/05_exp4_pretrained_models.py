"""Experiment 4 - which pre-trained backbone? (Section 6.2.5, Table 7.5)

Fine-tunes ViT-B/16, ViT-B/32, DeiT-B, Swin-B, BEiT-B and PVT on the same balanced DS9 sample and reports
accuracy, F1 and inference time per epoch.

Prerequisite: python scripts/01_build_images.py --settings gasf_128hz_dwt
"""
import argparse
from pathlib import Path

from sleepvit.config import add_common_args, load_config
from sleepvit.constants import PRETRAINED_MODELS
from sleepvit.experiment_utils import apply_split_keys, best_setting_name, fixed_split_keys, require_manifest, save_keys
from sleepvit.training.pipeline import run_classification, summarise
from sleepvit.utils import ensure_dir, get_logger, seed_everything

log = get_logger()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--models", nargs="*", default=list(PRETRAINED_MODELS), help="keys of PRETRAINED_MODELS")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)
    out = ensure_dir(Path(cfg.paths.results) / "exp4_pretrained_models")

    manifest = require_manifest(cfg, best_setting_name(cfg))
    keys = fixed_split_keys(cfg, manifest, cfg.experiments.epochs_per_class, cfg.experiments.val_fraction,
                            cfg.experiments.dataset)
    save_keys(keys, out / "split_keys.csv")
    train_df, val_df = apply_split_keys(manifest, keys)

    results = {}
    for key in args.models:
        log.info("=== %s (%s) ===", key, PRETRAINED_MODELS.get(key, key))
        results[key] = run_classification(cfg, train_df, val_df, key, True, out / key,
                                          Path(cfg.paths.checkpoints) / "exp4" / key)
    table = summarise(results, ("accuracy", "f1_macro", "inference_ms_per_epoch"))
    table.to_csv(out / "table_7_5_pretrained_models.csv")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
