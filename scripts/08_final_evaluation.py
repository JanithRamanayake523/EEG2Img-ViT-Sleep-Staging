"""Final model evaluation (Section 6.2.8, Figure 7.9, Table 7.8)

Loads the ViT-B/16 trained on original + synthetic images (Experiment 6) and reports
  * overall accuracy, per-stage F1 and the confusion matrix on the held-out real validation images;
  * accuracy on DS1-DS8 and DS9.

Table 7.8 in the thesis scores every epoch of each subset. Those subsets contain the training epochs too,
so this script reports two columns: ``all_epochs`` (thesis protocol) and ``held_out`` (validation rows only).
"""
import argparse
from pathlib import Path

import pandas as pd

from sleepvit.config import add_common_args, load_config
from sleepvit.constants import DATASET_GROUPS
from sleepvit.data.datasets import classifier_transform, make_loader
from sleepvit.data.manifest import select_dataset
from sleepvit.evaluation.plots import confusion_and_f1
from sleepvit.experiment_utils import apply_split_keys, best_setting_name, require_manifest
from sleepvit.training.pipeline import load_trained
from sleepvit.training.trainer import evaluate
from sleepvit.utils import ensure_dir, get_device, get_logger, save_json, seed_everything

log = get_logger()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--checkpoint", default=None,
                        help="default: checkpoints/exp6/original_plus_synthetic/best.pt")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    out = ensure_dir(Path(cfg.paths.results) / "final")
    ckpt = args.checkpoint or Path(cfg.paths.checkpoints) / "exp6" / "original_plus_synthetic" / "best.pt"

    model, mean, std = load_trained(cfg, cfg.classifier.pretrained_model, ckpt, device)
    tf = classifier_transform(mean, std, "none", cfg.images.size)

    def run(df):
        return evaluate(model, make_loader(df, tf, cfg.classifier.batch_size, False, cfg.classifier.num_workers), device)

    manifest = require_manifest(cfg, best_setting_name(cfg))
    keys = pd.read_csv(Path(cfg.paths.results) / "exp5_dcgan" / "split_keys.csv")
    _, val = apply_split_keys(manifest, keys)

    overall = run(val)
    save_json(overall, out / "overall_metrics.json")
    confusion_and_f1(overall, out / "figure_7_9.png", title="- DS9 (held-out)")
    log.info("held-out accuracy %.2f%%", 100 * overall["accuracy"])

    rows = {}
    val_keys = set(zip(val["subject"], val["epoch"]))
    for ds in DATASET_GROUPS:
        subset = select_dataset(manifest, ds)
        if subset.empty:
            continue
        held = subset[[(s, e) in val_keys for s, e in zip(subset["subject"], subset["epoch"])]]
        rows[ds] = {"all_epochs": 100 * run(subset)["accuracy"],
                    "held_out": 100 * run(held)["accuracy"] if len(held) else float("nan"),
                    "n_all": len(subset), "n_held_out": len(held)}
        log.info("%s %s", ds, rows[ds])
    table = pd.DataFrame(rows).T
    table.to_csv(out / "table_7_8_datasets.csv")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
