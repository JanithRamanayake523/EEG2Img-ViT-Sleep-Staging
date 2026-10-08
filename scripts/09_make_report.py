"""Collect the CSV/JSON outputs of all experiments into one markdown report and redraw the bar charts.

    python scripts/09_make_report.py
Missing experiments are skipped.
"""
import argparse
from pathlib import Path

import pandas as pd

from sleepvit.config import add_common_args, load_config
from sleepvit.evaluation.plots import bar_chart, grouped_bar_chart
from sleepvit.utils import get_logger

log = get_logger()

TABLES = [
    ("Table 7.1 - image transformations (Exp. 1)", "exp1_transforms/table_7_1_performance.csv"),
    ("Table 7.2 - clustering of ViT embeddings (Exp. 1)", "exp1_transforms/table_7_2_clustering.csv"),
    ("Table 7.3 - sampling frequency and resampling (Exp. 2)", "exp2_sampling/table_7_3_sampling.csv"),
    ("Figure 7.5 - scratch vs pre-trained (Exp. 3)", "exp3_scratch_vs_pretrained/figure_7_5_scratch_vs_pretrained.csv"),
    ("Table 7.5 - pre-trained models (Exp. 4)", "exp4_pretrained_models/table_7_5_pretrained_models.csv"),
    ("DCGAN image quality (Exp. 5)", "exp5_dcgan/gan_quality.csv"),
    ("Figure 7.8 - GAN augmentation (Exp. 6)", "exp6_gan_augmentation/figure_7_8_gan_augmentation.csv"),
    ("Table 7.8 - final model per dataset", "final/table_7_8_datasets.csv"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    root = Path(cfg.paths.results)

    lines = ["# Results summary\n"]
    for title, rel in TABLES:
        path = root / rel
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        lines += [f"## {title}\n", df.round(2).to_markdown() + "\n"]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    exp3 = root / "exp3_scratch_vs_pretrained/figure_7_5_scratch_vs_pretrained.csv"
    if exp3.exists():
        t = pd.read_csv(exp3, index_col=0)["accuracy"]
        grouped_bar_chart({"Pre-trained": {"Balanced": t["pretrained_balanced"], "Imbalanced": t["pretrained_imbalanced"]},
                           "From scratch": {"Balanced": t["scratch_balanced"], "Imbalanced": t["scratch_imbalanced"]}},
                          "Accuracy (%)", root / "exp3_scratch_vs_pretrained/figure_7_5.png")
    exp4 = root / "exp4_pretrained_models/table_7_5_pretrained_models.csv"
    if exp4.exists():
        bar_chart(pd.read_csv(exp4, index_col=0)["accuracy"].to_dict(), "Accuracy (%)",
                  root / "exp4_pretrained_models/figure_7_6.png", rotate=45)
    log.info("wrote %s", root / "REPORT.md")


if __name__ == "__main__":
    main()
