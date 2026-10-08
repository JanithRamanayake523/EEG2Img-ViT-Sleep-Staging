"""Experiment 1 - which image transformation is best? (Section 6.2.2, Tables 7.1 and 7.2)

Fine-tunes ViT-B/16 on 12,000 balanced DS9 epochs (2,000 per stage) for each transformation, reports
accuracy / F1 / processing time per epoch, then clusters the final-layer embeddings (t-SNE, silhouette,
Davies-Bouldin, Calinski-Harabasz).

Prerequisite:  python scripts/01_build_images.py --transforms gasf gadf mtf spectrogram --fs 512 --resample orig
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sleepvit.config import add_common_args, load_config
from sleepvit.constants import TRANSFORMS
from sleepvit.data.build_images import setting_name
from sleepvit.data.datasets import classifier_transform, make_loader
from sleepvit.data.discovery import discover_subjects
from sleepvit.data.edf_preprocess import epoch_signal, load_recording, standardize
from sleepvit.data.image_transforms import epoch_to_image
from sleepvit.evaluation.clustering import analyse_embeddings
from sleepvit.evaluation.plots import tsne_plot
from sleepvit.experiment_utils import apply_split_keys, fixed_split_keys, require_manifest, save_keys
from sleepvit.training.pipeline import load_trained, run_classification, summarise
from sleepvit.training.trainer import embed
from sleepvit.utils import ensure_dir, get_device, get_logger, save_json, seed_everything, timer

log = get_logger()


def benchmark_processing_time(cfg, transform: str, n_epochs: int = 20) -> float:
    """Seconds needed to turn one 16-channel epoch into an image (measured on one real subject)."""
    subject = discover_subjects(cfg.paths.raw_data, required_fs=cfg.data.required_fs)[0]
    rec = load_recording(subject.edf_path, cfg.data.required_fs, cfg.preprocessing.avg_before_filter)
    epochs = standardize(epoch_signal(rec.data, rec.fs, cfg.data.epoch_length_s)[:n_epochs])
    with timer() as t:
        for e in epochs:
            epoch_to_image(e, transform, rec.fs, cfg)
    return t() / len(epochs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--transforms", nargs="*", default=list(TRANSFORMS), choices=TRANSFORMS)
    parser.add_argument("--no-benchmark", action="store_true", help="skip the processing-time measurement")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(cfg.seed)
    out = ensure_dir(Path(cfg.paths.results) / "exp1_transforms")
    device = get_device(cfg.device)
    model_key = cfg.classifier.pretrained_model

    names = {t: setting_name(t, cfg.data.required_fs, "orig") for t in args.transforms}
    ref = require_manifest(cfg, names[args.transforms[0]])
    keys = fixed_split_keys(cfg, ref, cfg.experiments.epochs_per_class, cfg.experiments.val_fraction, cfg.experiments.dataset)
    save_keys(keys, out / "split_keys.csv")

    results, clustering = {}, {}
    for transform, name in names.items():
        log.info("=== %s ===", name)
        train_df, val_df = apply_split_keys(require_manifest(cfg, name), keys)
        run_dir = ensure_dir(out / transform)
        metrics = run_classification(cfg, train_df, val_df, model_key, True, run_dir, Path(cfg.paths.checkpoints) / "exp1" / transform)
        if not args.no_benchmark:
            metrics["processing_s_per_epoch"] = benchmark_processing_time(cfg, transform)
        results[transform] = metrics

        model, mean, std = load_trained(cfg, model_key, metrics["checkpoint"], device)
        loader = make_loader(val_df, classifier_transform(mean, std, "none", cfg.images.size), cfg.classifier.batch_size, False, cfg.classifier.num_workers)
        feats, labels = embed(model, loader, device)
        points, scores = analyse_embeddings(feats, labels, cfg.seed)
        clustering[transform] = scores
        np.save(run_dir / "tsne_points.npy", points)
        tsne_plot(points, labels, run_dir / "tsne.png", title=transform.upper())
        save_json(metrics, run_dir / "metrics.json")

    table = summarise(results, ("accuracy", "f1_macro", "processing_s_per_epoch"))
    table.to_csv(out / "table_7_1_performance.csv")
    pd.DataFrame(clustering).T.to_csv(out / "table_7_2_clustering.csv")
    print(table.round(2).to_string())
    print(pd.DataFrame(clustering).T.round(2).to_string())


if __name__ == "__main__":
    main()
