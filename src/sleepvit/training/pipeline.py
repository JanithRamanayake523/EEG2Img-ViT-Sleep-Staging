"""One complete classification run: build model -> fit -> evaluate -> save artefacts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch

from ..data.datasets import classifier_transform, make_loader
from ..models import build_classifier
from ..utils import ensure_dir, get_device, get_logger, save_json
from .trainer import evaluate, fit, save_history

log = get_logger()


def make_loaders(cfg, train_df: pd.DataFrame, val_df: pd.DataFrame, mean, std):
    c = cfg.classifier
    train_tf = classifier_transform(mean, std, c.augment, cfg.images.size)
    eval_tf = classifier_transform(mean, std, "none", cfg.images.size)
    train_loader = make_loader(train_df, train_tf, c.batch_size, True, c.num_workers)
    val_loader = make_loader(val_df, eval_tf, c.batch_size, False, c.num_workers)
    return train_loader, val_loader, eval_tf


def run_classification(cfg, train_df: pd.DataFrame, val_df: pd.DataFrame, model_key: str, pretrained: bool,
                       run_dir: str | Path, ckpt_dir: Optional[str | Path] = None,
                       extra_eval: Optional[Dict[str, pd.DataFrame]] = None, epochs: Optional[int] = None) -> Dict:
    """Train ``model_key`` on ``train_df`` and evaluate on ``val_df`` (plus any ``extra_eval`` sets).

    Writes ``metrics.json`` and ``history.json`` into ``run_dir`` and the best weights into ``ckpt_dir``.
    Validation data must always be *real* PSG images, also when the training data is synthetic.
    """
    run_dir = ensure_dir(run_dir)
    ckpt_dir = ensure_dir(ckpt_dir or Path(cfg.paths.checkpoints) / run_dir.name)
    device = get_device(cfg.device)

    model, mean, std = build_classifier(cfg, model_key, pretrained=pretrained)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model=%s pretrained=%s params=%.1fM train=%d val=%d", model_key, pretrained, n_params / 1e6,
             len(train_df), len(val_df))
    train_loader, val_loader, eval_tf = make_loaders(cfg, train_df, val_df, mean, std)

    ckpt = Path(ckpt_dir) / "best.pt"
    history = fit(model, train_loader, val_loader, cfg, device, ckpt, epochs=epochs)
    save_history(history, run_dir / "history.json")

    metrics = evaluate(model, val_loader, device)
    metrics.update({"model": model_key, "pretrained": pretrained, "params_millions": n_params / 1e6,
                    "n_train": len(train_df), "checkpoint": str(ckpt)})
    for name, df in (extra_eval or {}).items():
        loader = make_loader(df, eval_tf, cfg.classifier.batch_size, False, cfg.classifier.num_workers)
        metrics.setdefault("extra_eval", {})[name] = evaluate(model, loader, device)
    save_json(metrics, run_dir / "metrics.json")
    log.info("%s | acc %.4f | macro-F1 %.4f", run_dir.name, metrics["accuracy"], metrics["f1_macro"])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def load_trained(cfg, model_key: str, checkpoint: str | Path, device: torch.device):
    """Rebuild ``model_key`` and load fine-tuned weights. Returns ``(model, mean, std)``."""
    model, mean, std = build_classifier(cfg, model_key, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval(), mean, std


def summarise(rows: Dict[str, Dict], keys=("accuracy", "f1_macro", "f1_weighted", "inference_ms_per_epoch")) -> pd.DataFrame:
    """Collect several runs' headline metrics into one table (accuracy and F1 as percentages)."""
    table = pd.DataFrame({name: {k: m.get(k) for k in keys} for name, m in rows.items()}).T
    for k in ("accuracy", "f1_macro", "f1_weighted"):
        if k in table:
            table[k] = table[k] * 100.0
    return table
