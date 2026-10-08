"""Generic fine-tuning / training loop for the six-class sleep-stage classifiers."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.pretrained import extract_features
from ..utils import get_logger, save_json
from .metrics import classification_metrics

log = get_logger()


def _epoch_pass(model, loader, device, criterion, optimizer=None, scaler=None, amp=False) -> Tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, leave=False, desc="train" if train else "val"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / count, correct / count


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg, device: torch.device,
        checkpoint_path: str | Path, epochs: int | None = None) -> Dict[str, List[float]]:
    """Train with AdamW, ReduceLROnPlateau (val loss) and early stopping (val accuracy).

    The weights with the best validation accuracy are written to ``checkpoint_path`` and restored at the
    end, mirroring the Keras callbacks of the original notebooks.
    """
    c = cfg.classifier
    epochs = epochs or c.epochs
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=c.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=c.lr_factor, patience=c.lr_patience, min_lr=c.min_lr)
    scaler = torch.cuda.amp.GradScaler() if (c.amp and device.type == "cuda") else None

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, stale = -1.0, 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        tr_loss, tr_acc = _epoch_pass(model, train_loader, device, criterion, optimizer, scaler, c.amp)
        va_loss, va_acc = _epoch_pass(model, val_loader, device, criterion, amp=c.amp)
        scheduler.step(va_loss)
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, optimizer.param_groups[0]["lr"])):
            history[k].append(v)
        log.info("epoch %d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | %.0fs",
                 epoch, epochs, tr_loss, tr_acc, va_loss, va_acc, time.time() - start)
        if va_acc > best_acc:
            best_acc, stale = va_acc, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1
            if stale >= c.patience:
                log.info("early stopping (no val-accuracy gain for %d epochs)", c.patience)
                break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return ``(y_true, y_pred, ms_per_epoch)``; the timing covers the whole forward pass over the loader."""
    model.eval().to(device)
    y_true, y_pred = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for images, labels in tqdm(loader, leave=False, desc="predict"):
        logits = model(images.to(device, non_blocking=True))
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_true.append(labels.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    return y_true, y_pred, 1000.0 * elapsed / max(len(y_true), 1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    y_true, y_pred, ms = predict(model, loader, device)
    out = classification_metrics(y_true, y_pred)
    out["inference_ms_per_epoch"] = ms
    return out


@torch.no_grad()
def embed(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Penultimate-layer embeddings and labels for t-SNE / clustering analysis."""
    model.eval().to(device)
    feats, labels = [], []
    for images, y in tqdm(loader, leave=False, desc="embed"):
        feats.append(extract_features(model, images.to(device)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def save_history(history: Dict, path: str | Path) -> None:
    save_json(history, path)
