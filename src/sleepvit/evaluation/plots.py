"""Figures for the thesis chapters. Every function draws from real result files / arrays."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from PIL import Image  # noqa: E402

from ..constants import STAGE_NAMES  # noqa: E402

BAR_COLOR = plt.cm.viridis(0.2)


def bar_chart(values: Dict[str, float], ylabel: str, path: str | Path, ylim=None, rotate: int = 0) -> None:
    """Simple labelled bar chart (Figures 7.5, 7.6, 7.8)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(list(values), list(values.values()), color=BAR_COLOR, width=0.6)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right" if rotate else "center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def grouped_bar_chart(groups: Dict[str, Dict[str, float]], ylabel: str, path: str | Path, ylim=None) -> None:
    """``groups = {"Pre-trained": {"Balanced": 78.1, "Imbalanced": 68.5}, ...}`` (Figure 7.5)."""
    series = list(next(iter(groups.values())))
    x = np.arange(len(groups))
    width = 0.8 / len(series)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(series)))
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, s in enumerate(series):
        vals = [groups[g][s] for g in groups]
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width, label=s, color=colors[i])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(list(groups))
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion_and_f1(metrics: Dict, path: str | Path, title: str = "") -> None:
    """Row-normalised confusion matrix next to per-class F1 scores (Figure 7.9)."""
    cm = np.array(metrics["confusion_matrix_pct"])
    f1 = [metrics["per_class"][c]["f1"] for c in STAGE_NAMES]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5, xticklabels=STAGE_NAMES,
                yticklabels=STAGE_NAMES, ax=axes[0])
    axes[0].set(xlabel="Predicted class", ylabel="True class", title=f"Confusion matrix (%) {title}".strip())
    bars = sns.barplot(x=STAGE_NAMES, y=f1, hue=STAGE_NAMES, palette="viridis", legend=False, ax=axes[1])
    for p in bars.patches:
        axes[1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom")
    axes[1].set(xlabel="Class", ylabel="F1 score", ylim=(0, 1), title="F1 score by class")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def tsne_plot(points: np.ndarray, labels: np.ndarray, path: str | Path, title: str = "", hulls: bool = False) -> None:
    """t-SNE map coloured by sleep stage (optionally with convex hulls, as in the notebooks)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(STAGE_NAMES)))
    for k, name in enumerate(STAGE_NAMES):
        pts = points[labels == k]
        if len(pts) == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.7, color=cmap[k], label=name)
        if hulls and len(pts) >= 3:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(pts)
            for simplex in hull.simplices:
                ax.plot(pts[simplex, 0], pts[simplex, 1], "k-", lw=0.5)
    ax.set(xlabel="t-SNE component 1", ylabel="t-SNE component 2", title=title)
    ax.legend(title="Stage", loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def image_grid(paths: Sequence[str | Path], titles: Sequence[str], path: str | Path, ncols: int = 4) -> None:
    """Show example images side by side (Figures 7.1, 7.3, 7.4, 7.7)."""
    nrows = int(np.ceil(len(paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p, t in zip(axes.ravel(), paths, titles):
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(t)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def pixel_histograms(real: np.ndarray, fake: np.ndarray, path: str | Path, bins: int = 50) -> None:
    """Histogram of pixel intensities for real vs. generated images (measured, not hand-drawn)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, data, name in zip(axes, (real, fake), ("Real images", "Generated images")):
        ax.hist((data * 255).ravel(), bins=bins, color="steelblue", alpha=0.7)
        ax.set_title(f"Pixel histogram - {name}")
        ax.set_xlabel("Pixel intensity")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def training_curves(history: Dict[str, Sequence[float]], path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_acc"], label="train")
    axes[0].plot(history["val_acc"], label="val")
    axes[0].set(title="Accuracy", xlabel="epoch")
    axes[1].plot(history["train_loss"], label="train")
    axes[1].plot(history["val_loss"], label="val")
    axes[1].set(title="Loss", xlabel="epoch")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
