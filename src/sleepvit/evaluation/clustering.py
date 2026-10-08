"""t-SNE and cluster-quality scores of ViT embeddings (thesis Section 7.1, Table 7.2)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def tsne_embedding(features: np.ndarray, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    perplexity = min(perplexity, max(5.0, (len(features) - 1) / 3))
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca").fit_transform(features)


def clustering_scores(points: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Silhouette (higher better), Davies-Bouldin (lower better), Calinski-Harabasz (higher better)."""
    return {
        "silhouette": float(silhouette_score(points, labels)),
        "davies_bouldin": float(davies_bouldin_score(points, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(points, labels)),
    }


def analyse_embeddings(features: np.ndarray, labels: np.ndarray, seed: int = 42):
    """Return ``(tsne_points, scores)``; scores are computed on the 2-D t-SNE map, as in the notebooks,
    and additionally on the raw feature space (``*_raw`` keys)."""
    points = tsne_embedding(features, seed=seed)
    scores = clustering_scores(points, labels)
    scores.update({f"{k}_raw": v for k, v in clustering_scores(features, labels).items()})
    return points, scores
