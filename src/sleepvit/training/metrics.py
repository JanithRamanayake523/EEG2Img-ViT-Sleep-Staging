"""Classification metrics used throughout the experiments."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ..constants import STAGE_NAMES


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names=STAGE_NAMES) -> Dict:
    """Accuracy, macro/weighted F1, per-class precision/recall/F1 and the (row-normalised) confusion matrix."""
    labels = list(range(len(class_names)))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_pct = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True) * 100.0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for name, p, r, f, s in zip(class_names, prec, rec, f1, support)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_pct": cm_pct.tolist(),
        "n": int(len(y_true)),
    }
