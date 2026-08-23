"""Per-fuser metric suite: accuracy, macro P/R/F1, balanced accuracy.

Milestone 7 of the cronograma asks for "acurácia, precisão, revocação,
F1-score e acurácia balanceada". Accuracy alone hides the cost on the
minority class, and several bases here are strongly imbalanced — in Blood,
always predicting the majority already scores ~0.76, the same order as the
fusers being compared (ADR 0018).

Macro averaging is deliberate: it weights every class equally, which is what
exposes a fuser that buys accuracy by abandoning a rare class.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

METRIC_NAMES = ("acc", "precision", "recall", "f1", "balanced_acc")


def prediction_metrics(y_true, y_pred) -> dict[str, float]:
    """The five metrics for one set of predictions.

    `zero_division=0`: on an imbalanced base a fuser may never predict some
    class, leaving its precision undefined. Scoring it 0 is the honest
    reading — the fuser got none of that class right — and it keeps the
    macro average comparable across methods.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    kw = {"average": "macro", "zero_division": 0}
    return {
        "acc": float(np.mean(y_pred == y_true)),
        "precision": float(precision_score(y_true, y_pred, **kw)),
        "recall": float(recall_score(y_true, y_pred, **kw)),
        "f1": float(f1_score(y_true, y_pred, **kw)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def metric_columns(prefix: str, metrics: dict | None) -> dict[str, float | None]:
    """Flatten one metric dict into `<prefix>_<metric>` summary columns.

    A None dict (the fuser could not run) yields None for every metric, so
    the CSV keeps a stable header across modes.
    """
    metrics = metrics or {}
    return {f"{prefix}_{name}": metrics.get(name) for name in METRIC_NAMES}


def oracle_balanced_recall(matrix: np.ndarray, y, n: int = 1) -> float:
    """Macro recall of the event "at least n classifiers were correct".

    The Oracle has no predicted label, so precision and F1 are undefined for
    it — but recall per class is not, and that is the question that matters:
    is the Oracle ceiling itself propped up by the majority class? Averaging
    the per-class hit rates answers it (ADR 0018).
    """
    y = np.asarray(y)
    hit = (np.asarray(matrix).sum(axis=1) >= n)
    classes = np.unique(y)
    return float(np.mean([hit[y == c].mean() for c in classes]))
