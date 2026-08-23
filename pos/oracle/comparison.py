"""Real combination methods for Oracle_N comparison.

Implements the baseline fusion methods that Oracle_N is compared against
(cronograma milestone 6). The dynamic-selection family lives in
`des_comparison.py`.

Since ADR 0018 the prediction logic itself lives in `fusion_predict`; these
are the accuracy-only wrappers, kept because most callers want the scalar.
"""

from __future__ import annotations

import numpy as np

from pos.oracle.fusion_predict import (
    majority_vote_predict,
    mean_decision_predict,
    mean_probs_predict,
    soft_fusion_predict,
)


def _accuracy(y_pred, y) -> float:
    return float(np.mean(y_pred == y))


def majority_vote_accuracy(pool: list, X: np.ndarray, y: np.ndarray) -> float:
    """Hard majority vote accuracy."""
    return _accuracy(majority_vote_predict(pool, X), y)


def mean_probs_accuracy(pool: list, X: np.ndarray, y: np.ndarray) -> float:
    """Mean of predicted probabilities → argmax accuracy."""
    return _accuracy(mean_probs_predict(pool, X), y)


def mean_decision_accuracy(pool: list, X: np.ndarray, y: np.ndarray) -> float:
    """Mean of ||w||-normalised margins → argmax accuracy (ADR 0016)."""
    return _accuracy(mean_decision_predict(pool, X), y)


def soft_fusion_accuracy(pool: list, X: np.ndarray, y: np.ndarray):
    """Best available soft-combination accuracy. Returns (accuracy, rule)."""
    preds, rule = soft_fusion_predict(pool, X)
    return (None if preds is None else _accuracy(preds, y)), rule
