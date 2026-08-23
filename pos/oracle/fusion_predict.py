"""Ensemble predictions for the static fusion rules.

Split out of `comparison.py` in ADR 0018. Milestone 7 needs precision,
recall, F1 and balanced accuracy, not just accuracy, so the fusion rules
have to hand back *labels* — the accuracy functions in `comparison.py` are
now thin wrappers over these.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def majority_vote_predict(pool: list, X: np.ndarray) -> np.ndarray:
    """Hard majority vote. Ties broken by scipy.stats.mode (lowest label)."""
    preds = np.array([clf.predict(X) for clf in pool])
    return stats.mode(preds, axis=0, keepdims=False).mode


def mean_probs_predict(pool: list, X: np.ndarray) -> np.ndarray:
    """Mean of `predict_proba` across the pool, then argmax."""
    probs_sum = None
    for clf in pool:
        probs = clf.predict_proba(X)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    classes = pool[0].classes_
    return classes[np.argmax(probs_sum / len(pool), axis=1)]


def mean_decision_predict(pool: list, X: np.ndarray) -> np.ndarray:
    """Soft fusion for margin classifiers with no `predict_proba`.

    Each margin is divided by ||w||, turning it into the signed distance to
    that classifier's hyperplane — scale-free, so it can be averaged across
    the pool (ADR 0016). Raises ValueError when members disagree on
    `classes_`, since then the per-class columns are not aligned.
    """
    classes = pool[0].classes_
    total = None
    for clf in pool:
        if not np.array_equal(clf.classes_, classes):
            raise ValueError("pool members disagree on classes_")
        dec = np.asarray(clf.decision_function(X), dtype=float)
        norms = np.linalg.norm(np.atleast_2d(clf.coef_), axis=1)
        norms = np.where(norms > 0, norms, 1.0)
        if dec.ndim == 1:  # binary: decision_function > 0 means classes_[1]
            dec = dec / norms[0]
            dec = np.column_stack([-dec, dec])
        else:
            dec = dec / norms
        total = dec if total is None else total + dec
    return classes[np.argmax(total, axis=1)]


def soft_fusion_predict(pool: list, X: np.ndarray):
    """Best available soft combination. Returns (predictions, rule).

    rule is "mean_probs" when the pool exposes predict_proba, otherwise
    "mean_decision_norm", otherwise "none" with None predictions.
    """
    try:
        return mean_probs_predict(pool, X), "mean_probs"
    except (AttributeError, ValueError):
        pass
    try:
        return mean_decision_predict(pool, X), "mean_decision_norm"
    except (AttributeError, ValueError):
        return None, "none"
