"""Real combination methods for Oracle_N comparison.

Implements the baseline fusion methods that Oracle_N is compared against
(cronograma milestone 6). DCS/DES (OLA, LCA, Rank, META-DES via DESlib)
are deferred to Fase 5.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def majority_vote_accuracy(pool: list, X: np.ndarray, y: np.ndarray) -> float:
    """Hard majority vote accuracy.

    Each classifier votes; the mode of votes is the ensemble prediction.
    Ties broken by scipy.stats.mode (first sorted label wins).
    """
    # Collect predictions: shape (n_classifiers, n_samples)
    preds = np.array([clf.predict(X) for clf in pool])
    # mode along axis=0 (across classifiers) → (n_samples,)
    mode_result = stats.mode(preds, axis=0, keepdims=False)
    ensemble_preds = mode_result.mode
    return float(np.mean(ensemble_preds == y))


def mean_probs_accuracy(pool: list, X: np.ndarray, y: np.ndarray) -> float:
    """Mean of predicted probabilities → argmax accuracy.

    Each classifier must implement `.predict_proba(X)`. Probabilities are
    averaged across the pool, then argmax gives the ensemble prediction.
    """
    # Sum probabilities across pool, then divide
    probs_sum = None
    for clf in pool:
        probs = clf.predict_proba(X)
        if probs_sum is None:
            probs_sum = probs
        else:
            probs_sum += probs
    mean_probs = probs_sum / len(pool)
    # argmax → predicted class index (matches clf.classes_ ordering)
    ensemble_preds_idx = np.argmax(mean_probs, axis=1)
    # Map index → class label using first classifier's classes_
    classes = pool[0].classes_
    ensemble_preds = classes[ensemble_preds_idx]
    return float(np.mean(ensemble_preds == y))


def mean_decision_accuracy(pool: list, X: np.ndarray, y: np.ndarray) -> float:
    """Soft fusion for margin classifiers that have no `predict_proba`.

    The linear Perceptron of the reference thesis exposes only
    `decision_function`. Averaging those margins raw would let a classifier
    with large weights dominate, so each margin is divided by ||w||, turning
    it into the signed Euclidean distance from the sample to that
    classifier's hyperplane — a scale-free quantity that can be averaged
    across the pool (ADR 0016).

    Raises ValueError if the pool disagrees on `classes_`, since then the
    per-class columns are not aligned and the mean is meaningless.
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
    return float(np.mean(classes[np.argmax(total, axis=1)] == y))


def soft_fusion_accuracy(pool: list, X: np.ndarray, y: np.ndarray):
    """Best available soft-combination accuracy. Returns (accuracy, rule).

    rule is "mean_probs" when the pool exposes predict_proba, otherwise
    "mean_decision_norm", otherwise "none" with a None accuracy.
    """
    try:
        return mean_probs_accuracy(pool, X, y), "mean_probs"
    except (AttributeError, ValueError):
        pass
    try:
        return mean_decision_accuracy(pool, X, y), "mean_decision_norm"
    except (AttributeError, ValueError):
        return None, "none"
