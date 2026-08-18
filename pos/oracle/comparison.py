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
