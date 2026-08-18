"""Correctness matrix: 1/0 per (sample, classifier) pair.

Given a fitted pool and (X, y), produces an (n_samples, M) matrix where
entry [i, j] = 1 if classifier j predicts y[i] correctly, else 0.
"""

from __future__ import annotations

import numpy as np


def build_correctness_matrix(pool: list, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Build the (n_samples, n_classifiers) correctness matrix.

    Parameters
    ----------
    pool : list of fitted classifiers (must implement `.predict(X)`)
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)

    Returns
    -------
    matrix : np.ndarray of shape (n_samples, n_classifiers), dtype int
        matrix[i, j] = 1 if pool[j].predict(X)[i] == y[i], else 0
    """
    n_samples = len(y)
    n_classifiers = len(pool)
    matrix = np.zeros((n_samples, n_classifiers), dtype=int)
    for j, clf in enumerate(pool):
        preds = clf.predict(X)
        matrix[:, j] = (preds == y).astype(int)
    return matrix
