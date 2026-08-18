"""Oracle_N: generalization of the Oracle upper bound.

Oracle_N(x) = 1 if at least N classifiers in the pool correctly predict x,
else 0. The Oracle_N accuracy is the fraction of samples where Oracle_N = 1.

- Oracle_1 = traditional Oracle (>= 1 classifier correct)
- Oracle_M = unanimity (all M classifiers must be correct)
- Oracle_0 = 1.0 trivially

Monotonic: Oracle_1 >= Oracle_2 >= ... >= Oracle_M.
"""

from __future__ import annotations

import numpy as np


def oracle_n_accuracy(matrix: np.ndarray, n: int) -> float:
    """Fraction of samples correctly classified by at least N classifiers.

    Parameters
    ----------
    matrix : np.ndarray of shape (n_samples, n_classifiers), values 0/1
    n : int, minimum number of correct classifiers required

    Returns
    -------
    accuracy : float in [0, 1]
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    n_samples = matrix.shape[0]
    if n_samples == 0:
        return 0.0
    # Sum across classifiers (axis=1) → (n_samples,) of correct counts
    correct_counts = matrix.sum(axis=1)
    # Oracle_N = 1 if correct_counts >= n
    oracle = (correct_counts >= n).astype(int)
    return float(oracle.mean())


def oracle_n_vector(matrix: np.ndarray, n: int) -> np.ndarray:
    """Boolean vector: Oracle_N(x_i) for each sample i."""
    correct_counts = matrix.sum(axis=1)
    return correct_counts >= n
