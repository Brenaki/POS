"""Oracle curve: Oracle_1, Oracle_2, ..., Oracle_M accuracies.

Builds the full curve of Oracle_N accuracies for N = 1..M, where M is the
number of classifiers in the pool. The curve is monotonically non-increasing.
"""

from __future__ import annotations

import numpy as np

from pos.oracle.oracle_n import oracle_n_accuracy


def oracle_curve(matrix: np.ndarray) -> dict[int, float]:
    """Return {N: oracle_n_accuracy(matrix, N)} for N = 1..M.

    Parameters
    ----------
    matrix : np.ndarray of shape (n_samples, n_classifiers), values 0/1

    Returns
    -------
    curve : dict mapping N (1..M) → accuracy float in [0, 1]
    """
    n_classifiers = matrix.shape[1]
    return {n: oracle_n_accuracy(matrix, n) for n in range(1, n_classifiers + 1)}


def oracle_curve_array(matrix: np.ndarray) -> list[float]:
    """Return [acc_1, acc_2, ..., acc_M] as a list (ordered by N)."""
    curve = oracle_curve(matrix)
    n_classifiers = matrix.shape[1]
    return [curve[n] for n in range(1, n_classifiers + 1)]
