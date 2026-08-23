"""Pool evaluation: individual + ensemble metrics for a fitted pool.

Composes correctness_matrix, oracle_curve, majority_vote, mean_probs, and
pairwise double-fault diversity into a single evaluate_pool() call.
Used by the experiment runner (Ativ.3) to record the full picture of a
pool's behavior on a test set.

ADR 0006 (Oracle_N), ADR 0009 (reproducibility).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pos.diversity import double_fault_matrix
from pos.oracle.comparison import majority_vote_accuracy, mean_probs_accuracy
from pos.oracle.correctness_matrix import build_correctness_matrix
from pos.oracle.oracle_curve import oracle_curve_array


def _double_fault_matrix(y: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Pairwise double-fault matrix (M x M), diagonal = 0.

    preds has shape (M, n_samples). Entry [i, j] = fraction of samples
    where both clf_i and clf_j are wrong.

    Vectorised in ADR 0015 — this ran M*(M-1)/2 = 4950 calls into deslib's
    per-sample Python loop for every fold of every mode.
    """
    dfm = double_fault_matrix(y, preds)
    np.fill_diagonal(dfm, 0.0)
    return dfm


def evaluate_pool(pool: list, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Evaluate a fitted pool on (X, y) and return full metrics.

    Parameters
    ----------
    pool : list of fitted classifiers (must implement .predict(X))
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)

    Returns
    -------
    dict with keys:
        individual_accuracies : list[float], length M
        correctness_matrix : np.ndarray (n_samples, M) of 0/1
        oracle_curve : list[float], length M (Oracle_1..M)
        majority_vote : float
        mean_probs : float or None (None if pool lacks predict_proba)
        double_fault_matrix : np.ndarray (M, M)
        double_fault_mean : float (mean of off-diagonal entries)
        n_classifiers : int
    """
    matrix = build_correctness_matrix(pool, X, y)
    M = len(pool)

    individual_accuracies = [float(matrix[:, j].mean()) for j in range(M)]
    oracle_curve = oracle_curve_array(matrix)
    majority = majority_vote_accuracy(pool, X, y)

    mean_probs: float | None = None
    try:
        mean_probs = mean_probs_accuracy(pool, X, y)
    except (AttributeError, ValueError):
        mean_probs = None

    preds = np.array([clf.predict(X) for clf in pool])
    dfm = _double_fault_matrix(y, preds)
    off_diag = dfm[~np.eye(M, dtype=bool)]
    df_mean = float(off_diag.mean()) if off_diag.size else 0.0

    return {
        "individual_accuracies": individual_accuracies,
        "correctness_matrix": matrix,
        "oracle_curve": oracle_curve,
        "majority_vote": majority,
        "mean_probs": mean_probs,
        "double_fault_matrix": dfm,
        "double_fault_mean": df_mean,
        "n_classifiers": M,
    }
