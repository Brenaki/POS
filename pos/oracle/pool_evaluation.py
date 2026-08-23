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
from pos.oracle.correctness_matrix import build_correctness_matrix
from pos.oracle.fusion_predict import majority_vote_predict, soft_fusion_predict
from pos.oracle.metrics import oracle_balanced_recall, prediction_metrics
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
        majority_vote_metrics : dict (acc, macro P/R/F1, balanced accuracy)
        mean_probs : float or None (soft-combination accuracy, None if the
            pool supports neither predict_proba nor decision_function)
        mean_probs_metrics : dict or None (same five metrics)
        soft_fusion_rule : "mean_probs" | "mean_decision_norm" | "none"
        oracle_1_balanced : float (macro recall of "at least one correct")
        double_fault_matrix : np.ndarray (M, M)
        double_fault_mean : float (mean of off-diagonal entries)
        n_classifiers : int
    """
    matrix = build_correctness_matrix(pool, X, y)
    M = len(pool)

    individual_accuracies = [float(matrix[:, j].mean()) for j in range(M)]
    oracle_curve = oracle_curve_array(matrix)
    mv_metrics = prediction_metrics(y, majority_vote_predict(pool, X))

    soft_preds, soft_rule = soft_fusion_predict(pool, X)
    soft_metrics = None if soft_preds is None else prediction_metrics(y, soft_preds)

    preds = np.array([clf.predict(X) for clf in pool])
    dfm = _double_fault_matrix(y, preds)
    off_diag = dfm[~np.eye(M, dtype=bool)]
    df_mean = float(off_diag.mean()) if off_diag.size else 0.0

    return {
        "individual_accuracies": individual_accuracies,
        "correctness_matrix": matrix,
        "oracle_curve": oracle_curve,
        "majority_vote": mv_metrics["acc"],
        "majority_vote_metrics": mv_metrics,
        "mean_probs": None if soft_metrics is None else soft_metrics["acc"],
        "mean_probs_metrics": soft_metrics,
        "soft_fusion_rule": soft_rule,
        "oracle_1_balanced": oracle_balanced_recall(matrix, y, 1),
        "double_fault_matrix": dfm,
        "double_fault_mean": df_mean,
        "n_classifiers": M,
    }
