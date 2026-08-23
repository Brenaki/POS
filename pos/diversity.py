"""Pairwise double-fault diversity (extracted from Cpx.py during Fase 2 refactor).

Returns one mean double-fault value per classifier (mean over all pairs that
include that classifier).

Vectorised in ADR 0015. `deslib.util.diversity.double_fault` loops over every
sample in pure Python, and `diversitys` called it M*(M-1) times: for a Magic
fold (M=100, n_val=3423) that is ~34 million interpreted iterations, ~18 s per
call and ~26% of a whole GA generation. The identity below replaces it with a
single boolean matmul while producing exactly the same numbers.

    double_fault(y, p_i, p_j) = N00 / n = |{s : p_i[s] != y[s] and p_j[s] != y[s]}| / n

so, with W[i, s] = (p_i[s] != y[s]),

    N00 = (W @ W.T) / n
"""

from __future__ import annotations

import numpy as np


def double_fault_matrix(y_test: np.ndarray, predicts: np.ndarray) -> np.ndarray:
    """Full M×M matrix of pairwise double-fault values (diagonal = own error rate)."""
    y_test = np.asarray(y_test)
    predicts = np.asarray(predicts)
    n = predicts.shape[1]
    wrong = (predicts != y_test).astype(np.float64)
    return (wrong @ wrong.T) / n


def diversitys(y_test: np.ndarray, predicts: np.ndarray) -> list[float]:
    """Mean pairwise double-fault per classifier.

    For each classifier i, computes double_fault(y_test, predicts[i],
    predicts[j]) for every j != i, then returns the mean. Output length
    equals the number of classifiers (len(predicts)).
    """
    predicts = np.asarray(predicts)
    m = predicts.shape[0]
    if m < 2:
        return [float("nan")] * m
    df = double_fault_matrix(y_test, predicts)
    # exclude the diagonal (i == j) from each row's mean
    row_sums = df.sum(axis=1) - np.diag(df)
    return (row_sums / (m - 1)).tolist()
