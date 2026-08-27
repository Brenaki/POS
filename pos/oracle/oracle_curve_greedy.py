"""Greedy Oracle_N curve construction (ADR 2020 §5).

At each step N=1..M, select the classifier that most increases the oracle
accuracy. Guarantees monotonicity Oracle_1 <= Oracle_2 <= ... <= Oracle_M.

Oracle_M (all classifiers) is the only point directly comparable to Table 7
(classic pool oracle). Oracle_1..Oracle_99 is a POS contribution and depends
on the greedy ordering; it is NOT comparable to any published number.

A second, order-free reading of "Oracle_N" is also recorded: `atleast_curve`
gives the fraction of samples on which **at least N** of the M classifiers
were correct. It is decreasing, needs no ordering of the pool, and its N=1
point coincides with the greedy curve's N=M point (both are the classic
pool oracle). Both curves are reported; only that shared point is comparable
to Table 7.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def greedy_oracle_curve(matrix: np.ndarray) -> list[float]:
    """Build a greedy Oracle_1..Oracle_M curve.

    Parameters
    ----------
    matrix : np.ndarray of shape (n_samples, n_classifiers), values 0/1

    Returns
    -------
    curve : list[float] of length M, where curve[N-1] = oracle accuracy
            with the first N classifiers selected by the greedy strategy.
    """
    n_samples, M = matrix.shape
    if M == 0:
        return []
    remaining = list(range(M))
    selected: list[int] = []
    covered = np.zeros(n_samples, dtype=bool)
    curve: list[float] = []

    for _ in range(M):
        best_idx = -1
        best_gain = -1
        for idx in remaining:
            gain = int(matrix[:, idx].sum() - covered.dot(matrix[:, idx]))
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
        covered = covered | (matrix[:, best_idx].astype(bool))
        curve.append(float(covered.mean()))
    return curve


def extract_checkpoints(
    curve: list[float], checkpoints: Sequence[int]
) -> dict[int, float]:
    """Extract {N: curve[N-1]} for each N in checkpoints.

    If a checkpoint exceeds the curve length, it is clipped to the last value.
    """
    M = len(curve)
    result: dict[int, float] = {}
    for n in checkpoints:
        idx = min(n, M) - 1
        result[n] = curve[idx] if idx >= 0 else 0.0
    return result

def atleast_curve(matrix: np.ndarray) -> list[float]:
    """Fraction of samples where at least N of the M classifiers are correct.

    `curve[N-1]` for N = 1..M. Order-free and decreasing:
    `curve[0]` is the classic pool oracle, `curve[-1]` is unanimity.
    """
    n_samples, M = matrix.shape
    if M == 0:
        return []
    correct = np.asarray(matrix).sum(axis=1)
    return [float((correct >= n).mean()) for n in range(1, M + 1)]
