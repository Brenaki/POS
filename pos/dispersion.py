"""Dispersion metrics (extracted from Cpx.py during Fase 2 refactor).

Two functions preserved with exact original behavior:
- dispersion_linear: manual pairwise |a-b| mean per measure, then min-max norm.
- dispersion: sklearn pairwise_distances mean per row.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from sklearn.metrics import pairwise_distances

from pos.normalization import min_max_norm

ArrayLike = Union[list, np.ndarray]


def dispersion_linear(complexity: ArrayLike) -> list[list]:
    """Manual pairwise |a-b| dispersion per complexity measure, then min-max norm.

    Input shape: (n_bags, n_measures).
    Output shape: (n_bags, n_measures) as list of lists, after an internal
    transpose so the per-measure dispersion is normalized across bags.
    """
    complexity = np.asarray(list(complexity), dtype=float)
    n = len(complexity) - 1
    measures = complexity.T  # (n_measures, n_bags)

    # sum_l |x_j - x_l| for every j, per measure — the l == j term is 0, so
    # excluding it explicitly (as the original loop did) changes nothing.
    result = np.abs(measures[:, :, None] - measures[:, None, :]).sum(axis=2) / n

    result1 = np.array([min_max_norm(row) for row in result])
    return result1.T.tolist()


def dispersion(complexity: ArrayLike) -> list:
    """Mean pairwise distance per row using sklearn pairwise_distances."""
    complexity = np.asarray(complexity, dtype=float)
    # n_jobs=6 spawned joblib workers for a 100x3 matrix — pure overhead.
    n_jobs = 6 if complexity.shape[0] > 2000 else None
    dista = pairwise_distances(complexity, n_jobs=n_jobs)
    return dista.mean(axis=1).tolist()
