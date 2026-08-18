"""Dispersion metrics (extracted from Cpx.py during Fase 2 refactor).

Two functions preserved with exact original behavior:
- dispersion_linear: manual pairwise |a-b| mean per measure, then min-max norm.
- dispersion: sklearn pairwise_distances mean per row.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from sklearn.metrics import pairwise_distances

from pos.normalization import min_max_norm

ArrayLike = Union[list, np.ndarray]


def dispersion_linear(complexity: ArrayLike) -> List[list]:
    """Manual pairwise |a-b| dispersion per complexity measure, then min-max norm.

    Input shape: (n_bags, n_measures).
    Output shape: (n_bags, n_measures) as list of lists, after an internal
    transpose so the per-measure dispersion is normalized across bags.
    """
    result: list = []
    result1: list = []

    complexity = list(complexity)
    complexity = np.array(complexity)
    n = (len(complexity)) - 1
    complexity = complexity.T

    for i in complexity:
        dist = []
        for j in range(len(i)):
            dista = 0
            for l in range(len(i)):
                if j == l:
                    continue
                else:
                    dista += abs(i[j] - i[l])
            dist.append((dista) / n)
        result.append(dist)
    result = np.array(result)
    for i in result:
        r = min_max_norm(i)
        result1.append(r)
    result1 = np.array(result1)
    del result, r, dist, complexity  # noqa: F821 — legacy cleanup, preserves behavior
    result1 = result1.T
    result1 = result1.tolist()
    return result1


def dispersion(complexity: ArrayLike) -> list:
    """Mean pairwise distance per row using sklearn pairwise_distances."""
    result: list = []
    dista = pairwise_distances(complexity, n_jobs=6)
    dista = dista.tolist()

    for i in dista:
        result.append(np.mean(i))
    return result
