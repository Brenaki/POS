"""Fast complexity adapter: dispatches to overlapping/neighborhood modules.

Default complexity backend (ADR 0023). Computes overlapping, neighborhood,
balance and dimensionality; linearity and network return 0.0. Which values
match R/ECoL 0.3.0 exactly is pinned by tests/characterization/test_ecol_golden.py.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from pos.complexity.base import GROUP_MEASURES
from pos.complexity.neighborhood_fast import neighborhood_measure_fast
from pos.complexity.overlapping_measures import overlapping_measures
from pos.complexity.simple_measures import simple_measure


def complexity_data3(
    X_data: np.ndarray,
    y_data: np.ndarray,
    group: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Compute complexity measures for the given groups (KD-tree accelerated).

    When types is None: compute ALL measures in each group (legacy mode).
    When types is a list: types[i] is the single measure name for group[i]
    (one-to-one correspondence, matching ECoL R behavior). Returns one scalar
    per group element.
    """
    X = np.asarray(X_data, dtype=float)
    y = np.asarray(y_data)
    result: list = []

    if types is not None:
        for i, grp in enumerate(group):
            m_name = types[i] if i < len(types) else None
            if m_name is None:
                result.append(0.0)
                continue
            if grp == "overlapping":
                result.append(overlapping_measures(X, y, m_name))
            elif grp == "neighborhood":
                result.append(neighborhood_measure_fast(X, y, m_name))
            else:
                result.append(simple_measure(X, y, grp, m_name))
        return result

    for grp in group:
        for m_name in GROUP_MEASURES.get(grp, []):
            if grp == "overlapping":
                result.append(overlapping_measures(X, y, m_name))
            elif grp == "neighborhood":
                result.append(neighborhood_measure_fast(X, y, m_name))
            else:
                result.append(simple_measure(X, y, grp, m_name))
    return result