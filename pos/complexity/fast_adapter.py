"""Fast complexity adapter: dispatches to overlapping/neighborhood modules.

Drop-in replacement for pos.complexity.pyhard_adapter.complexity_data3.
Only computes the 8 measures the GA uses (F1-F4, N1/N2/N4/LSC).
Placeholders (F1v, N3, T1) return 0.0.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from pos.complexity.base import GROUP_MEASURES
from pos.complexity.neighborhood_measures import dist_matrix, neighborhood_measure
from pos.complexity.overlapping_measures import overlapping_measures


def complexity_data3(
    X_data: np.ndarray,
    y_data: np.ndarray,
    group: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Compute complexity measures for the given groups (fast numpy-only)."""
    X = np.asarray(X_data, dtype=float)
    y = np.asarray(y_data)
    result: list = []

    D = dist_matrix(X) if "neighborhood" in group else None

    for grp in group:
        for m_name in GROUP_MEASURES.get(grp, []):
            if grp == "overlapping":
                result.append(overlapping_measures(X, y, m_name))
            elif grp == "neighborhood":
                result.append(neighborhood_measure(D, y, m_name))
            else:
                result.append(0.0)
    return result