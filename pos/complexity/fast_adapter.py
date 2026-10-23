"""Fast complexity adapter: dispatches to overlapping/neighborhood modules.

Default complexity backend (ADR 0023). Computes overlapping, neighborhood,
balance and dimensionality; linearity and network return 0.0.
`backend="ecol-0.3.0"` swaps T1/N1/LSC/F4 for the exact ECoL ports (ADR 0024). Which values
match R/ECoL 0.3.0 exactly is pinned by tests/characterization/test_ecol_golden.py.
"""

from __future__ import annotations

import numpy as np

from pos.complexity.backends import check_backend, port_for
from pos.complexity.base import GROUP_MEASURES
from pos.complexity.neighborhood_fast import neighborhood_measure_fast
from pos.complexity.overlapping_measures import overlapping_measures
from pos.complexity.simple_measures import simple_measure


def _one(X, y, grp: str, m_name: str, backend: str) -> float:
    port = port_for(backend, m_name)
    if port is not None:
        return port(X, y)
    if grp == "overlapping":
        return overlapping_measures(X, y, m_name)
    if grp == "neighborhood":
        return neighborhood_measure_fast(X, y, m_name)
    return simple_measure(X, y, grp, m_name)


def complexity_data3(
    X_data: np.ndarray,
    y_data: np.ndarray,
    group: list[str],
    types: list[str] | None = None,
    backend: str = "fast",
) -> list[float]:
    """Compute complexity measures for the given groups (KD-tree accelerated).

    When types is None: compute ALL measures in each group (legacy mode).
    When types is a list: types[i] is the single measure name for group[i]
    (one-to-one correspondence, matching ECoL R behavior). Returns one scalar
    per group element. `backend` picks the definitions (ADR 0024).
    """
    check_backend(backend)
    X = np.asarray(X_data, dtype=float)
    y = np.asarray(y_data)
    if types is not None:
        names = (list(types) + [None] * len(group))[: len(group)]
        return [0.0 if m is None else _one(X, y, grp, m, backend)
                for grp, m in zip(group, names, strict=True)]
    return [_one(X, y, grp, m, backend)
            for grp in group for m in GROUP_MEASURES.get(grp, [])]
