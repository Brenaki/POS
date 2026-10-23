"""Complexity backends: which code computes each measure (ADR 0024).

* `fast` — the fast_adapter definitions every published run used;
* `ecol-0.3.0` — the same, except T1, N1, LSC and F4, which come from the
  exact ports of ECoL 0.3.0 (`t1_ecol`, `ecol_exact`; ADR 0023). N4 has no
  port (ECoL's interpolates random points) and stays kDN.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pos.complexity.ecol_exact import f4_ecol, lsc_ecol, n1_ecol
from pos.complexity.t1_ecol import t1_ecol

BACKENDS = ("fast", "ecol-0.3.0")

_PORTS: dict[str, dict[str, Callable[[np.ndarray, np.ndarray], float]]] = {
    "fast": {},
    "ecol-0.3.0": {
        "F4": f4_ecol,
        "N1": n1_ecol,
        "LSC": lsc_ecol,
        "T1": lambda X, y: t1_ecol(X, y, "0.3.0"),
    },
}


def check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"unknown complexity backend {backend!r} (known: {BACKENDS})")


def ported_measures(backend: str) -> set[str]:
    """Measure names the backend computes with an exact ECoL port."""
    check_backend(backend)
    return set(_PORTS[backend])


def port_for(backend: str, m_name: str):
    """The exact port for this measure under `backend`, or None."""
    return _PORTS[backend].get(m_name)
