"""ECoL 0.3.0 balance (C1, C2) and dimensionality (T2, T3, T4) measures.

Translated from ECoL's R/balance.R and R/dimensionality.R (ADR 0023); checked
against tests/_ecol_legacy_golden.json. T3/T4 count the principal components
needed for 95% of the variance exactly as `summary(prcomp(x))` does, which
rounds the cumulative proportion to 5 decimals before the comparison.
"""

from __future__ import annotations

import numpy as np


def _c1(y: np.ndarray) -> float:
    """Entropy of class proportions normalised by log(#classes)."""
    p = np.unique(y, return_counts=True)[1] / len(y)
    return float(-np.sum(p * np.log(p)) / np.log(len(p))) if len(p) > 1 else 0.0


def _c2(y: np.ndarray) -> float:
    """1 - 1/IR, IR = (nc-1)/nc * sum(n_i / (n - n_i))."""
    counts = np.unique(y, return_counts=True)[1]
    nc = len(counts)
    if nc < 2:
        return 0.0
    ir = (nc - 1) / nc * np.sum(counts / (len(y) - counts))
    return float(1 - 1 / ir)


def _n_components_95(X: np.ndarray) -> int:
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)[: min(X.shape)]
    var = s**2 / np.sum(s**2)
    return int(np.sum(np.round(np.cumsum(var), 5) < 0.95)) + 1


def simple_measure(X: np.ndarray, y: np.ndarray, group: str, m_name: str) -> float:
    """Dataset-level value of one balance/dimensionality measure (else 0.0)."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if group == "balance":
        return {"C1": _c1, "C2": _c2}.get(m_name, lambda _: 0.0)(np.asarray(y))
    if group == "dimensionality":
        if m_name == "T2":
            return p / n
        if m_name in ("T3", "T4"):
            return _n_components_95(X) / (n if m_name == "T3" else p)
    return 0.0
