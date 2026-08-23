"""ECoL overlapping measures: F2, F3, F4 + dispatcher (dataset-level scalars).

Translated from ECoL R package (github.com/lpfgarcia/ECoL, R/feature-based.R).
F1 and F1v live in fisher_measures.py (split for LOC cap).
"""

from __future__ import annotations

import numpy as np

from pos.complexity.fisher_measures import f1, f1v, _ovo_pairs, _branch


def _region_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Volume of overlap for one class pair — product of per-feature overlap/range."""
    overlap = np.maximum(np.minimum(a.max(axis=0), b.max(axis=0)) -
                         np.maximum(a.min(axis=0), b.min(axis=0)), 0)
    total_range = np.maximum(a.max(axis=0), b.max(axis=0)) - \
                  np.minimum(a.min(axis=0), b.min(axis=0))
    total_range[total_range == 0] = 1e-15
    return float(np.prod(overlap / total_range))


def _f2(X: np.ndarray, y: np.ndarray) -> float:
    """Volume of Overlap Region (mean over one-vs-one pairs)."""
    pairs = _ovo_pairs(y)
    vals = [_region_overlap(_branch(X, y, c1), _branch(X, y, c2))
            for c1, c2 in pairs]
    return float(np.mean(vals)) if vals else 0.0


def _f3(X: np.ndarray, y: np.ndarray) -> float:
    """Max Individual Feature Efficiency: 1 - max(non-overlap fraction)."""
    pairs = _ovo_pairs(y)
    vals = []
    for c1, c2 in pairs:
        a, b = _branch(X, y, c1), _branch(X, y, c2)
        X_sub = X[(y == c1) | (y == c2)]
        minmax = np.minimum(a.max(axis=0), b.max(axis=0))
        maxmin = np.maximum(a.min(axis=0), b.min(axis=0))
        non_ovlp = (X_sub < maxmin) | (X_sub > minmax)
        frac = non_ovlp.sum(axis=0) / len(X_sub)
        vals.append(1.0 - float(np.max(frac)))
    return float(np.mean(vals)) if vals else 0.0


def _f4(X: np.ndarray, y: np.ndarray) -> float:
    """Collective Feature Efficiency: fraction remaining after iterative removal."""
    pairs = _ovo_pairs(y)
    vals = []
    for c1, c2 in pairs:
        a, b = _branch(X, y, c1), _branch(X, y, c2)
        X_sub = X[(y == c1) | (y == c2)].copy()
        n_total = len(X_sub)
        remaining = list(range(X.shape[1]))
        while remaining and len(X_sub) > 0:
            cols = X_sub[:, remaining]
            a_r, b_r = a[:, remaining], b[:, remaining]
            minmax = np.minimum(a_r.max(axis=0), b_r.max(axis=0))
            maxmin = np.maximum(a_r.min(axis=0), b_r.min(axis=0))
            non_ovlp = (cols < maxmin) | (cols > minmax)
            best_col = int(np.argmax(non_ovlp.sum(axis=0)))
            X_sub = X_sub[~non_ovlp[:, best_col]]
            remaining.pop(best_col)
            if len(X_sub) == 0:
                break
        vals.append(len(X_sub) / n_total)
    return float(np.mean(vals)) if vals else 0.0


def overlapping_measures(X: np.ndarray, y: np.ndarray, m_name: str) -> float:
    """Return dataset-level scalar for one overlapping measure (ECoL-compatible)."""
    if m_name == "F1":
        return f1(X, y)
    if m_name == "F1v":
        return f1v(X, y)
    if m_name == "F2":
        return _f2(X, y)
    if m_name == "F3":
        return _f3(X, y)
    if m_name == "F4":
        return _f4(X, y)
    return 0.0