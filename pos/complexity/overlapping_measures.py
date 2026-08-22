"""Overlapping measures: F1, F2, F3, F4 (numpy-only, no pyhard).

F1 = fraction of features in overlap region per instance.
F2/F3/F4 = min/mean/max overlap degree (do_t) per instance.
"""

from __future__ import annotations

import itertools

import numpy as np


def _f1(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fisher overlap F1 — fraction of features in overlapping region."""
    n, n_feat = X.shape
    classes = np.unique(y)
    F1 = np.zeros(n)
    for c1, c2 in itertools.combinations(classes, 2):
        mask = (y == c1) | (y == c2)
        Xs, ys = X[mask], y[mask]
        indicator = np.zeros(mask.sum())
        for j in range(n_feat):
            f = Xs[:, j]
            max_min = max(f[ys == c1].min(), f[ys == c2].min())
            min_max = min(f[ys == c1].max(), f[ys == c2].max())
            indicator += ((f >= max_min) & (f <= min_max)).astype(float)
        F1[mask] += indicator / n_feat
    return F1 / max(len(classes) - 1, 1)


def _do_t(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Overlap degree (used by F2, F3, F4)."""
    n, n_feat = X.shape
    classes = np.unique(y)
    dot = np.zeros((n, n_feat))
    for c1, c2 in itertools.combinations(classes, 2):
        mask = (y == c1) | (y == c2)
        Xs, ys = X[mask], y[mask]
        for j in range(n_feat):
            f = Xs[:, j]
            max_min = max(f[ys == c1].min(), f[ys == c2].min())
            min_max = min(f[ys == c1].max(), f[ys == c2].max())
            denom = min_max - max_min
            if denom == 0:
                continue
            do = (min_max - f) / denom
            dot[mask, j] += 1.0 / (1.0 + np.abs(0.5 - do))
    k = len(classes)
    dot /= max(k * (k - 1) / 2, 1)
    return dot


def overlapping_measures(X: np.ndarray, y: np.ndarray, m_name: str) -> float:
    """Return dataset-level scalar for one overlapping measure."""
    if m_name == "F1":
        return float(np.nanmean(_f1(X, y)))
    dot = _do_t(X, y)
    if m_name == "F2":
        return float(np.nanmean(dot.min(axis=1)))
    if m_name == "F3":
        return float(np.nanmean(dot.mean(axis=1)))
    if m_name == "F4":
        return float(np.nanmean(dot.max(axis=1)))
    return 0.0