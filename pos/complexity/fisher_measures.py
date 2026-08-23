"""ECoL overlapping measures: F1, F1v (Fisher-based, dataset-level scalars).

Translated from ECoL R package (github.com/lpfgarcia/ECoL, R/feature-based.R).
"""

from __future__ import annotations

import itertools

import numpy as np


def _ovo_pairs(y: np.ndarray) -> list:
    return list(itertools.combinations(np.unique(y), 2))


def _branch(X: np.ndarray, y: np.ndarray, cls) -> np.ndarray:
    return X[y == cls]


def f1(X: np.ndarray, y: np.ndarray) -> float:
    """Maximum Fisher's Discriminant Ratio: 1/(F+1) per feature, mean."""
    classes = np.unique(y)
    n_feat = X.shape[1]
    overall_mean = X.mean(axis=0)
    num = np.zeros(n_feat)
    den = np.zeros(n_feat)
    for c in classes:
        Xc = _branch(X, y, c)
        nc = len(Xc)
        num += nc * (Xc.mean(axis=0) - overall_mean) ** 2
        den += np.sum((Xc - Xc.mean(axis=0)) ** 2, axis=0)
    fisher = num / np.maximum(den, 1e-15)
    return float(np.nanmean(1.0 / (fisher + 1.0)))


def f1v(X: np.ndarray, y: np.ndarray) -> float:
    """Directional Fisher ratio via LDA projection (one-vs-one, mean)."""
    pairs = _ovo_pairs(y)
    if not pairs:
        return 0.0
    vals = []
    for c1, c2 in pairs:
        a = _branch(X, y, c1)
        b = _branch(X, y, c2)
        if len(a) < 2 or len(b) < 2:
            continue
        c1m, c2m = a.mean(axis=0), b.mean(axis=0)
        n = len(a) + len(b)
        W = (len(a) / n) * np.cov(a, rowvar=False) + \
            (len(b) / n) * np.cov(b, rowvar=False)
        B = np.outer(c1m - c2m, c1m - c2m)
        d = np.linalg.pinv(W) @ (c1m - c2m)
        num = float(d @ B @ d)
        den = float(d @ W @ d)
        vals.append(1.0 if den < 1e-15 else 1.0 / (num / den + 1.0))
    return float(np.mean(vals)) if vals else 0.0