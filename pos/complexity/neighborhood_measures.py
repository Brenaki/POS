"""Neighborhood measures: N1, N2, kDN, LSC (numpy-only, no pyhard).

All use a normalized Manhattan distance matrix (Gower equivalent for numeric).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors


def dist_matrix(X: np.ndarray) -> np.ndarray:
    """Normalized Manhattan distance matrix (Gower equivalent for numeric)."""
    X = X.astype(float)
    ranges = np.ptp(X, axis=0)
    ranges[ranges == 0] = 1.0
    Xn = X / ranges
    return np.abs(Xn[:, None, :] - Xn[None, :, :]).sum(axis=2)


def _n1(D: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Borderline points N1 — fraction of MST neighbors of different class."""
    D_sparse = (D + np.eye(len(D)) * 1e10).astype(float)
    Tcsr = minimum_spanning_tree(D_sparse)
    mst = Tcsr.toarray()
    mst = np.where(mst > 0, mst, np.inf)
    N1 = np.zeros(len(y))
    for i in range(len(y)):
        neighbors = np.argwhere(np.minimum(mst[i, :], mst[:, i]) < np.inf).ravel()
        N1[i] = np.sum(y[neighbors] != y[i]) / max(len(neighbors), 1)
    return N1


def _n2(D: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Intra/extra ratio N2."""
    n = len(y)
    N2 = np.zeros(n)
    for i in range(n):
        same = D[i, y == y[i]].copy()
        same[same == 0] = np.inf
        diff = D[i, y != y[i]].copy()
        diff[diff == 0] = np.inf
        intra = same[1] if len(same) > 1 else 1e-15
        extra = diff.min() if len(diff) > 0 else 1e-15
        N2[i] = intra / max(extra, 1e-15)
    return 1 - 1 / (N2 + 1)


def _kdn(D: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
    """k-Disagreeing Neighbors."""
    n = len(y)
    k = min(k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=k, metric="precomputed").fit(D)
    _, indices = nbrs.kneighbors(D)
    kDN = np.zeros(n)
    for i in range(n):
        kDN[i] = np.sum(y[indices[i, 1:]] != y[i]) / max(k - 1, 1)
    return kDN


def _lsc(D: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Local Set Cardinality."""
    classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(classes, counts))
    indices = np.argsort(D, axis=1)
    LSC = np.zeros(len(y))
    for i in range(len(y)):
        nn = y[indices[i]]
        first_enemy = np.argmax(nn != y[i])
        LSC[i] = first_enemy / max(class_counts[y[i]], 1)
    return 1 - LSC


def neighborhood_measure(D: np.ndarray, y: np.ndarray, m_name: str) -> float:
    """Return dataset-level scalar for one neighborhood measure."""
    if m_name == "N1":
        return float(np.nanmean(_n1(D, y)))
    if m_name == "N2":
        return float(np.nanmean(_n2(D, y)))
    if m_name == "N4":
        return float(np.nanmean(_kdn(D, y)))
    if m_name == "LSC":
        return float(np.nanmean(_lsc(D, y)))
    return 0.0