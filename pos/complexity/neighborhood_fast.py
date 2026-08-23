"""KD-tree-based neighborhood measures — O(n·k·log n) instead of O(n²).

Exact for N2, kDN, LSC, T1. Approximate for N1 (kNN-graph MST).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse._base import SparseEfficiencyWarning  # noqa
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import KDTree

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


def _normalize(X: np.ndarray) -> np.ndarray:
    X = X.astype(float)
    ranges = np.ptp(X, axis=0)
    ranges[ranges == 0] = 1.0
    return X / ranges


def _n1_fast(Xn: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    n = len(y)
    k = min(k, n - 1)
    tree = KDTree(Xn, metric="manhattan")
    dists, indices = tree.query(Xn, k=k + 1)

    # Vectorized edge construction: n*k edges (skip column 0 = self)
    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].ravel()
    vals = dists[:, 1:].ravel()
    # Filter self-loops (i==j) to avoid setdiag (slow on CSR)
    mask = rows != cols
    rows, cols, vals = rows[mask], cols[mask], vals[mask]

    # Symmetrize: take min(i→j, j→i) as edge weight
    sparse = csr_matrix((vals, (rows, cols)), shape=(n, n))
    sym = sparse.minimum(sparse.T)
    sym.eliminate_zeros()

    mst = minimum_spanning_tree(sym)
    # Use sparse COO directly — MST has exactly n-1 edges
    mst_coo = mst.tocoo()
    mst_t = mst.T.tocoo()
    # Combine both directions
    edges_i = np.concatenate([mst_coo.row, mst_t.row])
    edges_j = np.concatenate([mst_coo.col, mst_t.col])

    # Count neighbors per node and different-class neighbors
    n_neighbors = np.bincount(edges_i, minlength=n)
    n_diff = np.bincount(edges_i[y[edges_j] != y[edges_i]], minlength=n)
    N1 = n_diff / np.maximum(n_neighbors, 1)
    return float(np.nanmean(N1))


def _n2_fast(Xn: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    classes = np.unique(y)
    N2 = np.zeros(n)

    for c in classes:
        mask_c = y == c
        mask_other = ~mask_c
        if not mask_other.any():
            continue

        # Same-class 1-NN (excluding self)
        if mask_c.sum() > 1:
            tree_same = KDTree(Xn[mask_c], metric="manhattan")
            d_same, _ = tree_same.query(Xn[mask_c], k=2)
            intra = d_same[:, 1]
        else:
            intra = np.full(mask_c.sum(), 1e15)

        # Different-class 1-NN
        tree_diff = KDTree(Xn[mask_other], metric="manhattan")
        d_diff, _ = tree_diff.query(Xn[mask_c], k=1)
        extra = d_diff[:, 0]

        ratio = intra / np.maximum(extra, 1e-15)
        N2[mask_c] = 1 - 1 / (ratio + 1)

    return float(np.nanmean(N2))


def _kdn_fast(Xn: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    n = len(y)
    k = min(k + 1, n)
    tree = KDTree(Xn, metric="manhattan")
    _, indices = tree.query(Xn, k=k)
    kDN = np.zeros(n)
    for i in range(n):
        kDN[i] = np.sum(y[indices[i, 1:]] != y[i]) / max(k - 1, 1)
    return float(np.nanmean(kDN))


def _lsc_fast(Xn: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(classes, counts))
    tree_all = KDTree(Xn, metric="manhattan")
    LSC = np.zeros(n)
    for c in classes:
        mask_c = y == c
        idx_c = np.where(mask_c)[0]
        if not (~mask_c).any() or len(idx_c) == 0:
            continue
        tree_diff = KDTree(Xn[~mask_c], metric="manhattan")
        d_enemy, _ = tree_diff.query(Xn[mask_c], k=1)
        counts_in_radius = tree_all.query_radius(
            Xn[mask_c], r=d_enemy[:, 0] * (1 - 1e-10)
        )
        for pos, i in enumerate(idx_c):
            n_closer = len(counts_in_radius[pos])  # includes self
            LSC[i] = n_closer / max(class_counts[y[i]], 1)
    return float(np.nanmean(1 - LSC))


def _t1_fast(Xn: np.ndarray, y: np.ndarray) -> float:
    """T1 = Fraction of Hyper-spheres Covering Data (ECoL N5). Delegates."""
    from pos.complexity.neighborhood_extra import _t1_fast as _t1
    return _t1(Xn, y)


def neighborhood_measure_fast(
    X: np.ndarray, y: np.ndarray, m_name: str
) -> float:
    """KD-tree version of neighborhood_measure — no n² dist matrix."""
    Xn = _normalize(X)
    y = np.asarray(y)
    if m_name == "N1":
        return _n1_fast(Xn, y)
    if m_name == "N2":
        return _n2_fast(Xn, y)
    if m_name == "N4":
        return _kdn_fast(Xn, y)
    if m_name == "LSC":
        return _lsc_fast(Xn, y)
    if m_name == "T1":
        return _t1_fast(Xn, y)
    if m_name == "N3":
        from pos.complexity.neighborhood_extra import _n3_fast
        return _n3_fast(Xn, y)
    return 0.0