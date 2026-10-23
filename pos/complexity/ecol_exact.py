"""Exact ports of ECoL 0.3.0 N1, LSC and F4 (ADR 0023).

The fast backend's `_lsc_fast` and `_f4` keep the definitions the GA has been
run with; these follow ECoL's R code (`c.LSC`, `removing`/`c.F4`) to the
letter and are checked against it through `ecol_bridge`:

* N1 = share of points with at least one cross-class edge in the minimum
  spanning tree of the full Gower distance matrix (zero distances are not
  edges, as in igraph's `graph.adjacency`); dense, so validation-only;
* LSC = 1 - sum_i |LS(i)| / n^2, LS(i) = points strictly closer to i than i's
  nearest enemy (the fast version divides each |LS(i)| by i's class size);
* F4 removes, per one-vs-one pair, the points that the most efficient
  remaining feature separates, *recomputing the class ranges on what is
  left* each round, until no point, no feature or a single class remains.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KDTree

from pos.complexity.neighborhood_extra import _normalize


def n1_ecol(X: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y)
    if len(y) < 2:
        return 0.0
    D = squareform(pdist(_normalize(np.asarray(X, dtype=float)), "cityblock"))
    mst = minimum_spanning_tree(D).tocoo()
    cross = y[mst.row] != y[mst.col]
    border = np.union1d(mst.row[cross], mst.col[cross])
    return float(len(border) / len(y))


def lsc_ecol(X: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y)
    n = len(y)
    if n < 2 or len(np.unique(y)) < 2:
        return 0.0
    Xn = _normalize(np.asarray(X, dtype=float))
    total = 0
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        enemy = KDTree(Xn[y != c], metric="manhattan")
        r = enemy.query(Xn[rows], k=1)[0][:, 0]
        # strict `<` and r = d(nearest enemy): the local set is all same-class
        # (a point sharing coordinates with an enemy has r = 0: empty set).
        own = KDTree(Xn[rows], metric="manhattan")
        counts = own.query_radius(Xn[rows], r=np.nextafter(r, 0.0), count_only=True)
        total += int(counts[r > 0].sum())
    return float(1 - total / n**2)


def _non_overlap(Xs: np.ndarray, ys: np.ndarray, c1, c2) -> np.ndarray:
    a, b = Xs[ys == c1], Xs[ys == c2]
    minmax = np.minimum(a.max(axis=0), b.max(axis=0))
    maxmin = np.maximum(a.min(axis=0), b.min(axis=0))
    return (Xs < maxmin) | (Xs > minmax)


def _remaining_fraction(Xs: np.ndarray, ys: np.ndarray, c1, c2) -> float:
    n0 = len(ys)
    while True:
        tmp = _non_overlap(Xs, ys, c1, c2)
        col = int(np.argmax(tmp.sum(axis=0)))
        keep = ~tmp[:, col]
        Xs, ys = np.delete(Xs[keep], col, axis=1), ys[keep]
        if len(ys) == 0 or Xs.shape[1] == 0 or len(np.unique(ys)) == 1:
            return len(ys) / n0


def f4_ecol(X: np.ndarray, y: np.ndarray) -> float:
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2:
        return 0.0
    vals = [_remaining_fraction(X[(y == c1) | (y == c2)], y[(y == c1) | (y == c2)], c1, c2)
            for c1, c2 in combinations(classes, 2)]
    return float(np.mean(vals))
