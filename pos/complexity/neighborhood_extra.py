"""N3 (1-NN LOO error rate) and T1/N5 (Fraction of Hyper-spheres).

T1 = Fraction of Hyper-spheres Covering Data (ECoL N5).
Algorithm:
1. For each point i, find nearest enemy distance de[i]
2. Radius r[i] = de[i] / 2 (simplified — ECoL uses recursive radios,
   but de/2 is the common approximation used in the complexity library)
3. Adherence matrix: adh[i,j] = True if dist(i,j) < r[i]
4. Covering: iteratively pick the sphere covering most uncovered points,
   remove them, repeat until all covered
5. T1 = #spheres in covering / n
"""

from __future__ import annotations

import heapq

import numpy as np
from sklearn.neighbors import KDTree


def _normalize(X: np.ndarray) -> np.ndarray:
    X = X.astype(float)
    ranges = np.ptp(X, axis=0)
    ranges[ranges == 0] = 1.0
    return X / ranges


def _n3_fast(Xn: np.ndarray, y: np.ndarray) -> float:
    """N3 = 1-NN leave-one-out error rate."""
    n = len(y)
    k = min(2, n)
    tree = KDTree(Xn, metric="manhattan")
    _, indices = tree.query(Xn, k=k)
    nn = indices[:, 1] if k > 1 else indices[:, 0]
    errors = np.sum(y[nn] != y)
    return float(errors / n)


def _t1_fast(Xn: np.ndarray, y: np.ndarray) -> float:
    """T1/N5 = Fraction of Hyper-spheres Covering Data."""
    n = len(y)
    classes = np.unique(y)
    if n < 2 or len(classes) < 2:
        return 0.0

    # One KD-tree per class, reused for both steps below (ADR 0015).
    idx_of = {c: np.where(y == c)[0] for c in classes}
    trees = {c: KDTree(Xn[idx_of[c]], metric="manhattan") for c in classes}

    # Step 1: nearest enemy distance = min over the other classes' trees.
    enemy_dist = np.full(n, np.inf)
    for c in classes:
        idx_c = idx_of[c]
        for other in classes:
            if other == c:
                continue
            d = trees[other].query(Xn[idx_c], k=1)[0][:, 0]
            enemy_dist[idx_c] = np.minimum(enemy_dist[idx_c], d)

    # Step 2: radius = enemy_dist / 2 (simplified radios)
    radii = enemy_dist / 2.0
    radii[~np.isfinite(radii)] = 0.0

    # Step 3: adherence sets.
    # For r_i > 0 every point within r_i is necessarily same-class: an enemy
    # sits at d >= 2*r_i, so d <= r_i would force r_i <= 0. Querying the
    # point's own class tree is therefore exact and searches ~n/k points
    # instead of n. r_i == 0 is the one exception (a duplicate carrying a
    # different label sits at distance 0), so those fall back to a full tree.
    neighbors = np.empty(n, dtype=object)
    zero = radii <= 0.0
    for c in classes:
        idx_c = idx_of[c]
        sel = idx_c[~zero[idx_c]]
        if not len(sel):
            continue
        res = trees[c].query_radius(Xn[sel], r=radii[sel])
        for pos, i in enumerate(sel):
            neighbors[i] = idx_c[res[pos]]
    if zero.any():
        sel = np.where(zero)[0]
        res = KDTree(Xn, metric="manhattan").query_radius(Xn[sel], r=radii[sel])
        for pos, i in enumerate(sel):
            neighbors[i] = res[pos]

    return _greedy_cover_fraction(neighbors, n)


def _greedy_cover_fraction(neighbors, n: int) -> float:
    """Greedy set cover over `neighbors`, returning #spheres / n.

    Lazy (CELF) greedy: marginal coverage is submodular, so a popped
    candidate whose recomputed key is still <= the next heap key is the true
    lexicographic maximum. The heap key is `(-gain, index)`, which reproduces
    the naive loop's tie-break (largest gain, lowest index) exactly while
    dropping its O(n^3) rescan (ADR 0014).
    """
    covered = np.zeros(n, dtype=bool)
    heap = [(-len(neighbors[i]), i) for i in range(n)]
    heapq.heapify(heap)
    n_uncovered = n
    n_spheres = 0

    while n_uncovered > 0:
        best_i = -1
        while heap:
            key = heapq.heappop(heap)
            i = key[1]
            if covered[i]:
                continue
            nb = neighbors[i]
            gain = int(np.count_nonzero(~covered[nb])) if len(nb) else 0
            if not heap or (-gain, i) <= heap[0]:
                best_i = i
                break
            heapq.heappush(heap, (-gain, i))
        if best_i < 0:
            break
        nb = neighbors[best_i]
        n_uncovered -= int(np.count_nonzero(~covered[nb]))
        covered[nb] = True
        n_spheres += 1

    return float(n_spheres / n)
