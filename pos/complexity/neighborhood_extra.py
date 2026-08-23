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

    # Step 1: nearest enemy distance for each point
    enemy_dist = np.full(n, np.inf)
    for c in classes:
        mask_c = y == c
        if not (~mask_c).any():
            continue
        tree_diff = KDTree(Xn[~mask_c], metric="manhattan")
        d, _ = tree_diff.query(Xn[mask_c], k=1)
        enemy_dist[mask_c] = d[:, 0]

    # Step 2: radius = enemy_dist / 2 (simplified radios)
    radii = enemy_dist / 2.0
    radii[~np.isfinite(radii)] = 0.0

    # Step 3: adherence sets, one batched KDTree call
    tree_all = KDTree(Xn, metric="manhattan")
    neighbors = tree_all.query_radius(Xn, r=radii)

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
