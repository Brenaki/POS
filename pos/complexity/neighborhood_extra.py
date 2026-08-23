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

    # Step 3: build adherence matrix (sparse via KDTree radius_neighbors)
    tree_all = KDTree(Xn, metric="manhattan")
    covered = np.zeros(n, dtype=bool)
    n_spheres = 0

    # Greedy set cover: pick sphere covering most uncovered points
    neighbors = [set(tree_all.query_radius([Xn[i]], r=radii[i])[0]) for i in range(n)]

    while not covered.all():
        best_i = -1
        best_count = -1
        for i in range(n):
            if covered[i]:
                continue
            uncovered = neighbors[i] - set(np.where(covered)[0])
            count = len(uncovered)
            if count > best_count:
                best_count = count
                best_i = i
        if best_i < 0:
            break
        covered[list(neighbors[best_i])] = True
        n_spheres += 1

    return float(n_spheres / n)