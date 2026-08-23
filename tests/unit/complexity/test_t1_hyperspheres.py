"""Tests for ADR 0014: the lazy-greedy T1 must match the naive greedy exactly.

The naive implementation rebuilt `set(np.where(covered)[0])` once per candidate
per round — O(n^3) — which made Magic-sized bags (~6.8k instances) impossible.
The replacement is a CELF lazy greedy with key `(-gain, index)`, which yields
the identical cover (largest gain, lowest index on ties).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KDTree

from pos.complexity.neighborhood_extra import _greedy_cover_fraction, _t1_fast


def _t1_naive(Xn: np.ndarray, y: np.ndarray) -> float:
    """The pre-ADR-0014 implementation, kept as the reference oracle."""
    n = len(y)
    classes = np.unique(y)
    if n < 2 or len(classes) < 2:
        return 0.0
    enemy_dist = np.full(n, np.inf)
    for c in classes:
        mask_c = y == c
        if not (~mask_c).any():
            continue
        tree_diff = KDTree(Xn[~mask_c], metric="manhattan")
        enemy_dist[mask_c] = tree_diff.query(Xn[mask_c], k=1)[0][:, 0]
    radii = enemy_dist / 2.0
    radii[~np.isfinite(radii)] = 0.0
    tree_all = KDTree(Xn, metric="manhattan")
    neighbors = [set(tree_all.query_radius([Xn[i]], r=radii[i])[0]) for i in range(n)]
    covered = np.zeros(n, dtype=bool)
    n_spheres = 0
    while not covered.all():
        best_i, best_count = -1, -1
        for i in range(n):
            if covered[i]:
                continue
            count = len(neighbors[i] - set(np.where(covered)[0]))
            if count > best_count:
                best_count, best_i = count, i
        if best_i < 0:
            break
        covered[list(neighbors[best_i])] = True
        n_spheres += 1
    return float(n_spheres / n)


def _normalize(X):
    r = np.ptp(X, axis=0)
    return X / np.where(r == 0, 1.0, r)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("kind", ["separated", "overlapping", "duplicates"])
def test_matches_naive_greedy(seed, kind):
    rng = np.random.default_rng(seed)
    n, d = 180, 4
    if kind == "separated":
        y = rng.integers(0, 3, n)
        X = rng.normal(size=(n, d)) + y[:, None] * 6.0
    elif kind == "overlapping":
        y = rng.integers(0, 2, n)
        X = rng.normal(size=(n, d))
    else:  # heavy ties — the case where the greedy tie-break matters
        y = rng.integers(0, 2, n)
        X = rng.integers(0, 4, size=(n, d)).astype(float)
    Xn = _normalize(X)
    assert _t1_fast(Xn, y) == pytest.approx(_t1_naive(Xn, y), abs=1e-12)


class TestGreedyCoverFraction:
    def test_single_sphere_covers_everything(self):
        neighbors = [np.arange(5) for _ in range(5)]
        assert _greedy_cover_fraction(neighbors, 5) == pytest.approx(1 / 5)

    def test_no_sphere_covers_a_neighbour(self):
        neighbors = [np.array([i]) for i in range(5)]
        assert _greedy_cover_fraction(neighbors, 5) == pytest.approx(1.0)

    def test_ties_pick_the_lowest_index(self):
        neighbors = [np.array([0, 1]), np.array([0, 1]), np.array([2])]
        assert _greedy_cover_fraction(neighbors, 3) == pytest.approx(2 / 3)


def test_degenerate_inputs():
    X = _normalize(np.random.default_rng(0).normal(size=(10, 2)))
    assert _t1_fast(X, np.zeros(10, dtype=int)) == 0.0
    assert _t1_fast(X[:1], np.array([0])) == 0.0


@pytest.mark.slow
def test_scales_to_magic_sized_bag():
    """A Magic bag is ~6.8k instances; the naive version needed hours."""
    import time

    rng = np.random.default_rng(0)
    n = 7000
    X = rng.normal(size=(n, 10))
    y = (X[:, 0] + rng.normal(scale=2, size=n) > 0).astype(int)
    t0 = time.perf_counter()
    _t1_fast(_normalize(X), y)
    assert time.perf_counter() - t0 < 30.0
