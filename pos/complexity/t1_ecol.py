"""Exact port of ECoL's T1/N5, Fraction of Hyper-spheres (ADR 0023).

`neighborhood_extra._t1_fast` approximates every radius by half the
nearest-enemy distance. ECoL (`radios`) does not: when x's nearest enemy z
has a *different* nearest enemy, x gets d(x, z) - radius(z), which is
>= d(x, z)/2, so its sphere can swallow more same-class points. This module
reproduces ECoL step by step, with KD-trees instead of dense matrices:

1. nearest enemy, ties broken by the lowest index (R's stable `sort`);
2. recursive radius, iterated with memoisation instead of R recursion;
3. sphere membership with ECoL's strict `d < r` (`translate`);
4. ECoL's `adherence` greedy: largest count of still-uncovered points, lowest
   index on ties, and the chosen centre leaves the pool even when it lies
   outside its own (zero-radius) sphere.

ECoL 0.4.4 reports N5 = #spheres / n. ECoL 0.3.0 — whose names `Cpx.py` uses
— reports T1 = mean(sphere sizes) / n, where the last sphere is recorded as
size 1. Distances are Manhattan on range-normalised features; ECoL's Gower is
that divided by the number of features, which no comparison here depends on.
"""

from __future__ import annotations

import heapq

import numpy as np
from sklearn.neighbors import KDTree

from pos.complexity.neighborhood_extra import _normalize

VERSIONS = ("0.3.0", "0.4.4")


def _nearest_enemy(Xn: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ne, de = np.empty(len(y), dtype=int), np.empty(len(y))
    for c in np.unique(y):
        rows, enemies = np.flatnonzero(y == c), np.flatnonzero(y != c)
        tree = KDTree(Xn[enemies], metric="manhattan")
        d = tree.query(Xn[rows], k=1)[0][:, 0]
        ties = tree.query_radius(Xn[rows], r=d)
        ne[rows] = [enemies[t].min() for t in ties]
        de[rows] = d
    return ne, de


def _radii(ne: np.ndarray, de: np.ndarray) -> np.ndarray:
    n = len(ne)
    r = np.full(n, np.nan)
    for i in range(n):
        chain, j = [], i
        while np.isnan(r[j]):
            if ne[ne[j]] == j or len(chain) > n:  # mutual pair, or tie cycle
                r[j] = de[j] / 2
                break
            chain.append(j)
            j = ne[j]
        for m in reversed(chain):
            r[m] = de[m] - r[ne[m]]
    return r


def _spheres(Xn: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
    # r <= d(nearest enemy), so with a strict `<` a sphere never holds an
    # enemy and the point's own class tree is enough.
    members = np.empty(len(y), dtype=object)
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        tree = KDTree(Xn[rows], metric="manhattan")
        strict = np.nextafter(r[rows], 0.0)
        hits = tree.query_radius(Xn[rows], r=np.maximum(strict, 0.0))
        for pos, i in enumerate(rows):
            members[i] = rows[hits[pos]] if r[i] > 0 else np.empty(0, dtype=int)
    return members


def _adherence(members: np.ndarray) -> list[int]:
    """ECoL's `adherence` loop as a lazy (CELF) greedy; returns ECoL's `h`."""
    n = len(members)
    alive = np.ones(n, dtype=bool)
    heap = [(-len(members[i]), i) for i in range(n)]
    heapq.heapify(heap)
    n_alive, h = n, []
    while n_alive > 0:
        while True:
            _, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            gain = int(np.count_nonzero(alive[members[i]]))
            if not heap or (-gain, i) <= heap[0]:
                break
            heapq.heappush(heap, (-gain, i))
        tmp = members[i][alive[members[i]]]
        alive[tmp] = False
        n_alive -= len(tmp)
        if alive[i]:
            alive[i] = False
            n_alive -= 1
        h.append(len(tmp) if n_alive > 0 else 1)
    return h


def t1_ecol(X: np.ndarray, y: np.ndarray, version: str = "0.4.4") -> float:
    """ECoL's T1 (`version="0.3.0"`) or N5 (`"0.4.4"`) on raw features X."""
    if version not in VERSIONS:
        raise ValueError(f"unknown ECoL version {version!r}; expected {VERSIONS}")
    y = np.asarray(y)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return 0.0
    Xn = _normalize(np.asarray(X, dtype=float))
    ne, de = _nearest_enemy(Xn, y)
    h = _adherence(_spheres(Xn, y, _radii(ne, de)))
    n = len(y)
    return len(h) / n if version == "0.4.4" else float(np.mean(np.asarray(h) / n))
