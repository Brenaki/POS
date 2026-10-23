"""Tests for the exact port of ECoL's T1/N5 (ADR 0023).

`_t1_dense` transcribes ECoL's R code line by line (dense Gower matrix,
recursive `radios`, strict `<` in `translate`, `adherence` loop with its
last-sphere-counts-as-1 quirk); `t1_ecol` must reproduce it exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from pos.complexity.t1_ecol import t1_ecol


def _t1_dense(X, y):
    X = X.astype(float)
    rng = np.ptp(X, axis=0)
    rng[rng == 0] = 1.0
    D = cdist(X / rng, X / rng, "cityblock") / X.shape[1]
    n = len(y)
    De = np.where(y[None, :] != y[:, None], D, np.inf)
    ne = De.argmin(1)
    de = De[np.arange(n), ne]
    r = np.full(n, np.nan)

    def rad(i):
        if np.isnan(r[i]):
            j = ne[i]
            r[i] = de[i] / 2 if ne[j] == i else de[i] - rad(j)
        return r[i]

    for i in range(n):
        rad(i)
    adh = D < r[:, None]
    alive = np.ones(n, bool)
    h = []
    while alive.any():
        sums = adh[:, alive].sum(1)
        sums[~alive] = -1
        a = int(np.argmax(sums))
        tmp = adh[a] & alive
        alive &= ~tmp
        alive[a] = False
        h.append(int(tmp.sum()) if alive.any() else 1)
    return {"0.4.4": len(h) / n, "0.3.0": float(np.mean(np.array(h) / n))}


def _data(kind, seed, n=150, d=4):
    rng = np.random.default_rng(seed)
    if kind == "separated":
        y = rng.integers(0, 3, n)
        return rng.normal(size=(n, d)) + y[:, None] * 6.0, y
    if kind == "overlapping":
        y = rng.integers(0, 2, n)
        return rng.normal(size=(n, d)), y
    y = rng.integers(0, 3, n)  # discrete grid: ties in distances and radii
    return rng.integers(0, 4, size=(n, d)).astype(float), y


@pytest.mark.parametrize("version", ["0.3.0", "0.4.4"])
@pytest.mark.parametrize("kind", ["separated", "overlapping", "ties"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_the_dense_transcription_of_ecol(version, kind, seed):
    X, y = _data(kind, seed)
    assert t1_ecol(X, y, version) == pytest.approx(_t1_dense(X, y)[version], abs=1e-12)


def test_recursive_radius_differs_from_half_the_enemy_distance():
    # 0.0's nearest enemy is 1.0, whose own nearest enemy is 0.6 (mutual, so
    # radius 0.2). ECoL gives 0.0 the radius 1.0 - 0.2 = 0.8, which covers
    # 0.6; the fast backend's d/2 = 0.5 does not, and needs a third sphere.
    from pos.complexity.neighborhood_extra import _normalize, _t1_fast
    X = np.array([[0.0], [0.6], [1.0], [1.1]])
    y = np.array([0, 0, 1, 1])
    assert t1_ecol(X, y, "0.4.4") == pytest.approx(2 / 4)
    assert _t1_fast(_normalize(X), y) == pytest.approx(3 / 4)


def test_degenerate_inputs_return_zero():
    X = np.random.default_rng(0).normal(size=(6, 2))
    assert t1_ecol(X, np.zeros(6, int)) == 0.0
    assert t1_ecol(X[:1], np.array([0])) == 0.0


def test_unknown_version_is_rejected():
    X, y = _data("separated", 0, n=10)
    with pytest.raises(ValueError, match="version"):
        t1_ecol(X, y, "1.0")


@pytest.mark.requires_r
@pytest.mark.parametrize("version", ["0.3.0", "0.4.4"])
def test_matches_r_ecol(version):
    from pos.complexity.ecol_bridge import ecol_measures
    bags = [_data(k, s, n=120) for k in ("overlapping", "ties") for s in (0, 1)]
    r = ecol_measures(bags, version, groups=("neighborhood",))
    r = r[r["measure"] == "T1"].set_index("bag")["value"]
    for b, (X, y) in enumerate(bags):
        assert t1_ecol(X, y, version) == pytest.approx(r[b], abs=1e-12)
