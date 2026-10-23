"""Exact ports of ECoL 0.3.0 N1, LSC and F4 (ADR 0023)."""

from __future__ import annotations

import numpy as np
import pytest

from pos.complexity.ecol_exact import f4_ecol, lsc_ecol, n1_ecol
from pos.complexity.neighborhood_extra import _normalize
from pos.complexity.neighborhood_fast import _lsc_fast
from pos.complexity.overlapping_measures import _f4


def _data(kind, seed, n=120, d=4):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    if kind == "ties":
        return rng.integers(0, 4, size=(n, d)).astype(float), y
    return rng.normal(size=(n, d)) + y[:, None] * 0.8, y


def test_lsc_divides_local_sets_by_n_squared():
    # 1-D, two classes; nearest-enemy radii: 0 -> 1.0, 0.1 -> 0.9, 1.0 -> 0.9,
    # 1.1 -> 1.0. Each strict local set holds the point and its classmate,
    # so sum |LS| = 8 and ECoL's LSC = 1 - 8/16.
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([0, 0, 1, 1])
    assert lsc_ecol(X, y) == pytest.approx(0.5)
    # the fast backend divides by the class size instead: 1 - mean(2/2) = 0
    assert _lsc_fast(_normalize(X), y) == pytest.approx(0.0)


def test_f4_recomputes_class_ranges_after_each_removal():
    # Feature 1 goes first (3 non-overlapping points vs 2) and removes the
    # f0 outliers 6 and -1. ECoL then recomputes the ranges: f0 becomes
    # [0, 1] vs [3, 5] and separates the rest, so nothing is left. Keeping the
    # full-pair ranges ([0, 6] vs [-1, 5]) separates nothing more: 5/8 remain.
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [6.0, 10.0],
                  [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [-1.0, -10.0]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert f4_ecol(X, y) == pytest.approx(0.0)
    assert _f4(X, y) == pytest.approx(5 / 8)


def test_degenerate_inputs():
    X = np.random.default_rng(0).normal(size=(6, 2))
    assert lsc_ecol(X, np.zeros(6, int)) == 0.0
    assert f4_ecol(X, np.zeros(6, int)) == 0.0


@pytest.mark.requires_r
def test_match_r_ecol_030():
    from pos.complexity.ecol_bridge import ecol_measures
    bags = [_data(k, s) for k in ("normal", "ties") for s in (0, 1, 2)]
    r = ecol_measures(bags, "0.3.0").set_index(["bag", "measure"])["value"]
    for b, (X, y) in enumerate(bags):
        assert lsc_ecol(X, y) == pytest.approx(r[(b, "LSC")], abs=1e-12)
        assert f4_ecol(X, y) == pytest.approx(r[(b, "F4")], abs=1e-12)


def test_n1_counts_points_with_a_cross_class_mst_edge():
    # MST on a line is the chain 0-1-2-3; only the 0.1-1.0 edge crosses
    # classes, so 2 of 4 points are borderline.
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    assert n1_ecol(X, np.array([0, 0, 1, 1])) == pytest.approx(0.5)


@pytest.mark.requires_r
def test_n1_matches_r_ecol_030_on_continuous_data():
    from pos.complexity.ecol_bridge import ecol_measures
    bags = [_data("normal", s) for s in (0, 1, 2)]
    r = ecol_measures(bags, "0.3.0", groups=("neighborhood",)).set_index(["bag", "measure"])
    if r.xs("N1", level="measure")["value"].isna().any():
        pytest.skip("R igraph not installed")
    for b, (X, y) in enumerate(bags):
        assert n1_ecol(X, y) == pytest.approx(r.loc[(b, "N1"), "value"], abs=1e-12)
