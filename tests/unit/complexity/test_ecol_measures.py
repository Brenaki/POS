"""TDD tests for P0 #3: ECoL-correct F1, F1v, F2, F3, F4, T1 measures.

Definitions from ECoL R package (lpfgarcia/ECoL):
- F1: Maximum Fisher's Discriminant Ratio → 1/(F+1), scalar per feature, max over features
- F1v: Directional Fisher → LDA projection maximizing class separation
- F2: Volume of Overlap Region → product of per-feature overlap / range
- F3: Max Individual Feature Efficiency → 1 - max(fraction non-overlapping)
- F4: Collective Feature Efficiency → fraction removed by iterative best-F3 elimination
- T1: Not in ECoL neighborhood! ECoL uses N1-N6. T1 was from the original
  complexity library (Orriols-Puig). It measures the fraction of instances
  that have a nearest neighbor of a different class — same as N3 (1-NN error).
  PGDCS uses 'T1' as a neighborhood measure alias for N3 (error rate of 1-NN).

The ECoL R code for F1:
  numerator: n_c * (mean_c - mean_all)^2  per class, summed
  denominator: sum of (x - mean_c)^2 per class
  F1 = 1 / (numerator/denominator + 1)  -- per feature, then max

The ECoL R code for F1v (one-vs-one):
  d = inv(W) * (c1 - c2)  — LDA direction
  F1v = 1 / ((d'Bd)/(d'Wd) + 1)
"""

from __future__ import annotations

import numpy as np
import pytest

from pos.complexity.overlapping_measures import overlapping_measures
from pos.complexity.neighborhood_fast import neighborhood_measure_fast


@pytest.fixture
def well_separated():
    """Two classes with clear separation — F1 should be high (low complexity)."""
    rng = np.random.RandomState(42)
    X = np.zeros((100, 3))
    X[:50] = rng.randn(50, 3) + np.array([5, 5, 5])
    X[50:] = rng.randn(50, 3) + np.array([-5, -5, -5])
    y = np.array([0]*50 + [1]*50)
    return X, y


@pytest.fixture
def heavy_overlap():
    """Two classes with heavy overlap — F1 should be low (high complexity)."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    y = rng.randint(0, 2, size=100)
    return X, y


class TestF1Fisher:
    """F1 = Maximum Fisher's Discriminant Ratio → 1/(F+1), dataset-level scalar."""

    def test_f1_returns_scalar(self, well_separated):
        v = overlapping_measures(well_separated[0], well_separated[1], "F1")
        assert isinstance(v, float)

    def test_f1_well_separated_lower(self, well_separated, heavy_overlap):
        """Well-separated data → Fisher ratio high → 1/(F+1) LOW.
        Wait: ECoL F1 returns 1/(F+1) where F is the Fisher ratio.
        High Fisher = good separation = LOW F1 value (less complex).
        """
        v_sep = overlapping_measures(well_separated[0], well_separated[1], "F1")
        v_ovl = overlapping_measures(heavy_overlap[0], heavy_overlap[1], "F1")
        assert v_sep < v_ovl, f"Well-separated F1={v_sep} should be < overlap F1={v_ovl}"

    def test_f1_range_0_to_1(self, well_separated, heavy_overlap):
        for X, y in [well_separated, heavy_overlap]:
            v = overlapping_measures(X, y, "F1")
            assert 0.0 <= v <= 1.0, f"F1={v} out of [0,1]"


class TestF1vDirectional:
    """F1v = Directional Fisher via LDA projection."""

    def test_f1v_returns_scalar(self, well_separated):
        v = overlapping_measures(well_separated[0], well_separated[1], "F1v")
        assert isinstance(v, float)

    def test_f1v_well_separated_lower(self, well_separated, heavy_overlap):
        v_sep = overlapping_measures(well_separated[0], well_separated[1], "F1v")
        v_ovl = overlapping_measures(heavy_overlap[0], heavy_overlap[1], "F1v")
        assert v_sep < v_ovl

    def test_f1v_not_zero(self, well_separated):
        """F1v must NOT return 0.0 (was placeholder before)."""
        v = overlapping_measures(well_separated[0], well_separated[1], "F1v")
        assert v != 0.0, "F1v returned 0.0 — placeholder not replaced"

    def test_f1v_range_0_to_1(self, well_separated, heavy_overlap):
        for X, y in [well_separated, heavy_overlap]:
            v = overlapping_measures(X, y, "F1v")
            assert 0.0 <= v <= 1.0


class TestF2VolumeOverlap:
    """F2 = Volume of Overlap Region → product of per-feature overlap/range."""

    def test_f2_returns_scalar(self, well_separated):
        v = overlapping_measures(well_separated[0], well_separated[1], "F2")
        assert isinstance(v, float)

    def test_f2_well_separated_lower(self, well_separated, heavy_overlap):
        v_sep = overlapping_measures(well_separated[0], well_separated[1], "F2")
        v_ovl = overlapping_measures(heavy_overlap[0], heavy_overlap[1], "F2")
        assert v_sep < v_ovl

    def test_f2_range_0_to_1(self, well_separated, heavy_overlap):
        for X, y in [well_separated, heavy_overlap]:
            v = overlapping_measures(X, y, "F2")
            assert 0.0 <= v <= 1.0


class TestF3FeatureEfficiency:
    """F3 = Max Individual Feature Efficiency → 1 - max(non-overlap fraction)."""

    def test_f3_returns_scalar(self, well_separated):
        v = overlapping_measures(well_separated[0], well_separated[1], "F3")
        assert isinstance(v, float)

    def test_f3_well_separated_lower(self, well_separated, heavy_overlap):
        v_sep = overlapping_measures(well_separated[0], well_separated[1], "F3")
        v_ovl = overlapping_measures(heavy_overlap[0], heavy_overlap[1], "F3")
        assert v_sep < v_ovl

    def test_f3_range_0_to_1(self, well_separated, heavy_overlap):
        for X, y in [well_separated, heavy_overlap]:
            v = overlapping_measures(X, y, "F3")
            assert 0.0 <= v <= 1.0


class TestF4CollectiveEfficiency:
    """F4 = Collective Feature Efficiency → fraction remaining after iterative removal."""

    def test_f4_returns_scalar(self, well_separated):
        v = overlapping_measures(well_separated[0], well_separated[1], "F4")
        assert isinstance(v, float)

    def test_f4_well_separated_lower(self, well_separated, heavy_overlap):
        v_sep = overlapping_measures(well_separated[0], well_separated[1], "F4")
        v_ovl = overlapping_measures(heavy_overlap[0], heavy_overlap[1], "F4")
        assert v_sep < v_ovl

    def test_f4_range_0_to_1(self, well_separated, heavy_overlap):
        for X, y in [well_separated, heavy_overlap]:
            v = overlapping_measures(X, y, "F4")
            assert 0.0 <= v <= 1.0


class TestT1Measure:
    """T1 = Fraction of Hyper-spheres Covering Data (ECoL N5).

    PGDCS uses 'T1' which is Fraction of Hyper-spheres Covering Data.
    This corresponds to ECoL N5, NOT N3 (1-NN error rate).
    T1 measures how many hyperspheres are needed to cover all data points.
    Lower T1 = fewer spheres needed = simpler problem.
    """

    def test_t1_returns_scalar(self, well_separated):
        v = neighborhood_measure_fast(well_separated[0], well_separated[1], "T1")
        assert isinstance(v, float)

    def test_t1_well_separated_lower(self, well_separated, heavy_overlap):
        """Well-separated data needs fewer hyperspheres → lower T1."""
        v_sep = neighborhood_measure_fast(well_separated[0], well_separated[1], "T1")
        v_ovl = neighborhood_measure_fast(heavy_overlap[0], heavy_overlap[1], "T1")
        assert v_sep < v_ovl

    def test_t1_not_zero(self, heavy_overlap):
        """T1 must NOT return 0.0 (was placeholder before)."""
        v = neighborhood_measure_fast(heavy_overlap[0], heavy_overlap[1], "T1")
        assert v != 0.0, "T1 returned 0.0 — placeholder not replaced"

    def test_t1_range_0_to_1(self, well_separated, heavy_overlap):
        for X, y in [well_separated, heavy_overlap]:
            v = neighborhood_measure_fast(X, y, "T1")
            assert 0.0 < v <= 1.0

    def test_t1_is_not_n3(self, heavy_overlap):
        """T1 should differ from N3 (1-NN LOO error) — they are different measures."""
        t1 = neighborhood_measure_fast(heavy_overlap[0], heavy_overlap[1], "T1")
        n3 = neighborhood_measure_fast(heavy_overlap[0], heavy_overlap[1], "N3")
        assert t1 != n3, f"T1={t1} equals N3={n3} — T1 should be Hyper-spheres, not 1-NN error"