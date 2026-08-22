"""TDD tests: KD-tree neighborhood measures must match brute-force results.

Tests are written FIRST (before implementation). They validate that the
KD-tree-based implementations produce identical results to the current
brute-force (n^2) versions on various dataset sizes and class distributions.
"""

from __future__ import annotations

import numpy as np
import pytest

from pos.complexity.neighborhood_measures import (
    dist_matrix,
    neighborhood_measure,
)
from pos.complexity.neighborhood_fast import neighborhood_measure_fast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 4))
    y = rng.integers(0, 3, size=50)
    return X, y


@pytest.fixture
def medium_dataset():
    rng = np.random.default_rng(99)
    X = rng.standard_normal((500, 8))
    y = rng.integers(0, 4, size=500)
    return X, y


@pytest.fixture
def binary_dataset():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 3))
    y = rng.integers(0, 2, size=200)
    return X, y


# ---------------------------------------------------------------------------
# N2 — exact match (only needs 1-NN same-class and 1-NN different-class)
# ---------------------------------------------------------------------------

class TestN2Exact:
    def test_n2_small(self, small_dataset):
        X, y = small_dataset
        D = dist_matrix(X)
        brute = neighborhood_measure(D, y, "N2")
        fast = neighborhood_measure_fast(X, y, "N2")
        assert fast == pytest.approx(brute, abs=1e-10)

    def test_n2_medium(self, medium_dataset):
        X, y = medium_dataset
        D = dist_matrix(X)
        brute = neighborhood_measure(D, y, "N2")
        fast = neighborhood_measure_fast(X, y, "N2")
        assert fast == pytest.approx(brute, abs=1e-10)

    def test_n2_binary(self, binary_dataset):
        X, y = binary_dataset
        D = dist_matrix(X)
        brute = neighborhood_measure(D, y, "N2")
        fast = neighborhood_measure_fast(X, y, "N2")
        assert fast == pytest.approx(brute, abs=1e-10)


# ---------------------------------------------------------------------------
# kDN — exact match (only needs k-NN)
# ---------------------------------------------------------------------------

class TestKDNExact:
    def test_kdn_small(self, small_dataset):
        X, y = small_dataset
        D = dist_matrix(X)
        brute = neighborhood_measure(D, y, "N4")
        fast = neighborhood_measure_fast(X, y, "N4")
        assert fast == pytest.approx(brute, abs=1e-10)

    def test_kdn_medium(self, medium_dataset):
        X, y = medium_dataset
        D = dist_matrix(X)
        brute = neighborhood_measure(D, y, "N4")
        fast = neighborhood_measure_fast(X, y, "N4")
        assert fast == pytest.approx(brute, abs=1e-10)

    def test_kdn_binary(self, binary_dataset):
        X, y = binary_dataset
        D = dist_matrix(X)
        brute = neighborhood_measure(D, y, "N4")
        fast = neighborhood_measure_fast(X, y, "N4")
        assert fast == pytest.approx(brute, abs=1e-10)


# ---------------------------------------------------------------------------
# LSC — exact match (only needs 1-NN enemy)
# ---------------------------------------------------------------------------

class TestLSCExact:
    def test_lsc_small(self, small_dataset):
        X, y = small_dataset
        D = dist_matrix(X)
        brute = neighborhood_measure(D, y, "LSC")
        fast = neighborhood_measure_fast(X, y, "LSC")
        assert fast == pytest.approx(brute, abs=1e-10)

    def test_lsc_medium(self, medium_dataset):
        X, y = medium_dataset
        D = dist_matrix(X)
        brute = neighborhood_measure(D, y, "LSC")
        fast = neighborhood_measure_fast(X, y, "LSC")
        assert fast == pytest.approx(brute, abs=1e-10)

    def test_lsc_binary(self, binary_dataset):
        X, y = binary_dataset
        D = dist_matrix(X)
        brute = neighborhood_measure(D, y, "LSC")
        fast = neighborhood_measure_fast(X, y, "LSC")
        assert fast == pytest.approx(brute, abs=1e-10)


# ---------------------------------------------------------------------------
# N1 — approximate match (kNN-graph MST vs full MST)
# ---------------------------------------------------------------------------

class TestN1Approximate:
    def test_n1_small(self, small_dataset):
        X, y = small_dataset
        D = dist_matrix(X)
        brute = neighborhood_measure(D, y, "N1")
        fast = neighborhood_measure_fast(X, y, "N1")
        # N1 uses kNN-graph MST approximation — allow small deviation
        assert fast == pytest.approx(brute, abs=0.05)

    def test_n1_medium(self, medium_dataset):
        X, y = medium_dataset
        D = dist_matrix(X)
        brute = neighborhood_measure(D, y, "N1")
        fast = neighborhood_measure_fast(X, y, "N1")
        assert fast == pytest.approx(brute, abs=0.05)

    def test_n1_binary(self, binary_dataset):
        X, y = binary_dataset
        D = dist_matrix(X)
        brute = neighborhood_measure(D, y, "N1")
        fast = neighborhood_measure_fast(X, y, "N1")
        assert fast == pytest.approx(brute, abs=0.05)


# ---------------------------------------------------------------------------
# All four measures via neighborhood_measure_fast
# ---------------------------------------------------------------------------

class TestAllMeasuresFast:
    @pytest.mark.parametrize("measure", ["N1", "N2", "N4", "LSC"])
    def test_fast_returns_float(self, small_dataset, measure):
        X, y = small_dataset
        result = neighborhood_measure_fast(X, y, measure)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("measure", ["N2", "N4", "LSC"])
    def test_exact_measures_match(self, medium_dataset, measure):
        X, y = medium_dataset
        D = dist_matrix(X)
        brute = neighborhood_measure(D, y, measure)
        fast = neighborhood_measure_fast(X, y, measure)
        assert fast == pytest.approx(brute, abs=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_class(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 3))
        y = np.ones(30, dtype=int)
        for m in ["N2", "N4", "LSC"]:
            result = neighborhood_measure_fast(X, y, m)
            # Single class → no enemies → degenerate but should not crash
            assert isinstance(result, float)

    def test_kdn_k_exceeds_n(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 1, 0])
        result = neighborhood_measure_fast(X, y, "N4")
        assert isinstance(result, float)

    def test_two_samples(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1])
        for m in ["N2", "LSC"]:
            result = neighborhood_measure_fast(X, y, m)
            assert isinstance(result, float)