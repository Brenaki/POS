"""TDD tests for P0 #2: fast_adapter must respect the `types` parameter."""

from __future__ import annotations

import numpy as np
import pytest

from pos.complexity.fast_adapter import complexity_data3


@pytest.fixture
def simple_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y = rng.randint(0, 2, size=100)
    return X, y


class TestFastAdapterTypes:
    """fast_adapter must only compute the measures specified in `types`."""

    def test_types_f1_t1_returns_two_values(self, simple_data):
        """types=['F1','T1'] with group=['overlapping','neighborhood']
        should return exactly 2 values: [F1, T1]."""
        X, y = simple_data
        result = complexity_data3(X, y, group=["overlapping", "neighborhood"],
                                   types=["F1", "T1"])
        assert len(result) == 2, f"Expected 2 values, got {len(result)}"

    def test_types_f1_f2_returns_two_values(self, simple_data):
        X, y = simple_data
        result = complexity_data3(X, y, group=["overlapping", "overlapping"],
                                   types=["F1", "F2"])
        assert len(result) == 2

    def test_types_none_computes_all(self, simple_data):
        """When types=None, compute all measures in the groups (legacy mode)."""
        X, y = simple_data
        result = complexity_data3(X, y, group=["overlapping", "neighborhood"],
                                   types=None)
        # overlapping has 5 measures, neighborhood has 6 = 11 total
        assert len(result) == 11

    def test_types_f1_t1_second_value_is_t1_not_f1v(self, simple_data):
        """Critical: the second value must be T1, not F1v (which is 0.0)."""
        X, y = simple_data
        result = complexity_data3(X, y, group=["overlapping", "neighborhood"],
                                   types=["F1", "T1"])
        assert result[1] != 0.0, "T1 should be non-zero — got 0.0 (F1v placeholder)"

    def test_types_preserves_order(self, simple_data):
        """types order must match group order — types[i] is for group[i]."""
        X, y = simple_data
        r1 = complexity_data3(X, y, group=["overlapping", "neighborhood"],
                               types=["F1", "T1"])
        r2 = complexity_data3(X, y, group=["overlapping", "neighborhood"],
                               types=["F2", "N1"])
        assert r1[0] != r2[0], "F1 and F2 should give different values"