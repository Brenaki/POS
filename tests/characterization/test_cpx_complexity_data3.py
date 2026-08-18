"""Characterization tests for complexity_data3 (pyhard backend, Fase 3).

As of ADR 0003, complexity_data3 uses pyhard instead of R/ECoL. These tests
assert PROPERTIES (shape, finiteness, range) rather than exact ECoL values,
because pyhard measures differ from ECoL. The legacy ECoL golden values are
captured separately in test_cpx_complexity_data3_ecol.py (requires R).
"""

from __future__ import annotations

import numpy as np
import pytest

# conftest.py rpy2 mock is irrelevant now (pyhard doesn't use R), but harmless.
import Cpx  # noqa: E402


class TestComplexityData3Pyhard:
    def test_overlapping_group_returns_5_values(self, wine_split):
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["overlapping"])
        assert len(result) == 5
        assert all(isinstance(v, float) for v in result)

    def test_neighborhood_group_returns_6_values(self, wine_split):
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["neighborhood"])
        assert len(result) == 6

    def test_linearity_group_returns_3_zeros(self, wine_split):
        # pyhard has no linearity equivalent → 0.0 placeholders
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["linearity"])
        assert len(result) == 3
        assert result == [0.0, 0.0, 0.0]

    def test_dimensionality_group_returns_3_zeros(self, wine_split):
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["dimensionality"])
        assert len(result) == 3
        assert result == [0.0, 0.0, 0.0]

    def test_balance_group_returns_2_values(self, wine_split):
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["balance"])
        assert len(result) == 2

    def test_network_group_returns_3_zeros(self, wine_split):
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["network"])
        assert len(result) == 3
        assert result == [0.0, 0.0, 0.0]

    def test_combined_overlapping_neighborhood_returns_11(self, wine_split):
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["overlapping", "neighborhood"])
        assert len(result) == 11

    def test_all_six_groups_returns_22(self, wine_split):
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(
            X, y, ["overlapping", "neighborhood", "linearity",
                   "dimensionality", "balance", "network"]
        )
        assert len(result) == 22

    def test_all_values_are_finite(self, wine_split):
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["overlapping", "neighborhood", "balance"])
        assert all(np.isfinite(result))

    def test_pyhard_measures_are_in_reasonable_range(self, wine_split):
        # pyhard measures are aggregated instance-hardness values in [0, 1]
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["overlapping", "neighborhood", "balance"])
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_reproducible_on_same_data(self, wine_split):
        X, y = wine_split["X_train"], wine_split["y_train"]
        r1 = Cpx.complexity_data3(X, y, ["overlapping", "neighborhood"])
        r2 = Cpx.complexity_data3(X, y, ["overlapping", "neighborhood"])
        assert r1 == r2

    def test_different_data_gives_different_values(self, wine_split):
        # A subset of the data should generally produce different complexity
        X, y = wine_split["X_train"], wine_split["y_train"]
        full = Cpx.complexity_data3(X, y, ["overlapping"])
        half = Cpx.complexity_data3(X[:30], y[:30], ["overlapping"])
        # At least one measure should differ (not all identical)
        assert full != half
