"""Tests for the five-mode pool dispatch (ADR 0019)."""

from __future__ import annotations

import numpy as np
import pytest

from pos.oracle.fold_splitter import BAG_MODES, check_dataset_viability
from pos.oracle.pool_modes import MODES, build_pool_for_mode


@pytest.fixture
def split_data():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 4)
    y = np.array([0] * 60 + [1] * 60)
    return X[:90], y[:90], X[90:], y[90:]


def _build(mode, data, M=8):
    X_tr, y_tr, X_val, y_val = data
    return build_pool_for_mode(mode, X_tr, y_tr, X_val, y_val, M, 2, 42)


class TestDispatch:
    @pytest.mark.parametrize("mode", ["bagging", "rf", "randbag"])
    def test_builds_pool_of_size_M(self, mode, split_data):
        pool, extra = _build(mode, split_data)
        assert len(pool) == 8
        assert extra == {}

    def test_unknown_mode_is_not_an_error(self, split_data):
        pool, extra = _build("nope", split_data)
        assert pool is None and extra == {}

    def test_modes_are_the_documented_five(self):
        assert set(MODES) == {"pgdcs", "ga", "randbag", "bagging", "rf"}


class TestViabilityGate:
    """Bag-based modes need >= 2 of the rarest class per bag, not just k-fold."""

    def test_bag_modes_cover_every_generator_that_draws_bags(self):
        assert set(BAG_MODES) == {"ga", "pgdcs", "randbag"}

    @pytest.mark.parametrize("mode", ["ga", "pgdcs", "randbag"])
    def test_bag_mode_rejects_what_tree_modes_accept(self, mode):
        # 5 >= n_folds so k-fold is fine, but 5 -> 4 -> 3 -> 1 per bag.
        y = np.array([0] * 200 + [1] * 5)
        ok_bag, reason = check_dataset_viability(y, 5, [mode])
        ok_tree, _ = check_dataset_viability(y, 5, ["rf"])
        assert ok_tree is True
        assert ok_bag is False and "per bag" in reason
