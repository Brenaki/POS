"""Tests for the random-bag pool: the generation control of ADR 0019."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Perceptron

from pos.oracle.randbag_pool import build_randbag_pool
from pos.pool.bag_generator import generate_bags


@pytest.fixture
def split_data():
    rng = np.random.RandomState(42)
    X = rng.randn(120, 4)
    y = np.array([0] * 60 + [1] * 60)
    return X[:90], y[:90], X[90:], y[90:]


class TestRandbagPool:
    def test_builds_M_perceptrons(self, split_data):
        X_tr, y_tr, X_val, y_val = split_data
        pool = build_randbag_pool(X_tr, y_tr, X_val, y_val, M=12, random_state=42)
        assert len(pool) == 12
        assert all(isinstance(clf, Perceptron) for clf in pool)

    def test_reproducible(self, split_data):
        X_tr, y_tr, X_val, y_val = split_data
        a = build_randbag_pool(X_tr, y_tr, X_val, y_val, M=8, random_state=42)
        b = build_randbag_pool(X_tr, y_tr, X_val, y_val, M=8, random_state=42)
        pa = np.array([c.predict(X_val) for c in a])
        pb = np.array([c.predict(X_val) for c in b])
        assert np.array_equal(pa, pb)

    def test_different_seed_gives_different_pool(self, split_data):
        X_tr, y_tr, X_val, y_val = split_data
        a = build_randbag_pool(X_tr, y_tr, X_val, y_val, M=8, random_state=42)
        b = build_randbag_pool(X_tr, y_tr, X_val, y_val, M=8, random_state=7)
        pa = np.array([c.predict(X_val) for c in a])
        pb = np.array([c.predict(X_val) for c in b])
        assert not np.array_equal(pa, pb)

    def test_bags_are_half_the_training_split(self, split_data):
        """It is the GA's generation-0 population, so the bags must match."""
        X_tr, y_tr, _, _ = split_data
        bags = generate_bags(X_tr, y_tr, 8, 0.5, random_state=42)
        assert len(bags["inst"]) == 8
        assert all(len(inst) == len(X_tr) // 2 for inst in bags["inst"])

    def test_validation_split_is_never_trained_on(self, split_data):
        """X_val is only scored; a pool fitted on tr must not memorise val."""
        X_tr, y_tr, X_val, y_val = split_data
        pool = build_randbag_pool(X_tr, y_tr, X_val, y_val, M=6, random_state=0)
        assert all(clf.n_features_in_ == X_tr.shape[1] for clf in pool)
