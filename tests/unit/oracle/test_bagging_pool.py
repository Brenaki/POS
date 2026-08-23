"""Tests for Bagging pool builder (controlled baseline, all features)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from pos.oracle.bagging_pool import build_bagging_pool


@pytest.fixture
def simple_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y = rng.randint(0, 2, size=100)
    return X, y


class TestBaggingPool:
    def test_returns_list_of_trees(self, simple_data):
        X, y = simple_data
        pool = build_bagging_pool(X, y, M=10, random_state=42)
        assert isinstance(pool, list)
        assert len(pool) == 10
        assert all(isinstance(clf, DecisionTreeClassifier) for clf in pool)

    def test_reproducible(self, simple_data):
        X, y = simple_data
        pool1 = build_bagging_pool(X, y, M=10, random_state=42)
        pool2 = build_bagging_pool(X, y, M=10, random_state=42)
        preds1 = np.array([clf.predict(X) for clf in pool1])
        preds2 = np.array([clf.predict(X) for clf in pool2])
        assert np.array_equal(preds1, preds2)

    def test_can_predict(self, simple_data):
        X, y = simple_data
        pool = build_bagging_pool(X, y, M=5, random_state=42)
        for clf in pool:
            preds = clf.predict(X)
            assert len(preds) == len(X)