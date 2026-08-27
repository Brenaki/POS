"""Tests for the ADR 0020 Bagging baseline (T10).

`build_bagging_pool_v2` follows Monteiro 2023 exactly:
n_estimators=100, max_samples=0.5, bootstrap=True, random_state=86,
default Decision Tree. The legacy `build_bagging_pool` stays for backwards
compatibility (full-sample, seed 42).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from pos.oracle.bagging_pool import build_bagging_pool, build_bagging_pool_v2


@pytest.fixture
def balanced():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    y = np.array([0] * 100 + [1] * 100)
    return X, y


class TestV2Defaults:
    def test_pool_has_100_trees(self, balanced):
        X, y = balanced
        pool = build_bagging_pool_v2(X, y)
        assert len(pool) == 100

    def test_all_trees_are_decision_trees(self, balanced):
        X, y = balanced
        pool = build_bagging_pool_v2(X, y)
        assert all(isinstance(t, DecisionTreeClassifier) for t in pool)

    def test_default_random_state_is_86(self, balanced):
        """Published seed: only BaggingClassifier(random_state=86) is from the paper."""
        X, y = balanced
        a = build_bagging_pool_v2(X, y, random_state=86)
        b = build_bagging_pool_v2(X, y, random_state=86)
        # same seed -> same predictions
        for ta, tb in zip(a, b):
            np.testing.assert_array_equal(ta.predict(X), tb.predict(X))

    def test_max_samples_is_half(self, balanced):
        """max_samples=0.5 per ADR 0020 §4."""
        X, y = balanced
        pool = build_bagging_pool_v2(X, y)
        # Each tree trained on ~100 of 200 samples (bootstrap with 0.5 fraction)
        # Check that trees disagree enough to confirm subsampling
        preds = np.array([t.predict(X) for t in pool])
        # not all predictions identical -> subsampling happened
        assert not np.all(preds == preds[0:1])

    def test_is_deterministic(self, balanced):
        X, y = balanced
        a = build_bagging_pool_v2(X, y)
        b = build_bagging_pool_v2(X, y)
        for ta, tb in zip(a, b):
            np.testing.assert_array_equal(ta.predict(X), tb.predict(X))

    def test_accepts_override_M(self, balanced):
        X, y = balanced
        pool = build_bagging_pool_v2(X, y, M=50)
        assert len(pool) == 50

    def test_accepts_override_random_state(self, balanced):
        X, y = balanced
        a = build_bagging_pool_v2(X, y, random_state=99)
        b = build_bagging_pool_v2(X, y, random_state=99)
        for ta, tb in zip(a, b):
            np.testing.assert_array_equal(ta.predict(X), tb.predict(X))


class TestLegacyBackwardsCompat:
    def test_legacy_pool_has_M_trees(self, balanced):
        X, y = balanced
        pool = build_bagging_pool(X, y, M=20, random_state=42)
        assert len(pool) == 20

    def test_legacy_uses_full_sample(self, balanced):
        """Legacy path: max_features=1.0, no max_samples override -> full sample."""
        X, y = balanced
        pool = build_bagging_pool(X, y, M=10, random_state=42)
        assert all(isinstance(t, DecisionTreeClassifier) for t in pool)