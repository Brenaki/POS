"""E2E reproducibility test: seed=42 → same pool → same correctness → same Oracle_N.

Tests the full GA pipeline: generate_bags → GA → pool → evaluate_pool.
Two runs with random_state=42 must produce identical correctness matrices
and Oracle_N values.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from pos.oracle.run_helpers import build_pool_ga
from pos.oracle.pool_evaluation import evaluate_pool


@pytest.fixture
def wine_split():
    X, y = load_wine(return_X_y=True, as_frame=False)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


class TestGAE2EReproducibility:
    """Same random_state → identical pool → identical Oracle_N."""

    @pytest.mark.slow
    def test_same_seed_same_pool(self, wine_split):
        """Two GA runs with seed=42 produce identical predictions."""
        X_tr, y_tr, X_val, y_val = wine_split
        pool1 = build_pool_ga(X_tr, y_tr, X_val, y_val, nr_generation=3,
                              random_state=42, jobs=1)
        pool2 = build_pool_ga(X_tr, y_tr, X_val, y_val, nr_generation=3,
                              random_state=42, jobs=1)
        preds1 = np.array([clf.predict(X_val) for clf in pool1])
        preds2 = np.array([clf.predict(X_val) for clf in pool2])
        assert np.array_equal(preds1, preds2), \
            "Same seed produced different pool predictions — GA not reproducible"

    @pytest.mark.slow
    def test_same_seed_same_oracle_n(self, wine_split):
        """Two GA runs with seed=42 produce identical Oracle_N values."""
        X_tr, y_tr, X_val, y_val = wine_split
        pool1 = build_pool_ga(X_tr, y_tr, X_val, y_val, nr_generation=3,
                              random_state=42, jobs=1)
        pool2 = build_pool_ga(X_tr, y_tr, X_val, y_val, nr_generation=3,
                              random_state=42, jobs=1)
        m1 = evaluate_pool(pool1, X_val, y_val)
        m2 = evaluate_pool(pool2, X_val, y_val)
        assert m1["oracle_curve"] == m2["oracle_curve"], \
            "Same seed produced different Oracle_N curves — not reproducible"
        assert m1["correctness_matrix"].tolist() == m2["correctness_matrix"].tolist()

    @pytest.mark.slow
    def test_different_seed_different_pool(self, wine_split):
        """Two GA runs with different seeds produce different predictions."""
        X_tr, y_tr, X_val, y_val = wine_split
        pool1 = build_pool_ga(X_tr, y_tr, X_val, y_val, nr_generation=3,
                              random_state=42, jobs=1)
        pool2 = build_pool_ga(X_tr, y_tr, X_val, y_val, nr_generation=3,
                              random_state=99, jobs=1)
        preds1 = np.array([clf.predict(X_val) for clf in pool1])
        preds2 = np.array([clf.predict(X_val) for clf in pool2])
        assert not np.array_equal(preds1, preds2), \
            "Different seeds produced same pool — GA not seed-dependent"