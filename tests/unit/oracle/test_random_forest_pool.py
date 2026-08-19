"""Tests for pos.oracle.random_forest_pool (baseline without GA)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from pos.oracle.random_forest_pool import build_rf_pool


@pytest.fixture(scope="module")
def wine_split():
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def test_build_rf_pool_returns_list_of_estimators(wine_split):
    X_train, _, y_train, _ = wine_split
    pool = build_rf_pool(X_train, y_train, M=10, random_state=42)
    assert isinstance(pool, list)
    assert len(pool) == 10
    for clf in pool:
        assert isinstance(clf, DecisionTreeClassifier)


def test_build_rf_pool_each_clf_fitted(wine_split):
    """Each estimator must be fitted (have classes_ and tree_)."""
    X_train, X_test, y_train, y_test = wine_split
    pool = build_rf_pool(X_train, y_train, M=5, random_state=42)
    for clf in pool:
        assert hasattr(clf, "classes_")
        assert hasattr(clf, "tree_")
        assert clf.classes_ is not None


def test_build_rf_pool_predictions_match_forest(wine_split):
    """Sum of pool predictions via majority should match RF.predict."""
    from scipy import stats
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, _ = wine_split
    M = 7
    forest = RandomForestClassifier(
        n_estimators=M, random_state=42, bootstrap=True, n_jobs=1,
    ).fit(X_train, y_train)
    pool = list(forest.estimators_)

    preds = np.array([clf.predict(X_test) for clf in pool])
    pool_majority = stats.mode(preds, axis=0, keepdims=False).mode
    forest_pred = forest.predict(X_test)
    # Majority of the sub-trees should match the forest's own prediction
    # (RF uses majority vote over estimators by default).
    assert np.array_equal(pool_majority, forest_pred)


def test_build_rf_pool_reproducible(wine_split):
    """Same random_state must produce identical predictions."""
    X_train, X_test, y_train, _ = wine_split
    p1 = build_rf_pool(X_train, y_train, M=5, random_state=42)
    p2 = build_rf_pool(X_train, y_train, M=5, random_state=42)
    preds1 = np.array([c.predict(X_test) for c in p1])
    preds2 = np.array([c.predict(X_test) for c in p2])
    assert np.array_equal(preds1, preds2)


def test_build_rf_pool_different_seeds_differ(wine_split):
    """Different random_state should (almost surely) produce different trees."""
    X_train, X_test, y_train, _ = wine_split
    p1 = build_rf_pool(X_train, y_train, M=5, random_state=42)
    p2 = build_rf_pool(X_train, y_train, M=5, random_state=999)
    preds1 = np.array([c.predict(X_test) for c in p1])
    preds2 = np.array([c.predict(X_test) for c in p2])
    assert not np.array_equal(preds1, preds2)


def test_build_rf_pool_predict_proba_available(wine_split):
    """Each estimator in the RF pool must expose predict_proba (for mean_probs)."""
    X_train, X_test, y_train, _ = wine_split
    pool = build_rf_pool(X_train, y_train, M=3, random_state=42)
    for clf in pool:
        probs = clf.predict_proba(X_test)
        assert probs.shape == (len(X_test), len(clf.classes_))
