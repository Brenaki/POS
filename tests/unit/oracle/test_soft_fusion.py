"""Soft-combination fallback for margin classifiers (ADR 0016).

The GA builds Perceptron pools (thesis sec. 5), which have no predict_proba,
so objective 6 of the subproject ("média das probabilidades preditas") needs a
defined fallback instead of a hole in summary.csv.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

from pos.oracle.comparison import (
    mean_decision_accuracy,
    mean_probs_accuracy,
    soft_fusion_accuracy,
)


@pytest.fixture(scope="module")
def wine():
    return load_wine(return_X_y=True)


def _perceptron_pool(X, y, n=8):
    pool = []
    for i in range(n):
        idx = np.random.RandomState(i).choice(len(y), int(0.7 * len(y)), replace=True)
        if len(np.unique(y[idx])) < len(np.unique(y)):
            continue
        pool.append(Perceptron(max_iter=100, tol=1.0, random_state=i).fit(X[idx], y[idx]))
    return pool


def test_tree_pool_uses_predict_proba(wine):
    X, y = wine
    pool = [DecisionTreeClassifier(random_state=i).fit(X, y) for i in range(5)]
    acc, rule = soft_fusion_accuracy(pool, X, y)
    assert rule == "mean_probs"
    assert acc == mean_probs_accuracy(pool, X, y)


def test_perceptron_pool_falls_back_to_decision_function(wine):
    X, y = wine
    pool = _perceptron_pool(X, y)
    acc, rule = soft_fusion_accuracy(pool, X, y)
    assert rule == "mean_decision_norm"
    assert acc is not None and 0.0 <= acc <= 1.0


def test_pool_with_neither_returns_none():
    class Dummy:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    acc, rule = soft_fusion_accuracy([Dummy()], np.zeros((4, 2)), np.zeros(4, dtype=int))
    assert acc is None
    assert rule == "none"


def test_normalisation_makes_the_fusion_scale_invariant(wine):
    """A classifier with 10x larger weights must not get 10x the vote.

    This is the whole reason the margin is divided by ||w||: raw
    decision_function values are not comparable across Perceptrons.
    """
    X, y = wine
    pool = _perceptron_pool(X, y)
    before = mean_decision_accuracy(pool, X, y)
    pool[0].coef_ = pool[0].coef_ * 10.0
    pool[0].intercept_ = pool[0].intercept_ * 10.0
    assert mean_decision_accuracy(pool, X, y) == before


def test_binary_pool_orientation_matches_classes(wine):
    """decision_function > 0 must map to classes_[1], not classes_[0]."""
    X, y = wine
    yb = (y > 0).astype(int)
    pool = [Perceptron(max_iter=200, tol=1.0, random_state=i).fit(X, yb) for i in range(5)]
    acc = mean_decision_accuracy(pool, X, yb)
    # a flipped orientation would land near 1 - acc, well under chance here
    assert acc > 0.5


def test_disagreeing_classes_raise(wine):
    X, y = wine
    pool = [Perceptron(max_iter=100, tol=1.0, random_state=0).fit(X, y),
            Perceptron(max_iter=100, tol=1.0, random_state=1).fit(X, (y > 0).astype(int))]
    with pytest.raises(ValueError, match="classes_"):
        mean_decision_accuracy(pool, X, y)
