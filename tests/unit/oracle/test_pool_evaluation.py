"""Tests for pos.oracle.pool_evaluation.

Uses a deterministic mock pool of 3 DecisionTreeClassifiers fitted on
load_wine to validate structure, monotonicity, and invariants.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from pos.oracle.pool_evaluation import evaluate_pool


@pytest.fixture(scope="module")
def wine_pool_and_test():
    """3-tree pool fitted on wine train split; returns (pool, X_test, y_test)."""
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    pool = []
    for rs in (42, 43, 44):
        clf = DecisionTreeClassifier(random_state=rs, max_depth=3)
        clf.fit(X_train, y_train)
        pool.append(clf)
    return pool, X_test, y_test


def test_evaluate_pool_returns_dict_with_expected_keys(wine_pool_and_test):
    pool, X_test, y_test = wine_pool_and_test
    result = evaluate_pool(pool, X_test, y_test)
    expected = {
        "individual_accuracies", "correctness_matrix", "oracle_curve",
        "majority_vote", "mean_probs", "double_fault_matrix",
        "double_fault_mean", "n_classifiers",
    }
    assert expected.issubset(result.keys())


def test_n_classifiers_matches_pool_size(wine_pool_and_test):
    pool, X_test, y_test = wine_pool_and_test
    result = evaluate_pool(pool, X_test, y_test)
    assert result["n_classifiers"] == len(pool) == 3


def test_individual_accuracies_length_matches_M(wine_pool_and_test):
    pool, X_test, y_test = wine_pool_and_test
    result = evaluate_pool(pool, X_test, y_test)
    assert len(result["individual_accuracies"]) == 3
    for acc in result["individual_accuracies"]:
        assert 0.0 <= acc <= 1.0


def test_correctness_matrix_shape(wine_pool_and_test):
    pool, X_test, y_test = wine_pool_and_test
    result = evaluate_pool(pool, X_test, y_test)
    matrix = result["correctness_matrix"]
    assert matrix.shape == (len(y_test), 3)
    assert set(np.unique(matrix)).issubset({0, 1})


def test_oracle_curve_is_monotonic_non_increasing(wine_pool_and_test):
    pool, X_test, y_test = wine_pool_and_test
    result = evaluate_pool(pool, X_test, y_test)
    curve = result["oracle_curve"]
    for i in range(len(curve) - 1):
        assert curve[i] >= curve[i + 1], (
            f"Monotonicidade violada: Oracle_{i+1}={curve[i]} < Oracle_{i+2}={curve[i+1]}"
        )


def test_oracle_1_geq_majority_vote(wine_pool_and_test):
    pool, X_test, y_test = wine_pool_and_test
    result = evaluate_pool(pool, X_test, y_test)
    oracle_1 = result["oracle_curve"][0]
    majority = result["majority_vote"]
    assert oracle_1 >= majority, (
        f"Oracle_1 ({oracle_1}) should be >= majority vote ({majority})"
    )


def test_double_fault_matrix_shape_and_symmetric(wine_pool_and_test):
    pool, X_test, y_test = wine_pool_and_test
    result = evaluate_pool(pool, X_test, y_test)
    dfm = result["double_fault_matrix"]
    assert dfm.shape == (3, 3)
    assert np.allclose(dfm, dfm.T)
    assert np.all(np.diag(dfm) == 0.0)


def test_double_fault_mean_in_valid_range(wine_pool_and_test):
    pool, X_test, y_test = wine_pool_and_test
    result = evaluate_pool(pool, X_test, y_test)
    dfm = result["double_fault_mean"]
    assert 0.0 <= dfm <= 1.0


def test_mean_probs_is_float(wine_pool_and_test):
    pool, X_test, y_test = wine_pool_and_test
    result = evaluate_pool(pool, X_test, y_test)
    assert isinstance(result["mean_probs"], float)
    assert 0.0 <= result["mean_probs"] <= 1.0


def test_evaluate_pool_with_no_proba_clf_returns_none_mean_probs():
    """Pool whose classifiers lack predict_proba should yield mean_probs=None."""
    class _NoProbaClf:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    X = np.random.rand(10, 2)
    y = np.array([0, 1] * 5)
    result = evaluate_pool([_NoProbaClf(), _NoProbaClf()], X, y)
    assert result["mean_probs"] is None
