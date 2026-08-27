"""Tests for the v2 treatment evaluation (ADR 0020 §5 and the mean-probability
combination restored now that both treatments are Decision Tree pools)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from pos.oracle.oracle_curve_greedy import atleast_curve, greedy_oracle_curve
from pos.oracle.profiles import ORACLE_CHECKPOINTS
from pos.oracle.treatment_eval import evaluate_treatment


@pytest.fixture
def pool_and_data():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(60, 3), rng.randn(60, 3) + 2.0])
    y = np.array([0] * 60 + [1] * 60)
    pool = []
    for i in range(8):
        idx = rng.choice(len(X), 40, replace=True)
        pool.append(DecisionTreeClassifier(max_depth=2, random_state=i).fit(X[idx], y[idx]))
    return pool, X, y


class TestCheckpoints:
    def test_twenty_one_checkpoints(self):
        assert len(ORACLE_CHECKPOINTS) == 21

    def test_first_is_one_then_every_five(self):
        assert ORACLE_CHECKPOINTS[0] == 1
        assert ORACLE_CHECKPOINTS[1] == 5
        assert ORACLE_CHECKPOINTS[-1] == 100


class TestBothOracleReadings:
    def test_greedy_is_non_decreasing(self, pool_and_data):
        pool, X, y = pool_and_data
        m = evaluate_treatment(pool, X, y, X, y, [], 0)
        curve = m["oracle_greedy_curve"]
        assert all(curve[i] >= curve[i - 1] for i in range(1, len(curve)))

    def test_atleast_is_non_increasing(self, pool_and_data):
        pool, X, y = pool_and_data
        m = evaluate_treatment(pool, X, y, X, y, [], 0)
        curve = m["oracle_curve"]
        assert all(curve[i] <= curve[i - 1] for i in range(1, len(curve)))

    def test_the_two_readings_meet_at_the_classic_oracle(self, pool_and_data):
        """greedy Oracle_M == at-least-1 == the classic pool oracle (Table 7)."""
        pool, X, y = pool_and_data
        m = evaluate_treatment(pool, X, y, X, y, [], 0)
        assert m["oracle_greedy_curve"][-1] == pytest.approx(m["oracle_curve"][0])

    def test_atleast_last_point_is_unanimity(self):
        matrix = np.array([[1, 1], [1, 0], [0, 0]])
        assert atleast_curve(matrix)[-1] == pytest.approx(1 / 3)
        assert greedy_oracle_curve(matrix)[-1] == pytest.approx(2 / 3)


class TestMeanProbability:
    def test_mean_probability_metrics_present(self, pool_and_data):
        pool, X, y = pool_and_data
        m = evaluate_treatment(pool, X, y, X, y, [], 0)
        assert m["mean_probs_metrics"] is not None
        assert set(m["mean_probs_metrics"]) >= {"acc", "f1", "balanced_acc"}

    def test_majority_vote_metrics_present(self, pool_and_data):
        pool, X, y = pool_and_data
        m = evaluate_treatment(pool, X, y, X, y, [], 0)
        assert m["majority_vote"] == pytest.approx(m["majority_vote_metrics"]["acc"])


class TestDiversity:
    def test_double_fault_mean_is_off_diagonal(self, pool_and_data):
        pool, X, y = pool_and_data
        m = evaluate_treatment(pool, X, y, X, y, [], 0)
        assert 0.0 <= m["double_fault_mean"] <= 1.0

    def test_individual_accuracies_match_matrix(self, pool_and_data):
        pool, X, y = pool_and_data
        m = evaluate_treatment(pool, X, y, X, y, [], 0)
        assert len(m["individual_accuracies"]) == len(pool)
        assert m["individual_accuracies"][0] == pytest.approx(
            m["correctness_matrix"][:, 0].mean())
