"""Tests for the greedy Oracle_N curve construction (ADR 2020 T11).

Greedy Oracle_N: at each step N=1..M, select the classifier that most
increases the oracle accuracy. Guarantees monotonicity
Oracle_1 <= Oracle_2 <= ... <= Oracle_M. Oracle_M == Oracle_100 is the only
point directly comparable to Table 7 (classic pool oracle).
"""

from __future__ import annotations

import numpy as np
import pytest

from pos.oracle.oracle_curve_greedy import greedy_oracle_curve


class TestGreedyMonotonicity:
    def test_curve_is_monotonically_non_decreasing(self):
        rng = np.random.RandomState(0)
        matrix = rng.randint(0, 2, size=(50, 10))
        curve = greedy_oracle_curve(matrix)
        for i in range(1, len(curve)):
            assert curve[i] >= curve[i - 1], f"non-monotonic at {i}: {curve}"

    def test_curve_length_equals_M(self):
        matrix = np.ones((20, 8), dtype=int)
        curve = greedy_oracle_curve(matrix)
        assert len(curve) == 8

    def test_oracle_M_equals_classic_oracle(self):
        """Oracle_M (all classifiers selected) == classic oracle (>= 1 correct).

        Greedy Oracle_N is INCREASING (nested sets C1⊂...⊂CM); Oracle_M
        covers all M classifiers, so it equals the classic pool oracle.
        This is the only point comparable to Table 7 (ADR 0020 §5).
        """
        rng = np.random.RandomState(42)
        matrix = rng.randint(0, 2, size=(100, 100))
        curve = greedy_oracle_curve(matrix)
        classic_oracle = float((matrix.sum(axis=1) >= 1).mean())
        assert curve[-1] == pytest.approx(classic_oracle)

    def test_oracle_1_is_max_individual_accuracy(self):
        """Greedy Oracle_1 = best single classifier accuracy (first pick)."""
        rng = np.random.RandomState(42)
        matrix = rng.randint(0, 2, size=(100, 50))
        curve = greedy_oracle_curve(matrix)
        max_individual = float(max(matrix[:, j].mean() for j in range(50)))
        assert curve[0] == pytest.approx(max_individual)

    def test_perfect_pool_gives_all_ones(self):
        """If every classifier is always correct, every Oracle_N == 1.0."""
        matrix = np.ones((30, 5), dtype=int)
        curve = greedy_oracle_curve(matrix)
        assert all(c == 1.0 for c in curve)

    def test_terrible_pool_gives_zero_for_unanimity(self):
        """If no classifier is ever correct, Oracle_M == 0.0."""
        matrix = np.zeros((30, 5), dtype=int)
        curve = greedy_oracle_curve(matrix)
        assert curve[-1] == 0.0
        assert curve[0] == 0.0


class TestGreedyProperty:
    def test_greedy_oracle_1_is_max_individual(self):
        """Oracle_1 should be the max individual accuracy (greedy picks best first)."""
        matrix = np.array([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        # individual accuracies: clf0=0.75, clf1=0.25, clf2=0.25, clf3=0.25
        curve = greedy_oracle_curve(matrix)
        assert curve[0] == pytest.approx(0.75)

    def test_greedy_picks_complementary_classifiers(self):
        """Greedy should prefer classifiers that cover new samples."""
        # clf0 correct on samples 0,1; clf1 correct on 2,3
        matrix = np.array([
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ])
        curve = greedy_oracle_curve(matrix)
        assert curve[0] == 0.5  # best single covers 2/4
        assert curve[1] == 1.0  # together they cover all 4


class TestCheckpoints:
    def test_checkpoint_extraction(self):
        from pos.oracle.oracle_curve_greedy import extract_checkpoints
        rng = np.random.RandomState(0)
        matrix = rng.randint(0, 2, size=(50, 100))
        curve = greedy_oracle_curve(matrix)
        checkpoints = (1, 5, 10, 20, 50, 100)
        values = extract_checkpoints(curve, checkpoints)
        assert values[1] == curve[0]
        assert values[5] == curve[4]
        assert values[100] == curve[99]
        assert values[100] == curve[-1]