"""Unit tests for pos.oracle.correctness_matrix.

TDD — tests written BEFORE implementation.
"""

from __future__ import annotations

import numpy as np


def test_correctness_matrix_shape():
    """Matrix should be (n_samples, n_classifiers)."""
    from pos.oracle.correctness_matrix import build_correctness_matrix

    # 3 fake classifiers that always predict a fixed label
    class FakeClf:
        def __init__(self, label):
            self.label = label
            self.classes_ = np.array([0, 1, 2])

        def predict(self, X):
            return np.array([self.label] * len(X))

    pool = [FakeClf(0), FakeClf(1), FakeClf(2)]
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 2, 0])

    matrix = build_correctness_matrix(pool, X, y)
    assert matrix.shape == (4, 3)
    assert matrix.dtype in (np.int64, np.int32, int)


def test_correctness_matrix_all_correct():
    """All classifiers correct → all ones."""
    from pos.oracle.correctness_matrix import build_correctness_matrix

    class PerfectClf:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.array([0, 1, 0, 1])

    pool = [PerfectClf(), PerfectClf()]
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 1, 0, 1])

    matrix = build_correctness_matrix(pool, X, y)
    expected = np.ones((4, 2), dtype=int)
    np.testing.assert_array_equal(matrix, expected)


def test_correctness_matrix_all_wrong():
    """All classifiers wrong → all zeros."""
    from pos.oracle.correctness_matrix import build_correctness_matrix

    class WrongClf:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.array([1, 0, 1, 0])

    pool = [WrongClf(), WrongClf()]
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 1, 0, 1])

    matrix = build_correctness_matrix(pool, X, y)
    expected = np.zeros((4, 2), dtype=int)
    np.testing.assert_array_equal(matrix, expected)


def test_correctness_matrix_mixed():
    """Known pattern: clf1 gets samples 0,1 right; clf2 gets 1,2 right."""
    from pos.oracle.correctness_matrix import build_correctness_matrix

    class ClfA:
        classes_ = np.array([0, 1, 2])

        def predict(self, X):
            return np.array([0, 1, 1, 0])  # right on 0,1; wrong on 2,3

    class ClfB:
        classes_ = np.array([0, 1, 2])

        def predict(self, X):
            return np.array([1, 1, 2, 2])  # wrong on 0; right on 1,2; wrong on 3

    pool = [ClfA(), ClfB()]
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 1, 2, 0])

    matrix = build_correctness_matrix(pool, X, y)
    # sample 0: clfA=0 (right), clfB=1 (wrong) → [1, 0]
    # sample 1: clfA=1 (right), clfB=1 (right) → [1, 1]
    # sample 2: clfA=1 (wrong), clfB=2 (right) → [0, 1]
    # sample 3: clfA=0 (right), clfB=2 (wrong) → [1, 0]
    expected = np.array([[1, 0], [1, 1], [0, 1], [1, 0]])
    np.testing.assert_array_equal(matrix, expected)


def test_correctness_matrix_single_classifier():
    """One classifier → matrix is (n, 1)."""
    from pos.oracle.correctness_matrix import build_correctness_matrix

    class Clf:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.array([0, 1, 1])

    pool = [Clf()]
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])

    matrix = build_correctness_matrix(pool, X, y)
    assert matrix.shape == (3, 1)
    np.testing.assert_array_equal(matrix, np.array([[1], [1], [0]]))
