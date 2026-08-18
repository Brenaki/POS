"""Unit tests for pos.oracle.oracle_n.

TDD — tests written BEFORE implementation.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_oracle_n_traditional_is_oracle_1():
    """Oracle_1 = Oracle tradicional: amostra correta se >= 1 classificador acertar."""
    from pos.oracle.oracle_n import oracle_n_accuracy

    # 4 samples, 3 classifiers
    # row sums: [1, 2, 1, 0]
    matrix = np.array([
        [1, 0, 0],  # 1 correct → Oracle_1 = 1
        [1, 1, 0],  # 2 correct → Oracle_1 = 1
        [0, 0, 1],  # 1 correct → Oracle_1 = 1
        [0, 0, 0],  # 0 correct → Oracle_1 = 0
    ])
    # Oracle_1: 3 of 4 samples have >= 1 correct → 0.75
    assert oracle_n_accuracy(matrix, 1) == 0.75


def test_oracle_n_unanimity_is_oracle_m():
    """Oracle_M (M = n_classifiers) = unanimidade: todos devem acertar."""
    from pos.oracle.oracle_n import oracle_n_accuracy

    matrix = np.array([
        [1, 1, 1],  # 3 correct → Oracle_3 = 1
        [1, 1, 0],  # 2 correct → Oracle_3 = 0
        [1, 0, 1],  # 2 correct → Oracle_3 = 0
        [0, 0, 0],  # 0 correct → Oracle_3 = 0
    ])
    # Oracle_3: 1 of 4 samples has all 3 correct → 0.25
    assert oracle_n_accuracy(matrix, 3) == 0.25


def test_oracle_n_zero_is_trivially_one():
    """Oracle_0: trivialmente todas as amostras passam (>= 0 acertos)."""
    from pos.oracle.oracle_n import oracle_n_accuracy

    matrix = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
    ])
    # Oracle_0: all 3 samples have >= 0 correct → 1.0
    assert oracle_n_accuracy(matrix, 0) == 1.0


def test_oracle_n_intermediate():
    """Oracle_2 with known matrix."""
    from pos.oracle.oracle_n import oracle_n_accuracy

    # row sums: [3, 2, 1, 0]
    matrix = np.array([
        [1, 1, 1],  # 3 correct → >= 2 → 1
        [1, 1, 0],  # 2 correct → >= 2 → 1
        [0, 0, 1],  # 1 correct  → >= 2 → 0
        [0, 0, 0],  # 0 correct  → >= 2 → 0
    ])
    # Oracle_2: 2 of 4 → 0.5
    assert oracle_n_accuracy(matrix, 2) == 0.5


def test_oracle_n_n_greater_than_m_is_zero():
    """Oracle_N com N > M (n_classifiers) → sempre 0 (ninguém pode ter >= N acertos)."""
    from pos.oracle.oracle_n import oracle_n_accuracy

    matrix = np.array([
        [1, 1, 1],  # M=3, asking N=4 → impossible
        [1, 1, 1],
    ])
    assert oracle_n_accuracy(matrix, 4) == 0.0


def test_oracle_n_all_correct_all_ones():
    """Se todos acertam tudo, Oracle_N = 1.0 para qualquer N <= M."""
    from pos.oracle.oracle_n import oracle_n_accuracy

    matrix = np.ones((5, 4), dtype=int)
    for n in range(1, 5):
        assert oracle_n_accuracy(matrix, n) == 1.0


def test_oracle_n_all_wrong_all_zeros():
    """Se todos erram tudo, Oracle_N = 0.0 para qualquer N >= 1."""
    from pos.oracle.oracle_n import oracle_n_accuracy

    matrix = np.zeros((5, 4), dtype=int)
    for n in range(1, 5):
        assert oracle_n_accuracy(matrix, n) == 0.0


def test_oracle_n_single_sample():
    """Edge case: matriz 1x1."""
    from pos.oracle.oracle_n import oracle_n_accuracy

    assert oracle_n_accuracy(np.array([[1]]), 1) == 1.0
    assert oracle_n_accuracy(np.array([[0]]), 1) == 0.0


def test_oracle_n_vector_returns_boolean():
    """oracle_n_vector returns boolean array of Oracle_N(x_i) per sample."""
    from pos.oracle.oracle_n import oracle_n_vector

    matrix = np.array([
        [1, 1, 1],  # sum=3 → >= 2 → True
        [1, 1, 0],  # sum=2 → >= 2 → True
        [1, 0, 0],  # sum=1 → >= 2 → False
        [0, 0, 0],  # sum=0 → >= 2 → False
    ])
    vec = oracle_n_vector(matrix, 2)
    assert vec.dtype == bool
    np.testing.assert_array_equal(vec, np.array([True, True, False, False]))


def test_oracle_n_negative_n_raises():
    """N < 0 should raise ValueError."""
    from pos.oracle.oracle_n import oracle_n_accuracy

    with pytest.raises(ValueError):
        oracle_n_accuracy(np.array([[1, 0]]), -1)
