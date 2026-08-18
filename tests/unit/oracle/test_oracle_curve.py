"""Unit tests for pos.oracle.oracle_curve.

TDD — tests written BEFORE implementation.
"""

from __future__ import annotations

import numpy as np


def test_oracle_curve_returns_dict_with_all_levels():
    """oracle_curve(matrix) deve retornar dict {1: acc, 2: acc, ..., M: acc}."""
    from pos.oracle.oracle_curve import oracle_curve

    # 4 samples, 3 classifiers
    matrix = np.array([
        [1, 1, 1],  # sum=3
        [1, 1, 0],  # sum=2
        [1, 0, 0],  # sum=1
        [0, 0, 0],  # sum=0
    ])
    curve = oracle_curve(matrix)
    # M = 3 classifiers → curve has keys 1, 2, 3
    assert set(curve.keys()) == {1, 2, 3}
    # Oracle_1: 3 of 4 have >= 1 → 0.75
    # Oracle_2: 2 of 4 have >= 2 → 0.5
    # Oracle_3: 1 of 4 have >= 3 → 0.25
    assert curve[1] == 0.75
    assert curve[2] == 0.5
    assert curve[3] == 0.25


def test_oracle_curve_monotonic_non_increasing():
    """Oracle_1 >= Oracle_2 >= ... >= Oracle_M (propriedade chave)."""
    from pos.oracle.oracle_curve import oracle_curve

    # Random matrix with known shape
    rng = np.random.default_rng(42)
    matrix = rng.integers(0, 2, size=(20, 7))
    curve = oracle_curve(matrix)
    values = [curve[n] for n in range(1, 8)]
    for i in range(len(values) - 1):
        assert values[i] >= values[i + 1], (
            f"Monotonicidade violada: Oracle_{i+1}={values[i]} < Oracle_{i+2}={values[i+1]}"
        )


def test_oracle_curve_all_correct():
    """Todos acertam → todas as Oracle_N = 1.0."""
    from pos.oracle.oracle_curve import oracle_curve

    matrix = np.ones((5, 4), dtype=int)
    curve = oracle_curve(matrix)
    for n in range(1, 5):
        assert curve[n] == 1.0


def test_oracle_curve_all_wrong():
    """Todos erram → todas as Oracle_N = 0.0 (para N >= 1)."""
    from pos.oracle.oracle_curve import oracle_curve

    matrix = np.zeros((5, 4), dtype=int)
    curve = oracle_curve(matrix)
    for n in range(1, 5):
        assert curve[n] == 0.0


def test_oracle_curve_single_classifier():
    """Pool com 1 classificador → curve tem só key=1."""
    from pos.oracle.oracle_curve import oracle_curve

    matrix = np.array([[1], [0], [1], [1]])
    curve = oracle_curve(matrix)
    assert set(curve.keys()) == {1}
    # 3 of 4 correct → 0.75
    assert curve[1] == 0.75


def test_oracle_curve_array_returns_list():
    """oracle_curve_array(matrix) retorna lista ordenada [acc_1, acc_2, ..., acc_M]."""
    from pos.oracle.oracle_curve import oracle_curve_array

    matrix = np.array([
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    arr = oracle_curve_array(matrix)
    assert isinstance(arr, list)
    assert len(arr) == 3
    assert arr == [0.75, 0.5, 0.25]
