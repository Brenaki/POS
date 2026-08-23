"""ADR 0015: the vectorised double-fault must match deslib exactly.

`diversitys` implements Eq. 2 of Monteiro et al. (2022):
    DDV[C_i] = sum_j DF(C_i, C_j) / (N - 1)
where DF is the Double Fault measure. The reference implementation is
`deslib.util.diversity.double_fault`, which loops over every sample in
Python; these tests pin the matmul reformulation against it.
"""

from __future__ import annotations

import numpy as np
import pytest
from deslib.util import diversity as deslib_diversity

from pos.diversity import diversitys, double_fault_matrix


def _reference_diversitys(y_test, predicts):
    """The pre-ADR-0015 implementation, kept as the oracle."""
    out = []
    for i in range(len(predicts)):
        db = [deslib_diversity.double_fault(y_test, predicts[i], predicts[j])
              for j in range(len(predicts)) if j != i]
        out.append(np.mean(db))
    return out


@pytest.mark.parametrize("m,n,k", [(2, 10, 2), (5, 40, 2), (20, 200, 3), (60, 300, 5)])
def test_matches_deslib(m, n, k):
    rng = np.random.default_rng(m * 1000 + n)
    y = rng.integers(0, k, n)
    preds = rng.integers(0, k, (m, n))
    expected = np.array(_reference_diversitys(y, preds))
    got = np.array(diversitys(y, preds))
    assert np.allclose(got, expected, atol=1e-12)


def test_pairwise_entry_matches_deslib():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 150)
    preds = rng.integers(0, 3, (6, 150))
    dfm = double_fault_matrix(y, preds)
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            assert dfm[i, j] == pytest.approx(
                deslib_diversity.double_fault(y, preds[i], preds[j]), abs=1e-12)


def test_all_correct_gives_zero_double_fault():
    y = np.array([0, 1, 0, 1])
    preds = np.array([y, y, y])
    assert diversitys(y, preds) == [0.0, 0.0, 0.0]


def test_all_wrong_gives_one():
    y = np.array([0, 1, 0, 1])
    preds = np.array([1 - y, 1 - y, 1 - y])
    assert diversitys(y, preds) == [1.0, 1.0, 1.0]


def test_symmetric_and_diagonal_is_own_error_rate():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 80)
    preds = rng.integers(0, 2, (4, 80))
    dfm = double_fault_matrix(y, preds)
    assert np.allclose(dfm, dfm.T)
    for i in range(4):
        assert dfm[i, i] == pytest.approx(np.mean(preds[i] != y))


def test_single_classifier_is_undefined():
    y = np.array([0, 1, 0])
    assert np.isnan(diversitys(y, np.array([[0, 1, 0]]))).all()
