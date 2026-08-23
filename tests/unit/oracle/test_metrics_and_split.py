"""Metric suite and the three-way split (ADR 0018, cronograma milestone 7).

Milestone 7 asks for precision, recall, F1 and balanced accuracy on top of
accuracy, and the split has to keep the GA's fitness set disjoint from the
DSEL — that disjointness is the whole reason for the rerun.
"""

from __future__ import annotations

import numpy as np
import pytest

from pos.oracle.fold_splitter import stratified_three_way_split, stratified_val_split
from pos.oracle.metrics import (
    METRIC_NAMES,
    metric_columns,
    oracle_balanced_recall,
    prediction_metrics,
)


def test_perfect_prediction_scores_one_everywhere():
    y = np.array([0, 0, 1, 1, 2])
    got = prediction_metrics(y, y)
    assert set(got) == set(METRIC_NAMES)
    assert all(v == pytest.approx(1.0) for v in got.values())


def test_majority_only_predictor_is_caught_by_the_balanced_metrics():
    """The reason milestone 7 asks for more than accuracy.

    Nine of ten samples are class 0. Always predicting 0 scores 0.9 accuracy
    and would look competitive against the fusers being compared, while
    balanced accuracy and macro F1 expose that the minority class is gone.
    """
    y = np.array([0] * 9 + [1])
    y_pred = np.zeros_like(y)
    got = prediction_metrics(y, y_pred)
    assert got["acc"] == pytest.approx(0.9)
    assert got["balanced_acc"] == pytest.approx(0.5)
    assert got["recall"] == pytest.approx(0.5)
    # Class 0: 9 of the 10 predicted are right. Class 1 is never predicted, so
    # its precision is undefined -> scored 0. Macro = (0.9 + 0) / 2.
    assert got["precision"] == pytest.approx(0.45)
    assert got["f1"] < 0.5


def test_metric_columns_keep_a_stable_header_when_the_fuser_could_not_run():
    cols = metric_columns("des_metades", None)
    assert set(cols) == {f"des_metades_{n}" for n in METRIC_NAMES}
    assert all(v is None for v in cols.values())


def test_oracle_balanced_recall_separates_the_classes():
    """Oracle_1 can be propped up entirely by the majority class."""
    y = np.array([0, 0, 0, 0, 1])
    # Every majority sample is covered; the single minority one is not.
    matrix = np.array([[1, 0], [1, 0], [1, 1], [1, 0], [0, 0]])
    assert oracle_balanced_recall(matrix, y, 1) == pytest.approx(0.5)
    plain = (matrix.sum(axis=1) >= 1).mean()
    assert plain == pytest.approx(0.8)


def test_three_way_split_keeps_the_training_slice_identical():
    """Bagging/RF pools must stay bit-identical to the two-way run (ADR 0018)."""
    rng = np.random.RandomState(0)
    X = rng.rand(200, 4)
    y = np.array([0] * 100 + [1] * 100)
    X_tr2, y_tr2, _, _ = stratified_val_split(X, y, 0.2, 7)
    X_tr3, y_tr3, X_val, y_val, X_dsel, y_dsel = stratified_three_way_split(
        X, y, 0.2, 7)
    assert np.array_equal(X_tr2, X_tr3)
    assert np.array_equal(y_tr2, y_tr3)
    assert len(y_val) + len(y_dsel) == len(y) - len(y_tr3)
    assert abs(len(y_val) - len(y_dsel)) <= 1


def test_three_way_split_leaves_fitness_and_dsel_disjoint():
    """The bias ADR 0018 removes: the GA used to fit on its own DSEL."""
    rng = np.random.RandomState(1)
    X = np.arange(200, dtype=float).reshape(200, 1) + rng.rand(200, 1) * 1e-6
    y = np.array([0] * 100 + [1] * 100)
    _, _, X_val, _, X_dsel, _ = stratified_three_way_split(X, y, 0.2, 3)
    assert not (set(X_val.ravel().tolist()) & set(X_dsel.ravel().tolist()))


def test_three_way_split_is_deterministic_for_a_seed():
    rng = np.random.RandomState(2)
    X, y = rng.rand(120, 3), np.array([0] * 60 + [1] * 60)
    a = stratified_three_way_split(X, y, 0.2, 11)
    b = stratified_three_way_split(X, y, 0.2, 11)
    assert all(np.array_equal(x, z) for x, z in zip(a, b, strict=True))
