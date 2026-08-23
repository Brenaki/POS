"""Tests for ADR 0014: stratified validation split + dataset viability gate."""

from __future__ import annotations

import collections

import numpy as np

from pos.oracle.fold_splitter import (
    check_dataset_viability,
    min_class_count,
    min_instances_per_class_in_bag,
    stratified_val_split,
)


class TestMinClassCount:
    def test_counts_rarest_class(self):
        assert min_class_count(np.array([0] * 10 + [1] * 3)) == 3

    def test_empty(self):
        assert min_class_count(np.array([])) == 0


class TestBagArithmetic:
    def test_ecoli_leaves_nothing_in_a_bag(self):
        assert min_instances_per_class_in_bag(2, 10, 0.2, 0.5) < 2

    def test_healthy_class_survives(self):
        assert min_instances_per_class_in_bag(60, 10, 0.2, 0.5) >= 2


class TestViability:
    def test_rejects_class_smaller_than_n_folds(self):
        y = np.array([0] * 300 + [1] * 34 + [2] * 2)
        ok, reason = check_dataset_viability(y, 10, ["ga", "rf"])
        assert not ok and "min_class_count=2" in reason

    def test_accepts_balanced_dataset(self):
        y = np.array([0] * 100 + [1] * 100)
        ok, reason = check_dataset_viability(y, 10, ["ga", "bagging", "rf"])
        assert ok and reason == ""

    def test_ga_check_is_stricter_than_rf(self):
        """min_class == n_folds passes k-fold but can still starve GA bags.

        With n_folds=4 and 4 minority instances: k-fold is fine, but
        4 -> 3 (train) -> 2 (after val) -> 1 per bag, below verify_bag's 2.
        """
        y = np.array([0] * 400 + [1] * 4)
        ok_rf, _ = check_dataset_viability(y, 4, ["rf"])
        ok_ga, reason = check_dataset_viability(y, 4, ["ga"])
        assert ok_rf
        assert not ok_ga and "verify_bag" in reason

    def test_ten_folds_is_dominated_by_the_kfold_check(self):
        """At n_folds=10 any class passing k-fold also survives to a bag."""
        y = np.array([0] * 400 + [1] * 10)
        ok, reason = check_dataset_viability(y, 10, ["ga"])
        assert ok, reason


class TestStratifiedValSplit:
    def test_val_keeps_class_proportions(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 500 + [1] * 50)
        X = rng.normal(size=(len(y), 3))
        _, y_tr, _, y_val = stratified_val_split(X, y, 0.2, 42)
        val_ratio = collections.Counter(y_val)[1] / len(y_val)
        assert 0.05 <= val_ratio <= 0.14, f"minority fraction in val: {val_ratio}"
        assert set(y_tr) == set(y_val) == {0, 1}

    def test_sizes(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 100 + [1] * 100)
        X = rng.normal(size=(200, 3))
        X_tr, y_tr, X_val, y_val = stratified_val_split(X, y, 0.2, 7)
        assert len(X_val) == len(y_val) == 40
        assert len(X_tr) == len(y_tr) == 160

    def test_reproducible(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 100 + [1] * 100)
        X = rng.normal(size=(200, 3))
        a = stratified_val_split(X, y, 0.2, 7)
        b = stratified_val_split(X, y, 0.2, 7)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[3], b[3])

    def test_falls_back_when_a_class_is_singleton(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 60 + [1] * 60 + [2])
        X = rng.normal(size=(121, 3))
        X_tr, y_tr, X_val, y_val = stratified_val_split(X, y, 0.2, 3)
        assert len(y_tr) + len(y_val) == 121
