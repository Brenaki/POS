"""Tests for the Monteiro 2023 holdout splitter (ADR 0020 T2).

20 independent stratified holdouts 50/25/25, new split per replication,
deterministic given a root seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pos.oracle.monteiro_splitter import monteiro_holdout_splits


@pytest.fixture
def balanced_data():
    rng = np.random.RandomState(0)
    X = rng.randn(400, 5)
    y = np.array([0] * 200 + [1] * 200)
    return X, y


@pytest.fixture
def imbalanced_data():
    rng = np.random.RandomState(0)
    X = rng.randn(400, 5)
    y = np.array([0] * 300 + [1] * 100)
    return X, y


class TestMonteiroHoldout:
    def test_produces_twenty_replications(self, balanced_data):
        X, y = balanced_data
        splits = monteiro_holdout_splits(X, y, n_replications=20, root_seed=86)
        assert len(splits) == 20

    def test_fractions_are_50_25_25(self, balanced_data):
        X, y = balanced_data
        n = len(y)
        splits = monteiro_holdout_splits(X, y, n_replications=1, root_seed=86)
        s = splits[0]
        assert s["X_train"].shape[0] == int(n * 0.5)
        assert s["X_val"].shape[0] == int(n * 0.25)
        assert s["X_test"].shape[0] == n - int(n * 0.5) - int(n * 0.25)

    def test_partitions_are_disjoint(self, balanced_data):
        X, y = balanced_data
        splits = monteiro_holdout_splits(X, y, n_replications=5, root_seed=86)
        for s in splits:
            tr = set(map(tuple, s["X_train"]))
            va = set(map(tuple, s["X_val"]))
            te = set(map(tuple, s["X_test"]))
            assert not (tr & va), "train/val overlap"
            assert not (tr & te), "train/test overlap"
            assert not (va & te), "val/test overlap"

    def test_partitions_are_stratified(self, imbalanced_data):
        X, y = imbalanced_data
        splits = monteiro_holdout_splits(X, y, n_replications=5, root_seed=86)
        # imbalanced: 75/25; check each partition preserves roughly the ratio
        for s in splits:
            for yy in (s["y_train"], s["y_val"], s["y_test"]):
                _, counts = np.unique(yy, return_counts=True)
                assert len(counts) == 2, "a class is missing from a partition"
                ratio = counts.min() / counts.sum()
                assert 0.20 <= ratio <= 0.30, f"stratification lost: {ratio:.3f}"

    def test_splits_are_deterministic_with_seed(self, balanced_data):
        X, y = balanced_data
        a = monteiro_holdout_splits(X, y, n_replications=3, root_seed=86)
        b = monteiro_holdout_splits(X, y, n_replications=3, root_seed=86)
        for sa, sb in zip(a, b):
            np.testing.assert_array_equal(sa["X_train"], sb["X_train"])
            np.testing.assert_array_equal(sa["X_test"], sb["X_test"])

    def test_different_seeds_produce_different_splits(self, balanced_data):
        X, y = balanced_data
        a = monteiro_holdout_splits(X, y, n_replications=1, root_seed=86)
        b = monteiro_holdout_splits(X, y, n_replications=1, root_seed=87)
        assert not np.array_equal(a[0]["X_train"], b[0]["X_train"])

    def test_new_split_per_replication(self, balanced_data):
        X, y = balanced_data
        splits = monteiro_holdout_splits(X, y, n_replications=20, root_seed=86)
        # pair-wise: at least one replication differs from another
        distinct = set()
        for s in splits:
            distinct.add(s["X_train"].tobytes())
        assert len(distinct) > 1, "all 20 replications produced the same split"

    def test_validation_and_dsel_overlap(self, balanced_data):
        """Monteiro profile: validation == DSEL (ADR 0020 §2.2)."""
        X, y = balanced_data
        splits = monteiro_holdout_splits(X, y, n_replications=1, root_seed=86)
        s = splits[0]
        np.testing.assert_array_equal(s["X_val"], s["X_dsel"])
        np.testing.assert_array_equal(s["y_val"], s["y_dsel"])

    def test_seed_for_replication_is_derived(self, balanced_data):
        """rep r uses seed root+r; replication 0 != replication 1."""
        X, y = balanced_data
        splits = monteiro_holdout_splits(X, y, n_replications=2, root_seed=86)
        assert not np.array_equal(splits[0]["X_train"], splits[1]["X_train"])

    def test_returns_labels_in_each_partition(self, balanced_data):
        X, y = balanced_data
        splits = monteiro_holdout_splits(X, y, n_replications=1, root_seed=86)
        s = splits[0]
        for key in ("y_train", "y_val", "y_test", "y_dsel"):
            assert key in s
            assert s[key].shape[0] == s[key.replace("y_", "X_")].shape[0]