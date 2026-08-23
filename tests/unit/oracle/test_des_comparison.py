"""DCS/DES baselines over a fitted pool (ADR 0017, cronograma milestone 6).

Objective 6 of the subproject compares Oracle_N against real combination
methods. These tests pin the contract of `evaluate_des`: it never raises for
a pool a method cannot handle, it records why instead, and the neighbourhood
adapts to a small validation split.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from pos.oracle.correctness_matrix import build_correctness_matrix
from pos.oracle.des_comparison import (
    DES_METHODS,
    best_des,
    des_columns,
    effective_k,
    evaluate_des,
)
from pos.oracle.oracle_n import oracle_n_accuracy


@pytest.fixture(scope="module")
def split():
    X, y = load_wine(return_X_y=True)
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    X_tr, X_dsel, y_tr, y_dsel = train_test_split(
        X_tr, y_tr, test_size=0.25, random_state=0, stratify=y_tr)
    return X_tr, y_tr, X_dsel, y_dsel, X_test, y_test


def _pool(make, X_tr, y_tr, n=20):
    pool = []
    n_classes = len(np.unique(y_tr))
    for i in range(n):
        idx = np.random.RandomState(i).choice(len(y_tr), int(0.7 * len(y_tr)), replace=True)
        if len(np.unique(y_tr[idx])) < n_classes:
            continue
        pool.append(make(i).fit(X_tr[idx], y_tr[idx]))
    return pool


def test_tree_pool_runs_every_method(split):
    X_tr, y_tr, X_dsel, y_dsel, X_test, y_test = split
    pool = _pool(lambda i: DecisionTreeClassifier(random_state=i), X_tr, y_tr)
    accs, notes = evaluate_des(pool, X_dsel, y_dsel, X_test, y_test)
    assert notes == {}
    assert set(accs) == set(DES_METHODS)
    assert all(0.0 <= a <= 1.0 for a in accs.values())


def test_perceptron_pool_records_why_metades_cannot_run(split):
    """METADES needs predict_proba — the same gap ADR 0016 hit on soft fusion.

    The other four must still produce a number: a run must not lose the whole
    DCS/DES comparison because one method is inapplicable.
    """
    X_tr, y_tr, X_dsel, y_dsel, X_test, y_test = split
    pool = _pool(lambda i: Perceptron(max_iter=100, tol=1.0, random_state=i), X_tr, y_tr)
    accs, notes = evaluate_des(pool, X_dsel, y_dsel, X_test, y_test)
    assert accs["metades"] is None
    assert "probability" in notes["metades"]
    assert all(accs[m] is not None for m in ("ola", "lca", "knorae", "knorau"))


def test_knora_survives_the_removed_numpy_aliases(split):
    """Regression: deslib 0.3.5 calls np.float/np.int, gone since numpy 1.24.

    Without `deslib_compat` these two come back as None with an
    AttributeError note instead of an accuracy.
    """
    X_tr, y_tr, X_dsel, y_dsel, X_test, y_test = split
    pool = _pool(lambda i: DecisionTreeClassifier(random_state=i), X_tr, y_tr)
    accs, _ = evaluate_des(pool, X_dsel, y_dsel, X_test, y_test,
                           methods=("knorae", "knorau"))
    assert accs["knorae"] is not None
    assert accs["knorau"] is not None


def test_effective_k_shrinks_to_fit_a_tiny_dsel():
    assert effective_k(1000) == 7
    assert effective_k(5) == 4
    assert effective_k(1) == 1


def test_des_columns_always_has_every_method():
    cols = des_columns({"ola": 0.9})
    assert cols["des_ola"] == 0.9
    assert cols["des_metades"] is None
    assert set(cols) == {f"des_{m}" for m in DES_METHODS}


def test_best_des_picks_the_maximum_and_tolerates_an_empty_pool():
    assert best_des({"ola": 0.7, "lca": 0.9, "metades": None}) == ("lca", 0.9)
    assert best_des({"ola": None}) == (None, None)
    assert best_des(None) == (None, None)


def test_deslib_oracle_agrees_with_our_oracle_1(split):
    """Independent check of the Oracle_1 definition (objective 2).

    DESlib ships its own implementation of the traditional Oracle; if our
    correctness matrix and curve are right, Oracle_1 must equal it exactly.
    """
    from pos.oracle.deslib_compat import install_numpy_aliases

    install_numpy_aliases()
    from deslib.static.oracle import Oracle

    X_tr, y_tr, _, _, X_test, y_test = split
    pool = _pool(lambda i: DecisionTreeClassifier(random_state=i), X_tr, y_tr)
    theirs = Oracle(pool_classifiers=pool).fit(X_tr, y_tr).score(X_test, y_test)
    ours = oracle_n_accuracy(build_correctness_matrix(pool, X_test, y_test), 1)
    assert ours == pytest.approx(theirs)
