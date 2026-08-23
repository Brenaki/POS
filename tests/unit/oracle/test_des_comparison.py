"""DCS/DES baselines over a fitted pool (ADR 0017/0018, milestone 6).

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
    best_des,
    des_accuracies,
    des_columns,
    effective_k,
    evaluate_des,
)
from pos.oracle.des_methods import DES_METHODS, NEEDS_PROBA
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
    metrics, preds, notes = evaluate_des(pool, X_dsel, y_dsel, X_test, y_test)
    assert notes == {}
    assert set(metrics) == set(DES_METHODS)
    assert all(0.0 <= m["acc"] <= 1.0 for m in metrics.values())
    # ADR 0018: predictions are stored so a new metric never costs a rerun.
    assert set(preds) == set(DES_METHODS)
    assert all(p.shape == y_test.shape for p in preds.values())


def test_recorded_accuracy_matches_the_recorded_predictions(split):
    """The scalar and the stored labels must not drift apart."""
    X_tr, y_tr, X_dsel, y_dsel, X_test, y_test = split
    pool = _pool(lambda i: DecisionTreeClassifier(random_state=i), X_tr, y_tr)
    metrics, preds, _ = evaluate_des(pool, X_dsel, y_dsel, X_test, y_test,
                                     methods=("ola", "knorau"))
    for name in ("ola", "knorau"):
        assert metrics[name]["acc"] == pytest.approx(np.mean(preds[name] == y_test))


def test_perceptron_pool_records_why_the_proba_methods_cannot_run(split):
    """META-DES and KNOP need predict_proba — the ADR 0016 gap again.

    Everything else must still produce a number: a run must not lose the whole
    DCS/DES comparison because two methods are inapplicable.
    """
    X_tr, y_tr, X_dsel, y_dsel, X_test, y_test = split
    pool = _pool(lambda i: Perceptron(max_iter=100, tol=1.0, random_state=i), X_tr, y_tr)
    metrics, _, notes = evaluate_des(pool, X_dsel, y_dsel, X_test, y_test)
    for name in NEEDS_PROBA:
        assert metrics[name] is None
        assert "probability" in notes[name]
    runnable = [m for m in DES_METHODS if m not in NEEDS_PROBA]
    assert all(metrics[m] is not None for m in runnable)


def test_knora_survives_the_removed_numpy_aliases(split):
    """Regression: deslib 0.3.5 calls np.float/np.int, gone since numpy 1.24.

    Without `deslib_compat` these two come back as None with an
    AttributeError note instead of an accuracy.
    """
    X_tr, y_tr, X_dsel, y_dsel, X_test, y_test = split
    pool = _pool(lambda i: DecisionTreeClassifier(random_state=i), X_tr, y_tr)
    metrics, _, _ = evaluate_des(pool, X_dsel, y_dsel, X_test, y_test,
                                 methods=("knorae", "knorau"))
    assert metrics["knorae"] is not None
    assert metrics["knorau"] is not None


def test_effective_k_shrinks_to_fit_a_tiny_dsel():
    assert effective_k(1000) == 7
    assert effective_k(5) == 4
    assert effective_k(1) == 1


def test_des_columns_carry_accuracy_under_the_bare_name(split):
    """`des_ola` must stay accuracy so the ADR 0017 run still lines up."""
    cols = des_columns({"ola": {"acc": 0.9, "f1": 0.8, "balanced_acc": 0.85,
                                "precision": 0.7, "recall": 0.6}})
    assert cols["des_ola"] == 0.9
    assert cols["des_ola_f1"] == 0.8
    assert cols["des_metades"] is None
    assert cols["des_metades_balanced_acc"] is None
    assert all(f"des_{m}" in cols for m in DES_METHODS)


def test_des_accuracies_flattens_the_metric_dicts():
    got = des_accuracies({"ola": {"acc": 0.9}, "lca": None}, methods=("ola", "lca"))
    assert got == {"ola": 0.9, "lca": None}


def test_best_des_picks_the_maximum_and_tolerates_an_empty_pool():
    metrics = {"ola": {"acc": 0.7}, "lca": {"acc": 0.9}, "metades": None}
    assert best_des(metrics) == ("lca", 0.9)
    # A different metric can crown a different method — the point of ADR 0018.
    scored = {"ola": {"acc": 0.7, "balanced_acc": 0.9},
              "lca": {"acc": 0.9, "balanced_acc": 0.5}}
    assert best_des(scored, metric="balanced_acc") == ("ola", 0.9)
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
