"""Tests for the pruning instrumentation (ADR 0019).

The ADR 0018 run concluded that how aggressively a method prunes the pool is
the axis that orders the fusers, but the pruning was inferred from the name of
the algorithm. These assert the measured quantity against methods whose
selection size is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import BaggingClassifier

from pos.oracle.des_comparison import evaluate_des
from pos.oracle.des_methods import method_classes, takes_k
from pos.oracle.selection_size import SelectionSpy, static_summary

M = 10


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.RandomState(0)
    X = rng.randn(400, 6)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    pool = list(BaggingClassifier(n_estimators=M, random_state=0)
                .fit(X[:200], y[:200]).estimators_)
    return pool, X[200:300], y[200:300], X[300:], y[300:]


def _measure(name, fitted):
    pool, X_dsel, y_dsel, X_test, _ = fitted
    cls = method_classes()[name]
    kwargs = {"pool_classifiers": pool, "random_state": 42}
    if takes_k(name):
        kwargs["k"] = 7
    ds = cls(**kwargs)
    ds.fit(X_dsel, y_dsel)
    spy = SelectionSpy()
    spy.attach(ds)
    ds.predict(X_test)
    return spy.summary(M, len(X_test), ds=ds)


class TestMeasuredSelection:
    @pytest.mark.parametrize("name", ["ola", "lca", "mcb", "rank"])
    def test_dcs_keeps_exactly_one_classifier(self, name, fitted):
        assert _measure(name, fitted)["sel_frac"] == pytest.approx(1.0 / M)

    @pytest.mark.parametrize("name", ["knorae", "desp", "desknn"])
    def test_des_keeps_more_than_one(self, name, fitted):
        frac = _measure(name, fitted)["sel_frac"]
        assert 1.0 / M < frac <= 1.0

    @pytest.mark.parametrize("name", ["knorau", "knop"])
    def test_weighting_methods_keep_the_whole_pool(self, name, fitted):
        """They never call `select` — every classifier votes, weighted."""
        assert _measure(name, fitted)["sel_frac"] == 1.0

    def test_routed_fraction_is_a_share_of_the_queries(self, fitted):
        summary = _measure("ola", fitted)
        assert 0.0 < summary["routed_frac"] <= 1.0


class TestStaticMethods:
    def test_single_best_is_one_classifier(self):
        assert static_summary("single_best", M)["sel_frac"] == pytest.approx(1.0 / M)

    def test_static_selection_is_the_pct_default(self):
        assert static_summary("static_sel", M)["sel_frac"] == 0.5


class TestEvaluateDesCarriesTheMeasurement:
    def test_sel_frac_reaches_the_metrics_dict(self, fitted):
        pool, X_dsel, y_dsel, X_test, y_test = fitted
        metrics, _, notes = evaluate_des(pool, X_dsel, y_dsel, X_test, y_test,
                                         methods=("ola", "knorau", "static_sel"))
        assert metrics["ola"]["sel_frac"] == pytest.approx(1.0 / M)
        assert metrics["knorau"]["sel_frac"] == 1.0
        assert metrics["static_sel"]["sel_frac"] == 0.5
        assert notes == {}

    def test_instrumentation_never_masks_the_accuracy(self, fitted):
        pool, X_dsel, y_dsel, X_test, y_test = fitted
        metrics, _, _ = evaluate_des(pool, X_dsel, y_dsel, X_test, y_test,
                                     methods=("ola",))
        assert 0.0 <= metrics["ola"]["acc"] <= 1.0
