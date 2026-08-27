"""Tests for the real PGDCS trajectory (ADR 0020 §3.2).

`PgdcsEvolution` is the piece that connects the GA to `PgdcsTracker`: it must
record every generation, never let generation 0 win, hand PGDCS_A and PGDCS_D
their own generation, and return the classifiers the GA itself fitted.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from pos.pool.pgdcs_evolution import PgdcsEvolution


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(60, 4), rng.randn(60, 4) + 2.0])
    y = np.array([0] * 60 + [1] * 60)
    Xv = np.vstack([rng.randn(30, 4), rng.randn(30, 4) + 2.0])
    yv = np.array([0] * 30 + [1] * 30)
    return X, y, Xv, yv


def _run(data, ngen=3, M=6, seed=7):
    X, y, Xv, yv = data
    pg = PgdcsEvolution(nr_generation=ngen, nr_individual=M, nr_pop=M,
                        nr_child=M, nr_bags=M, classifier="tree",
                        types=["F1", "N1"], jobs=1, random_state=seed)
    return pg, pg.run(X, y, Xv, yv)


class TestTrajectory:
    def test_records_every_generation_including_zero(self, data):
        _, tracker = _run(data, ngen=3)
        gens = [e["gen"] for e in tracker.generation_trace]
        assert gens == [0, 1, 2, 3]

    def test_candidate_count_matches_nr_generation(self, data):
        _, tracker = _run(data, ngen=3)
        candidates = [e for e in tracker.generation_trace if e["gen"] > 0]
        assert len(candidates) == 3

    def test_generation_zero_never_selected(self, data):
        _, tracker = _run(data, ngen=3)
        assert tracker.best_a["gen"] > 0
        assert tracker.best_d["gen"] > 0

    def test_best_a_maximises_mvr(self, data):
        _, tracker = _run(data, ngen=3)
        cand = [e for e in tracker.generation_trace if e["gen"] > 0]
        assert tracker.best_a["mvr"] == max(e["mvr"] for e in cand)

    def test_best_d_maximises_gdisp(self, data):
        _, tracker = _run(data, ngen=3)
        cand = [e for e in tracker.generation_trace if e["gen"] > 0]
        assert tracker.best_d["gdisp"] == max(e["gdisp"] for e in cand)


class TestPools:
    def test_pools_have_M_fitted_trees(self, data):
        _, tracker = _run(data, ngen=3, M=6)
        for best in (tracker.best_a, tracker.best_d):
            assert len(best["pool"]) == 6
            assert all(isinstance(c, DecisionTreeClassifier) for c in best["pool"])

    def test_classifiers_are_the_ones_the_ga_fitted(self, data):
        """Not rebuilt from bags: every member is already fitted."""
        _, tracker = _run(data, ngen=3)
        X, y, Xv, yv = data
        for clf in tracker.best_a["pool"]:
            assert clf.predict(Xv).shape == yv.shape

    def test_mvr_matches_the_recorded_pool(self, data):
        """The recorded MVR is the majority vote of the recorded pool."""
        from scipy import stats
        X, y, Xv, yv = data
        _, tracker = _run(data, ngen=3)
        pool = tracker.best_a["pool"]
        preds = np.array([c.predict(Xv) for c in pool])
        voted = stats.mode(preds, axis=0, keepdims=False).mode
        assert float(np.mean(voted == yv)) == pytest.approx(tracker.best_a["mvr"])


class TestSampling:
    def test_bags_are_drawn_with_replacement(self, data):
        """ADR 0020: PGDCS bags are random with replacement, not stratified."""
        X, y, _, _ = data
        pg = PgdcsEvolution(nr_bags=20, tam_bags=0.5, classifier="tree",
                            types=["F1", "N1"], random_state=3)
        bags = pg.generate_bags(X, y)
        sizes = {len(b) for b in bags["inst"]}
        assert sizes == {int(len(X) * 0.5)}
        assert any(len(set(b)) < len(b) for b in bags["inst"]), "no duplicates drawn"


class TestReproducibility:
    def test_same_seed_same_generations(self, data):
        _, a = _run(data, ngen=3, seed=11)
        _, b = _run(data, ngen=3, seed=11)
        assert [e["gen"] for e in a.generation_trace] == [e["gen"] for e in b.generation_trace]
        assert a.best_a["gen"] == b.best_a["gen"]
        assert a.best_a["mvr"] == pytest.approx(b.best_a["mvr"])
