"""Tests for the PGDCS generation tracker (ADR 0020 T6-T9).

The tracker collects (generation, MVR accuracy, Gdisp, pool) for gen=1..20
and selects the best generation per criterion. Tests cover:
- T6: gen 0 is evaluated but never selected; 20 candidates
- T7: PGDCS_A = max MVR, PGDCS_D = max Gdisp, tie -> oldest
- T8: fitted classifiers are kept from the best generation
- T9: both pools have exactly 100 classifiers
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from pos.pool.pgdcs_tracker import PgdcsTracker


def _fake_pool(n, seed=0):
    """A pool of n fitted dummy trees (trained on 2-point data)."""
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    pool = []
    for i in range(n):
        t = DecisionTreeClassifier(random_state=seed + i)
        t.fit(X, y)
        pool.append(t)
    return pool


class TestTrackerInit:
    def test_starts_empty(self):
        t = PgdcsTracker()
        assert t.generation_trace == []
        assert t.best_a is None
        assert t.best_d is None


class TestGenerationRecording:
    def test_records_generation_with_mvr_and_gdisp(self):
        t = PgdcsTracker()
        t.record(gen=1, mvr=0.8, gdisp=0.3, pool=_fake_pool(100))
        assert len(t.generation_trace) == 1
        assert t.generation_trace[0]["gen"] == 1
        assert t.generation_trace[0]["mvr"] == 0.8
        assert t.generation_trace[0]["gdisp"] == 0.3

    def test_gen_zero_is_evaluated_but_not_a_candidate(self):
        t = PgdcsTracker()
        t.record(gen=0, mvr=0.5, gdisp=0.1, pool=_fake_pool(100))
        t.record(gen=1, mvr=0.7, gdisp=0.2, pool=_fake_pool(100))
        t.finalize()
        assert t.generation_trace[0]["gen"] == 0  # recorded
        assert t.best_a["gen"] == 1  # gen 0 not selected
        assert t.best_d["gen"] == 1

    def test_exactly_20_candidate_generations(self):
        t = PgdcsTracker()
        for g in range(0, 21):  # gen 0..20
            t.record(gen=g, mvr=0.5 + g * 0.01, gdisp=0.1 + g * 0.01,
                     pool=_fake_pool(100))
        t.finalize()
        candidates = [e for e in t.generation_trace if e["gen"] > 0]
        assert len(candidates) == 20


class TestPgdcsASelection:
    def test_pgdcs_a_picks_max_mvr(self):
        t = PgdcsTracker()
        t.record(gen=1, mvr=0.7, gdisp=0.5, pool=_fake_pool(100))
        t.record(gen=2, mvr=0.9, gdisp=0.2, pool=_fake_pool(100))
        t.record(gen=3, mvr=0.8, gdisp=0.6, pool=_fake_pool(100))
        t.finalize()
        assert t.best_a["gen"] == 2
        assert t.best_a["mvr"] == pytest.approx(0.9)

    def test_pgdcs_a_tie_keeps_oldest(self):
        t = PgdcsTracker()
        t.record(gen=1, mvr=0.94, gdisp=0.5, pool=_fake_pool(100))
        t.record(gen=8, mvr=0.94, gdisp=0.5, pool=_fake_pool(100))
        t.finalize()
        assert t.best_a["gen"] == 1  # oldest wins


class TestPgdcsDSelection:
    def test_pgdcs_d_picks_max_gdisp(self):
        t = PgdcsTracker()
        t.record(gen=1, mvr=0.7, gdisp=0.5, pool=_fake_pool(100))
        t.record(gen=2, mvr=0.9, gdisp=0.2, pool=_fake_pool(100))
        t.record(gen=3, mvr=0.8, gdisp=0.6, pool=_fake_pool(100))
        t.finalize()
        assert t.best_d["gen"] == 3
        assert t.best_d["gdisp"] == pytest.approx(0.6)

    def test_pgdcs_d_tie_keeps_oldest(self):
        t = PgdcsTracker()
        t.record(gen=1, mvr=0.5, gdisp=0.42, pool=_fake_pool(100))
        t.record(gen=5, mvr=0.5, gdisp=0.42, pool=_fake_pool(100))
        t.finalize()
        assert t.best_d["gen"] == 1


class TestFittedClassifiers:
    def test_best_a_pool_is_kept(self):
        pool_a = _fake_pool(100, seed=10)
        t = PgdcsTracker()
        t.record(gen=1, mvr=0.9, gdisp=0.2, pool=pool_a)
        t.record(gen=2, mvr=0.8, gdisp=0.6, pool=_fake_pool(100, seed=20))
        t.finalize()
        assert t.best_a["pool"] is pool_a

    def test_best_d_pool_is_kept(self):
        pool_d = _fake_pool(100, seed=30)
        t = PgdcsTracker()
        t.record(gen=1, mvr=0.9, gdisp=0.2, pool=_fake_pool(100, seed=10))
        t.record(gen=2, mvr=0.8, gdisp=0.6, pool=pool_d)
        t.finalize()
        assert t.best_d["pool"] is pool_d

    def test_pools_are_distinct_when_different_generations_win(self):
        t = PgdcsTracker()
        t.record(gen=1, mvr=0.9, gdisp=0.2, pool=_fake_pool(100, seed=10))
        t.record(gen=2, mvr=0.8, gdisp=0.6, pool=_fake_pool(100, seed=20))
        t.finalize()
        assert t.best_a["pool"] is not t.best_d["pool"]


class TestPoolSize:
    def test_both_pools_have_100_classifiers(self):
        t = PgdcsTracker()
        for g in range(0, 21):
            t.record(gen=g, mvr=0.5 + g * 0.01, gdisp=0.1 + g * 0.01,
                     pool=_fake_pool(100, seed=g))
        t.finalize()
        assert len(t.best_a["pool"]) == 100
        assert len(t.best_d["pool"]) == 100


class TestFinalizeGuards:
    def test_finalize_with_no_candidates_raises(self):
        t = PgdcsTracker()
        t.record(gen=0, mvr=0.5, gdisp=0.1, pool=_fake_pool(100))
        with pytest.raises(ValueError, match="no candidate generations"):
            t.finalize()

    def test_finalize_with_one_candidate_works(self):
        t = PgdcsTracker()
        t.record(gen=0, mvr=0.5, gdisp=0.1, pool=_fake_pool(100))
        t.record(gen=1, mvr=0.7, gdisp=0.3, pool=_fake_pool(100))
        t.finalize()
        assert t.best_a["gen"] == 1
        assert t.best_d["gen"] == 1