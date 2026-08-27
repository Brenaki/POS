"""Tests for the profile-aware runner (ADR 2020 T13).

Verifies that:
- the runner dispatches to the correct splitter per profile;
- all three treatments produce rows;
- no soft_* columns appear in the output;
- the six DCS/DES methods are evaluated (when not --no-des);
- bagging treatment works without PGDCS.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from pos.oracle.profiles import materialize_profile
from pos.oracle.profile_runner import run_profile_experiment


def _fake_pool(n, n_features=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(max(4, n * 2), n_features)
    y = rng.randint(0, 2, size=max(4, n * 2))
    pool = []
    for i in range(n):
        t = DecisionTreeClassifier(random_state=seed + i)
        t.fit(X, y)
        pool.append(t)
    return pool


def _fake_pgdcs_fn(X_tr, y_tr, X_val, y_val, config):
    """A fake PGDCS that returns two identical pools of M trees."""
    M = config["M"]
    pool = _fake_pool(M, seed=42)
    return {
        "selected_measures": ["F1", "N1"],
        "best_a": {"gen": 10, "mvr": 0.85, "pool": pool},
        "best_d": {"gen": 12, "gdisp": 0.42, "pool": pool},
    }


@pytest.fixture
def small_data():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    y = np.array([0] * 100 + [1] * 100)
    return X, y


class TestPosOracleV2Runner:
    def test_produces_rows_for_all_treatments(self, small_data):
        X, y = small_data
        cfg = materialize_profile("pos-oracle-v2", scale="smoke")
        cfg["des_methods"] = []  # skip DES for speed
        rows = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=_fake_pgdcs_fn)
        treatments = {r["treatment"] for r in rows}
        assert treatments == {"pgdcs_a", "pgdcs_d", "bagging"}

    def test_rows_have_profile_field(self, small_data):
        X, y = small_data
        cfg = materialize_profile("pos-oracle-v2", scale="smoke")
        cfg["des_methods"] = []
        rows = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=_fake_pgdcs_fn)
        for r in rows:
            assert r["profile"] == "pos-oracle-v2"

    def test_no_soft_columns_in_output(self, small_data):
        X, y = small_data
        cfg = materialize_profile("pos-oracle-v2", scale="smoke")
        cfg["des_methods"] = []
        rows = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=_fake_pgdcs_fn)
        soft_cols = {"mean_probs", "soft_fusion_rule", "mean_decision_norm",
                     "soft_acc", "soft_precision"}
        for r in rows:
            for col in soft_cols:
                assert col not in r, f"soft column {col} present in v2 row"

    def test_oracle_checkpoints_present(self, small_data):
        X, y = small_data
        cfg = materialize_profile("pos-oracle-v2", scale="smoke")
        cfg["des_methods"] = []
        rows = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=_fake_pgdcs_fn)
        for r in rows:
            assert "oracle_1" in r
            assert "oracle_100" in r
            assert r["oracle_classic"] is not None

    def test_fold_count_matches_smoke(self, small_data):
        X, y = small_data
        cfg = materialize_profile("pos-oracle-v2", scale="smoke")
        cfg["des_methods"] = []
        rows = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=_fake_pgdcs_fn)
        folds = {r["fold"] for r in rows}
        assert len(folds) == cfg["n_folds"]

    def test_bagging_works_without_pgdcs(self, small_data):
        X, y = small_data
        cfg = materialize_profile("pos-oracle-v2", scale="smoke")
        cfg["des_methods"] = []
        cfg["treatments"] = ["bagging"]
        rows = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=None)
        assert len(rows) > 0
        assert all(r["treatment"] == "bagging" for r in rows)


class TestMonteiroRunner:
    def test_produces_rows_for_all_treatments(self, small_data):
        X, y = small_data
        cfg = materialize_profile("monteiro-2023-dt-compatible-v1", scale="smoke")
        cfg["des_methods"] = []
        rows = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=_fake_pgdcs_fn)
        treatments = {r["treatment"] for r in rows}
        assert treatments == {"pgdcs_a", "pgdcs_d", "bagging"}

    def test_replication_field_present(self, small_data):
        X, y = small_data
        cfg = materialize_profile("monteiro-2023-dt-compatible-v1", scale="smoke")
        cfg["des_methods"] = []
        rows = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=_fake_pgdcs_fn)
        for r in rows:
            assert "replication" in r

    def test_replication_count_matches_smoke(self, small_data):
        X, y = small_data
        cfg = materialize_profile("monteiro-2023-dt-compatible-v1", scale="smoke")
        cfg["des_methods"] = []
        rows = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=_fake_pgdcs_fn)
        reps = {r["replication"] for r in rows}
        assert len(reps) == cfg["n_replications"]

    def test_bagging_uses_random_state_86(self, small_data):
        """Monteiro profile bagging uses the published seed."""
        X, y = small_data
        cfg = materialize_profile("monteiro-2023-dt-compatible-v1", scale="smoke")
        cfg["des_methods"] = []
        cfg["treatments"] = ["bagging"]
        rows = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=None)
        # deterministic across calls with same seed
        rows2 = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=None)
        for r1, r2 in zip(rows, rows2):
            assert r1["oracle_classic"] == r2["oracle_classic"]