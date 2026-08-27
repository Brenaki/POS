"""Tests for order-independence (T16), schema v1 loader (T17), and smoke (T18).

T16: seeds and results are independent of dataset/treatment ordering.
T17: schema v1 runs remain readable; legacy `pgdcs` maps to `pgdcs_d`.
T18: smoke of both profiles concludes and generates all manifests.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

from pos.oracle.profiles import materialize_profile
from pos.oracle.profile_runner import run_profile_experiment
from pos.oracle.summary_v2 import load_summary_v1, empty_summary_columns_v2


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
    M = config["M"]
    pool = _fake_pool(M, n_features=X_tr.shape[1], seed=42)
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


class TestOrderIndependence:
    """T16: results don't depend on the order datasets or treatments are run."""

    def test_treatment_order_does_not_change_results(self, small_data):
        X, y = small_data
        cfg = materialize_profile("pos-oracle-v2", scale="smoke")
        cfg["des_methods"] = []
        cfg["treatments"] = ["bagging", "pgdcs_a", "pgdcs_d"]
        rows_a = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=_fake_pgdcs_fn)
        cfg["treatments"] = ["pgdcs_d", "pgdcs_a", "bagging"]
        rows_b = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=_fake_pgdcs_fn)
        # same classic oracle per (fold, treatment), regardless of order
        for r_a in rows_a:
            match = next(r_b for r_b in rows_b
                         if r_b["fold"] == r_a["fold"]
                         and r_b["treatment"] == r_a["treatment"])
            assert r_a["oracle_classic"] == match["oracle_classic"]


class TestSchemaV1Loader:
    """T17: schema v1 runs remain readable."""

    def test_load_v1_maps_pgdcs_to_pgdcs_d(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame([{
            "dataset": "Wine", "fold": 0, "mode": "pgdcs",
            "M": 100, "oracle_M": 0.99, "majority_vote": 0.95,
            "mean_probs": 0.93, "soft_fusion_rule": "mean_probs",
        }])
        path = tmp_path / "summary.csv"
        df.to_csv(path, index=False)
        rows = load_summary_v1(path)
        assert rows[0]["treatment"] == "pgdcs_d"
        assert rows[0]["mode"] == "pgdcs"

    def test_load_v1_preserves_soft_columns(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame([{
            "dataset": "Wine", "fold": 0, "mode": "bagging",
            "M": 100, "oracle_M": 0.99, "majority_vote": 0.95,
            "mean_probs": 0.93, "soft_fusion_rule": "mean_probs",
        }])
        path = tmp_path / "summary.csv"
        df.to_csv(path, index=False)
        rows = load_summary_v1(path)
        assert "mean_probs" in rows[0]
        assert "soft_fusion_rule" in rows[0]

    def test_load_v1_non_pgdcs_keeps_treatment(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame([{
            "dataset": "Wine", "fold": 0, "mode": "bagging",
            "M": 100, "oracle_M": 0.99, "majority_vote": 0.95,
        }])
        path = tmp_path / "summary.csv"
        df.to_csv(path, index=False)
        rows = load_summary_v1(path)
        assert rows[0]["treatment"] == "bagging"


class TestEmptySchemaV2:
    def test_empty_header_has_no_soft_columns(self):
        cols = empty_summary_columns_v2()
        assert "mean_probs" not in cols
        assert "soft_fusion_rule" not in cols

    def test_empty_header_has_profile(self):
        cols = empty_summary_columns_v2()
        assert "profile" in cols
        assert "treatment" in cols


class TestSmokeIntegration:
    """T18: smoke of both profiles produces rows for all treatments."""

    def test_pos_oracle_smoke_produces_all_manifests(self, small_data):
        X, y = small_data
        cfg = materialize_profile("pos-oracle-v2", scale="smoke")
        cfg["des_methods"] = []
        rows = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=_fake_pgdcs_fn)
        assert len(rows) == cfg["n_folds"] * len(cfg["treatments"])
        for r in rows:
            assert r["profile"] == "pos-oracle-v2"
            assert "oracle_1" in r
            assert r["oracle_1"] is not None  # at least 1 classifier exists

    def test_monteiro_smoke_produces_all_manifests(self, small_data):
        X, y = small_data
        cfg = materialize_profile("monteiro-2023-dt-compatible-v1", scale="smoke")
        cfg["des_methods"] = []
        rows = run_profile_experiment(cfg, X, y, "test", build_pgdcs_fn=_fake_pgdcs_fn)
        assert len(rows) == cfg["n_replications"] * len(cfg["treatments"])
        for r in rows:
            assert r["profile"] == "monteiro-2023-dt-compatible-v1"
            assert "replication" in r
            assert r["oracle_1"] is not None