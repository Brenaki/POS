"""Tests for the schema v2 recorder (ADR 0020).

Covers the conformance gate, per-dataset measure selection, artifact layout,
resume, and the aggregation the published tables are read from.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pos.oracle.profiles import materialize_profile
from pos.oracle.run_recorder_v2 import DEFAULT_DATASETS, record_run_v2
from pos.oracle.run_records_v2 import per_dataset_summary_v2


@pytest.fixture
def cfg():
    c = materialize_profile("monteiro-2023-dt-compatible-v1", scale="smoke")
    c["des_methods"] = []
    c["datasets"] = ["Wine"]
    c["_selected_measures"] = ["F1", "N1"]
    c["measure_selection"] = "preset"  # skip the 10x100 vote in tests
    c["jobs"] = 1
    return c


class TestDefaults:
    def test_default_datasets_are_the_paper_28(self):
        assert len(DEFAULT_DATASETS) == 28
        assert DEFAULT_DATASETS[0] == "Australian"


class TestRecordRun:
    def test_writes_summary_and_manifest(self, cfg, tmp_path):
        manifest = record_run_v2(cfg, tmp_path)
        assert (tmp_path / "summary.csv").exists()
        assert (tmp_path / "run_manifest.json").exists()
        assert manifest["n_summary_rows"] == cfg["n_replications"] * 3

    def test_manifest_carries_conformance_and_provenance(self, cfg, tmp_path):
        record_run_v2(cfg, tmp_path)
        m = json.loads((tmp_path / "run_manifest.json").read_text())
        assert m["schema_version"] == 2
        assert m["profile"] == "monteiro-2023-dt-compatible-v1"
        assert m["conformance"]["Wine"]["conformance_status"] == "conformant"
        assert m["conformance"]["Wine"]["snapshot_sha256"].startswith("sha256:")
        assert "git_sha" in m

    def test_fold_artifacts_are_written_per_treatment(self, cfg, tmp_path):
        record_run_v2(cfg, tmp_path)
        fold = tmp_path / "Wine" / "fold_0"
        for t in ("pgdcs_a", "pgdcs_d", "bagging"):
            assert (fold / f"fold_manifest_{t}.json").exists()
            assert (fold / f"correctness_matrix_{t}.npy").exists()

    def test_summary_has_no_soft_columns(self, cfg, tmp_path):
        record_run_v2(cfg, tmp_path)
        cols = pd.read_csv(tmp_path / "summary.csv").columns
        for bad in ("mean_probs", "soft_fusion_rule", "mean_decision_norm", "soft_acc"):
            assert bad not in cols

    def test_summary_has_both_oracle_readings(self, cfg, tmp_path):
        record_run_v2(cfg, tmp_path)
        cols = set(pd.read_csv(tmp_path / "summary.csv").columns)
        assert {"oracle_1", "oracle_greedy_1", "oracle_classic",
                "oracle_curve_json", "oracle_greedy_curve_json"} <= cols


class TestConformanceGate:
    def test_mismatched_dataset_is_skipped_by_default(self, cfg, tmp_path):
        cfg["datasets"] = ["P2"]
        manifest = record_run_v2(cfg, tmp_path)
        assert manifest["n_summary_rows"] == 0
        assert manifest["skipped_datasets"][0]["dataset"] == "P2"

    def test_allow_unverified_data_lets_it_run(self, cfg, tmp_path):
        cfg["datasets"] = ["P2"]
        cfg["allow_unverified_data"] = True
        cfg["treatments"] = ["bagging"]
        manifest = record_run_v2(cfg, tmp_path)
        assert manifest["n_summary_rows"] == cfg["n_replications"]

    def test_gate_reference_is_recorded_per_dataset(self, cfg, tmp_path):
        manifest = record_run_v2(cfg, tmp_path)
        assert manifest["conformance"]["Wine"]["gate_reference"] == "monteiro-2023-table-2"


class TestPerProfileGate:
    """ADR 0021: pos-oracle-v2 gates on its own snapshot catalog."""

    @pytest.fixture
    def pos_cfg(self):
        c = materialize_profile("pos-oracle-v2", scale="smoke")
        c.update(des_methods=[], datasets=["P2"], treatments=["bagging"], jobs=1)
        return c

    def test_p2_runs_under_pos_without_the_flag(self, pos_cfg, tmp_path):
        manifest = record_run_v2(pos_cfg, tmp_path)
        assert manifest["n_summary_rows"] == pos_cfg["n_folds"]
        assert "skipped_datasets" not in manifest
        assert manifest["comparable"] == pos_cfg["comparable"]
        gate = manifest["conformance"]["P2"]
        assert gate["gate_reference"] == "pos-catalog"
        assert gate["gate_status"] == "conformant"


class TestIncrementalFlush:
    def test_rows_survive_a_failure_on_a_later_dataset(self, cfg, tmp_path, monkeypatch):
        """summary.csv is flushed per dataset: a crash on dataset 2 must not
        lose dataset 1, which resume would then skip without recovering."""
        cfg["datasets"] = ["Wine", "Banana"]
        import pos.oracle.run_recorder_v2 as rec
        real = rec.run_profile_experiment
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("boom")
            return real(*args, **kwargs)

        monkeypatch.setattr(rec, "run_profile_experiment", flaky)
        manifest = record_run_v2(cfg, tmp_path)
        df = pd.read_csv(tmp_path / "summary.csv")
        assert set(df["dataset"]) == {"Wine"}
        assert len(df) == cfg["n_replications"] * 3
        assert manifest["errors"][0]["dataset"] == "Banana"


class TestResume:
    def test_resume_adds_no_duplicate_rows(self, cfg, tmp_path):
        first = record_run_v2(cfg, tmp_path)
        again = record_run_v2(cfg, tmp_path, resume=True)
        assert again["n_summary_rows"] == first["n_summary_rows"]


class TestAggregation:
    def test_per_dataset_summary_groups_by_treatment(self, cfg, tmp_path):
        record_run_v2(cfg, tmp_path)
        agg = per_dataset_summary_v2(pd.read_csv(tmp_path / "summary.csv"))
        assert {r["treatment"] for r in agg} == {"pgdcs_a", "pgdcs_d", "bagging"}
        assert all(r["n_splits"] == cfg["n_replications"] for r in agg)

    def test_aggregation_reports_selected_generation(self, cfg, tmp_path):
        record_run_v2(cfg, tmp_path)
        agg = per_dataset_summary_v2(pd.read_csv(tmp_path / "summary.csv"))
        a = next(r for r in agg if r["treatment"] == "pgdcs_a")
        assert not np.isnan(a["pgdcs_best_generation_mean"])

    def test_empty_frame_aggregates_to_nothing(self):
        assert per_dataset_summary_v2(pd.DataFrame()) == []
