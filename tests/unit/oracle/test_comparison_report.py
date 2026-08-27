"""Tests for the summary -> published-table comparison (ADR 0020 §10)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pos.oracle.comparison_report import (
    build_report,
    conformance_from_manifest,
    pos_results_from_summary,
)


def _summary():
    return pd.DataFrame([
        {"dataset": "Wine", "treatment": "pgdcs_a", "mv_acc": 0.99,
         "oracle_classic": 1.0, "pgdcs_best_generation": 12},
        {"dataset": "Wine", "treatment": "pgdcs_a", "mv_acc": 0.97,
         "oracle_classic": 1.0, "pgdcs_best_generation": 14},
        {"dataset": "Wine", "treatment": "bagging", "mv_acc": 0.95,
         "oracle_classic": 1.0, "pgdcs_best_generation": None},
    ])


class TestAggregation:
    def test_mvr_is_a_percentage_mean(self):
        out = pos_results_from_summary(_summary())
        assert out[("Wine", "pgdcs_a", "mvr")]["mean"] == pytest.approx(98.0)

    def test_mvr_std_is_the_sample_std(self):
        out = pos_results_from_summary(_summary())
        assert out[("Wine", "pgdcs_a", "mvr")]["std"] == pytest.approx(1.4142, abs=1e-3)

    def test_oracle_uses_at_least_one_correct(self):
        """Table 7 is the classic pool oracle: at least one classifier right."""
        out = pos_results_from_summary(_summary())
        assert out[("Wine", "pgdcs_a", "oracle")]["mean"] == pytest.approx(100.0)

    def test_generation_is_not_scaled(self):
        out = pos_results_from_summary(_summary())
        assert out[("Wine", "pgdcs_a", "generation")]["mean"] == pytest.approx(13.0)

    def test_bagging_has_no_generation_entry(self):
        out = pos_results_from_summary(_summary())
        assert ("Wine", "bagging", "generation") not in out

    def test_n_is_the_number_of_replications(self):
        out = pos_results_from_summary(_summary())
        assert out[("Wine", "pgdcs_a", "mvr")]["n"] == 2

    def test_empty_frame_is_empty(self):
        assert pos_results_from_summary(pd.DataFrame()) == {}


class TestManifest:
    def test_reads_status_per_dataset(self, tmp_path):
        (tmp_path / "run_manifest.json").write_text(json.dumps(
            {"conformance": {"Wine": {"conformance_status": "conformant"},
                             "P2": {"conformance_status": "mismatch"}}}))
        assert conformance_from_manifest(tmp_path) == {
            "Wine": "conformant", "P2": "mismatch"}

    def test_missing_manifest_is_empty(self, tmp_path):
        assert conformance_from_manifest(tmp_path) == {}


class TestBuildReport:
    def _run_dir(self, tmp_path, status):
        _summary().to_csv(tmp_path / "summary.csv", index=False)
        (tmp_path / "run_manifest.json").write_text(json.dumps(
            {"conformance": {"Wine": {"conformance_status": status}}}))
        return tmp_path

    def test_reports_every_comparable_metric(self, tmp_path):
        rows = build_report(self._run_dir(tmp_path, "conformant"))
        assert {r["method"] for r in rows} == {"mvr", "oracle", "generation"}

    def test_delta_is_pos_minus_published(self, tmp_path):
        rows = build_report(self._run_dir(tmp_path, "conformant"))
        mvr = next(r for r in rows
                   if r["method"] == "mvr" and r["treatment"] == "bagging")
        assert mvr["published_mean"] == 95.2
        assert mvr["delta_percentage_points"] == pytest.approx(-0.2)

    def test_approved_with_note_is_included(self, tmp_path):
        rows = build_report(self._run_dir(tmp_path, "approved_with_note"))
        assert rows and all(r["dataset_conformance"] == "approved_with_note"
                            for r in rows)

    def test_mismatch_is_excluded_by_default(self, tmp_path):
        assert build_report(self._run_dir(tmp_path, "mismatch")) == []

    def test_mismatch_is_included_when_allowed(self, tmp_path):
        rows = build_report(self._run_dir(tmp_path, "mismatch"),
                            allow_unverified=True)
        assert rows
