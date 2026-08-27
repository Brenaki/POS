"""Tests for the summary.csv v2 schema (ADR 2020 T12).

The v2 schema adds profile/replication identification, oracle checkpoints
1,5,10,...,100, and removes the soft_fusion / mean_probs columns.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from pos.oracle.summary_v2 import (
    ORACLE_CHECKPOINTS_V2,
    build_summary_row_v2,
    empty_summary_columns_v2,
)


def _fake_metrics():
    """A minimal metrics dict that build_summary_row_v2 accepts."""
    curve = list(np.linspace(0.5, 0.95, 100))
    return {
        "n_classifiers": 100,
        "oracle_curve": curve,
        "majority_vote": 0.82,
        "majority_vote_metrics": {
            "acc": 0.82, "precision": 0.81, "recall": 0.80,
            "f1": 0.80, "balanced_acc": 0.79,
        },
        "individual_accuracies": [0.7] * 100,
        "double_fault_mean": 0.15,
        "des": {"ola": {"acc": 0.75}, "lca": {"acc": 0.73}},
        "pgdcs_types": ["F1", "N1"],
        "pgdcs_best_generation": 14,
        "pgdcs_best_score": 0.82,
        "pgdcs_best_metric": "mvr",
    }


class TestSchemaV2Columns:
    def test_has_profile_and_treatment(self):
        row = build_summary_row_v2(
            profile="pos-oracle-v2", dataset="Wine", fold=0,
            treatment="pgdcs_a", metrics=_fake_metrics(), n_test=36,
        )
        assert row["profile"] == "pos-oracle-v2"
        assert row["treatment"] == "pgdcs_a"

    def test_has_dataset_and_fold(self):
        row = build_summary_row_v2(
            profile="pos-oracle-v2", dataset="Wine", fold=3,
            treatment="bagging", metrics=_fake_metrics(), n_test=36,
        )
        assert row["dataset"] == "Wine"
        assert row["fold"] == 3

    def test_has_replication_for_monteiro(self):
        row = build_summary_row_v2(
            profile="monteiro-2023-dt-compatible-v1", dataset="Wine",
            fold=7, treatment="pgdcs_d", metrics=_fake_metrics(), n_test=45,
        )
        assert row["replication"] == 7

    def test_oracle_checkpoints_present(self):
        row = build_summary_row_v2(
            profile="pos-oracle-v2", dataset="Wine", fold=0,
            treatment="pgdcs_a", metrics=_fake_metrics(), n_test=36,
        )
        for n in ORACLE_CHECKPOINTS_V2:
            assert f"oracle_{n}" in row, f"missing oracle_{n}"

    def test_oracle_classic_equals_curve_first(self):
        metrics = _fake_metrics()
        row = build_summary_row_v2(
            profile="pos-oracle-v2", dataset="Wine", fold=0,
            treatment="pgdcs_a", metrics=metrics, n_test=36,
        )
        assert row["oracle_classic"] == metrics["oracle_curve"][0]

    def test_oracle_curve_json_present(self):
        row = build_summary_row_v2(
            profile="pos-oracle-v2", dataset="Wine", fold=0,
            treatment="pgdcs_a", metrics=_fake_metrics(), n_test=36,
        )
        curve = json.loads(row["oracle_curve_json"])
        assert len(curve) == 100

    def test_has_majority_vote_metrics(self):
        row = build_summary_row_v2(
            profile="pos-oracle-v2", dataset="Wine", fold=0,
            treatment="pgdcs_a", metrics=_fake_metrics(), n_test=36,
        )
        assert row["mv_acc"] == 0.82
        assert "mv_precision" in row
        assert "mv_recall" in row
        assert "mv_f1" in row
        assert "mv_balanced_acc" in row

    def test_has_double_fault_and_individual_acc(self):
        row = build_summary_row_v2(
            profile="pos-oracle-v2", dataset="Wine", fold=0,
            treatment="pgdcs_a", metrics=_fake_metrics(), n_test=36,
        )
        assert "double_fault_mean" in row
        assert "mean_individual_acc" in row

    def test_has_pgdcs_fields(self):
        row = build_summary_row_v2(
            profile="pos-oracle-v2", dataset="Wine", fold=0,
            treatment="pgdcs_a", metrics=_fake_metrics(), n_test=36,
        )
        assert row["pgdcs_types"] == "F1|N1"
        assert row["pgdcs_best_generation"] == 14
        assert row["pgdcs_best_score"] == 0.82
        assert row["pgdcs_best_metric"] == "mvr"


class TestSchemaV2NoSoftColumns:
    SOFT_COLUMNS = (
        "mean_probs", "soft_fusion_rule", "mean_decision_norm",
        "soft_acc", "soft_precision", "soft_recall", "soft_f1",
        "soft_balanced_acc", "mean_probs_metrics",
    )

    def test_no_soft_columns_in_row(self):
        row = build_summary_row_v2(
            profile="pos-oracle-v2", dataset="Wine", fold=0,
            treatment="pgdcs_a", metrics=_fake_metrics(), n_test=36,
        )
        for col in self.SOFT_COLUMNS:
            assert col not in row, f"soft column {col} should be removed in v2"

    def test_no_soft_columns_in_empty_header(self):
        cols = empty_summary_columns_v2()
        for col in self.SOFT_COLUMNS:
            assert col not in cols, f"soft column {col} in empty header"

    def test_empty_header_has_all_oracle_checkpoints(self):
        cols = empty_summary_columns_v2()
        for n in ORACLE_CHECKPOINTS_V2:
            assert f"oracle_{n}" in cols