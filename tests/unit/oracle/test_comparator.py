"""Tests for the reference comparator (ADR 2020 §10)."""

from __future__ import annotations

import numpy as np
import pytest

from pos.oracle.comparator import compare_reference, build_comparison_row


class TestBuildComparisonRow:
    def test_returns_all_required_fields(self):
        row = build_comparison_row(
            dataset="Wine", treatment="pgdcs_a", method="mvr",
            published_mean=99.3, published_std=1.0,
            pos_mean=98.5, pos_std=2.1,
            dataset_conformance="conformant",
            protocol_compatibility="compatible_modern",
        )
        assert row["dataset"] == "Wine"
        assert row["treatment"] == "pgdcs_a"
        assert row["method"] == "mvr"
        assert row["published_mean"] == 99.3
        assert row["published_std"] == 1.0
        assert row["pos_mean"] == 98.5
        assert row["pos_std"] == 2.1
        assert row["delta_percentage_points"] == pytest.approx(-0.8)
        assert row["dataset_conformance"] == "conformant"
        assert row["protocol_compatibility"] == "compatible_modern"
        assert "pos_ci95" in row
        assert "source_table" in row

    def test_delta_is_pos_minus_published(self):
        row = build_comparison_row(
            dataset="Wine", treatment="bagging", method="mvr",
            published_mean=95.2, published_std=4.4,
            pos_mean=93.0, pos_std=3.0,
            dataset_conformance="conformant",
            protocol_compatibility="compatible_modern",
        )
        assert row["delta_percentage_points"] == pytest.approx(-2.2)

    def test_published_std_null_for_oracle(self):
        row = build_comparison_row(
            dataset="Wine", treatment="pgdcs_a", method="oracle",
            published_mean=100, published_std=None,
            pos_mean=99.5, pos_std=0.5,
            dataset_conformance="conformant",
            protocol_compatibility="compatible_modern",
        )
        assert row["published_std"] is None
        assert row["source_table"] == "7"

    def test_ci95_computed_from_pos_std(self):
        row = build_comparison_row(
            dataset="Wine", treatment="pgdcs_a", method="mvr",
            published_mean=99.3, published_std=1.0,
            pos_mean=98.5, pos_std=2.1,
            dataset_conformance="conformant",
            protocol_compatibility="compatible_modern",
            n_pos=20,
        )
        ci = row["pos_ci95"]
        assert ci is not None
        assert "low" in ci
        assert "high" in ci


class TestCompareReference:
    def test_returns_rows_for_each_dataset_treatment_method(self):
        # minimal fake POS results
        pos_results = {
            ("Wine", "pgdcs_a", "mvr"): {"mean": 98.5, "std": 2.1, "n": 20},
            ("Wine", "pgdcs_d", "mvr"): {"mean": 96.0, "std": 2.5, "n": 20},
            ("Wine", "bagging", "mvr"): {"mean": 93.0, "std": 3.0, "n": 20},
            ("Wine", "pgdcs_a", "oracle"): {"mean": 99.5, "std": 0.5, "n": 20},
        }
        conformance = {"Wine": "conformant"}
        rows = compare_reference(
            pos_results=pos_results,
            conformance=conformance,
            protocol_compatibility="compatible_modern",
        )
        assert len(rows) == 4
        treatments = {r["treatment"] for r in rows}
        assert treatments == {"pgdcs_a", "pgdcs_d", "bagging"}

    def test_skips_non_conformant_datasets(self):
        pos_results = {
            ("P2", "pgdcs_a", "mvr"): {"mean": 90.0, "std": 2.0, "n": 20},
        }
        conformance = {"P2": "mismatch"}
        rows = compare_reference(
            pos_results=pos_results,
            conformance=conformance,
            protocol_compatibility="compatible_modern",
        )
        # mismatch datasets are excluded from the reference report
        assert len(rows) == 0

    def test_handles_missing_pos_results_gracefully(self):
        conformance = {"Wine": "conformant"}
        rows = compare_reference(
            pos_results={},
            conformance=conformance,
            protocol_compatibility="compatible_modern",
        )
        assert len(rows) == 0