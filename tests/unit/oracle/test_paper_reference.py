"""Tests for paper reference table fixtures (ADR 2020 §9).

Fixtures contain the Decision Tree tables extracted from Monteiro 2023:
- Table 4: measure votes
- Table 6: selected generations
- Table 7: Oracle (mean only, no std)
- Table A.10: DT MVR
- Table A.12: DT DCS
- Table A.14: DT DES
"""

from __future__ import annotations

import pytest

from pos.oracle.paper_reference import (
    DATASET_NAMES_PAPER,
    TABLE_4_VOTES,
    TABLE_6_GENERATIONS,
    TABLE_7_ORACLE,
    TABLE_A10_MVR,
    get_reference_value,
)


class TestDatasetNames:
    def test_has_28_names(self):
        assert len(DATASET_NAMES_PAPER) == 28

    def test_first_is_australian(self):
        assert DATASET_NAMES_PAPER[0] == "Australian"

    def test_last_is_wine(self):
        assert DATASET_NAMES_PAPER[-1] == "Wine"


class TestTable4Votes:
    def test_has_28_entries(self):
        assert len(TABLE_4_VOTES) == 28

    def test_wine_votes(self):
        wine = TABLE_4_VOTES["Wine"]
        assert wine["F1"] == 1
        assert wine["F3"] == 8
        assert wine["N3"] == 5

    def test_each_entry_has_11_measures(self):
        for name, votes in TABLE_4_VOTES.items():
            assert len(votes) == 11, f"{name} has {len(votes)} measures"


class TestTable6Generations:
    def test_has_28_entries(self):
        assert len(TABLE_6_GENERATIONS) == 28

    def test_wine_dt_pgdcs_a(self):
        wine = TABLE_6_GENERATIONS["Wine"]
        assert wine["dt_pgdcs_a_mean"] == 14.8
        assert wine["dt_pgdcs_a_std"] == 5.6

    def test_wine_dt_pgdcs_d(self):
        wine = TABLE_6_GENERATIONS["Wine"]
        assert wine["dt_pgdcs_d_mean"] == 12.8


class TestTable7Oracle:
    def test_has_28_entries(self):
        assert len(TABLE_7_ORACLE) == 28

    def test_wine_dt_pgdcs_a_oracle(self):
        wine = TABLE_7_ORACLE["Wine"]
        assert wine["dt_pgdcs_a"] == 100

    def test_wine_dt_bagging_oracle(self):
        wine = TABLE_7_ORACLE["Wine"]
        assert wine["dt_bagging"] == 100

    def test_no_std_field(self):
        """Table 7 publishes only mean, not std (ADR 0020 §5)."""
        wine = TABLE_7_ORACLE["Wine"]
        assert "dt_pgdcs_a_std" not in wine


class TestTableA10MVR:
    def test_has_28_entries(self):
        assert len(TABLE_A10_MVR) == 28

    def test_wine_pgdcs_a(self):
        wine = TABLE_A10_MVR["Wine"]
        assert wine["pgdcs_a_mean"] == 99.3
        assert wine["pgdcs_a_std"] == 1.0

    def test_wine_bagging(self):
        wine = TABLE_A10_MVR["Wine"]
        assert wine["bagging_mean"] == 95.2
        assert wine["bagging_std"] == 4.4


class TestGetReferenceValue:
    def test_returns_mean_and_std(self):
        result = get_reference_value("Wine", "pgdcs_a", "mvr")
        assert result["published_mean"] == 99.3
        assert result["published_std"] == 1.0
        assert result["source_table"] == "A.10"

    def test_oracle_has_null_std(self):
        result = get_reference_value("Wine", "pgdcs_a", "oracle")
        assert result["published_mean"] == 100
        assert result["published_std"] is None
        assert result["source_table"] == "7"

    def test_unknown_dataset_returns_none(self):
        result = get_reference_value("Unknown", "pgdcs_a", "mvr")
        assert result is None