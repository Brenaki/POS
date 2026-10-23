"""v1 (fast measures) vs v2 (ECoL 0.3.0 measures) run comparison (ADR 0024)."""

from __future__ import annotations

import pandas as pd
import pytest

from pos.analysis.backend_compare import (
    chosen_measures,
    paired_by_dataset,
    table4_agreement,
    table4_winners,
)


def _summary(rows):
    return pd.DataFrame(rows, columns=["dataset", "treatment", "replication",
                                       "pgdcs_types", "mv_acc"])


def test_chosen_measures_reads_one_pair_per_dataset():
    s = _summary([("A", "pgdcs_a", 0, "F3|N2", 0.9), ("A", "bagging", 0, None, 0.8),
                  ("B", "pgdcs_d", 0, "F4|T1", 0.7)])
    assert chosen_measures(s) == {"A": ("F3", "N2"), "B": ("F4", "T1")}


def test_table4_winners_keep_ties_per_family():
    votes = {"F1": 2, "F1v": 2, "F2": 0, "F3": 1, "F4": 0,
             "N1": 0, "N2": 5, "N3": 0, "N4": 1, "T1": 0, "LSC": 0}
    assert table4_winners(votes) == ({"F1", "F1v"}, {"N2"})


def test_table4_agreement_counts_hits_per_family():
    table = {"A": {"F1": 3, "F1v": 0, "F2": 0, "F3": 0, "F4": 0,
                   "N1": 0, "N2": 0, "N3": 0, "N4": 0, "T1": 4, "LSC": 0}}
    agree = table4_agreement({"A": ("F1", "N2"), "Z": ("F1", "N1")}, table)
    assert agree.to_dict("records") == [
        {"dataset": "A", "chosen": "F1|N2", "table4": "F1|T1",
         "overlapping_hit": True, "neighborhood_hit": False}]


def test_paired_by_dataset_averages_replications_then_pairs():
    a = _summary([("A", "pgdcs_a", 0, "x", 0.8), ("A", "pgdcs_a", 1, "x", 1.0),
                  ("B", "pgdcs_a", 0, "x", 0.5), ("C", "pgdcs_a", 0, "x", 0.4)])
    b = _summary([("A", "pgdcs_a", 0, "x", 0.7), ("B", "pgdcs_a", 0, "x", 0.6),
                  ("C", "bagging", 0, "x", 0.3)])
    out = paired_by_dataset(a, b, "pgdcs_a", "pgdcs_a")
    assert out["n"] == 2
    assert out["mean_delta_pp"] == pytest.approx(((0.9 - 0.7) + (0.5 - 0.6)) / 2 * 100)
    assert out["wins"] == 1
    assert 0 <= out["p"] <= 1
