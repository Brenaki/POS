"""Tests for fast-vs-ECoL agreement metrics (ADR 0023).

The GA ranks bags by complexity, so what matters is whether the fast backend
orders the bags of one dataset the way ECoL does — not the absolute values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pos.analysis.ecol_agreement import agreement_by_measure, ranking_agreement, to_dispersion


def _frame(fast, ecol, dataset="D", measure="T1"):
    return pd.DataFrame({"dataset": dataset, "bag": range(len(fast)),
                         "measure": measure, "fast": fast, "ecol": ecol})


class TestRankingAgreement:
    def test_monotone_transform_keeps_the_ranking(self):
        df = _frame([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.3, 0.35, 0.5, 0.6, 0.61, 0.9])
        out = ranking_agreement(df, k=2)
        assert out["kendall"] == pytest.approx(1.0)
        assert out["argmin_agree"] == 1.0
        assert out["topk_overlap"] == 1.0

    def test_reversed_order_disagrees(self):
        df = _frame([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1])
        out = ranking_agreement(df, k=2)
        assert out["kendall"] == pytest.approx(-1.0)
        assert out["argmin_agree"] == 0.0
        assert out["topk_overlap"] == 0.0

    def test_constant_column_gives_nan_kendall_not_a_crash(self):
        df = _frame([0.2, 0.2, 0.2], [0.1, 0.2, 0.3])
        assert np.isnan(ranking_agreement(df, k=1)["kendall"])

    def test_argmin_ties_count_as_agreement_when_ecol_best_is_among_them(self):
        df = _frame([0.1, 0.1, 0.5], [0.3, 0.2, 0.9])
        assert ranking_agreement(df, k=1)["argmin_agree"] == 1.0


class TestAgreementByMeasure:
    def test_one_row_per_measure_with_error_and_correlations(self):
        a = _frame([0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45], "A")
        b = _frame([0.5, 0.6, 0.7, 0.8], [0.55, 0.65, 0.75, 0.85], "B")
        f1 = _frame([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], "A", "F1")
        out = agreement_by_measure(pd.concat([a, b, f1]), k=1).set_index("measure")
        assert out.loc["T1", "mae"] == pytest.approx(0.05)
        assert out.loc["T1", "pearson"] == pytest.approx(1.0)
        assert out.loc["T1", "kendall_mean"] == pytest.approx(1.0)
        assert out.loc["T1", "n_datasets"] == 2
        assert out.loc["F1", "mae"] == pytest.approx(0.0)

    def test_missing_ecol_values_are_dropped(self):
        df = _frame([0.1, 0.2, 0.3], [0.1, np.nan, 0.3])
        out = agreement_by_measure(df, k=1).iloc[0]
        assert out["n_bags"] == 2


class TestToDispersion:
    def test_replaces_values_by_the_ga_dispersion_within_each_dataset(self):
        from pos.dispersion import dispersion_linear
        a = _frame([0.1, 0.2, 0.9], [0.3, 0.1, 0.5], "A")
        b = _frame([0.5, 0.5, 0.6], [0.2, 0.4, 0.4], "B")
        out = to_dispersion(pd.concat([a, b])).set_index(["dataset", "bag"])
        expect = np.array(dispersion_linear([[v] for v in [0.1, 0.2, 0.9]]))[:, 0]
        np.testing.assert_allclose(out.loc["A", "fast"].to_numpy(), expect)
        assert out.loc["A", "ecol"].max() == pytest.approx(1.0)
        assert len(out) == 6
