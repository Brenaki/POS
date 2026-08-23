"""Difference-in-differences between the two split designs (ADR 0018).

The bias estimate this produces is a headline claim of the ADR, so the
arithmetic behind it is pinned here: which methods are comparable, which rows
are paired, and how the GA delta is split into a size effect and a bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pos.analysis.run_diff import (
    common_methods,
    decompose,
    did_table,
    pair_runs,
    pool_identity,
    recovered_over,
)

POOL = {"oracle_2": 0.9, "oracle_5": 0.95, "oracle_M": 1.0,
        "mean_individual_acc": 0.6, "double_fault_mean": 0.1, "M": 100}


def _frame(rows, extra_cols=()):
    """rows = (dataset, fold, mode, oracle_1, majority_vote, {method: acc})."""
    out = []
    for ds, fold, mode, o1, mv, des in rows:
        rec = {"dataset": ds, "fold": fold, "mode": mode,
               "oracle_1": o1, "majority_vote": mv, **POOL}
        rec.update({f"des_{k}": v for k, v in des.items()})
        for col in extra_cols:
            rec.setdefault(col, np.nan)
        out.append(rec)
    return pd.DataFrame(out)


def test_only_methods_present_in_both_runs_are_compared():
    """The new run ranks more methods; a wider max would fake an improvement."""
    old = _frame([("a", 0, "ga", 0.9, 0.7, {"ola": 0.75, "knorau": 0.78})])
    new = _frame([("a", 0, "ga", 0.9, 0.7, {"ola": 0.75, "knorau": 0.78,
                                            "desknn": 0.89})])
    assert common_methods(old, new) == ["des_ola", "des_knorau"]
    # des_desknn would push recovered to 0.95; restricted to the common pair
    # both runs must land on the same 0.4.
    merged = pair_runs(old, new)
    assert merged["rec_old"].iat[0] == pytest.approx(0.4)
    assert merged["rec_new"].iat[0] == pytest.approx(0.4)


def test_recovered_is_undefined_when_the_oracle_adds_nothing():
    df = _frame([("a", 0, "ga", 0.7, 0.7, {"ola": 0.7})])
    assert np.isnan(recovered_over(df, ["des_ola"]).iat[0])


def test_paired_means_stay_consistent_with_the_delta():
    """A fold missing on one side must drop from both means, not just one."""
    old = _frame([("a", 0, "ga", 0.9, 0.7, {"ola": 0.8}),
                  ("a", 1, "ga", 0.7, 0.7, {"ola": 0.7})])   # no gap -> NaN
    new = _frame([("a", 0, "ga", 0.9, 0.7, {"ola": 0.75}),
                  ("a", 1, "ga", 0.9, 0.7, {"ola": 0.9})])
    did = did_table(pair_runs(old, new))
    assert did.loc["ga", "pares"] == 1
    assert did.loc["ga", "d_rec"] == pytest.approx(
        did.loc["ga", "rec_new"] - did.loc["ga", "rec_old"])
    assert did.loc["ga", "d_rec"] == pytest.approx(0.25 - 0.5)


def test_identical_pools_report_zero_discrepancy():
    """ADR 0018 keeps X_tr intact, so bagging/rf pool columns must not move."""
    rows = [("a", 0, "bagging", 0.9, 0.7, {"ola": 0.8})]
    ident = pool_identity(pair_runs(_frame(rows), _frame(rows)))
    assert (ident.loc["bagging"].drop("folds") == 0).all()
    assert ident.loc["ga", "folds"] == 0


def test_bias_is_the_ga_delta_net_of_the_control_delta():
    old = _frame([("a", 0, m, 0.9, 0.7, {"ola": 0.8}) for m in
                  ("ga", "bagging", "rf")])
    # Controls lose 0.25 of recovered; the GA loses 0.5 -> bias -0.25.
    new = _frame([("a", 0, "ga", 0.9, 0.7, {"ola": 0.7}),
                  ("a", 0, "bagging", 0.9, 0.7, {"ola": 0.75}),
                  ("a", 0, "rf", 0.9, 0.7, {"ola": 0.75})])
    dec = decompose(did_table(pair_runs(old, new)))
    assert dec["dsel_size_effect"] == pytest.approx(-0.25)
    assert dec["ga_delta"] == pytest.approx(-0.5)
    assert dec["dsel_bias"] == pytest.approx(-0.25)


def test_decomposition_refuses_an_incomplete_run():
    df = _frame([("a", 0, "ga", 0.9, 0.7, {"ola": 0.8})])
    assert decompose(did_table(pair_runs(df, df))) == {}
