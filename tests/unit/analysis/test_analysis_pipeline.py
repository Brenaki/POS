"""End-to-end check of the results analysis (loader, tests, figures).

Guards the derived quantities the report leans on — N*, DF/e^2 and the
Oracle-curve identity — and makes sure every figure still renders.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pos.analysis.figures_compare import (
    plot_diversity_vs_gap,
    plot_gap_per_dataset,
    plot_nstar,
)
from pos.analysis.figures_curves import plot_mean_curves, plot_per_dataset_grid
from pos.analysis.figures_des import plot_fuser_accuracy, plot_recovered_gap
from pos.analysis.fusers import available_fusers, fuser_comparison, recovery_summary
from pos.analysis.loader import MODES, load_run, mean_curve, per_dataset
from pos.analysis.stats_tests import (
    compare,
    compare_table,
    critical_difference,
    format_comparison,
)
from pos.oracle.des_comparison import DES_METHODS

M = 8
N_DATASETS = 6
N_FOLDS = 2


def _row(ds, fold, mode, rng):
    """One synthetic summary row with a monotone Oracle curve."""
    curve = np.sort(rng.uniform(0.2, 1.0, M))[::-1]
    acc = float(curve.mean())
    majority = float(curve[M // 2])
    # DES lands between majority vote and the Oracle, as it does in practice.
    des = {f"des_{d}": float(rng.uniform(majority, curve[0])) for d in DES_METHODS}
    if mode == "ga":
        des["des_metades"] = np.nan  # needs predict_proba; a Perceptron has none
    return {
        "dataset": ds, "fold": fold, "mode": mode, "M": M, "n_test": 50,
        "oracle_M": float(curve[-1]), "oracle_curve_json": json.dumps(curve.tolist()),
        "majority_vote": majority, "mean_probs": majority - 0.01,
        "soft_fusion_rule": "mean_probs",
        "double_fault_mean": float(rng.uniform(0.05, 0.3)),
        "mean_individual_acc": acc,
        **{f"oracle_{n}": float(curve[n - 1]) for n in range(1, 6)},
        **des,
    }


@pytest.fixture()
def run_dir(tmp_path):
    rng = np.random.RandomState(0)
    rows = [_row(f"DS{d}", f, m, rng)
            for d in range(N_DATASETS) for f in range(N_FOLDS) for m in MODES]
    pd.DataFrame(rows).to_csv(tmp_path / "summary.csv", index=False)
    return tmp_path


def test_load_run_derives_gap_nstar_and_df_ratio(run_dir):
    df = load_run(run_dir)
    assert len(df) == N_DATASETS * N_FOLDS * len(MODES)
    assert (df["gap_1"] == df["oracle_1"] - df["majority_vote"]).all()
    # curve[M//2] is the majority vote, so the first N below it is M//2 + 2
    assert set(df["nstar"]) <= set(range(1, M + 1))
    err = 1.0 - df["mean_individual_acc"]
    assert np.allclose(df["df_ratio"], df["double_fault_mean"] / err**2)


def test_curve_mean_equals_mean_individual_accuracy(run_dir):
    """mean(Oracle_1..M) == mean individual accuracy — the identity the report uses."""
    df = load_run(run_dir)
    auc = np.array([np.mean(c) for c in df["curve"]])
    assert np.abs(auc - df["mean_individual_acc"].values).max() < 1e-12


def test_per_dataset_keeps_all_modes_when_one_is_empty(run_dir):
    """A metric missing for one mode must return an all-NaN column, not vanish."""
    df = load_run(run_dir)
    df.loc[df["mode"] == "ga", "mean_probs"] = np.nan
    table = per_dataset(df, "mean_probs")
    assert list(table.columns) == MODES
    assert table["ga"].isna().all()


def test_compare_reports_ranks_and_pairs(run_dir):
    df = load_run(run_dir)
    res = compare(df, "oracle_1")
    assert res["n_datasets"] == N_DATASETS
    assert set(res["ranks"]) == set(MODES)
    assert len(res["pairs"]) == 3
    assert res["critical_difference"] == critical_difference(N_DATASETS)
    assert "Friedman" in format_comparison(res)


def test_every_figure_renders(run_dir, tmp_path):
    df = load_run(run_dir)
    assert mean_curve(df, "ga").shape == (M,)
    outputs = [
        plot_mean_curves(df, tmp_path / "f1.png"),
        plot_mean_curves(df, tmp_path / "f2.png", zoom=4),
        plot_nstar(df, tmp_path / "f3.png"),
        plot_gap_per_dataset(df, tmp_path / "f4.png"),
        plot_diversity_vs_gap(df, tmp_path / "f5.png"),
        plot_per_dataset_grid(df, tmp_path / "f6.png"),
    ]
    for path in outputs:
        assert path.exists() and path.stat().st_size > 1000


def test_des_columns_derive_best_and_recovered_share(run_dir):
    df = load_run(run_dir)
    assert (df["des_best"] <= df["oracle_1"] + 1e-12).all()
    assert (df["des_best"] >= df["majority_vote"] - 1e-12).all()
    expected = (df["des_best"] - df["majority_vote"]) / (df["oracle_1"] - df["majority_vote"])
    assert np.allclose(df["recovered"], expected)
    assert np.allclose(df["gap_des"], df["oracle_1"] - df["des_best"])


def test_metades_is_dropped_only_from_the_mode_that_cannot_run_it(run_dir):
    """An all-NaN fuser column must not silently empty the paired test."""
    df = load_run(run_dir)
    assert "des_metades" not in available_fusers(df, "ga")
    assert "des_metades" in available_fusers(df, "rf")
    assert fuser_comparison(df, "ga")["n_datasets"] == N_DATASETS
    assert len(fuser_comparison(df, "ga")["methods"]) == len(DES_METHODS) + 1
    assert len(fuser_comparison(df, "rf")["methods"]) == len(DES_METHODS) + 2


def test_recovery_summary_has_one_row_per_mode(run_dir):
    table = recovery_summary(load_run(run_dir))
    assert list(table.index) == MODES
    assert np.allclose(table["gap_mv"], table["oracle_1"] - table["majority_vote"])


def test_des_figures_render(run_dir, tmp_path):
    df = load_run(run_dir)
    for path in (plot_fuser_accuracy(df, tmp_path / "f7.png"),
                 plot_recovered_gap(df, tmp_path / "f8.png")):
        assert path.exists() and path.stat().st_size > 1000


def test_compare_table_handles_two_methods(run_dir):
    """A fuser against majority vote is k=2, where Friedman is undefined."""
    df = load_run(run_dir)
    sub = df[df["mode"] == "rf"]
    table = sub.pivot_table(index="dataset", values=["majority_vote", "des_ola"],
                            aggfunc="mean")[["majority_vote", "des_ola"]]
    res = compare_table(table, name="rf")
    assert np.isnan(res["friedman_chi2"])
    assert len(res["pairs"]) == 1
    assert "Friedman n/a (k=2)" in format_comparison(res)
