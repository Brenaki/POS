"""Tests for the schema-v2 battery analysis (ADR 0022).

A synthetic v2 summary is enough: the module only derives columns from what
`record_run_v2` writes (full Oracle curve as JSON, MVR, DES columns).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pos.analysis.v2_battery import battery_table, contrast, load_v2_run


def _row(dataset, fold, treatment, curve, mv, des, n_classes_hint=None):
    return {"profile": "pos-oracle-v2", "dataset": dataset, "fold": fold,
            "treatment": treatment, "M": len(curve),
            "oracle_curve_json": json.dumps(curve), "oracle_1": curve[0],
            "majority_vote": mv, "mv_acc": mv, "double_fault_mean": 0.04,
            "mean_individual_acc": 0.8, "des_knorau": des, "des_ola": des - 0.05}


@pytest.fixture
def run_dir(tmp_path):
    M = 4
    binary = [1.0, 0.9, 0.7, 0.5]       # MVR 0.8 sits between Oracle_3 and Oracle_2
    multi = [0.95, 0.85, 0.6, 0.3]
    rows = [_row("Bin", f, "bagging", binary, 0.8, 0.85) for f in range(2)]
    rows += [_row("Multi", f, "bagging", multi, 0.7, 0.75) for f in range(2)]
    pd.DataFrame(rows).to_csv(tmp_path / "summary.csv", index=False)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"conformance": {
        "Bin": {"snapshot_classes": 2}, "Multi": {"snapshot_classes": 10}}}))
    assert M == len(binary)
    return tmp_path


class TestLoad:
    def test_derives_curve_classes_and_nstar(self, run_dir):
        df = load_v2_run(run_dir)
        b = df[df["dataset"] == "Bin"].iloc[0]
        assert b["is_binary"] and b["n_classes"] == 2
        assert b["nstar"] == 3            # first N with Oracle_N < MVR
        assert b["mode"] == "bagging"

    def test_recovered_uses_the_best_dynamic_method(self, run_dir):
        df = load_v2_run(run_dir)
        b = df[df["dataset"] == "Bin"].iloc[0]
        assert b["des_best"] == pytest.approx(0.85)
        assert b["recovered"] == pytest.approx((0.85 - 0.8) / (1.0 - 0.8))

    def test_df_ratio_is_double_fault_over_squared_error(self, run_dir):
        df = load_v2_run(run_dir)
        assert df["df_ratio"].iloc[0] == pytest.approx(0.04 / 0.2**2)


class TestTables:
    def test_battery_table_has_one_row_per_dataset_and_mode(self, run_dir):
        t = battery_table(load_v2_run(run_dir))
        assert len(t) == 2
        assert {"oracle_1", "majority_vote", "des_best", "recovered", "nstar",
                "df_ratio", "n_classes"} <= set(t.columns)

    def test_contrast_reports_medians_per_group(self):
        a = pd.DataFrame({"recovered": [0.1, 0.2, 0.3], "nstar": [1, 2, 3]})
        b = pd.DataFrame({"recovered": [0.5, 0.6], "nstar": [4, 5]})
        out = contrast(a, b, ["recovered", "nstar"])
        assert out.loc["recovered", "median_a"] == pytest.approx(0.2)
        assert out.loc["nstar", "median_b"] == pytest.approx(4.5)
        assert 0 <= out.loc["recovered", "p_mannwhitney"] <= 1
        assert np.isfinite(out.loc["nstar", "p_mannwhitney"])
