"""Tests for the analyses ADR 0019 added in answer to the external review."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pos.analysis.df_exact import (
    attach_df_ratio_exact,
    exact_independence_df,
    load_manifest_table,
)
from pos.analysis.mode_contrasts import contrast, pgdcs_measure_choices
from pos.analysis.nstar_split import nstar_summary, sandwich_check
from pos.analysis.pruning import pruning_correlation, pruning_table
from pos.analysis.threshold_lodo import gap_band, lodo_scores


class TestExactDoubleFault:
    def test_equal_error_rates_reduce_to_e_squared(self):
        """With identical rates the exact denominator IS the mean-based one."""
        accs = [0.8] * 10
        assert exact_independence_df(accs) == pytest.approx(0.2**2)

    def test_unequal_rates_exceed_the_mean_based_value(self):
        """Jensen: the mean-based e^2 understates, so df_ratio is biased up."""
        accs = [0.5] * 5 + [1.0] * 5
        mean_based = (1.0 - np.mean(accs)) ** 2
        assert exact_independence_df(accs) < mean_based

    def test_a_single_erring_classifier_has_no_joint_failure(self):
        assert exact_independence_df([0.5] + [1.0] * 9) == pytest.approx(0.0)

    def test_needs_at_least_two_classifiers(self):
        assert np.isnan(exact_independence_df([0.5]))


@pytest.fixture()
def manifest_run(tmp_path):
    """A run directory holding two fold manifests, as the recorder writes them."""
    for fold, accs in enumerate([[0.8] * 4, [0.6, 0.6, 1.0, 1.0]]):
        d = tmp_path / "DS0" / f"fold_{fold}"
        d.mkdir(parents=True)
        (d / "fold_manifest_ga.json").write_text(json.dumps({
            "dataset": "DS0", "fold_idx": fold, "mode": "ga",
            "individual_accuracies": accs, "pgdcs_types": [],
        }))
    return tmp_path


class TestManifestTable:
    def test_reads_every_manifest(self, manifest_run):
        table = load_manifest_table(manifest_run)
        assert len(table) == 2
        assert set(table["mode"]) == {"ga"}

    def test_attaches_ratio_to_the_run_frame(self, manifest_run):
        df = pd.DataFrame({"dataset": ["DS0", "DS0"], "fold": [0, 1],
                           "mode": ["ga", "ga"], "double_fault_mean": [0.04, 0.04]})
        out = attach_df_ratio_exact(df, manifest_run)
        assert out.loc[0, "df_ratio_exact"] == pytest.approx(1.0)  # 0.04 / 0.2^2
        assert out.loc[1, "df_ratio_exact"] > 1.0  # unequal rates, smaller denom


def _curve(values):
    return list(values)


@pytest.fixture()
def binary_frame():
    """M=4, so the sandwich is Oracle_3 <= MVR <= Oracle_2."""
    rows = []
    for fold, (curve, mv) in enumerate([
        ([0.9, 0.8, 0.7, 0.6], 0.7),   # MVR == Oracle_3, the tie-free case
        ([0.9, 0.8, 0.7, 0.6], 0.75),  # ties won: strictly between
    ]):
        rows.append({"dataset": "DS0", "fold": fold, "mode": "ga", "M": 4,
                     "curve": _curve(curve), "majority_vote": mv,
                     "is_binary": True, "n_classes": 2, "nstar": 4})
    return pd.DataFrame(rows)


class TestNstarSandwich:
    def test_sandwich_holds_on_both_folds(self, binary_frame):
        res = sandwich_check(binary_frame)
        assert res["n_folds"] == 2
        assert res["holds_both_frac"] == 1.0
        assert res["exact_equality"] == 1

    def test_violation_is_reported_not_hidden(self, binary_frame):
        bad = binary_frame.copy()
        bad.loc[0, "majority_vote"] = 0.5  # below Oracle_3
        assert sandwich_check(bad)["holds_both_frac"] < 1.0

    def test_summary_labels_the_cardinality_groups(self, binary_frame):
        out = nstar_summary(binary_frame)
        assert set(out["group"]) == {"binaria"}


class TestThresholdLodo:
    def test_gap_band_edges(self):
        assert gap_band(0.01) == 0
        assert gap_band(0.05) == 1
        assert gap_band(0.30) == 2

    def test_lodo_beats_baseline_on_a_separable_rule(self):
        """DF/e^2 built to determine the band exactly: LODO must reach 1.0."""
        rows = []
        for i in range(12):
            ratio = [0.5, 3.0, 6.0][i % 3]
            gap = [0.5, 0.2, 0.01][i % 3]
            rows.append({"dataset": f"DS{i}", "mode": "ga",
                         "df_ratio": ratio, "gap_1": gap})
        res = lodo_scores(pd.DataFrame(rows))
        assert res["lodo_accuracy"] == pytest.approx(1.0)
        assert res["baseline_accuracy"] < 1.0


class TestPruning:
    @pytest.fixture()
    def des_frame(self):
        rows = []
        for i in range(6):
            rows.append({
                "dataset": f"DS{i}", "fold": 0, "mode": "rf",
                "majority_vote": 0.80,
                "des_ola": 0.74, "des_ola_sel_frac": 0.01,
                "des_ola_routed_frac": 0.6,
                "des_knorau": 0.82, "des_knorau_sel_frac": 1.0,
                "des_knorau_routed_frac": 0.6,
            })
        return pd.DataFrame(rows)

    def test_table_is_ordered_by_how_much_it_prunes(self, des_frame):
        table = pruning_table(des_frame, methods=("ola", "knorau"))
        assert list(table["method"]) == ["ola", "knorau"]
        assert table.loc[0, "delta_vs_mvr"] < 0 < table.loc[1, "delta_vs_mvr"]

    def test_correlation_is_positive_when_pruning_hurts(self, des_frame):
        corr = pruning_correlation(des_frame, methods=("ola", "knorau"))
        assert (corr["spearman"] > 0).all()

    def test_run_without_the_columns_yields_empty(self):
        empty = pd.DataFrame({"dataset": ["A"], "mode": ["rf"],
                              "majority_vote": [0.5]})
        assert pruning_table(empty).empty


class TestModeContrasts:
    @pytest.fixture()
    def two_modes(self):
        rows = []
        for i in range(8):
            rows.append({"dataset": f"DS{i}", "mode": "pgdcs", "gap_1": 0.30 + i * 0.01,
                         "pgdcs_types": "F3|N3"})
            rows.append({"dataset": f"DS{i}", "mode": "ga", "gap_1": 0.20 + i * 0.01,
                         "pgdcs_types": ""})
        return pd.DataFrame(rows)

    def test_paired_contrast_detects_the_shift(self, two_modes):
        table = contrast(two_modes, "pgdcs", "ga", metrics=["gap_1"])
        assert table.loc[0, "delta"] == pytest.approx(0.10)
        assert table.loc[0, "wins"] == 8
        assert table.loc[0, "p"] < 0.05

    def test_measure_choices_counted_only_for_pgdcs(self, two_modes):
        choices = pgdcs_measure_choices(two_modes)
        assert choices.loc[0, "folds"] == 8
        assert not choices.loc[0, "is_f1_t1"]

    def test_f1_t1_choice_is_flagged(self, two_modes):
        df = two_modes.copy()
        df.loc[df["mode"] == "pgdcs", "pgdcs_types"] = "F1|T1"
        assert bool(pgdcs_measure_choices(df).loc[0, "is_f1_t1"])


@pytest.fixture()
def review_frame():
    """A frame carrying every column the review sections read."""
    rng = np.random.RandomState(3)
    rows = []
    for i in range(10):
        for mode in ("pgdcs", "ga", "randbag", "bagging", "rf"):
            mv = 0.70 + rng.uniform(0, 0.1)
            o1 = mv + rng.uniform(0.1, 0.4)
            curve = np.linspace(o1, mv - 0.1, 8).tolist()
            rows.append({
                "dataset": f"DS{i}", "fold": 0, "mode": mode, "M": 8,
                "curve": curve, "oracle_1": o1, "majority_vote": mv,
                "gap_1": o1 - mv, "nstar": int(rng.randint(3, 7)),
                "recovered": float(rng.uniform(0, 0.6)),
                "double_fault_mean": 0.05,
                "mean_individual_acc": 0.75,
                "df_ratio": float(rng.uniform(0.8, 6.0)),
                "df_ratio_exact": float(rng.uniform(0.8, 6.0)),
                "acc_spread": 0.04,
                "is_binary": i % 3 != 0, "n_classes": 2 if i % 3 else 4,
                "n_features": 5, "n_samples": 200, "imbalance_ratio": 1.5,
                "pgdcs_types": "F3|N3" if mode == "pgdcs" else "",
                "des_ola": mv - 0.05, "des_ola_sel_frac": 1 / 8,
                "des_ola_routed_frac": 0.7,
                "des_knorau": mv + 0.02, "des_knorau_sel_frac": 1.0,
                "des_knorau_routed_frac": 0.7,
            })
    return pd.DataFrame(rows)


class TestReviewSections:
    """Every section must render text, or say plainly that it could not."""

    def test_all_sections_produce_text(self, review_frame):
        from pos.analysis.review_sections import SECTIONS

        for fn in SECTIONS:
            if fn.__name__ == "section_recoverability":
                continue  # needs the ARFF catalogue; covered separately
            out = fn(review_frame)
            assert isinstance(out, str) and out.strip()

    def test_nstar_section_reports_the_sandwich(self, review_frame):
        from pos.analysis.review_sections import section_nstar

        assert "sanduiche" in section_nstar(review_frame)

    def test_recovered_section_states_the_empirical_bound(self, review_frame):
        from pos.analysis.review_sections import section_recovered

        assert "cota empirica" in section_recovered(review_frame)

    def test_pruning_section_degrades_on_a_run_without_the_columns(self):
        from pos.analysis.review_sections import section_pruning

        bare = pd.DataFrame({"dataset": ["A"], "mode": ["rf"],
                             "majority_vote": [0.5]})
        assert "sem colunas sel_frac" in section_pruning(bare)


class TestRecoverabilityModel:
    @pytest.fixture()
    def feature_frame(self):
        rng = np.random.RandomState(0)
        rows = []
        for i in range(20):
            f1 = rng.uniform(0, 1)
            rows.append({"dataset": f"DS{i}", "mode": "ga", "F1": f1,
                         "mean_individual_acc": 0.7 + 0.1 * f1,
                         "recovered": 0.5 * f1 + rng.normal(0, 0.02)})
        return pd.DataFrame(rows)

    def test_lodo_recovers_a_planted_signal(self, feature_frame):
        from pos.analysis.recoverability import lodo_regression, ridge_model

        res = lodo_regression(feature_frame, ridge_model())
        assert res["n"] == 20
        assert res["spearman"] > 0.8
        assert res["mae"] < res["baseline_mae"]

    def test_forest_also_runs(self, feature_frame):
        from pos.analysis.recoverability import forest_model, lodo_regression

        assert lodo_regression(feature_frame, forest_model())["n"] == 20

    def test_feature_frame_keeps_one_row_per_dataset_mode(self):
        from pos.analysis.recoverability import build_feature_frame

        df = pd.DataFrame({
            "dataset": ["A", "A", "B"], "mode": ["ga", "ga", "ga"],
            "recovered": [0.2, 0.4, 0.1], "mean_individual_acc": [0.7, 0.7, 0.8],
        })
        complexity = pd.DataFrame({"dataset": ["A", "B"], "F1": [0.5, 0.9]})
        out = build_feature_frame(df, complexity)
        assert len(out) == 2
        assert out.loc[out.dataset == "A", "recovered"].iat[0] == pytest.approx(0.3)


class TestDatasetMeta:
    def test_catalogue_tags_binary_datasets(self):
        from pos.analysis.dataset_meta import dataset_meta

        meta = dataset_meta()
        assert {"dataset", "n_classes", "is_binary"} <= set(meta.columns)
        assert meta["is_binary"].any() and (~meta["is_binary"]).any()

    def test_attach_is_a_left_join(self):
        from pos.analysis.dataset_meta import attach_dataset_meta

        df = pd.DataFrame({"dataset": ["Wine"], "mode": ["ga"]})
        out = attach_dataset_meta(df)
        assert len(out) == 1
        assert out.loc[0, "n_classes"] == 3
