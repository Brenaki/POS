"""Tests for the scientific profile registry (ADR 0020).

A profile is a frozen dict of protocol parameters materialized from a single
source of truth. Tests guard three invariants:

* the two profiles exist and expose exactly the ADR-mandated keys;
* the materialized config matches the ADR values verbatim (no drift);
* the treatment list is exactly the three scientific treatments.
"""

from __future__ import annotations

import pytest

from pos.oracle.profiles import (
    MONTIERO_2023_DT,
    POS_ORACLE_V2,
    PROFILE_NAMES,
    SCIENTIFIC_TREATMENTS,
    materialize_profile,
    profile_for,
)


class TestRegistry:
    def test_exposes_the_three_profiles(self):
        assert set(PROFILE_NAMES) == {"pos-oracle-v2", "monteiro-2023-dt-compatible-v1",
                                      "monteiro-2023-dt-compatible-v2"}

    def test_treatments_are_exactly_the_three(self):
        assert SCIENTIFIC_TREATMENTS == ("pgdcs_a", "pgdcs_d", "bagging")


class TestPosOracleV2:
    def test_protocol_is_10_fold_stratified(self):
        assert POS_ORACLE_V2["n_folds"] == 10
        assert POS_ORACLE_V2["cv_shuffle"] is True
        assert POS_ORACLE_V2["random_state"] == 42

    def test_oracle_checkpoints_match_adr(self):
        """ADR 0020 revised: 21 checkpoints — 1, then every 5 up to M=100."""
        assert POS_ORACLE_V2["oracle_checkpoints"] == (1,) + tuple(range(5, 101, 5))
        assert len(POS_ORACLE_V2["oracle_checkpoints"]) == 21

    def test_des_methods_are_the_six_canonical(self):
        assert POS_ORACLE_V2["des_methods"] == (
            "lca", "ola", "rank", "knorae", "knorau", "metades",
        )

    def test_three_way_split_disjoint(self):
        assert POS_ORACLE_V2["split"] == "three_way_disjoint"


class TestMonteiro2023Dt:
    def test_replications_are_twenty(self):
        assert MONTIERO_2023_DT["n_replications"] == 20

    def test_holdout_fractions_match_paper(self):
        assert MONTIERO_2023_DT["train_frac"] == 0.5
        assert MONTIERO_2023_DT["val_frac"] == 0.25
        assert MONTIERO_2023_DT["test_frac"] == 0.25

    def test_root_seed_is_86(self):
        assert MONTIERO_2023_DT["reproducibility_root_seed"] == 86

    def test_bagging_random_state_is_86(self):
        assert MONTIERO_2023_DT["bagging_random_state"] == 86

    def test_reference_scope_is_decision_tree_slice(self):
        assert MONTIERO_2023_DT["reference_scope"] == (
            "decision-tree / pgdcs-a / pgdcs-d / bagging"
        )

    def test_validation_dsel_overlap_recorded(self):
        assert MONTIERO_2023_DT["validation_dsel_overlap"] is True

    def test_measure_selection_once_per_dataset(self):
        assert MONTIERO_2023_DT["measure_selection"] == "once_per_dataset"


class TestMaterialization:
    def test_pos_oracle_full_matches_adr_values(self):
        cfg = materialize_profile("pos-oracle-v2", scale="full")
        assert cfg["profile"] == "pos-oracle-v2"
        assert cfg["n_folds"] == 10
        assert cfg["nr_generation"] == 20
        assert cfg["M"] == 100
        assert cfg["nr_individual"] == 100
        assert cfg["nr_child"] == 100
        assert cfg["proba_crossover"] == 0.9
        assert cfg["proba_mutation"] == 0.1
        assert cfg["tam_bags"] == 0.5
        assert cfg["classifier"] == "tree"
        assert cfg["treatments"] == list(SCIENTIFIC_TREATMENTS)
        assert cfg["comparable"] is True

    def test_smoke_scale_reduces_generations_and_folds(self):
        cfg = materialize_profile("pos-oracle-v2", scale="smoke")
        assert cfg["nr_generation"] < 20
        assert cfg["n_folds"] < 10
        assert cfg["comparable"] is False

    def test_monteiro_full_matches_adr_values(self):
        cfg = materialize_profile("monteiro-2023-dt-compatible-v1", scale="full")
        assert cfg["profile"] == "monteiro-2023-dt-compatible-v1"
        assert cfg["n_replications"] == 20
        assert cfg["M"] == 100
        assert cfg["nr_generation"] == 20
        assert cfg["nr_individual"] == 100
        assert cfg["nr_child"] == 100
        assert cfg["proba_crossover"] == 0.9
        assert cfg["proba_mutation"] == 0.1
        assert cfg["tam_bags"] == 0.5
        assert cfg["classifier"] == "tree"
        # Step-1 sub-samples use u = 0.4 (paper §5.1); the GA bags stay at 50%
        # ("data subsets with just 50% of the training dataset size", §5.2).
        assert cfg["measure_fraction"] == 0.4
        assert cfg["shared_evolution_trajectory"] is True
        assert cfg["modernization"] is True
        assert cfg["comparable"] is True

    def test_monteiro_smoke_reduces_replications(self):
        cfg = materialize_profile("monteiro-2023-dt-compatible-v1", scale="smoke")
        assert cfg["n_replications"] < 20
        assert cfg["comparable"] is False

    def test_override_marks_not_comparable(self):
        cfg = materialize_profile(
            "pos-oracle-v2", scale="full", overrides={"M": 50}
        )
        assert cfg["comparable"] is False
        assert cfg["M"] == 50

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError):
            profile_for("does-not-exist")

    def test_fingerprint_is_stable_across_calls(self):
        a = materialize_profile("pos-oracle-v2", scale="full")
        b = materialize_profile("pos-oracle-v2", scale="full")
        assert a["fingerprint"] == b["fingerprint"]

    def test_fingerprint_changes_with_overrides(self):
        a = materialize_profile("pos-oracle-v2", scale="full")
        b = materialize_profile("pos-oracle-v2", scale="full", overrides={"M": 50})
        assert a["fingerprint"] != b["fingerprint"]
