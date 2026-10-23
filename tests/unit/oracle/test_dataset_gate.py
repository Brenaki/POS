"""Tests for the per-profile conformance gate (ADR 0021).

The Monteiro profile keeps the Table 2 gate unchanged; `pos-oracle-v2`,
whose reference scope is itself, gates against the POS snapshot catalog
(SHA-256 per file), so P2 and new datasets need no `--allow-unverified-data`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pos.oracle.dataset_gate import (
    POS_CATALOG,
    TABLE_2,
    dataset_reference_for,
    gate_decision,
    load_pos_catalog,
    pos_catalog_status,
)
from pos.oracle.run_records_v2 import survey_dataset

DATASET_DIR = Path(__file__).resolve().parents[3] / "Dataset"

CATALOG = {"Wine": {"sha256": "sha256:abc", "instances": 178}}


def _record(name="Wine", sha="sha256:abc", status="conformant"):
    return {"name": name, "snapshot_sha256": sha, "conformance_status": status,
            "conformance_reason": "table 2 says so"}


class TestReferenceForProfile:
    def test_pos_profile_gates_on_its_own_catalog(self):
        assert dataset_reference_for("pos-oracle-v2") == POS_CATALOG

    def test_monteiro_profile_gates_on_table_2(self):
        assert dataset_reference_for("monteiro-2023-dt-compatible-v1") == TABLE_2

    def test_unknown_profile_falls_back_to_the_stricter_gate(self):
        assert dataset_reference_for("something-else") == TABLE_2


class TestPosCatalogStatus:
    def test_matching_hash_is_conformant(self):
        status, _ = pos_catalog_status(_record(), CATALOG)
        assert status == "conformant"

    def test_different_hash_is_a_mismatch(self):
        status, reason = pos_catalog_status(_record(sha="sha256:zzz"), CATALOG)
        assert status == "mismatch"
        assert "sha256" in reason

    def test_dataset_outside_the_catalog_is_unverified(self):
        status, _ = pos_catalog_status(_record(name="Nope"), CATALOG)
        assert status == "unverified"

    def test_missing_file_is_unverified(self):
        status, _ = pos_catalog_status(_record(sha=None), CATALOG)
        assert status == "unverified"


class TestGateDecision:
    def test_table_2_mismatch_is_blocked_under_monteiro(self):
        d = gate_decision(_record(status="mismatch"),
                          "monteiro-2023-dt-compatible-v1", False, CATALOG)
        assert d["gate_passes"] is False
        assert d["gate_reference"] == TABLE_2

    def test_table_2_mismatch_passes_under_pos_when_hash_matches(self):
        d = gate_decision(_record(status="mismatch"), "pos-oracle-v2", False, CATALOG)
        assert d["gate_passes"] is True
        assert d["gate_status"] == "conformant"

    def test_hash_mismatch_is_blocked_under_pos(self):
        d = gate_decision(_record(sha="sha256:zzz"), "pos-oracle-v2", False, CATALOG)
        assert d["gate_passes"] is False

    def test_allow_unverified_opens_any_gate(self):
        d = gate_decision(_record(name="Nope"), "pos-oracle-v2", True, CATALOG)
        assert d["gate_passes"] is True


class TestShippedCatalog:
    def test_every_catalog_entry_matches_its_snapshot(self):
        catalog = load_pos_catalog(DATASET_DIR)
        assert "P2" in catalog and "Wine" in catalog
        for name in catalog:
            status, reason = pos_catalog_status(
                survey_dataset(name, DATASET_DIR), catalog)
            assert status == "conformant", f"{name}: {reason}"

    def test_missing_catalog_file_is_an_empty_catalog(self, tmp_path):
        assert load_pos_catalog(tmp_path) == {}


@pytest.mark.parametrize("profile", ["pos-oracle-v2"])
def test_p2_passes_the_pos_gate_without_the_flag(profile):
    catalog = load_pos_catalog(DATASET_DIR)
    d = gate_decision(survey_dataset("P2", DATASET_DIR), profile, False, catalog)
    assert d["gate_passes"] is True
