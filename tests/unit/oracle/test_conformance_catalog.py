"""Tests for the conformance-gated dataset catalog v2 (ADR 0020 T14, T15).

The catalog maps the 28 datasets of Table 2 (Monteiro 2023) to POS snapshots,
recording conformance status (conformant / mismatch / unverified) and SHA-256
of the POS-approved snapshot (not the paper, which does not publish hashes).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pos.oracle.conformance_catalog import (
    CONFORMANCE_MISMATCH,
    CONFORMANCE_UNVERIFIED,
    PAPER_TABLE_2,
    check_conformance,
    conformance_status,
)


class TestPaperTable2:
    def test_has_28_datasets(self):
        assert len(PAPER_TABLE_2) == 28

    def test_first_is_australian(self):
        assert PAPER_TABLE_2[0]["name"] == "Australian"

    def test_p2_has_5000_instances(self):
        p2 = next(d for d in PAPER_TABLE_2 if d["name"] == "P2")
        assert p2["instances"] == 5000

    def test_heart_has_10_features(self):
        heart = next(d for d in PAPER_TABLE_2 if d["name"] == "Heart")
        assert heart["features"] == 10

    def test_ilpd_has_34_features(self):
        ilpd = next(d for d in PAPER_TABLE_2 if d["name"] == "ILPD")
        assert ilpd["features"] == 34

    def test_ionosphere_has_16_features(self):
        iono = next(d for d in PAPER_TABLE_2 if d["name"] == "Ionosphere")
        assert iono["features"] == 16

    def test_laryngeal3_has_2_features(self):
        l3 = next(d for d in PAPER_TABLE_2 if d["name"] == "Laryngeal3")
        assert l3["features"] == 2

    def test_each_entry_has_required_fields(self):
        for d in PAPER_TABLE_2:
            assert "name" in d
            assert "instances" in d
            assert "features" in d
            assert "classes" in d
            assert "source" in d


class TestConformance:
    def test_conformant_when_metadata_matches(self):
        entry = {"name": "Banana", "instances": 2000, "features": 2,
                 "classes": 2, "source": "PRTools"}
        actual = {"instances": 2000, "features": 2, "classes": 2}
        status, reason = check_conformance(entry, actual)
        assert status == "conformant"
        assert reason == ""

    def test_mismatch_on_instance_count(self):
        entry = {"name": "P2", "instances": 5000, "features": 2,
                 "classes": 2, "source": "[33]"}
        actual = {"instances": 2000, "features": 2, "classes": 2}
        status, reason = check_conformance(entry, actual)
        assert status == CONFORMANCE_MISMATCH
        assert "instances" in reason.lower()

    def test_mismatch_on_features(self):
        entry = {"name": "Heart", "instances": 270, "features": 10,
                 "classes": 2, "source": "STATLOG"}
        actual = {"instances": 270, "features": 13, "classes": 2}
        status, reason = check_conformance(entry, actual)
        assert status == CONFORMANCE_MISMATCH
        assert "features" in reason.lower()

    def test_mismatch_on_classes(self):
        entry = {"name": "Wine", "instances": 178, "features": 13,
                 "classes": 3, "source": "UCI"}
        actual = {"instances": 178, "features": 13, "classes": 2}
        status, reason = check_conformance(entry, actual)
        assert status == CONFORMANCE_MISMATCH
        assert "classes" in reason.lower()

    def test_unverified_when_no_actual(self):
        entry = {"name": "X", "instances": 100, "features": 5,
                 "classes": 2, "source": "UCI"}
        status, reason = check_conformance(entry, None)
        assert status == CONFORMANCE_UNVERIFIED
        assert "not found" in reason.lower() or "no snapshot" in reason.lower()


class TestConformanceStatus:
    def test_returns_dict_with_required_fields(self):
        result = conformance_status(
            name="Banana",
            actual={"instances": 2000, "features": 2, "classes": 2},
            snapshot_sha256="abc123",
        )
        assert "conformance_status" in result
        assert "conformance_reason" in result
        assert "snapshot_sha256" in result
        assert "paper_instances" in result
        assert "paper_features" in result
        assert "paper_classes" in result