"""Tests for the Table 2 reconciliation (ADR 0020 §8).

Four snapshots are `approved_with_note` because the paper's Features column
is shifted by one row across rows 09-13; P2 stays a genuine mismatch.
"""

from __future__ import annotations

from pathlib import Path

from pos.oracle.dataset_reconciliation import (
    APPROVED_WITH_NOTE,
    comparable_statuses,
    dataset_stem,
    reconcile,
)
from pos.oracle.run_records_v2 import survey_dataset

DATASET_DIR = Path(__file__).resolve().parents[3] / "Dataset"


class TestAliases:
    def test_australian_lives_as_adult(self):
        assert dataset_stem("Australian") == "Adult"

    def test_other_names_pass_through(self):
        assert dataset_stem("Wine") == "Wine"


class TestReconcile:
    def test_lifts_a_known_mismatch_to_approved_with_note(self):
        status = {"name": "Heart", "conformance_status": "mismatch",
                  "conformance_reason": "features: paper=10 actual=13"}
        out = reconcile(status)
        assert out["conformance_status"] == APPROVED_WITH_NOTE
        assert out["reconciled"] is True
        assert "shifted" in out["conformance_reason"]

    def test_never_lifts_to_conformant(self):
        status = {"name": "ILPD", "conformance_status": "mismatch",
                  "conformance_reason": "features: paper=34 actual=10"}
        assert reconcile(status)["conformance_status"] != "conformant"

    def test_leaves_unlisted_mismatches_alone(self):
        status = {"name": "P2", "conformance_status": "mismatch",
                  "conformance_reason": "instances: paper=5000 actual=2000"}
        assert reconcile(status)["conformance_status"] == "mismatch"

    def test_does_not_touch_a_conformant_record(self):
        status = {"name": "Heart", "conformance_status": "conformant",
                  "conformance_reason": ""}
        assert reconcile(status)["conformance_status"] == "conformant"


class TestComparableStatuses:
    def test_default_excludes_mismatch(self):
        assert "mismatch" not in comparable_statuses()
        assert APPROVED_WITH_NOTE in comparable_statuses()

    def test_allow_unverified_includes_everything(self):
        assert "mismatch" in comparable_statuses(allow_unverified=True)


class TestSnapshotSurvey:
    def test_australian_snapshot_is_conformant(self):
        r = survey_dataset("Australian", DATASET_DIR)
        assert r["conformance_status"] == "conformant"
        assert r["snapshot_file"] == "Adult.arff"
        assert r["snapshot_sha256"].startswith("sha256:")

    def test_heart_is_approved_with_note(self):
        assert survey_dataset("Heart", DATASET_DIR)["conformance_status"] == APPROVED_WITH_NOTE

    def test_p2_stays_a_mismatch(self):
        r = survey_dataset("P2", DATASET_DIR)
        assert r["conformance_status"] == "mismatch"
        assert "instances" in r["conformance_reason"]

    def test_missing_snapshot_is_unverified(self, tmp_path):
        assert survey_dataset("Wine", tmp_path)["conformance_status"] == "unverified"
