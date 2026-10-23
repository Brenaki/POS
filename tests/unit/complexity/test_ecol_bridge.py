"""Tests for the R/ECoL subprocess bridge (ADR 0023)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pos.complexity import ecol_bridge
from pos.complexity.ecol_bridge import canonical_names, ecol_measures, ensure_ecol_source
from pos.complexity.fast_adapter import complexity_data3


def _two_clusters():
    X = np.array([[0.0], [0.01], [0.02], [0.03], [0.04],
                  [1.0], [1.01], [1.02], [1.03], [1.04]])
    return X, np.array([0] * 5 + [1] * 5)


class TestCanonicalNames:
    def test_ecol_044_names_map_back_to_header(self):
        df = pd.DataFrame({"group": ["featurebased", "neighborhood", "neighborhood"],
                           "measure": ["F1", "N5", "N6"]})
        out = canonical_names(df, "0.4.4")
        assert out["group"].tolist() == ["overlapping", "neighborhood", "neighborhood"]
        assert out["measure"].tolist() == ["F1", "T1", "LSC"]

    def test_ecol_030_names_are_already_canonical(self):
        df = pd.DataFrame({"group": ["overlapping"], "measure": ["T1"]})
        assert canonical_names(df, "0.3.0").equals(df)


class TestEnsureSource:
    def test_hash_mismatch_is_rejected_and_the_tarball_dropped(self, tmp_path, monkeypatch):
        (tmp_path / "ECoL_9.9.9.tar.gz").write_bytes(b"not ecol")
        monkeypatch.setitem(ecol_bridge.ECOL_TARBALLS, "9.9.9", ("unused", "0" * 64))
        with pytest.raises(ValueError, match="SHA-256"):
            ensure_ecol_source("9.9.9", cache=tmp_path)
        assert not (tmp_path / "ECoL_9.9.9.tar.gz").exists()

    def test_already_extracted_source_is_reused(self, tmp_path, monkeypatch):
        (tmp_path / "9.9.9" / "ECoL").mkdir(parents=True)
        monkeypatch.setitem(ecol_bridge.ECOL_TARBALLS, "9.9.9", ("unused", "0" * 64))
        assert ensure_ecol_source("9.9.9", cache=tmp_path) == tmp_path / "9.9.9" / "ECoL"


@pytest.mark.requires_r
class TestAgainstR:
    def test_t1_semantics_differ_between_ecol_versions(self):
        X, y = _two_clusters()
        t1 = {v: ecol_measures([(X, y)], v, groups=("neighborhood",))
              .set_index("measure").loc["T1", "value"] for v in ("0.3.0", "0.4.4")}
        # 0.4.4: N5 = #spheres / n = 2/10.
        assert t1["0.4.4"] == pytest.approx(0.2)
        # 0.3.0: mean(sphere sizes)/n, and the last sphere is recorded as 1.
        assert t1["0.3.0"] == pytest.approx(np.mean([5, 1]) / 10)

    def test_fast_t1_matches_ecol_044_on_separated_clusters(self):
        X, y = _two_clusters()
        ecol = ecol_measures([(X, y)], "0.4.4", groups=("neighborhood",))
        fast = complexity_data3(X, y, ["neighborhood"], ["T1"])[0]
        assert fast == pytest.approx(ecol.set_index("measure").loc["T1", "value"])
