"""Characterization tests for Cpx.complexity_data3 (R/ECoL-dependent).

These are skipped automatically when R is not on PATH (see conftest.py).
When R IS available, they capture golden values from ECoL so we can audit
the behavior change when we replace R with pyhard in Fase 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# conftest.py injects rpy2 mocks, but the REAL rpy2 must be importable for
# these tests. We re-import Cpx fresh after ensuring real rpy2 is loaded.
# The requires_r marker skips the whole module if R is missing.

pytestmark = pytest.mark.requires_r


# Header from Cpx.py — 22 measures in 6 groups
EXPECTED_HEADER = [
    "overlapping.F1", "overlapping.F1v", "overlapping.F2", "overlapping.F3", "overlapping.F4",
    "neighborhood.N1", "neighborhood.N2", "neighborhood.N3", "neighborhood.N4",
    "neighborhood.T1", "neighborhood.LSC",
    "linearity.L1", "linearity.L2", "linearity.L3",
    "dimensionality.T2", "dimensionality.T3", "dimensionality.T4",
    "balance.C1", "balance.C2",
    "network.Density", "network.ClsCoef", "network.Hubs",
]


def _import_real_cpx():
    """Import Cpx with the REAL rpy2 (not the mock). Requires R on PATH."""
    import sys
    # Remove the mock so the real rpy2 loads
    for mod in list(sys.modules):
        if mod.startswith("rpy2"):
            del sys.modules[mod]
    if "Cpx" in sys.modules:
        del sys.modules["Cpx"]
    import Cpx
    return Cpx


class TestComplexityData3:
    def test_overlapping_group_returns_5_measures(self, wine_split):
        Cpx = _import_real_cpx()
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["overlapping"])
        assert len(result) == 5
        assert all(isinstance(v, float) for v in result)

    def test_neighborhood_group_returns_6_measures(self, wine_split):
        Cpx = _import_real_cpx()
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["neighborhood"])
        assert len(result) == 6

    def test_linearity_group_returns_3_measures(self, wine_split):
        Cpx = _import_real_cpx()
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["linearity"])
        assert len(result) == 3

    def test_dimensionality_group_returns_3_measures(self, wine_split):
        Cpx = _import_real_cpx()
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["dimensionality"])
        assert len(result) == 3

    def test_balance_group_returns_2_measures(self, wine_split):
        Cpx = _import_real_cpx()
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["balance"])
        assert len(result) == 2

    def test_network_group_returns_3_measures(self, wine_split):
        Cpx = _import_real_cpx()
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["network"])
        assert len(result) == 3

    def test_combined_overlapping_neighborhood_returns_11(self, wine_split):
        Cpx = _import_real_cpx()
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["overlapping", "neighborhood"])
        assert len(result) == 11

    def test_all_six_groups_returns_22(self, wine_split):
        Cpx = _import_real_cpx()
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(
            X, y, ["overlapping", "neighborhood", "linearity",
                   "dimensionality", "balance", "network"]
        )
        assert len(result) == 22

    def test_golden_values_overlapping_neighborhood_wine(self, wine_split):
        """Capture golden ECoL values for load_wine train split (rs=42).

        These values are the REFERENCE that Fase 3 (pyhard replacement) will
        be compared against. If this test runs, it also writes the values to
        `tests/_ecol_legacy_golden.json` for archival.
        """
        Cpx = _import_real_cpx()
        X, y = wine_split["X_train"], wine_split["y_train"]
        result = Cpx.complexity_data3(X, y, ["overlapping", "neighborhood"])

        # The exact values depend on ECoL version + R version; we assert
        # finiteness and reasonable range here, and archive the actual values.
        assert all(np.isfinite(result))
        assert all(0.0 <= v <= 1.0 for v in result)  # most ECoL measures are in [0,1]

        # Archive for Fase 3 comparison
        golden_path = Path(__file__).resolve().parent.parent / "_ecol_legacy_golden.json"
        header_11 = EXPECTED_HEADER[:11]  # overlapping (5) + neighborhood (6)
        archive = {
            "dataset": "load_wine",
            "split": {"test_size": 0.4, "random_state": 42, "stratify": True},
            "group": ["overlapping", "neighborhood"],
            "measures": header_11,
            "values": result,
            "note": "Captured by test_golden_values_overlapping_neighborhood_wine with real R/ECoL.",
        }
        golden_path.write_text(json.dumps(archive, indent=2))
