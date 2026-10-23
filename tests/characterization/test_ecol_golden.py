"""Python backends against the frozen ECoL 0.3.0 golden values (ADR 0003, 0023).

`tests/_ecol_legacy_golden.json` holds what the legacy `Cpx.complexity_data3`
computed through R for the `sample.ipynb` wine split (regenerate with
`scripts/gen_ecol_golden.py`). Every one of the 22 measures is either
reproduced exactly by `fast_adapter` or listed in KNOWN_DEVIATIONS with why.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pos.complexity.base import GROUP_MEASURES, HEADER
from pos.complexity.ecol_exact import f4_ecol, lsc_ecol, n1_ecol
from pos.complexity.fast_adapter import complexity_data3
from pos.complexity.t1_ecol import t1_ecol

GOLDEN = json.loads((Path(__file__).parents[1] / "_ecol_legacy_golden.json").read_text())
VALUES = GOLDEN["values"]

EXACT = ["overlapping.F1", "overlapping.F1v", "overlapping.F2", "overlapping.F3",
         "neighborhood.N2", "neighborhood.N3",
         "dimensionality.T2", "dimensionality.T3", "dimensionality.T4",
         "balance.C1", "balance.C2"]

KNOWN_DEVIATIONS = {
    "overlapping.F4": "fast keeps the class ranges of the full pair; ECoL recomputes them "
                      "on the remaining points after each removal (exact port: f4_ecol)",
    "neighborhood.N1": "fast: mean share of cross-class MST neighbours on a 10-NN graph; "
                       "ECoL: share of points with any cross-class edge on the full MST "
                       "(exact port: n1_ecol)",
    "neighborhood.N4": "fast computes kDN; ECoL's N4 classifies random interpolated points",
    "neighborhood.T1": "fast: radius = d(nearest enemy)/2 and #spheres/n; ECoL 0.3.0: "
                       "recursive radius and mean(sphere size)/n (exact port: t1_ecol)",
    "neighborhood.LSC": "fast divides each local set by its class size; ECoL by n "
                        "(exact port: lsc_ecol)",
    "linearity.L1": "not implemented (0.0): needs ECoL's e1071 linear SVM",
    "linearity.L2": "not implemented (0.0)",
    "linearity.L3": "not implemented (0.0)",
    "network.Density": "not implemented (0.0): ECoL builds an epsilon-NN graph in igraph",
    "network.ClsCoef": "not implemented (0.0)",
    "network.Hubs": "not implemented (0.0)",
}


@pytest.fixture(scope="module")
def fast_values(wine_split_no_stratify):
    w = wine_split_no_stratify
    vals = complexity_data3(w["X_train"], w["y_train"], list(GROUP_MEASURES))
    return dict(zip(HEADER, vals, strict=True))


def test_every_measure_is_classified():
    assert sorted(EXACT + list(KNOWN_DEVIATIONS)) == sorted(HEADER)
    assert sorted(VALUES) == sorted(HEADER)


@pytest.mark.parametrize("name", EXACT)
def test_fast_backend_reproduces_ecol(fast_values, name):
    assert fast_values[name] == pytest.approx(VALUES[name], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("name, port", [
    ("neighborhood.T1", lambda X, y: t1_ecol(X, y, "0.3.0")),
    ("neighborhood.N1", n1_ecol), ("neighborhood.LSC", lsc_ecol), ("overlapping.F4", f4_ecol)])
def test_exact_ports_reproduce_ecol_030(wine_split_no_stratify, name, port):
    w = wine_split_no_stratify
    assert port(w["X_train"], w["y_train"]) == pytest.approx(VALUES[name], rel=1e-9, abs=1e-12)


@pytest.mark.requires_r
def test_golden_file_matches_a_fresh_r_run():
    from pos.complexity.ecol_bridge import ecol_measures
    from scripts.gen_ecol_golden import VERSION, wine_train
    X, y = wine_train()
    df = ecol_measures([(X, y)], VERSION, tuple(GROUP_MEASURES))
    for g, m, v in zip(df["group"], df["measure"], df["value"], strict=True):
        if VALUES[f"{g}.{m}"] is not None:
            assert v == pytest.approx(VALUES[f"{g}.{m}"], rel=1e-9), f"{g}.{m}"
