"""Complexity backend switch (ADR 0024).

`backend="fast"` is the historical fast_adapter; `"ecol-0.3.0"` swaps T1, N1,
LSC and F4 for the exact ECoL 0.3.0 ports and keeps every other measure.
"""

from __future__ import annotations

import numpy as np
import pytest

from pos.complexity.backends import BACKENDS, ported_measures
from pos.complexity.base import HEADER
from pos.complexity.ecol_exact import f4_ecol, lsc_ecol, n1_ecol
from pos.complexity.fast_adapter import complexity_data3
from pos.complexity.t1_ecol import t1_ecol

GROUPS = ["overlapping", "neighborhood"]
PORTS = {"F4": f4_ecol, "N1": n1_ecol, "LSC": lsc_ecol,
         "T1": lambda X, y: t1_ecol(X, y, "0.3.0")}


@pytest.fixture
def data():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 90)
    return rng.normal(size=(90, 4)) + y[:, None] * 0.7, y


def test_backends_are_named():
    assert BACKENDS == ("fast", "ecol-0.3.0")
    assert ported_measures("ecol-0.3.0") == {"F4", "N1", "LSC", "T1"}
    assert ported_measures("fast") == set()


def test_default_backend_is_fast(data):
    X, y = data
    assert complexity_data3(X, y, GROUPS) == complexity_data3(X, y, GROUPS, backend="fast")


def test_ecol_backend_swaps_only_the_ported_measures(data):
    X, y = data
    fast = complexity_data3(X, y, GROUPS)
    ecol = complexity_data3(X, y, GROUPS, backend="ecol-0.3.0")
    names = [h.split(".")[1] for h in HEADER if h.split(".")[0] in GROUPS]
    for name, f, e in zip(names, fast, ecol, strict=True):
        expected = PORTS[name](X, y) if name in PORTS else f
        assert e == pytest.approx(expected, abs=1e-12), name


@pytest.mark.parametrize("types", [["F4", "T1"], ["F1", "N1"], ["F3", "LSC"]])
def test_ecol_backend_with_one_measure_per_group(data, types):
    X, y = data
    got = complexity_data3(X, y, GROUPS, types, backend="ecol-0.3.0")
    fast = complexity_data3(X, y, GROUPS, types)
    for name, g, f in zip(types, got, fast, strict=True):
        assert g == pytest.approx(PORTS[name](X, y) if name in PORTS else f, abs=1e-12)


def test_unknown_backend_is_rejected(data):
    X, y = data
    with pytest.raises(ValueError, match="backend"):
        complexity_data3(X, y, GROUPS, backend="ecol-0.4.4")
