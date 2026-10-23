"""The `monteiro-2023-dt-compatible-v2` profile: v1 with ECoL 0.3.0 measures (ADR 0024).

v1 must stay byte-identical (its fingerprint is the one of the published
run), and the backend must reach both places that compute complexity: the
measure vote and the GA fitness.
"""

from __future__ import annotations

import numpy as np
import pytest

from pos.oracle import pgdcs_treatment
from pos.oracle.dataset_gate import TABLE_2, dataset_reference_for
from pos.oracle.profile_runner import profile_splits
from pos.oracle.profiles import (
    MONTEIRO_PROFILES,
    PROFILE_NAMES,
    materialize_profile,
    profile_for,
)

V1, V2 = "monteiro-2023-dt-compatible-v1", "monteiro-2023-dt-compatible-v2"


def test_v2_is_registered_as_a_monteiro_profile():
    assert V2 in PROFILE_NAMES
    assert MONTEIRO_PROFILES == (V1, V2)


def test_v2_is_v1_plus_the_ecol_backend():
    v1, v2 = profile_for(V1), profile_for(V2)
    assert v2.pop("complexity_backend") == "ecol-0.3.0"
    assert v2.pop("profile") == V2
    v1.pop("profile")
    assert v1 == v2


def test_published_fingerprints_do_not_move():
    # results/experiments/2026-08-27T19-47-09_7c5ec6d and ..T23-01-32_7c5ec6d
    assert "complexity_backend" not in profile_for(V1)
    assert materialize_profile(V1)["fingerprint"] == "sha256:f318d8ef6145758e"
    assert materialize_profile("pos-oracle-v2")["fingerprint"] == "sha256:3a3a672740722258"


def test_v2_runs_the_monteiro_protocol():
    assert dataset_reference_for(V2) == TABLE_2
    X = np.random.default_rng(0).normal(size=(80, 3))
    y = np.repeat([0, 1], 40)
    cfg = materialize_profile(V2, scale="smoke")
    assert cfg["comparable"] is False
    assert len(profile_splits(cfg, X, y)) == cfg["n_replications"]


def test_treatment_hands_the_backend_to_vote_and_ga(monkeypatch):
    seen = {}

    def fake_vote(X, y, frac, random_state=None, backend="fast"):
        seen["vote"] = backend
        return ["F4", "T1"]

    class FakeEvolution:
        def __init__(self, **kw):
            seen["ga"] = kw.get("complexity_backend")

        def run(self, *a):
            return type("T", (), {"best_a": {}, "best_d": {}, "generation_trace": []})()

    monkeypatch.setattr(pgdcs_treatment, "monteiro_get_best_types", fake_vote)
    monkeypatch.setattr(pgdcs_treatment, "PgdcsEvolution", FakeEvolution)
    cfg = materialize_profile(V2, scale="smoke")
    pgdcs_treatment.build_pgdcs_result(None, None, None, None, cfg)
    assert seen == {"vote": "ecol-0.3.0", "ga": "ecol-0.3.0"}

    seen.clear()
    pgdcs_treatment.build_pgdcs_result(None, None, None, None,
                                       materialize_profile(V1, scale="smoke"))
    assert seen == {"vote": "fast", "ga": "fast"}


@pytest.mark.parametrize("module", ["pos.pool.monteiro_voter", "pos.pool.fitness_evaluator"])
def test_vote_and_fitness_compute_with_the_backend(monkeypatch, module):
    import importlib

    from pos.complexity import fast_adapter
    mod = importlib.import_module(module)
    backends = set()

    def spy(*a, backend="fast", **kw):
        backends.add(backend)
        return fast_adapter.complexity_data3(*a, backend=backend, **kw)

    monkeypatch.setattr(mod, "complexity_data3", spy)
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(40, 3), rng.randn(40, 3) + 2.0])
    y = np.repeat([0, 1], 40)
    if module.endswith("monteiro_voter"):
        mod.monteiro_get_best_types(X, y, 0.4, random_state=1, backend="ecol-0.3.0")
    else:
        from pos.pool.pgdcs_evolution import PgdcsEvolution
        PgdcsEvolution(nr_generation=1, nr_individual=4, nr_pop=4, nr_child=4,
                       nr_bags=4, types=["F4", "T1"], jobs=2, random_state=1,
                       complexity_backend="ecol-0.3.0").run(X, y, X, y)
    assert backends == {"ecol-0.3.0"}
