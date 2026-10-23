"""PGDCS treatment adapter for the profile runner (ADR 0020 §3).

Turns a profile config plus one (train, validation) pair into the tracker
result the runner consumes:

    {"selected_measures": [...], "best_a": {...}, "best_d": {...}}

Measure selection follows Monteiro 2023: 10 rounds x 100 random-with-
replacement sub-samples, one Overlapping + one Neighborhood measure, ties
broken by mean dispersion. `measure_selection="once_per_dataset"` (the
Monteiro profile, whose Table 4 is per dataset) lets the recorder vote once
and pass the result back in `_selected_measures`; `"per_fold"` (the POS
profile) votes inside each fold's own training data. `complexity_backend`
(ADR 0024) picks the measure definitions for both the vote and the GA.
"""

from __future__ import annotations

from typing import Any

from pos.pool.monteiro_voter import monteiro_get_best_types
from pos.pool.pgdcs_evolution import PgdcsEvolution


def select_measures(X_tr, y_tr, cfg: dict[str, Any], seed) -> list[str]:
    """One Overlapping + one Neighborhood measure for this training data."""
    return monteiro_get_best_types(
        X_tr, y_tr, cfg.get("measure_fraction", 0.5), random_state=seed,
        backend=cfg.get("complexity_backend", "fast"),
    )


def build_pgdcs_result(
    X_tr, y_tr, X_val, y_val, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Run one PGDCS trajectory; return PGDCS_A and PGDCS_D from its tracker."""
    seed = cfg.get("_seed", cfg.get("random_state", 42))
    types = cfg.get("_selected_measures") or select_measures(X_tr, y_tr, cfg, seed)
    pg = PgdcsEvolution(
        nr_generation=cfg["nr_generation"],
        nr_individual=cfg["nr_individual"],
        nr_pop=cfg["nr_pop"],
        nr_child=cfg["nr_child"],
        proba_crossover=cfg["proba_crossover"],
        proba_mutation=cfg["proba_mutation"],
        tam_bags=cfg["tam_bags"],
        nr_bags=cfg["M"],
        classifier=cfg.get("classifier", "tree"),
        jobs=cfg.get("jobs", 1),
        types=list(types),
        random_state=seed,
        complexity_backend=cfg.get("complexity_backend", "fast"),
    )
    tracker = pg.run(X_tr, y_tr, X_val, y_val)
    return {
        "selected_measures": list(types),
        "best_a": tracker.best_a,
        "best_d": tracker.best_d,
        "generation_trace": [
            {"gen": e["gen"], "mvr": e["mvr"], "gdisp": e["gdisp"]}
            for e in tracker.generation_trace
        ],
    }
