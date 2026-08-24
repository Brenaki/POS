"""Pool generation modes and the dispatch the run recorder uses.

ADR 0019 took the number of modes from three to five. The dispatch lives here
rather than in `run_recorder._run_fold` so that adding a mode does not push
either module past the 150 LOC cap (ADR 0002), and so the list of modes that
generate bags — and therefore need the stricter viability gate — has one home.

    pgdcs    GA + per-dataset complexity-measure vote  (PGDCS as published)
    ga       GA with F1/T1 fixed                       (what every run so far used)
    randbag  the GA's generation-0 population, no search
    bagging  bootstrap + decision trees
    rf       bootstrap + feature subspace + decision trees
"""

from __future__ import annotations

from pos.oracle.fold_splitter import BAG_MODES  # noqa: F401  (re-export)
from pos.oracle.run_helpers import (
    build_pool_bagging,
    build_pool_ga,
    build_pool_rf,
)

MODES = ("pgdcs", "ga", "randbag", "bagging", "rf")


def build_pool_pgdcs(X_tr, y_tr, X_val, y_val, nr_generation, random_state,
                     jobs=1, classifier="perc"):
    """PGDCS as published: `types=None` re-enables the measure vote.

    Returns (pool, types). The chosen measures are a result, not telemetry —
    they answer which measures each dataset asks for, and how often that lands
    on the F1/T1 pair the other runs hardcode.
    """
    from pool_generation import poolGeneration

    pg = poolGeneration(nr_generation=nr_generation, iteration=1,
                        classifier=classifier, types=None, jobs=jobs,
                        random_state=random_state)
    pg.generate(X_tr, y_tr, X_val, y_val, iteration=1)
    return pg.get_pool(), list(pg.types or [])


def build_pool_randbag(X_tr, y_tr, X_val, y_val, M, random_state):
    """Random stratified bags + Perceptron — the generation control."""
    from pos.oracle.randbag_pool import build_randbag_pool
    return build_randbag_pool(X_tr, y_tr, X_val, y_val, M=M,
                              random_state=random_state)


def build_pool_for_mode(mode, X_tr, y_tr, X_val, y_val, M, nr_generation,
                        random_state, jobs=1, base_classifier="perc"):
    """Return (pool, extra_metrics). (None, {}) for an unknown mode."""
    if mode == "ga":
        return build_pool_ga(X_tr, y_tr, X_val, y_val, nr_generation,
                             random_state, jobs=jobs,
                             classifier=base_classifier), {}
    if mode == "pgdcs":
        pool, types = build_pool_pgdcs(X_tr, y_tr, X_val, y_val, nr_generation,
                                       random_state, jobs=jobs,
                                       classifier=base_classifier)
        return pool, {"pgdcs_types": types}
    if mode == "randbag":
        return build_pool_randbag(X_tr, y_tr, X_val, y_val, M, random_state), {}
    if mode == "rf":
        return build_pool_rf(X_tr, y_tr, M, random_state), {}
    if mode == "bagging":
        return build_pool_bagging(X_tr, y_tr, M, random_state), {}
    return None, {}
