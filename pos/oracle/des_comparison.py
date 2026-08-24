"""Dynamic selection (DCS/DES) baselines for a fitted pool — milestone 6.

Objective 6 of the subproject asks Oracle_N to be compared against real
combination methods. Majority vote and soft fusion live in `comparison.py`;
this module adds the dynamic-selection family via DESlib, which is exactly
what the first run motivated: the GA pool reached Oracle_1 = 0.9979 while its
majority vote reached 0.7824, so the information is in the pool and the
static fuser does not extract it.

Protocol: the region of competence is estimated on `X_dsel`, the slice that
`fold_splitter.stratified_three_way_split` holds out of both pool training
and the GA's fitness (ADR 0018). Under the two-way split of ADR 0017 the
DSEL was the GA's own fitness set, which made its numbers optimistic.

This returns *predictions*, not just accuracy: milestone 7 needs macro
precision/recall/F1 too, and storing only the accuracy scalar is what forced
this rerun in the first place.
"""

from __future__ import annotations

import numpy as np

from pos.oracle.des_methods import (
    DEFAULT_K,
    DES_METHODS,
    STATIC_METHODS,
    method_classes,
    takes_k,
)
from pos.oracle.metrics import METRIC_NAMES, prediction_metrics
from pos.oracle.selection_size import SelectionSpy, static_summary


def effective_k(n_dsel: int, k: int = DEFAULT_K) -> int:
    """Largest usable neighbourhood: DESlib needs k < len(DSEL)."""
    return max(1, min(k, n_dsel - 1))


def evaluate_des(pool, X_dsel, y_dsel, X_test, y_test, k: int = DEFAULT_K,
                 random_state: int = 42, methods=DES_METHODS,
                 n_jobs: int = 1) -> tuple[dict, dict, dict]:
    """Run every method on (X_test, y_test). Returns (metrics, preds, notes).

    A method that cannot run on this pool gets None in `metrics` and the
    reason in `notes`. That is expected, not an error: META-DES and KNOP need
    `predict_proba`, which the linear Perceptron of the thesis does not
    expose — the same limitation ADR 0016 handled for soft fusion — and a
    very small DSEL can leave fewer samples than the neighbourhood needs.
    """
    available = method_classes()
    metrics: dict[str, dict | None] = {}
    preds: dict[str, np.ndarray] = {}
    notes: dict[str, str] = {}
    k_eff = effective_k(len(y_dsel), k)
    for name in methods:
        metrics[name] = None
        cls = available.get(name)
        if cls is None:
            notes[name] = "unknown method"
            continue
        try:
            kwargs = {"pool_classifiers": list(pool), "random_state": random_state}
            if takes_k(name):
                kwargs["k"] = k_eff
                kwargs["n_jobs"] = n_jobs
            ds = cls(**kwargs)
            ds.fit(np.asarray(X_dsel), np.asarray(y_dsel))
            spy = _attach_spy(name, ds)
            y_pred = ds.predict(np.asarray(X_test))
            preds[name] = np.asarray(y_pred)
            got = prediction_metrics(y_test, y_pred)
            got.update(_selection_summary(name, spy, ds, len(pool), len(y_test)))
            metrics[name] = got
        except Exception as exc:  # recorded, never fatal — see docstring
            notes[name] = f"{type(exc).__name__}: {exc}"[:200]
    return metrics, preds, notes


def _attach_spy(name: str, ds):
    """Record what the method selects. Never let instrumentation kill a run."""
    if name in STATIC_METHODS:
        return None
    try:
        spy = SelectionSpy()
        spy.attach(ds)
        return spy
    except Exception:  # same policy as the caller: measured or noted, never fatal
        return None


def _selection_summary(name, spy, ds, M, n_queries) -> dict:
    """E[S]/M for this method, measured (dynamic) or known (static)."""
    if name in STATIC_METHODS:
        return static_summary(name, M)
    if spy is None:
        return {"sel_frac": None, "routed_frac": None}
    try:
        return spy.summary(M, n_queries, ds=ds)
    except Exception:
        return {"sel_frac": None, "routed_frac": None}


def des_accuracies(metrics: dict | None, methods=DES_METHODS) -> dict:
    """Just the accuracy of each method, for the code that only needs that."""
    metrics = metrics or {}
    return {m: (metrics.get(m) or {}).get("acc") for m in methods}


def des_columns(metrics: dict | None, methods=DES_METHODS) -> dict:
    """Flatten into summary.csv columns: `des_ola`, `des_ola_f1`, ...

    The bare `des_<method>` name stays accuracy, so every column the ADR 0017
    run wrote keeps its meaning and the two runs line up.
    """
    metrics = metrics or {}
    out: dict[str, float | None] = {}
    for m in methods:
        got = metrics.get(m) or {}
        out[f"des_{m}"] = got.get("acc")
        for name in METRIC_NAMES:
            if name != "acc":
                out[f"des_{m}_{name}"] = got.get(name)
        # ADR 0019: the pruning axis, measured instead of inferred.
        out[f"des_{m}_sel_frac"] = got.get("sel_frac")
        out[f"des_{m}_routed_frac"] = got.get("routed_frac")
    return out


def best_des(metrics: dict | None, methods=DES_METHODS, metric: str = "acc"):
    """(name, score) of the best method that ran, or (None, None).

    The per-fold maximum is optimistic as a *method* estimate, but that is
    not what it is used for: it answers "how much of the Oracle_1 - MV gap is
    reachable by dynamic selection at all", which is the question objective 8
    poses when it asks for a prior analysis of a pool's potential.
    """
    metrics = metrics or {}
    scored = [(m, (metrics.get(m) or {}).get(metric)) for m in methods]
    scored = [(m, v) for m, v in scored if v is not None]
    if not scored:
        return None, None
    return max(scored, key=lambda t: t[1])
