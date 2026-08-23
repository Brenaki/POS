"""Dynamic selection (DCS/DES) baselines for a fitted pool — milestone 6.

Objective 6 of the subproject asks Oracle_N to be compared against real
combination methods. Majority vote and soft fusion live in `comparison.py`;
this module adds the dynamic-selection family via DESlib, which is exactly
what the first run motivated: the GA pool reached Oracle_1 = 0.9979 while its
majority vote reached 0.7824, so the information is in the pool and the
static fuser does not extract it.

Protocol: the region of competence is estimated on the validation split that
`fold_splitter.stratified_val_split` already holds out of pool training
(DSEL). No test data and no pool-training data enters the competence
estimate, which is the standard DESlib setup (Cruz et al. 2018).
"""

from __future__ import annotations

import numpy as np

from pos.oracle.deslib_compat import install_numpy_aliases

DES_METHODS = ("ola", "lca", "knorae", "knorau", "metades")
DEFAULT_K = 7


def _method_classes() -> dict:
    """Lazy import: deslib is heavy and only needed when a run asks for it."""
    install_numpy_aliases()
    from deslib.dcs import LCA, OLA
    from deslib.des import KNORAE, KNORAU, METADES

    return {"ola": OLA, "lca": LCA, "knorae": KNORAE,
            "knorau": KNORAU, "metades": METADES}


def effective_k(n_dsel: int, k: int = DEFAULT_K) -> int:
    """Largest usable neighbourhood: DESlib needs k < len(DSEL)."""
    return max(1, min(k, n_dsel - 1))


def evaluate_des(pool, X_dsel, y_dsel, X_test, y_test, k: int = DEFAULT_K,
                 random_state: int = 42, methods=DES_METHODS,
                 n_jobs: int = 1) -> tuple[dict, dict]:
    """Accuracy of each DCS/DES method on (X_test, y_test).

    Returns (accuracies, notes). A method that cannot run on this pool gets
    None in `accuracies` and the reason in `notes`. That is expected, not an
    error: METADES needs `predict_proba`, which the linear Perceptron of the
    thesis does not expose — the same limitation ADR 0016 handled for soft
    fusion — and a very small validation split can leave fewer samples than
    the neighbourhood needs.
    """
    available = _method_classes()
    accs: dict[str, float | None] = {}
    notes: dict[str, str] = {}
    k_eff = effective_k(len(y_dsel), k)
    for name in methods:
        accs[name] = None
        cls = available.get(name)
        if cls is None:
            notes[name] = "unknown method"
            continue
        try:
            ds = cls(pool_classifiers=list(pool), k=k_eff,
                     random_state=random_state, n_jobs=n_jobs)
            ds.fit(np.asarray(X_dsel), np.asarray(y_dsel))
            accs[name] = float(np.mean(ds.predict(np.asarray(X_test)) == y_test))
        except Exception as exc:  # recorded, never fatal — see docstring
            notes[name] = f"{type(exc).__name__}: {exc}"[:200]
    return accs, notes


def des_columns(accs: dict | None, methods=DES_METHODS) -> dict:
    """Flatten the accuracy dict into summary.csv columns (`des_ola`, ...)."""
    accs = accs or {}
    return {f"des_{m}": accs.get(m) for m in methods}


def best_des(accs: dict | None, methods=DES_METHODS):
    """(name, accuracy) of the best method that ran, or (None, None).

    The per-fold maximum is optimistic as a *method* estimate, but that is
    not what it is used for: it answers "how much of the Oracle_1 - MV gap is
    reachable by dynamic selection at all", which is the question objective 8
    poses when it asks for a prior analysis of a pool's potential.
    """
    accs = accs or {}
    scored = [(m, a) for m, a in accs.items() if m in methods and a is not None]
    if not scored:
        return None, None
    return max(scored, key=lambda t: t[1])
