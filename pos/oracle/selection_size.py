"""Measure how much of the pool each dynamic-selection method actually keeps.

ADR 0019. The strongest conclusion of the ADR 0018 run — the axis that orders
the fusers is how aggressively each one prunes the pool — was read off the
*nature* of the algorithms (OLA keeps one classifier, KNORA-U keeps everyone
locally correct). This measures it: E[S]/M, the mean fraction of the pool
selected per query.

Every DESlib method routes its decision through `select(competences)`, so
wrapping that one method records what the method actually did, with no
reimplementation to drift out of sync. Two shapes come back:

    (n_queries, M) boolean mask  -> DES: count the True entries per row
    (n_queries,)   index array   -> DCS: exactly one classifier per query

DESlib also short-circuits queries whose region of competence agrees
unanimously; those never reach `select`. `routed_frac` reports how many did,
because E[S]/M without it is a mean over an undeclared subset.
"""

from __future__ import annotations

import numpy as np

# Static methods have no region of competence, so nothing to measure: their
# pruning is a constructor argument, not a per-query decision.
STATIC_SEL_FRAC = {"single_best": None, "static_sel": 0.5}


class SelectionSpy:
    """Records the size of each selection a fitted DS method makes."""

    def __init__(self) -> None:
        self.sizes: list[np.ndarray] = []

    def attach(self, ds) -> None:
        """Shadow `ds.select` on the instance; the class stays untouched."""
        original = ds.select

        def spy(competences, *args, **kwargs):
            out = original(competences, *args, **kwargs)
            self._record(out)
            return out

        ds.select = spy

    def _record(self, out) -> None:
        arr = np.asarray(out)
        if arr.ndim == 2:
            if arr.dtype == bool:
                self.sizes.append(arr.sum(axis=1).astype(float))
            else:  # index matrix (DES-KNN): one row of selected indices
                self.sizes.append(np.full(arr.shape[0], float(arr.shape[1])))
        elif arr.ndim == 1:
            self.sizes.append(np.ones(arr.shape[0], dtype=float))

    def summary(self, M: int, n_queries: int, ds=None) -> dict[str, float | None]:
        """Mean selected fraction, and the share of queries that reached DS."""
        if not self.sizes or M <= 0:
            return weighting_summary(ds, n_queries) if ds is not None else \
                {"sel_frac": None, "routed_frac": None}
        all_sizes = np.concatenate(self.sizes)
        routed = float(len(all_sizes) / n_queries) if n_queries else None
        return {"sel_frac": float(all_sizes.mean() / M), "routed_frac": routed}


def weighting_summary(ds, n_queries: int) -> dict[str, float | None]:
    """KNORA-U and KNOP never call `select` — they weight instead of prune.

    DESlib marks them `mode="weighting"`: every classifier votes, scaled by its
    local competence. So the selected fraction is 1.0 by construction, and the
    fact that the two least-pruning methods are also the two that never lose to
    majority vote is the finding, not an artefact of the instrumentation.
    """
    if getattr(ds, "mode", None) != "weighting":
        return {"sel_frac": None, "routed_frac": None}
    return {"sel_frac": 1.0, "routed_frac": 1.0 if n_queries else None}


def static_summary(name: str, M: int) -> dict[str, float | None]:
    """Selected fraction of a static method, known without measuring."""
    frac = STATIC_SEL_FRAC.get(name)
    if frac is None and name == "single_best":
        frac = 1.0 / M if M else None
    return {"sel_frac": frac, "routed_frac": 1.0 if frac is not None else None}
