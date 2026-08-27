"""PGDCS generation tracker (ADR 0020 T6-T9).

Collects (generation, MVR accuracy, Gdisp, fitted pool) for gen=0..20 and
selects the best generation per criterion:

- PGDCS_A = argmax MVR validation accuracy (gen 0 excluded)
- PGDCS_D = argmax Gdisp (gen 0 excluded)
- Tie -> oldest generation (strict `<` update, never `<=`)
- The fitted classifiers of the winning generation are kept (Algoritmo 2),
  not reconstructed from bags.

One evolution per fold/replication produces a single trajectory; PGDCS_A and
PGDCS_D diverge only at the exit criterion, so sharing the trajectory is an
algorithmically consistent cost optimization (`shared_evolution_trajectory`).
"""

from __future__ import annotations

from typing import Any


class PgdcsTracker:
    """Record per-generation results and select PGDCS_A / PGDCS_D pools.

    ADR 0020 §3.2:
    - 20 candidate generations (gen=1..20);
    - gen 0 is evaluated (record) but never selected;
    - tie-break: keep the oldest generation.
    """

    def __init__(self) -> None:
        self.generation_trace: list[dict[str, Any]] = []
        self.best_a: dict[str, Any] | None = None
        self.best_d: dict[str, Any] | None = None

    def record(self, gen: int, mvr: float, gdisp: float, pool: list) -> None:
        """Record one generation's (MVR, Gdisp, fitted pool). gen=0 allowed."""
        self.generation_trace.append({
            "gen": gen, "mvr": float(mvr), "gdisp": float(gdisp), "pool": pool,
        })

    def finalize(self) -> None:
        """Select PGDCS_A (max MVR) and PGDCS_D (max Gdisp) over gen=1..N."""
        candidates = [e for e in self.generation_trace if e["gen"] > 0]
        if not candidates:
            raise ValueError("no candidate generations (only gen 0 recorded)")
        self.best_a = _select_oldest_on_tie(candidates, key="mvr")
        self.best_d = _select_oldest_on_tie(candidates, key="gdisp")


def _select_oldest_on_tie(
    candidates: list[dict[str, Any]], key: str
) -> dict[str, Any]:
    """Pick max by `key`; on tie return the entry with the smallest `gen`.

    ADR 0020: update only when `value > current_best` (strict), so the first
    (oldest) generation with the max value wins.
    """
    best = candidates[0]
    for entry in candidates[1:]:
        if entry[key] > best[key]:
            best = entry
    return best