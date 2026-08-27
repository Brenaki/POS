"""PGDCS evolution with per-generation tracking (ADR 0020 §3.2).

`poolGeneration` keeps a single best generation under one stop criterion and
rebuilds the pool from the winning bags. Monteiro 2023 needs both criteria
from the same trajectory and the winning generation's *own* fitted
classifiers (Algorithm 2), so this subclass:

- draws bags randomly **with replacement**, not stratified (class coverage is
  the genetic operators' job);
- records `(gen, MVR validation accuracy, Gdisp, fitted pool)` for gen 0..N
  into a `PgdcsTracker`, which excludes gen 0 from selection and breaks ties
  toward the oldest generation;
- keeps the estimators the GA itself scored (`keep_estimators`), so the pool
  that is evaluated is the pool that was selected.

MVR is read from the cached per-bag validation predictions rather than
refitting an EnsembleVoteClassifier — the same numbers, without ~M refits per
generation (ADR 0015).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from pos.dispersion import dispersion
from pos.pool.bag_generator import generate_bags_random_replacement
from pos.pool.pgdcs_tracker import PgdcsTracker
from pos.pool.pool_generation import poolGeneration


class PgdcsEvolution(poolGeneration):
    """One PGDCS trajectory; PGDCS_A and PGDCS_D are read from its tracker."""

    def __init__(self, tracker: PgdcsTracker | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.keep_estimators = True
        self.tracker = tracker if tracker is not None else PgdcsTracker()

    def generate_bags(self, X_train, y_train):
        """Random sampling with replacement (ADR 0020 §3.3)."""
        return generate_bags_random_replacement(
            X_train, y_train, self.nr_bags, self.tam_bags,
            random_state=self.random_state,
        )

    def _population_mvr_and_pool(self, names: list[int]):
        """(MVR accuracy on validation, fitted pool) for these bag names."""
        entries = [self._eval_cache[n] for n in names]
        preds = np.array([e[2] for e in entries])
        voted = stats.mode(preds, axis=0, keepdims=False).mode
        mvr = float(np.mean(voted == np.asarray(self.y_val)))
        return mvr, [e[3] for e in entries]

    def the_function(self, population, gen, fitness):
        """Per-generation callback: record, then refresh fitness state."""
        self.generation = gen
        self.off = [ind[0] for ind in population]
        if isinstance(fitness, np.ndarray) and fitness.ndim == 2:
            gdisp = float(np.mean(dispersion(fitness)))
        else:
            gdisp = 0.0
        mvr, pool = self._population_mvr_and_pool(self.off)
        self.tracker.record(gen=gen, mvr=mvr, gdisp=gdisp, pool=pool)
        if self.method_disperse and gen != self.nr_generation:
            self.get_complexity(population=population)
        return population

    def run(self, X_train, y_train, X_val, y_val) -> PgdcsTracker:
        """Run one trajectory (20 generations) and finalize the tracker."""
        self.generate(X_train, y_train, X_val, y_val, iteration=1)
        self.tracker.finalize()
        return self.tracker
