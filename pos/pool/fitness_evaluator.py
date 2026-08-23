"""Fitness evaluation mixin: complexity, diversity, dispersion for GA.

Extracted from pool_generation.py during Fase 2.
Uses fast_adapter (numpy-only) instead of pyhard — no joblib nesting issues.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pos.classifiers import biuld_classifier, biuld_classifier_tree
from pos.complexity.fast_adapter import complexity_data3
from pos.dispersion import dispersion_linear
from pos.diversity import diversitys
from pos.normalization import min_max_norm
from pos.pool.bag_generator import build_bags
from pos.voting import voting_classifier


def _eval_one(i, bags_inst, X_train, y_train, X_val, y_val, classifier, group, types, random_state=None):
    """Module-level: evaluate one bag (thread-safe, no shared mutable state)."""
    indx_bag1 = bags_inst[i]
    X_bag, y_bag = build_bags(indx_bag1, X_train, y_train)
    cpx = complexity_data3(X_bag, y_bag, group, types)
    rs_i = random_state + i if isinstance(random_state, int) else None
    if classifier == "perc":
        estimator, score, pred = biuld_classifier(X_bag, y_bag, X_bag, y_bag, X_val, y_val,
                                                  random_state=rs_i)
    elif classifier == "tree":
        estimator, score, pred = biuld_classifier_tree(X_bag, y_bag, X_bag, y_bag, X_val, y_val, random_state=rs_i)
    return cpx, score, pred, estimator


class FitnessEvaluatorMixin:
    """Evaluate GA individuals on complexity/dispersion/diversity/accuracy."""

    def diversity_ga(self, pred, y):
        pred = np.array(pred)
        d = diversitys(y, pred)
        return d

    def parallel_distance2(self, i, bags, group, types):
        indx_bag1 = bags["inst"][i]
        X_bag, y_bag = build_bags(indx_bag1, self.X_train, self.y_train)
        cpx = complexity_data3(X_bag, y_bag, group, types)
        rs_i = self.random_state + i if isinstance(self.random_state, int) else None
        if self.classifier == "perc":
            estimator, score, pred = biuld_classifier(
                X_bag, y_bag, X_bag, y_bag, self.X_val, self.y_val,
                random_state=rs_i)
        elif self.classifier == "tree":
            estimator, score, pred = biuld_classifier_tree(
                X_bag, y_bag, X_bag, y_bag, self.X_val, self.y_val, random_state=rs_i)
        return cpx, score, pred, estimator

    def _eval_many(self, indices):
        """Evaluate parallel_distance2 for a list of bag indices, memoised.

        Bags are append-only and immutable once created, and the per-bag tree
        seed is `random_state + position`, which is also stable. So a bag's
        (complexity, score, predictions) is a pure function of its identity
        and can be cached by bag NAME.

        This matters because mu+lambda selection re-presents survivors every
        generation: `the_function` calls `get_complexity(population=...)` on
        100 individuals that were all evaluated already, so half of every
        generation's work was recomputation (ADR 0015).

        Fitted estimators are only retained under the `maxacc` stop criterion,
        the one path that consumes them — keeping ~2100 fitted trees alive
        otherwise would cost far more memory than the cache saves.
        """
        indices = list(indices)
        cache = self._eval_cache
        names = [self.bags["name"][i] for i in indices]
        todo = [(i, nm) for i, nm in zip(indices, names) if nm not in cache]
        keep_estimators = getattr(self, "stop_criteria", None) == "maxacc"

        if todo:
            if self.jobs > 1:
                with ThreadPoolExecutor(max_workers=self.jobs) as ex:
                    results = list(ex.map(
                        lambda t: _eval_one(t[0], self.bags["inst"], self.X_train,
                            self.y_train, self.X_val, self.y_val, self.classifier,
                            self.group, self.types, self.random_state),
                        todo))
            else:
                results = [self.parallel_distance2(i, self.bags, self.group,
                                                   self.types)
                           for i, _ in todo]
            for (_, nm), (cpx, score, pred, est) in zip(todo, results, strict=True):
                cache[nm] = (cpx, score, pred, est if keep_estimators else None)

        rows = [cache[nm] for nm in names]
        return tuple(zip(*rows)) if rows else ((), (), (), ())

    def get_complexity(self, first_evaluate=False, population=None):
        dist = {"name": list(), "dist": list(), "diver": list(),
                "score": list(), "score_g": list()}

        if first_evaluate and self.generation == 0:
            dist["name"] = self.pop
            c, score, pred, pool = self._eval_many(range(len(dist["name"])))
            self.c = c
        elif first_evaluate == False and population is None:
            # Offspring produced since the last call: names
            # [name_individual - nr_individual, name_individual).
            # Resolve each name to its position in self.bags instead of
            # assuming name == position (the old `range(100, nr_individual+100)`
            # only happened to be right for generation 1 — see ADR 0014).
            begin = self.name_individual - self.nr_individual
            names = list(range(begin, self.name_individual))
            for i in names:
                dist["name"].append([i])
            indices = [self.bags["name"].index(i) for i in names]
            c, score, pred, pool = self._eval_many(indices)
            self.c = c
        elif population is not None:
            dist["name"] = population
            indices = [self.bags["name"].index(i[0]) for i in population]
            c, score, pred, pool = self._eval_many(indices)
            self.c = c

        dist["dist"] = dispersion_linear(c)
        dist["score"] = score
        d = self.diversity_ga(pred, self.y_val)
        dist["diver"] = min_max_norm(d)
        # `score_g` is read only by the `maxacc` stop criterion. Computing it
        # unconditionally re-fitted all M classifiers on X_val on every call
        # (EnsembleVoteClassifier defaults to refit=True) — ~4 s per call on a
        # Magic fold, for a number that `maxdistance` never looks at (ADR 0015).
        if getattr(self, "stop_criteria", None) == "maxacc":
            dist["score_g"] = voting_classifier(pool, self.X_val, self.y_val)
        else:
            dist["score_g"] = None
        self.dist = dist
        return

    def evaluate_linear_dispersion(self, ind1):
        dist = self.dist
        dst1 = dist2 = diver = None
        for i in range(len(dist["name"])):
            if dist["name"][i][0] == ind1[0]:
                dst1 = dist["dist"][i][0]
                dist2 = dist["dist"][i][1]
                diver = dist["diver"][i]
                break
        return (dst1, dist2, diver)
