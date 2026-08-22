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


def _eval_one(i, bags_inst, X_train, y_train, X_val, y_val, classifier, group, types):
    """Module-level: evaluate one bag (thread-safe, no shared mutable state)."""
    indx_bag1 = bags_inst[i]
    X_bag, y_bag = build_bags(indx_bag1, X_train, y_train)
    cpx = complexity_data3(X_bag, y_bag, group, types)
    if classifier == "perc":
        estimator, score, pred = biuld_classifier(X_bag, y_bag, X_bag, y_bag, X_val, y_val)
    elif classifier == "tree":
        estimator, score, pred = biuld_classifier_tree(X_bag, y_bag, X_bag, y_bag, X_val, y_val)
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
        if self.classifier == "perc":
            estimator, score, pred = biuld_classifier(
                X_bag, y_bag, X_bag, y_bag, self.X_val, self.y_val)
        elif self.classifier == "tree":
            estimator, score, pred = biuld_classifier_tree(
                X_bag, y_bag, X_bag, y_bag, self.X_val, self.y_val)
        return cpx, score, pred, estimator

    def _eval_many(self, indices):
        """Evaluate parallel_distance2 for a list of bag indices.

        Uses ThreadPoolExecutor when self.jobs > 1. Safe now because
        fast_adapter uses pure numpy (no joblib nesting).
        """
        indices = list(indices)
        if self.jobs > 1:
            with ThreadPoolExecutor(max_workers=self.jobs) as ex:
                results = list(ex.map(
                    lambda i: _eval_one(i, self.bags["inst"], self.X_train,
                        self.y_train, self.X_val, self.y_val, self.classifier,
                        self.group, self.types),
                    indices))
            return list(zip(*results, strict=True))
        r = [self.parallel_distance2(i, self.bags, self.group, self.types)
             for i in indices]
        return zip(*r, strict=True) if r else ((), (), (), ())

    def get_complexity(self, first_evaluate=False, population=None):
        dist = {"name": list(), "dist": list(), "diver": list(),
                "score": list(), "score_g": list()}

        if first_evaluate and self.generation == 0:
            dist["name"] = self.pop
            c, score, pred, pool = self._eval_many(range(len(dist["name"])))
            self.c = c
        elif first_evaluate == False and population is None:
            begin = self.name_individual - self.nr_individual
            for i in range(begin, self.name_individual):
                dist["name"].append([i])
            c, score, pred, pool = self._eval_many(
                range(100, self.nr_individual + 100))
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
        dist["score_g"] = voting_classifier(pool, self.X_val, self.y_val)
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
