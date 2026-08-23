"""Pool builder mixin: construct fitted classifiers from saved bags.

Extracted from pool_generation.py during Fase 2.
"""

from __future__ import annotations

from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

from pos.pool.bag_generator import build_bags


class PoolBuilderMixin:
    """Build the classifier pool from the saved bags."""

    def get_bags(self):
        bags = []
        for bag in self.bags_saved:
            bags.append(build_bags(bag[1:], self.X_train, self.y_train))
        return bags

    def get_pool(self):
        bags = self.get_bags()
        pool = []
        if self.classifier == "tree":
            for i, bag in enumerate(bags):
                rs_i = (self.random_state + i
                        if isinstance(self.random_state, int) else None)
                tree = DecisionTreeClassifier(random_state=rs_i)
                X_bag = bag[0]
                y_bag = bag[1]
                pool.append(tree.fit(X_bag, y_bag))
        else:
            # Must match biuld_classifier's Perceptron exactly: the final pool
            # was being built with different hyperparameters (no max_iter, no
            # seed) than the one the GA scored during fitness (ADR 0015).
            for i, bag in enumerate(bags):
                rs_i = (self.random_state + i
                        if isinstance(self.random_state, int) else None)
                percP = Perceptron(max_iter=100, tol=1.0, random_state=rs_i)
                X_bag = bag[0]
                y_bag = bag[1]
                pool.append(percP.fit(X_bag, y_bag))
        self.pool_classificators = pool
        return pool
