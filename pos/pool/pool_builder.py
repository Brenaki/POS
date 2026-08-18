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
            for bag in bags:
                tree = DecisionTreeClassifier()
                X_bag = bag[0]
                y_bag = bag[1]
                pool.append(tree.fit(X_bag, y_bag))
        else:
            for bag in bags:
                percP = Perceptron(tol=1.0)
                X_bag = bag[0]
                y_bag = bag[1]
                pool.append(percP.fit(X_bag, y_bag))
        self.pool_classificators = pool
        return pool
