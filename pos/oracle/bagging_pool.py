"""Bagging-based pool builder (baseline without GA, controlled comparison).

Builds a pool of M DecisionTreeClassifiers via sklearn's BaggingClassifier
with bootstrap=True and max_features=1.0 (all features). Returns the
fitted estimators_ list — same .predict/.predict_proba interface as the
pool produced by poolGeneration.get_pool() and build_rf_pool().

Unlike RandomForest (max_features='sqrt'), this uses ALL features per tree,
isolating the effect of the GA from the effect of feature subsampling.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def build_bagging_pool(
    X_train: np.ndarray,
    y_train: np.ndarray,
    M: int = 100,
    random_state: int = 42,
) -> list[DecisionTreeClassifier]:
    """Build a pool of M DecisionTrees via BaggingClassifier (all features).

    Parameters
    ----------
    X_train, y_train : training data
    M : int, number of estimators (pool size)
    random_state : int, fixed seed for reproducibility

    Returns
    -------
    pool : list of M fitted DecisionTreeClassifier (bag.estimators_)
    """
    bag = BaggingClassifier(
        n_estimators=M,
        random_state=random_state,
        bootstrap=True,
        max_features=1.0,
        n_jobs=1,
    )
    bag.fit(X_train, y_train)
    return list(bag.estimators_)