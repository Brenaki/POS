"""RandomForest-based pool builder (baseline without the GA).

Builds a pool of M DecisionTreeClassifiers via sklearn's RandomForestClassifier
with config defaults (max_features='sqrt', bootstrap=True). Returns the
fitted estimators_ list — same .predict/.predict_proba interface as the
pool produced by poolGeneration.get_pool().

ADR 0009 — decision to use RF defaults as the 'no GA' baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def build_rf_pool(
    X_train: np.ndarray,
    y_train: np.ndarray,
    M: int = 100,
    random_state: int = 42,
) -> list[DecisionTreeClassifier]:
    """Build a pool of M DecisionTrees via RandomForestClassifier.

    Parameters
    ----------
    X_train, y_train : training data
    M : int, number of estimators (pool size)
    random_state : int, fixed seed for reproducibility

    Returns
    -------
    pool : list of M fitted DecisionTreeClassifier (forest.estimators_)
    """
    forest = RandomForestClassifier(
        n_estimators=M,
        random_state=random_state,
        bootstrap=True,
        n_jobs=1,
    )
    forest.fit(X_train, y_train)
    return list(forest.estimators_)


def rf_pool_majority_predict(pool: list[DecisionTreeClassifier], X: np.ndarray) -> np.ndarray:
    """Hard majority vote over the RF pool predictions.

    Convenience function. For metric computation use
    pos.oracle.comparison.majority_vote_accuracy instead.
    """
    preds = np.array([clf.predict(X) for clf in pool])
    from scipy import stats
    return stats.mode(preds, axis=0, keepdims=False).mode
