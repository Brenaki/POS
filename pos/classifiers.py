"""Base-learner builders (extracted from Cpx.py during Fase 2 refactor).

Preserves the legacy `biuld_*` misspelling (to be renamed by a future ADR) and
the EXACT branch behavior, including the known bug in `biuld_classifier` where
`X_test != None` raises ValueError for numpy arrays (see ADR 0004 and the
characterization tests).
"""

from __future__ import annotations

from typing import Optional, Union, Tuple

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

Classifier = Union[Perceptron, DecisionTreeClassifier]


def biuld_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    score_train: bool = False,
):
    """Build a Perceptron and return (estimator, score[, score2,] predict).

    KNOWN BUG (preserved): `X_test != None` raises ValueError when X_test is
    a numpy array. Use `biuld_classifier_tree` for the array-X_test path, or
    pass X_test=None to get the (estimator, score) return.
    """
    perc = Perceptron(n_jobs=4, max_iter=100, tol=1.0)
    perc.fit(X_train, y_train)
    score = perc.score(X_val, y_val)
    if X_test != None and y_test != None and score_train == False:  # noqa: E711 — legacy bug
        predict = perc.predict(X_test)
        return perc, score, predict
    elif score_train:
        score2 = perc.score(X_test, y_test)
        predict = perc.predict(X_test)
        return perc, score, score2, predict
    else:
        return perc, score


def biuld_classifier_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    score_train: bool = False,
):
    """Build a DecisionTree and return (estimator, score[, score2,] predict).

    Note: the legacy code uses `X_test.all() != None` which accidentally
    works for arrays (because `.all()` returns a scalar). Preserved as-is.
    """
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    score = tree.score(X_val, y_val)
    if X_test.all() != None and y_test.all() != None and score_train == False:  # noqa: E711
        predict = tree.predict(X_test)
        return tree, score, predict
    elif score_train:
        score2 = tree.score(X_test, y_test)
        predict = tree.predict(X_test)
        return tree, score, score2, predict
    else:
        return tree, score
