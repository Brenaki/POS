"""Hard voting classifier (extracted from Cpx.py during Fase 2 refactor).

Wraps mlxtend.EnsembleVoteClassifier with voting='hard'. The legacy
Cpx.voting_classifier used refit=False in a commented-out line and refit=True
in the active line; the active (refit=True) behavior is preserved here.
"""

from __future__ import annotations

from typing import List

import numpy as np
from mlxtend.classifier import EnsembleVoteClassifier


def voting_classifier(pool: List, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Fit a hard-voting ensemble on the pool and return its score on (X_val, y_val).

    Note: the legacy code created the voter with the default refit=True, so
    each call re-fits the pool. This is preserved.
    """
    voting = EnsembleVoteClassifier(clfs=pool, voting="hard")
    voting.fit(X_val, y_val)
    result = voting.score(X_val, y_val)
    return result
