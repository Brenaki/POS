"""Pairwise double-fault diversity (extracted from Cpx.py during Fase 2 refactor).

Uses deslib.util.diversity.double_fault. Returns one mean double-fault value
per classifier (mean over all pairs that include that classifier).
"""

from __future__ import annotations

from typing import List

import numpy as np
from deslib.util import diversity


def diversitys(y_test: np.ndarray, predicts: np.ndarray) -> List[float]:
    """Mean pairwise double-fault per classifier.

    For each classifier i, computes double_fault(y_test, predicts[i],
    predicts[j]) for every j != i, then returns the mean. Output length
    equals the number of classifiers (len(predicts)).
    """
    double_faults: list = []
    for i in range(len(predicts)):
        db = []
        for j in range(len(predicts)):
            if i == j:
                continue
            else:
                db.append(diversity.double_fault(y_test, predicts[i], predicts[j]))
        double_faults.append(np.mean(db))

    return double_faults
