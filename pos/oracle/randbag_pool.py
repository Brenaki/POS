"""Random-bag pool builder: the GA's generation-0 population, without the search.

ADR 0019. Until this mode existed, generation method and base classifier were
confounded in every comparison the project made: `ga` meant Perceptrons and
`bagging`/`rf` meant trees, so "the GA pool has the largest Oracle_1 - MVR gap"
could never be separated from "the Perceptron is the weakest base learner".

This builds the pool the GA starts from — the same stratified 50% bags from
`bag_generator.generate_bags`, the same linear Perceptron, the same M and the
same seed — and then stops. That gives the two clean contrasts:

    ga      vs randbag  -> the effect of the GA's search (base learner fixed)
    randbag vs bagging  -> the effect of the base learner (search absent in both)
"""

from __future__ import annotations

import numpy as np

from pos.classifiers import biuld_classifier
from pos.pool.bag_generator import build_bags, generate_bags

TAM_BAGS = 0.5


def build_randbag_pool(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    M: int = 100,
    random_state: int = 42,
    tam_bags: float = TAM_BAGS,
) -> list:
    """Build M Perceptrons on M random stratified bags of (X_train, y_train).

    `X_val`/`y_val` are only scored, never trained on — `biuld_classifier`
    returns (estimator, score) and the score is what the GA would have used as
    fitness. Here it is discarded: no selection happens, which is the point.
    """
    bags = generate_bags(X_train, y_train, M, tam_bags, random_state=random_state)
    pool = []
    for i, inst in enumerate(bags["inst"]):
        X_bag, y_bag = build_bags(inst, X_train, y_train)
        seed = random_state + i if isinstance(random_state, int) else random_state
        clf, _ = biuld_classifier(X_bag, y_bag, X_val, y_val, random_state=seed)
        pool.append(clf)
    return pool
