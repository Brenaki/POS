"""Bag generation and instance-index building (extracted from pool_generation.py).

Two samplers coexist (ADR 0020):

* `generate_bags` — stratified sub-sampling via `train_test_split` (legacy,
  no replacement). Preserved for backwards compatibility.
* `generate_bags_random_replacement` — random sampling with replacement,
  NOT stratified. This is the PGDCS path: Monteiro 2023 draws bags randomly
  with replacement; class coverage is enforced by the genetic operators
  (crossover requires every class present; mutation preserves the label),
  not by the sampler.
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split


def generate_bags_random_replacement(
    X_train: np.ndarray,
    y_train: np.ndarray,
    nr_bags: int,
    tam_bags: float,
    random_state: Union[int, None] = None,
) -> dict:
    """Create nr_bags random sub-samples with replacement (ADR 0020).

    Each bag draws `int(len(X_train) * tam_bags)` instance indices uniformly
    with replacement from `range(len(X_train))`. Sampling is NOT stratified;
    class coverage is enforced downstream by `GeneticOperatorsMixin`.

    Per-bag seed is `random_state + i` when `random_state` is an int, giving
    distinct-but-reproducible bags.

    Returns a dict {"name": [...], "inst": [...]} where inst[i] is the list
    of instance indices selected into bag i (with possible duplicates).
    """
    n = len(X_train)
    size = max(1, int(n * tam_bags))
    bags = {"name": list(), "inst": list()}
    for i in range(nr_bags):
        if isinstance(random_state, int):
            rng = np.random.RandomState(random_state + i)
        elif isinstance(random_state, np.random.RandomState):
            rng = random_state
        else:
            rng = np.random.RandomState()
        idx = rng.choice(n, size=size, replace=True)
        bags["name"].append(i)
        bags["inst"].append(idx.tolist())
    return bags


def generate_bags(
    X_train: np.ndarray,
    y_train: np.ndarray,
    nr_bags: int,
    tam_bags: float,
    random_state: Union[int, np.random.RandomState, None] = None,
) -> dict:
    """Create nr_bags stratified sub-samples of the training set.

    Returns a dict {"name": [...], "inst": [...]} where inst[i] is the list
    of instance indices selected into bag i.

    random_state: int, RandomState, or None. When None, uses the global
    numpy RNG (non-reproducible). Pass an int or RandomState for reproducibility.
    """
    indices = np.arange(len(X_train))
    bags = {"name": list(), "inst": list()}
    for i in range(nr_bags):
        rs_i = random_state + i if isinstance(random_state, int) else random_state
        _, _, _, _, id_bag, _ = train_test_split(
            X_train, y_train, indices, test_size=tam_bags,
            stratify=y_train, random_state=rs_i,
        )
        bags["name"].append(len(bags["name"]))
        bags["inst"].append(id_bag.tolist())
    return bags


def build_bags(
    indx_bag: List[int], X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct (X_data, y_data) for a bag given its instance indices.

    Vectorised in ADR 0015: the old list comprehension built a Python list of
    row views that every consumer immediately re-materialised with
    `np.asarray`. Fancy indexing produces the identical values in one step.
    """
    idx = np.asarray(indx_bag, dtype=int)
    return X_train[idx], y_train[idx]
