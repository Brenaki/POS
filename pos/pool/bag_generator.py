"""Bag generation and instance-index building (extracted from pool_generation.py)."""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split


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


def build_bags(indx_bag: List[int], X_train: np.ndarray, y_train: np.ndarray) -> Tuple[list, list]:
    """Reconstruct (X_data, y_data) for a bag given its instance indices."""
    X_data = [X_train[int(i)] for i in indx_bag]
    y_data = [y_train[int(i)] for i in indx_bag]
    return X_data, y_data
