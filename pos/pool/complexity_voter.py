"""Complexity voting for measure selection (extracted from pool_generation.py)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from pos.complexity.base import HEADER
from pos.complexity import complexity_data3
from pos.normalization import min_max_norm


def complexities(X_train, y_train, tam_bags, grupos) -> List[float]:
    """Sample one bag and compute its complexity vector."""
    _, X_bag, _, y_bag = train_test_split(X_train, y_train, test_size=tam_bags)
    return complexity_data3(X_bag, y_bag, grupos)


def vote_complexity(X_data, y_data, tam_bags, grupos) -> Tuple:
    """Vote over 100 sub-samples to pick the most-variant complexity measures.

    Returns (voto, text, max, stad) exactly as the legacy poolGeneration.vote_complexity.
    """
    voto = [0] * 11
    for _ in range(1, 12):
        stad = []
        comp = []
        cp = []
        for _ in range(100):
            cp.append(complexities(X_data, y_data, tam_bags, grupos))
        comp.append(cp)
        comp = np.array(comp)
        cpx = np.squeeze(comp)
        cpx = cpx.T
        for k in cpx:
            norm = min_max_norm(k)
            std = np.std(norm)
            std = std.tolist()
            stad.append(std)
        max_idx = np.argsort(stad)
        stad = np.array(stad)
        max_idx = max_idx[::-1]
        del cpx
        overlapping = stad[0:5]
        neighborhood = stad[5:11]
        o = np.argmax(overlapping)
        nei = np.argmax(neighborhood)
        voto[o] = voto[o] + 1
        voto[nei + 5] = voto[nei + 5] + 1
    text = ""
    for carro, cor in zip(HEADER, voto):
        text += "{} {}, ".format(carro, cor)
    return voto, text, max_idx, stad


def get_best_types(X_train, y_train, tam_bags, group, n: int = 2) -> List[str]:
    """Return the n most-voted complexity measure names (e.g. ['F1', 'T1'])."""
    print("votting complex...")
    res = vote_complexity(X_train, y_train, tam_bags, group)
    votes = res[0]
    ordened = np.argsort(votes)
    types = []
    for i in range(1, n + 1):
        feature = HEADER[ordened[-i]]
        types.append(feature.split(".")[1])
    return types
