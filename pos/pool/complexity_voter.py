"""Complexity voting for measure selection (extracted from pool_generation.py).

This is the step that makes PGDCS *PGDCS*: instead of fixing which complexity
measures drive the GA's fitness, it votes over sub-samples of the training set
and keeps the measures whose value varies the most across them — the ones that
actually discriminate bags for this dataset.

ADR 0019 fixed two defects that made the vote unusable:

1. **Backend.** This module imported `complexity_data3` from `pos.complexity`,
   which resolves to the pyhard adapter. pyhard has no mapping for `F1v`, `N3`
   and `T1` and returns a constant 0.0 for all three; a constant has zero
   standard deviation and can never win the argmax below, so the vote could
   never select T1 — one of the two measures the runs hardcode. The GA's own
   fitness (`pos/pool/fitness_evaluator.py`) already used `fast_adapter`, so
   selection and fitness disagreed numerically about the same measure. Both now
   read from `fast_adapter`, which computes all eleven and is what ADRs
   0010/0011/0013 validated against ECoL.
2. **Reproducibility.** The sub-sampling had no `random_state`, so two identical
   runs could pick different measures (against ADR 0012).

The legacy code hardcoded 11 measures and the 5/6 split between the two groups.
That is derived from `grupos` here instead: `HEADER` holds 22 names across six
groups, and slicing it globally only worked because overlapping and
neighborhood happen to come first.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.model_selection import train_test_split

from pos.complexity.base import GROUP_MEASURES
from pos.complexity.fast_adapter import complexity_data3
from pos.normalization import min_max_norm

logger = logging.getLogger(__name__)

VOTE_ROUNDS = 11
SAMPLES_PER_ROUND = 100

Seed = int | np.random.RandomState | None


def measure_names(grupos: list[str]) -> list[str]:
    """Flat `group.measure` names, in the order complexity_data3 returns them."""
    return [f"{g}.{m}" for g in grupos for m in GROUP_MEASURES[g]]


def _group_slices(grupos: list[str]) -> list[tuple[int, int]]:
    """(start, stop) of each group inside the flat measure vector."""
    bounds, start = [], 0
    for g in grupos:
        stop = start + len(GROUP_MEASURES[g])
        bounds.append((start, stop))
        start = stop
    return bounds


def _seed_for(random_state: Seed, draw: int) -> Seed:
    """Distinct-but-reproducible seed per sub-sample, as in bag_generator."""
    return random_state + draw if isinstance(random_state, int) else random_state


def complexities(X_train, y_train, tam_bags, grupos,
                 random_state: Seed = None) -> list[float]:
    """Sample one bag and compute its complexity vector."""
    _, X_bag, _, y_bag = train_test_split(
        X_train, y_train, test_size=tam_bags, random_state=random_state
    )
    return complexity_data3(X_bag, y_bag, grupos)


def vote_complexity(X_data, y_data, tam_bags, grupos,
                    random_state: Seed = None) -> tuple:
    """Vote over sub-samples to pick the most-variant complexity measures.

    Each of VOTE_ROUNDS rounds draws SAMPLES_PER_ROUND bags, min-max normalises
    every measure across them, and gives one vote to the highest-dispersion
    measure *of each group*. Returns (voto, text, max_idx, stad), same shape the
    legacy poolGeneration.vote_complexity returned.
    """
    names = measure_names(grupos)
    slices = _group_slices(grupos)
    voto = [0] * len(names)
    draw = 0
    max_idx = np.array([], dtype=int)
    stad = np.array([])
    for _ in range(VOTE_ROUNDS):
        cp = [
            complexities(X_data, y_data, tam_bags, grupos,
                         random_state=_seed_for(random_state, draw + i))
            for i in range(SAMPLES_PER_ROUND)
        ]
        draw += SAMPLES_PER_ROUND
        cpx = np.asarray(cp).T  # one row per measure, one column per sub-sample
        stad = np.array([float(np.std(min_max_norm(row))) for row in cpx])
        max_idx = np.argsort(stad)[::-1]
        for start, stop in slices:
            voto[start + int(np.argmax(stad[start:stop]))] += 1
    text = "".join(f"{name} {v}, " for name, v in zip(names, voto, strict=True))
    return voto, text, max_idx, stad


def get_best_types(X_train, y_train, tam_bags, group, n: int = 2,
                   random_state: Seed = None) -> list[str]:
    """Return the n most-voted complexity measure names (e.g. ['F1', 'T1'])."""
    names = measure_names(group)
    votes, text, _, _ = vote_complexity(X_train, y_train, tam_bags, group,
                                        random_state=random_state)
    ordened = np.argsort(votes)
    types = [names[ordened[-i]].split(".")[1] for i in range(1, n + 1)]
    logger.info("complexity vote picked %s (votes: %s)", types, text.strip(", "))
    return types
