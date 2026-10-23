"""Monteiro 2023 complexity voting path (ADR 2020 §3.4).

Split from complexity_voter.py to stay under the 150 LOC cap (ADR 0002).
10 rounds × 100 random-with-replacement sub-samples, exactly one Overlapping
+ one Neighborhood measure, tie-break by mean dispersion.
"""

from __future__ import annotations

import logging

import numpy as np

from pos.complexity.fast_adapter import complexity_data3
from pos.normalization import min_max_norm
from pos.pool.bag_generator import generate_bags_random_replacement
from pos.pool.voter_common import (
    MONTEIRO_ROUNDS,
    MONTEIRO_SAMPLES_PER_ROUND,
    Seed,
    _group_slices,
    _seed_for,
    measure_names,
    votes_from_sigmas,
)

logger = logging.getLogger(__name__)


def _resolve_winners(
    votes: list[int], dispersion: np.ndarray, groups: list[str]
) -> list[str]:
    """Pick exactly one measure per group: max votes, tie-break by dispersion."""
    names = measure_names(groups)
    slices = _group_slices(groups)
    winners: list[str] = []
    for start, stop in slices:
        family_votes = votes[start:stop]
        family_disp = dispersion[start:stop]
        family_names = names[start:stop]
        max_v = max(family_votes)
        tied = [i for i, v in enumerate(family_votes) if v == max_v]
        if len(tied) == 1:
            winners.append(family_names[tied[0]].split(".")[1])
        else:
            best = max(tied, key=lambda i: float(family_disp[i]))
            winners.append(family_names[best].split(".")[1])
    return winners


def monteiro_vote_complexity(
    X_data, y_data, tam_bags, grupos=None, random_state: Seed = None,
    backend: str = "fast",
) -> tuple:
    """Vote over 10×100 random-with-replacement sub-samples (Monteiro 2023)."""
    if grupos is None:
        grupos = ["overlapping", "neighborhood"]
    slices = _group_slices(grupos)
    draw = 0
    round_sigmas: list[list[float]] = []

    for _ in range(MONTEIRO_ROUNDS):
        bags = generate_bags_random_replacement(
            X_data, y_data, MONTEIRO_SAMPLES_PER_ROUND, tam_bags,
            random_state=_seed_for(random_state, draw),
        )
        draw += MONTEIRO_SAMPLES_PER_ROUND
        cp = []
        for inst in bags["inst"]:
            idx = np.asarray(inst, dtype=int)
            cp.append(complexity_data3(X_data[idx], y_data[idx], grupos,
                                       backend=backend))
        cpx = np.asarray(cp).T
        round_sigmas.append([float(np.std(min_max_norm(row))) for row in cpx])
    votes = votes_from_sigmas(round_sigmas, slices)
    dispersion = np.asarray(round_sigmas).mean(axis=0)
    types = _resolve_winners(votes, dispersion, grupos)
    return types, votes, dispersion


def monteiro_get_best_types(
    X_train, y_train, tam_bags, random_state: Seed = None, backend: str = "fast"
) -> list[str]:
    """Return the two ADR 0020 measures: one Overlapping + one Neighborhood."""
    groups = ["overlapping", "neighborhood"]
    types, _, _ = monteiro_vote_complexity(
        X_train, y_train, tam_bags, groups, random_state=random_state,
        backend=backend,
    )
    logger.info("monteiro vote picked %s", types)
    return types
