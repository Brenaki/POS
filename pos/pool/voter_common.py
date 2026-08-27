"""Shared helpers for the complexity-measure voters (ADR 0020).

`complexity_voter` (legacy vote) and `monteiro_voter` (Monteiro 2023 vote)
both need the flat measure-name vector, the per-group slices and the
per-draw seed rule. Keeping them here breaks the import cycle the ADR 0020
split introduced, and keeps both voters under the 150 LOC cap (ADR 0002).
"""

from __future__ import annotations

import numpy as np

from pos.complexity.base import GROUP_MEASURES

# Legacy POS vote (pre-ADR 0020).
VOTE_ROUNDS = 11
SAMPLES_PER_ROUND = 100

# Monteiro 2023: T=10 iterations of n=100 subsets => 1000 sub-samples.
MONTEIRO_ROUNDS = 10
MONTEIRO_SAMPLES_PER_ROUND = 100

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


def votes_from_sigmas(sigmas, slices: list[tuple[int, int]]) -> list[int]:
    """One vote per family per round, on that round's own dispersion.

    Monteiro 2023 Eq. (10)-(11): at each iteration `t` the measure with the
    largest sigma_j^t *inside that iteration* receives the vote. Voting on the
    running mean across iterations instead collapses a family's votes onto a
    single measure, which contradicts the spread shown in Tables 3 and 4.
    """
    rounds = list(sigmas)
    if not rounds:
        return []
    votes = [0] * len(rounds[0])
    for row in rounds:
        for start, stop in slices:
            votes[start + int(np.argmax(row[start:stop]))] += 1
    return votes


def _seed_for(random_state: Seed, draw: int) -> Seed:
    """Distinct-but-reproducible seed per sub-sample, as in bag_generator."""
    return random_state + draw if isinstance(random_state, int) else random_state
