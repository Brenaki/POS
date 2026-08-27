"""Tests for the complexity voter aligned with Monteiro 2023 (ADR 0020 T5).

10 rounds × 100 samples, exactly one Overlapping + one Neighborhood measure,
tie-break by mean dispersion across the T iterations.
"""

from __future__ import annotations

import numpy as np
import pytest

from pos.pool.complexity_voter import (
    MONTEIRO_ROUNDS,
    MONTEIRO_SAMPLES_PER_ROUND,
    monteiro_get_best_types,
    monteiro_vote_complexity,
)


@pytest.fixture
def wine_data():
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True, as_frame=False)
    return X, y


class TestMonteiroVoterParams:
    def test_rounds_match_paper(self):
        assert MONTEIRO_ROUNDS == 10

    def test_samples_per_round_match_paper(self):
        assert MONTEIRO_SAMPLES_PER_ROUND == 100


class TestMonteiroVoting:
    def test_returns_two_measures_one_per_family(self, wine_data):
        X, y = wine_data
        types, votes, dispersion = monteiro_vote_complexity(
            X, y, tam_bags=0.5, random_state=42
        )
        assert len(types) == 2
        # first from overlapping, second from neighborhood
        overlapping = {"F1", "F1v", "F2", "F3", "F4"}
        neighborhood = {"N1", "N2", "N3", "N4", "T1", "LSC"}
        assert types[0] in overlapping, f"{types[0]} not in overlapping"
        assert types[1] in neighborhood, f"{types[1]} not in neighborhood"

    def test_is_deterministic_with_seed(self, wine_data):
        X, y = wine_data
        a, _, _ = monteiro_vote_complexity(X, y, tam_bags=0.5, random_state=42)
        b, _, _ = monteiro_vote_complexity(X, y, tam_bags=0.5, random_state=42)
        assert a == b

    def test_vote_counts_sum_to_rounds(self, wine_data):
        """Each round gives exactly one vote per family."""
        X, y = wine_data
        _, votes, _ = monteiro_vote_complexity(X, y, tam_bags=0.5, random_state=42)
        # votes covers all 11 measures (5 overlapping + 6 neighborhood)
        assert len(votes) == 11
        # overlapping sub-sum == MONTEIRO_ROUNDS, neighborhood sub-sum == MONTEIRO_ROUNDS
        assert sum(votes[:5]) == MONTEIRO_ROUNDS
        assert sum(votes[5:]) == MONTEIRO_ROUNDS

    def test_returns_dispersion_per_measure(self, wine_data):
        X, y = wine_data
        _, _, dispersion = monteiro_vote_complexity(X, y, tam_bags=0.5,
                                                      random_state=42)
        assert len(dispersion) == 11


class TestTieBreak:
    def test_tie_breaks_by_mean_dispersion(self):
        """When two measures in the same family tie on votes, the one with
        higher mean dispersion wins."""
        # construct votes with a tie in overlapping family
        votes = [5, 5, 0, 0, 0, 10, 0, 0, 0, 0, 0]
        dispersion = np.array([0.8, 0.3, 0.1, 0.1, 0.1,
                               0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        groups = ["overlapping", "neighborhood"]
        from pos.pool.complexity_voter import _resolve_winners
        types = _resolve_winners(votes, dispersion, groups)
        assert types[0] == "F1"  # higher dispersion (0.8 > 0.3)
        assert types[1] == "N1"

    def test_no_tie_picks_max_votes(self):
        votes = [10, 0, 0, 0, 0, 0, 8, 2, 0, 0, 0]
        dispersion = np.zeros(11)
        groups = ["overlapping", "neighborhood"]
        from pos.pool.complexity_voter import _resolve_winners
        types = _resolve_winners(votes, dispersion, groups)
        assert types[0] == "F1"
        assert types[1] == "N2"


class TestMonteiroGetBestTypes:
    def test_returns_list_of_two_names(self, wine_data):
        X, y = wine_data
        types = monteiro_get_best_types(X, y, tam_bags=0.5, random_state=42)
        assert isinstance(types, list)
        assert len(types) == 2

    def test_is_deterministic(self, wine_data):
        X, y = wine_data
        a = monteiro_get_best_types(X, y, tam_bags=0.5, random_state=42)
        b = monteiro_get_best_types(X, y, tam_bags=0.5, random_state=42)
        assert a == b

class TestVotesFromSigmas:
    """Eq. (10)-(11): one vote per family per round, on that round's own σ."""

    def test_votes_follow_each_round_not_the_running_mean(self):
        from pos.pool.voter_common import votes_from_sigmas

        # Two measures in one family. Measure 0 wins rounds 1-2 by a wide
        # margin, measure 1 wins rounds 3-5 narrowly: per-round voting gives
        # 2 x 3, while voting on the cumulative mean would give 5 x 0.
        sigmas = [
            [0.90, 0.10],
            [0.90, 0.10],
            [0.10, 0.11],
            [0.10, 0.11],
            [0.10, 0.11],
        ]
        assert votes_from_sigmas(sigmas, [(0, 2)]) == [2, 3]

    def test_one_vote_per_family_per_round(self):
        from pos.pool.voter_common import votes_from_sigmas

        sigmas = [[0.5, 0.1, 0.2, 0.9], [0.1, 0.5, 0.9, 0.2]]
        votes = votes_from_sigmas(sigmas, [(0, 2), (2, 4)])
        assert sum(votes[0:2]) == 2
        assert sum(votes[2:4]) == 2
        assert votes == [1, 1, 1, 1]

    def test_empty_rounds_give_zero_votes(self):
        from pos.pool.voter_common import votes_from_sigmas

        assert votes_from_sigmas([], [(0, 2)]) == []


class TestMonteiroVoteSpread:
    def test_votes_are_not_forced_onto_a_single_measure(self, wine_data):
        """The running-mean bug made a family's votes collapse to ~10/10.

        Table 3/4 show the votes spread over several measures; with per-round
        voting the winner must not be the only measure with votes.
        """
        X, y = wine_data
        _, votes, _ = monteiro_vote_complexity(X, y, 0.5, random_state=86)
        overlapping = votes[:5]
        assert sum(1 for v in overlapping if v > 0) >= 2
