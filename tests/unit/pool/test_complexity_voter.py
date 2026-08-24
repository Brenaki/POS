"""Tests for the PGDCS complexity-measure vote (ADR 0019).

The vote was unusable before ADR 0019 for two reasons, and both are asserted
here: it ran on a backend that reported a constant 0.0 for T1 (so T1 could
never be voted, even though the runs hardcode it), and it sub-sampled without
a seed (so two identical runs could pick different measures).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_wine

from pos.pool import complexity_voter as cv

GROUPS = ["overlapping", "neighborhood"]


@pytest.fixture(scope="module")
def wine():
    data = load_wine()
    return data.data, data.target


@pytest.fixture
def fast_vote(monkeypatch):
    """Shrink the vote so a test does not pay the full 11 x 100 sub-samples."""
    monkeypatch.setattr(cv, "VOTE_ROUNDS", 3)
    monkeypatch.setattr(cv, "SAMPLES_PER_ROUND", 8)


class TestMeasureLayout:
    def test_names_follow_group_order(self):
        names = cv.measure_names(GROUPS)
        assert names[0] == "overlapping.F1"
        assert names[-1] == "neighborhood.LSC"
        assert len(names) == 11

    def test_slices_partition_the_vector(self):
        assert cv._group_slices(GROUPS) == [(0, 5), (5, 11)]

    def test_single_group_still_works(self):
        assert cv._group_slices(["overlapping"]) == [(0, 5)]
        assert len(cv.measure_names(["overlapping"])) == 5


class TestVoteIsReproducible:
    def test_same_seed_same_measures(self, wine, fast_vote):
        X, y = wine
        first = cv.get_best_types(X, y, 0.5, GROUPS, random_state=42)
        second = cv.get_best_types(X, y, 0.5, GROUPS, random_state=42)
        assert first == second

    def test_one_vote_per_group_per_round(self, wine, fast_vote):
        X, y = wine
        votes, _, _, stad = cv.vote_complexity(X, y, 0.5, GROUPS, random_state=0)
        assert sum(votes) == cv.VOTE_ROUNDS * len(GROUPS)
        assert sum(votes[0:5]) == cv.VOTE_ROUNDS
        assert sum(votes[5:11]) == cv.VOTE_ROUNDS
        assert len(stad) == 11

    def test_returns_requested_number_of_measures(self, wine, fast_vote):
        X, y = wine
        assert len(cv.get_best_types(X, y, 0.5, GROUPS, n=3, random_state=1)) == 3


class TestBackendExposesEveryMeasure:
    """The regression that ADR 0019 fixed: T1/N3/F1v were constant zeros."""

    @pytest.mark.parametrize("measure", ["neighborhood.T1", "neighborhood.N3",
                                         "overlapping.F1v"])
    def test_measure_is_not_a_constant_zero(self, wine, measure):
        X, y = wine
        idx = cv.measure_names(GROUPS).index(measure)
        values = [
            cv.complexities(X, y, 0.5, GROUPS, random_state=s)[idx]
            for s in range(6)
        ]
        assert np.std(values) > 0.0, f"{measure} is constant — vote can never pick it"
