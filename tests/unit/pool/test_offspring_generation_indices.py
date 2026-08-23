"""Regression tests for ADR 0014: offspring must be scored on their OWN bags.

The legacy `get_complexity(first_evaluate=False, population=None)` branch
evaluated `range(100, nr_individual + 100)` — a constant. It happened to be
right for generation 1 and was wrong for every generation after it, so a
20-generation run scored individuals 200..299, 300..399, ... using the data
of individuals 100..199.

A determinism test cannot catch that (a deterministic bug stays deterministic),
so these tests assert the name -> evaluated-bag correspondence directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

import pos.pool.fitness_evaluator as fe
from pos.pool.errors import StratificationError


@pytest.fixture
def wine_tr_val():
    X, y = load_wine(return_X_y=True, as_frame=False)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def _run_ga_recording_offspring_evals(X_tr, y_tr, X_val, y_val, nr_generation):
    """Run the GA, returning [(names_scored, bag_positions_evaluated), ...]
    for every offspring-evaluation call."""
    from pool_generation import poolGeneration

    records: list[tuple[list[int], list[int]]] = []
    orig_eval_many = fe.FitnessEvaluatorMixin._eval_many
    orig_get_cpx = fe.FitnessEvaluatorMixin.get_complexity
    last_indices: list[list[int]] = []

    def spy_eval_many(self, indices):
        indices = list(indices)
        last_indices.append(indices)
        return orig_eval_many(self, indices)

    def spy_get_complexity(self, first_evaluate=False, population=None):
        is_offspring_call = (not first_evaluate) and population is None
        out = orig_get_cpx(self, first_evaluate=first_evaluate, population=population)
        if is_offspring_call:
            names = [n[0] for n in self.dist["name"]]
            records.append((names, last_indices[-1]))
        return out

    fe.FitnessEvaluatorMixin._eval_many = spy_eval_many
    fe.FitnessEvaluatorMixin.get_complexity = spy_get_complexity
    try:
        pg = poolGeneration(nr_generation=nr_generation, iteration=1,
                            classifier="tree", types=["F1", "T1"], jobs=1,
                            random_state=42)
        pg.generate(X_tr, y_tr, X_val, y_val, iteration=1)
        bag_names = list(pg.bags["name"])
    finally:
        fe.FitnessEvaluatorMixin._eval_many = orig_eval_many
        fe.FitnessEvaluatorMixin.get_complexity = orig_get_cpx
    return records, bag_names


class TestOffspringEvaluatedOnOwnBags:
    @pytest.mark.slow
    def test_each_generation_scores_its_own_offspring(self, wine_tr_val):
        """Generation g must score names 100*g..100*g+99 on those same bags."""
        X_tr, X_val, y_tr, y_val = wine_tr_val
        records, bag_names = _run_ga_recording_offspring_evals(
            X_tr, y_tr, X_val, y_val, nr_generation=3)

        assert len(records) == 3, "expected one offspring evaluation per generation"

        for gen, (names, positions) in enumerate(records, start=1):
            expected_names = list(range(100 * gen, 100 * gen + 100))
            assert names == expected_names, (
                f"generation {gen} scored names {names[0]}..{names[-1]}, "
                f"expected {expected_names[0]}..{expected_names[-1]}"
            )
            evaluated_names = [bag_names[p] for p in positions]
            assert evaluated_names == expected_names, (
                f"generation {gen} scored names "
                f"{expected_names[0]}..{expected_names[-1]} using bags "
                f"{evaluated_names[0]}..{evaluated_names[-1]} — offspring "
                "evaluated against the wrong individuals"
            )

    @pytest.mark.slow
    def test_generation_two_does_not_reuse_generation_one_bags(self, wine_tr_val):
        """The exact legacy symptom: gen 2 evaluating bag positions 100..199."""
        X_tr, X_val, y_tr, y_val = wine_tr_val
        records, bag_names = _run_ga_recording_offspring_evals(
            X_tr, y_tr, X_val, y_val, nr_generation=2)
        _, gen2_positions = records[1]
        assert min(gen2_positions) >= 200, (
            f"generation 2 evaluated bag positions starting at "
            f"{min(gen2_positions)} — the legacy range(100, 200) bug"
        )


class TestGenerationRecordedByStopCriteria:
    @pytest.mark.slow
    def test_gen_temp_is_not_pinned_to_zero(self, wine_tr_val):
        """max_distance must record the real generation, not self.generation=0."""
        from pool_generation import poolGeneration

        X_tr, X_val, y_tr, y_val = wine_tr_val
        seen: list[int] = []
        pg = poolGeneration(nr_generation=3, iteration=1, classifier="tree",
                            types=["F1", "T1"], jobs=1, random_state=42)
        orig = pg.max_distance

        def spy(fitness_matrix, generation=None, population=None, bags=None):
            seen.append(generation)
            return orig(fitness_matrix, generation=generation,
                        population=population, bags=bags)

        pg.max_distance = spy
        pg.generate(X_tr, y_tr, X_val, y_val, iteration=1)
        assert seen == [0, 1, 2, 3], f"generations recorded: {seen}"
        assert pg.gen_temp in (0, 1, 2, 3)


class TestStratificationFailureIsCatchable:
    def test_raises_exception_not_systemexit(self):
        """`exit(0)` used to escape `except Exception` and kill the run."""
        assert issubclass(StratificationError, Exception)
        assert not issubclass(StratificationError, SystemExit)

    @pytest.mark.slow
    def test_unusable_split_raises_stratification_error(self):
        """A class with 2 instances cannot fill a 50% bag -> catchable error."""
        from pool_generation import poolGeneration

        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(size=(60, 4)), rng.normal(size=(60, 4)) + 4,
                       rng.normal(size=(2, 4)) + 12])
        y = np.array([0] * 60 + [1] * 60 + [2] * 2)
        pg = poolGeneration(nr_generation=1, iteration=1, classifier="tree",
                            types=["F1", "T1"], jobs=1, random_state=42)
        with pytest.raises(Exception) as exc:
            pg.generate(X, y, X[:40], y[:40], iteration=1)
        assert not isinstance(exc.value, SystemExit)
