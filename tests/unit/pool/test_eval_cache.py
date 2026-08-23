"""ADR 0015: memoising bag evaluation must not change any result.

mu+lambda selection re-presents survivors every generation, so
`the_function` asked `get_complexity` to re-evaluate 100 individuals that
had already been evaluated. The cache removes that recomputation; these
tests pin both halves of the claim — it is used, and it is invisible.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

import pos.pool.fitness_evaluator as fe


@pytest.fixture
def wine_tr_val():
    X, y = load_wine(return_X_y=True, as_frame=False)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def _run(nr_generation, jobs=1, classifier="perc"):
    from pool_generation import poolGeneration

    X, y = load_wine(return_X_y=True, as_frame=False)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    pg = poolGeneration(nr_generation=nr_generation, iteration=1,
                        classifier=classifier, types=["F1", "T1"], jobs=jobs,
                        random_state=42)
    pg.generate(X_tr, y_tr, X_val, y_val, iteration=1)
    pool = pg.get_pool()
    return pg, np.array([c.predict(X_val) for c in pool])


class TestCacheIsInvisible:
    @pytest.mark.slow
    def test_same_seed_same_pool(self):
        _, a = _run(3)
        _, b = _run(3)
        assert np.array_equal(a, b)

    @pytest.mark.slow
    def test_cache_holds_every_bag_created(self):
        pg, _ = _run(3)
        assert set(pg._eval_cache) == set(pg.bags["name"])


class TestCacheAvoidsRework:
    @pytest.mark.slow
    def test_survivors_are_not_re_evaluated(self):
        """Only the 100 new offspring per generation may hit the workers."""
        from pool_generation import poolGeneration

        X, y = load_wine(return_X_y=True, as_frame=False)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        computed: list[int] = []
        orig = fe.FitnessEvaluatorMixin.parallel_distance2

        def spy(self, i, bags, group, types):
            computed.append(i)
            return orig(self, i, bags, group, types)

        fe.FitnessEvaluatorMixin.parallel_distance2 = spy
        try:
            pg = poolGeneration(nr_generation=3, iteration=1, classifier="perc",
                                types=["F1", "T1"], jobs=1, random_state=42)
            pg.generate(X_tr, y_tr, X_val, y_val, iteration=1)
        finally:
            fe.FitnessEvaluatorMixin.parallel_distance2 = orig

        # 100 initial bags + 100 offspring per generation, each evaluated once
        assert len(computed) == len(set(computed)), "a bag was evaluated twice"
        assert len(computed) == 400, f"expected 400 evaluations, got {len(computed)}"


class TestCacheIsResetPerIteration:
    @pytest.mark.slow
    def test_two_iterations_do_not_share_bag_names(self):
        """Bag names restart at 0 each iteration — a stale cache would lie."""
        from pool_generation import poolGeneration

        X, y = load_wine(return_X_y=True, as_frame=False)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        pg = poolGeneration(nr_generation=1, iteration=2, classifier="perc",
                            types=["F1", "T1"], jobs=1, random_state=42)
        pg.generate(X_tr, y_tr, X_val, y_val, iteration=2)
        # after the final iteration the cache only holds that iteration's bags
        assert max(pg._eval_cache) == max(pg.bags["name"])
        assert len(pg._eval_cache) <= len(pg.bags["name"])
