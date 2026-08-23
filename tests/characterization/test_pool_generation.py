"""Characterization tests for pool_generation.poolGeneration.

End-to-end tests (TestPoolGenerationEndToEnd) require R because generate()
calls complexity_data3 which needs ECoL. Unit tests (TestPoolGenerationUnit)
only need the rpy2 mock from conftest.py and run without R installed.
"""

from __future__ import annotations

import numpy as np
import pytest

# No module-level pytestmark — each class picks its own markers.


def _import_pool_generation():
    """Import pool_generation using the rpy2 mock installed by conftest."""
    import pool_generation
    return pool_generation


class TestPoolGenerationEndToEnd:
    @pytest.mark.slow
    def test_generate_get_bags_get_pool_wine(self, wine_split):
        """End-to-end: poolGeneration.generate → get_bags → get_pool on wine.

        Uses DEFAULT pool sizes (nr_bags=nr_individual=nr_pop=nr_child=100).
        The hardcoded `range(100, nr_individual+100)` in
        fitness_evaluator.get_complexity() was removed in ADR 0014 — offspring
        are now resolved by bag name, so this no longer depends on 100.

        DEAP generation_function bug (ADR 0007) was fixed in ADR 0008.
        ~270s with pyhard backend, nr_generation=1, iteration=1.
        """
        pg_mod = _import_pool_generation()
        pool_gen = pg_mod.poolGeneration(
            nr_generation=1,
            iteration=1,
            classifier="tree",
            types=["F1", "T1"],  # skip get_best_types
        )
        pool_gen.generate(
            wine_split["X_train"], wine_split["y_train"],
            wine_split["X_test"], wine_split["y_test"],
            iteration=1,
        )
        bags = pool_gen.get_bags()
        pool = pool_gen.get_pool()

        # Structural assertions (not exact values — GA is stochastic)
        assert isinstance(bags, list)
        assert len(bags) >= 1
        for bag in bags:
            X_bag, y_bag = bag
            # build_bags returns plain lists, not arrays
            assert len(X_bag) == len(y_bag)
            assert len(X_bag[0]) == wine_split["X_train"].shape[1]

        assert isinstance(pool, list)
        assert len(pool) == len(bags)
        # Each pool member is a fitted classifier
        for clf in pool:
            assert hasattr(clf, "predict")
            assert hasattr(clf, "classes_")

    @pytest.mark.slow
    def test_get_pool_returns_decision_trees_when_classifier_tree(self, wine_split):
        """Same as above but checks classifier type. Uses defaults (100)."""
        pg_mod = _import_pool_generation()
        pool_gen = pg_mod.poolGeneration(
            nr_generation=1, iteration=1, classifier="tree",
            types=["F1", "T1"],
        )
        pool_gen.generate(
            wine_split["X_train"], wine_split["y_train"],
            wine_split["X_test"], wine_split["y_test"],
            iteration=1,
        )
        pool = pool_gen.get_pool()
        from sklearn.tree import DecisionTreeClassifier

        assert all(isinstance(c, DecisionTreeClassifier) for c in pool)


class TestPoolGenerationUnit:
    """Unit-level tests for poolGeneration methods that don't need the GA loop."""

    def test_generate_bags_returns_dict_with_name_and_inst(self, wine_split):
        pg_mod = _import_pool_generation()
        pool_gen = pg_mod.poolGeneration(nr_bags=5, tam_bags=0.5)
        pool_gen.X_train = wine_split["X_train"]
        pool_gen.y_train = wine_split["y_train"]
        bags = pool_gen.generate_bags(wine_split["X_train"], wine_split["y_train"])

        assert isinstance(bags, dict)
        assert "name" in bags and "inst" in bags
        assert len(bags["name"]) == 5
        assert len(bags["inst"]) == 5
        # Each bag is a list of instance indices
        for inst in bags["inst"]:
            assert isinstance(inst, list)
            assert all(isinstance(i, (int, np.integer)) for i in inst)

    def test_split_data_produces_three_disjoint_splits(self, wine_split):
        pg_mod = _import_pool_generation()
        pool_gen = pg_mod.poolGeneration()
        result = pool_gen.split_data(wine_split["X_train"], wine_split["y_train"])
        X_train, y_train, X_test, y_test, X_vali, y_vali, id_train, id_test, id_vali = result

        # 50% train, 25% test, 25% validation
        n = len(wine_split["X_train"])
        assert len(X_train) == n // 2
        assert len(X_test) == n // 4
        assert len(X_vali) == n - n // 2 - n // 4

        # Indices are disjoint
        all_ids = set(id_train) | set(id_test) | set(id_vali)
        assert len(all_ids) == n

    def test_build_bags_returns_subset_of_train(self, wine_split):
        pg_mod = _import_pool_generation()
        pool_gen = pg_mod.poolGeneration()
        pool_gen.X_train = wine_split["X_train"]
        pool_gen.y_train = wine_split["y_train"]

        indices = list(range(10))
        X_bag, y_bag = pool_gen.build_bags(indices)
        assert len(X_bag) == 10
        assert len(y_bag) == 10
        np.testing.assert_array_equal(X_bag, wine_split["X_train"][:10])

    def test_verify_bag_rejects_missing_class(self, wine_split):
        pg_mod = _import_pool_generation()
        pool_gen = pg_mod.poolGeneration()
        pool_gen.X_train = wine_split["X_train"]
        pool_gen.y_train = wine_split["y_train"]

        # A bag with only one class should fail verification
        single_class_indices = [i for i, y in enumerate(wine_split["y_train"]) if y == 0][:5]
        assert pool_gen.verify_bag(single_class_indices) is False

    def test_verify_bag_accepts_all_classes_with_min_2(self, wine_split):
        pg_mod = _import_pool_generation()
        pool_gen = pg_mod.poolGeneration()
        pool_gen.X_train = wine_split["X_train"]
        pool_gen.y_train = wine_split["y_train"]

        # Pick 2 instances per class (wine has 3 classes)
        indices = []
        for cls in np.unique(wine_split["y_train"]):
            cls_idx = [i for i, y in enumerate(wine_split["y_train"]) if y == cls][:2]
            indices.extend(cls_idx)
        assert pool_gen.verify_bag(indices) is True
