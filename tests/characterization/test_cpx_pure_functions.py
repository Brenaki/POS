"""Characterization tests for the Python-pure functions of Cpx.py.

These capture the CURRENT observable behavior of the functions that do NOT
depend on R/ECoL, so we have a regression net before the Fase 2 refactor
(Fowler: "run the code, capture behavior, then change structure").

The rpy2/ECoL mocks installed by conftest.py let us `import Cpx` without R;
only `complexity_data3` is R-dependent and is tested in a separate file under
@pytest.mark.requires_r.
"""

from __future__ import annotations

import numpy as np
import pytest

# conftest.py injects rpy2 mocks into sys.modules before this import lands.
import Cpx  # noqa: E402  — legacy module, top-level rpy2 import is mocked


# ---------------------------------------------------------------------------
# min_max_norm
# ---------------------------------------------------------------------------

class TestMinMaxNorm:
    def test_linear_range_maps_to_0_to_1(self):
        result = Cpx.min_max_norm([0, 1, 2, 3, 4])
        assert result == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_constant_input_returns_zeros(self):
        # when min == max, the function returns a list of 0 (int, not float)
        result = Cpx.min_max_norm([5, 5, 5, 5])
        assert result == [0, 0, 0, 0]

    def test_negative_values(self):
        result = Cpx.min_max_norm([-2, 0, 2])
        assert result == [0.0, 0.5, 1.0]

    def test_float_values(self):
        result = Cpx.min_max_norm([1.0, 2.0, 3.0])
        assert result == [0.0, 0.5, 1.0]

    def test_single_element_constant(self):
        # single element: min == max == that element → returns [0]
        result = Cpx.min_max_norm([42])
        assert result == [0]

    def test_numpy_array_input(self):
        result = Cpx.min_max_norm(np.array([10, 20, 30]))
        assert result == [0.0, 0.5, 1.0]

    def test_2d_array_returns_list_of_per_row_arrays(self):
        # KNOWN QUIRK: np.min/np.max on a 2d array return GLOBAL scalars, but
        # `for value in dataset` iterates ROWS, so each output element is a
        # 1d array (not a float). This characterizes the actual behavior and
        # must be preserved during the Fase 2 refactor (Fowler: capture what
        # the code DOES, not what it should do).
        result = Cpx.min_max_norm(np.array([[0, 10], [5, 15]]))
        assert isinstance(result, list)
        assert len(result) == 2  # one entry per row
        # global min=0, global max=15
        np.testing.assert_allclose(result[0], [0.0, 10 / 15])
        np.testing.assert_allclose(result[1], [5 / 15, 1.0])


# ---------------------------------------------------------------------------
# dispersion_linear
# ---------------------------------------------------------------------------

class TestDispersionLinear:
    def test_shape_3_bags_2_measures(self, small_complexity_matrix):
        result = Cpx.dispersion_linear(small_complexity_matrix)
        arr = np.array(result)
        # Output shape is (n_bags, n_measures) after the internal .T
        assert arr.shape == (3, 2)

    def test_golden_values(self, small_complexity_matrix):
        result = Cpx.dispersion_linear(small_complexity_matrix)
        expected = [
            [2 / 3, 0.75],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_returns_list_of_lists(self, small_complexity_matrix):
        result = Cpx.dispersion_linear(small_complexity_matrix)
        assert isinstance(result, list)
        assert all(isinstance(row, list) for row in result)


# ---------------------------------------------------------------------------
# dispersion (pairwise mean distance via sklearn)
# ---------------------------------------------------------------------------

class TestDispersion:
    def test_returns_one_value_per_row(self):
        complexity = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        result = Cpx.dispersion(complexity)
        assert len(result) == 3

    def test_identical_points_zero_distance(self):
        complexity = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        result = Cpx.dispersion(complexity)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-9)

    def test_outlier_has_larger_mean_distance(self):
        # point far from two clustered points should have the largest mean dist
        complexity = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0]])
        result = Cpx.dispersion(complexity)
        assert np.argmax(result) == 2


# ---------------------------------------------------------------------------
# diversitys (pairwise double-fault mean)
# ---------------------------------------------------------------------------

class TestDiversitys:
    def test_returns_one_value_per_classifier(self, dummy_predictions):
        y_test, preds = dummy_predictions
        result = Cpx.diversitys(y_test, preds)
        assert len(result) == 3

    def test_identical_classifiers_zero_double_fault_difference(self):
        # If all classifiers predict identically, double-fault rates are equal
        y_test = np.array([0, 0, 1, 1])
        preds = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ])
        result = Cpx.diversitys(y_test, preds)
        # double_fault(y, p_i, p_j) where p_i == p_j: counts cases where both wrong
        # All classifiers correct → double fault = 0 for all pairs
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-9)

    def test_perfect_classifiers_zero_double_fault(self, dummy_predictions):
        y_test, preds = dummy_predictions
        # classifier 0 is correct on 7/8; double faults are bounded ≥ 0
        result = Cpx.diversitys(y_test, preds)
        assert all(r >= 0.0 for r in result)
        assert all(r <= 1.0 for r in result)


# ---------------------------------------------------------------------------
# voting_classifier (mlxtend EnsembleVoteClassifier hard voting)
# ---------------------------------------------------------------------------

class TestVotingClassifier:
    def test_returns_float_score(self, wine_split):
        from sklearn.tree import DecisionTreeClassifier

        X_tr, y_tr = wine_split["X_train"], wine_split["y_train"]
        pool = [
            DecisionTreeClassifier(random_state=1).fit(X_tr, y_tr),
            DecisionTreeClassifier(random_state=2).fit(X_tr, y_tr),
            DecisionTreeClassifier(random_state=3).fit(X_tr, y_tr),
        ]
        score = Cpx.voting_classifier(pool, X_tr, y_tr)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_single_classifier_matches_individual_score(self, wine_split):
        from sklearn.tree import DecisionTreeClassifier

        X_tr, y_tr = wine_split["X_train"], wine_split["y_train"]
        tree = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
        score = Cpx.voting_classifier([tree], X_tr, y_tr)
        individual = tree.score(X_tr, y_tr)
        assert score == pytest.approx(individual)


# ---------------------------------------------------------------------------
# biuld_classifier (Perceptron) and biuld_classifier_tree (DecisionTree)
# ---------------------------------------------------------------------------

class TestBiuldClassifier:
    def test_perceptron_with_xtest_array_raises_valueerror(self, wine_split):
        # KNOWN BUG: `biuld_classifier` uses `X_test != None` which, for a
        # numpy array, raises "truth value of an array is ambiguous". This
        # means the Perceptron path is BROKEN whenever X_test is passed as an
        # array (which is what pool_generation.parallel_distance2 does when
        # classifier="perc"). Preserved during Fase 2; fix is a separate ADR.
        X_tr, y_tr = wine_split["X_train"], wine_split["y_train"]
        X_val, y_val = wine_split["X_valid"], wine_split["y_valid"]
        with pytest.raises(ValueError, match="ambiguous"):
            Cpx.biuld_classifier(X_tr, y_tr, X_tr, y_tr, X_val, y_val)

    def test_perceptron_score_train_with_xtest_array_also_raises(self, wine_split):
        # Same root bug affects the score_train=True branch.
        X_tr, y_tr = wine_split["X_train"], wine_split["y_train"]
        X_val, y_val = wine_split["X_valid"], wine_split["y_valid"]
        with pytest.raises(ValueError, match="ambiguous"):
            Cpx.biuld_classifier(X_tr, y_tr, X_tr, y_tr, X_val, y_val, score_train=True)

    def test_tree_returns_estimator_score_predict(self, wine_split):
        X_tr, y_tr = wine_split["X_train"], wine_split["y_train"]
        X_val, y_val = wine_split["X_valid"], wine_split["y_valid"]
        est, score, pred = Cpx.biuld_classifier_tree(X_tr, y_tr, X_tr, y_tr, X_val, y_val)
        from sklearn.tree import DecisionTreeClassifier

        assert isinstance(est, DecisionTreeClassifier)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert len(pred) == len(y_val)

    def test_perceptron_without_test_returns_estimator_score(self, wine_split):
        X_tr, y_tr = wine_split["X_train"], wine_split["y_train"]
        # X_test=None, y_test=None → returns (est, score) only (the bug path
        # is not triggered because `None != None` is False → falls to else)
        result = Cpx.biuld_classifier(X_tr, y_tr, X_tr, y_tr)
        assert len(result) == 2
        est, score = result
        from sklearn.linear_model import Perceptron

        assert isinstance(est, Perceptron)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# save_bag (CSV IO) — exercised minimally to characterize the path logic
# ---------------------------------------------------------------------------

class TestSaveBag:
    def test_save_validation_bag_writes_csv(self, tmp_path, monkeypatch):
        # The function uses os.system('mkdir -p ...') and writes relative paths.
        # We chdir to a tmp_path and verify a CSV is produced.
        monkeypatch.chdir(tmp_path)
        Cpx.save_bag([1, 2, 3], "validation", str(tmp_path), "bag_test", 0)
        # The function writes to base_name + '.csv' in cwd
        csv_path = tmp_path / "bag_test.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "1" in content and "2" in content and "3" in content
