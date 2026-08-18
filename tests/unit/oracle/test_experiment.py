"""Unit tests for pos.oracle.experiment (10-fold CV protocol).

TDD — tests written BEFORE implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DATASET_DIR = Path(__file__).resolve().parents[3] / "Dataset"

pytestmark = pytest.mark.slow  # all tests in this module run the GA (~270s/fold)


def test_run_experiment_returns_dict_with_expected_keys():
    """run_experiment should return a dict with oracle_curve, majority, mean_probs."""
    from pos.oracle.experiment import run_experiment

    path = DATASET_DIR / "Wine.arff"
    if not path.exists():
        pytest.skip(f"Dataset não encontrado: {path}")

    result = run_experiment(
        dataset_path=path,
        n_folds=3,  # small for test speed
        nr_generation=1,
        random_state=42,
    )

    assert isinstance(result, dict)
    expected_keys = {"dataset", "n_folds", "oracle_curve_mean", "oracle_curve_std",
                     "majority_vote_mean", "majority_vote_std",
                     "mean_probs_mean", "mean_probs_std", "n_classifiers"}
    assert expected_keys.issubset(result.keys())


def test_run_experiment_oracle_curve_is_monotonic():
    """The mean Oracle curve over folds must be non-increasing."""
    from pos.oracle.experiment import run_experiment

    path = DATASET_DIR / "Wine.arff"
    if not path.exists():
        pytest.skip(f"Dataset não encontrado: {path}")

    result = run_experiment(
        dataset_path=path,
        n_folds=3,
        nr_generation=1,
        random_state=42,
    )
    curve = result["oracle_curve_mean"]
    for i in range(len(curve) - 1):
        assert curve[i] >= curve[i + 1], (
            f"Monotonicidade violada: Oracle_{i+1}={curve[i]} < Oracle_{i+2}={curve[i+1]}"
        )


def test_run_experiment_oracle_1_geq_majority_vote():
    """Oracle_1 (traditional Oracle) must be >= majority vote accuracy."""
    from pos.oracle.experiment import run_experiment

    path = DATASET_DIR / "Wine.arff"
    if not path.exists():
        pytest.skip(f"Dataset não encontrado: {path}")

    result = run_experiment(
        dataset_path=path,
        n_folds=3,
        nr_generation=1,
        random_state=42,
    )
    oracle_1 = result["oracle_curve_mean"][0]  # Oracle_1 = first element
    majority = result["majority_vote_mean"]
    assert oracle_1 >= majority, (
        f"Oracle_1 ({oracle_1}) should be >= majority vote ({majority})"
    )


def test_run_experiment_reproducible_with_seed():
    """Same random_state should produce same results."""
    from pos.oracle.experiment import run_experiment

    path = DATASET_DIR / "Wine.arff"
    if not path.exists():
        pytest.skip(f"Dataset não encontrado: {path}")

    r1 = run_experiment(
        dataset_path=path, n_folds=3, nr_generation=1, random_state=42,
    )
    r2 = run_experiment(
        dataset_path=path, n_folds=3, nr_generation=1, random_state=42,
    )
    assert r1["oracle_curve_mean"] == r2["oracle_curve_mean"]
    assert r1["majority_vote_mean"] == r2["majority_vote_mean"]
