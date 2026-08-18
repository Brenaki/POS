"""10-fold stratified CV experiment for Oracle_N.

For each fold:
1. Train poolGeneration.generate() on the training split
2. Build correctness matrix on the test split
3. Compute oracle_curve, majority_vote, mean_probs
Aggregate mean ± std over folds.

Random state is fixed (42) for reproducibility (ADR 0006).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from pos.oracle.arff_loader import load_arff_dataset
from pos.oracle.comparison import majority_vote_accuracy, mean_probs_accuracy
from pos.oracle.correctness_matrix import build_correctness_matrix
from pos.oracle.oracle_curve import oracle_curve_array


def _build_pool(X_train, y_train, X_val, y_val, nr_generation: int):
    """Build a classifier pool via poolGeneration (legacy API)."""
    from pool_generation import poolGeneration

    pg = poolGeneration(
        nr_generation=nr_generation,
        iteration=1,
        classifier="tree",
        types=["F1", "T1"],  # skip get_best_types (pyhard expensive)
    )
    pg.generate(X_train, y_train, X_val, y_val, iteration=1)
    return pg.get_pool()


def run_experiment(
    dataset_path: Path | str,
    n_folds: int = 10,
    nr_generation: int = 1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run Oracle_N experiment with stratified k-fold CV.

    Parameters
    ----------
    dataset_path : Path or str
        Path to .arff dataset.
    n_folds : int (default 10)
        Number of stratified CV folds.
    nr_generation : int (default 1)
        GA generations for poolGeneration (keep low for speed; increase for quality).
    random_state : int (default 42)
        Fixed seed for reproducibility.

    Returns
    -------
    result : dict with keys:
        - dataset: str (file stem)
        - n_folds: int
        - oracle_curve_mean: list of floats (length = M)
        - oracle_curve_std:  list of floats
        - majority_vote_mean, majority_vote_std: float
        - mean_probs_mean, mean_probs_std: float
        - n_classifiers: int (M, should be consistent across folds)
    """
    X, y = load_arff_dataset(dataset_path)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_curves: list[list[float]] = []
    fold_majority: list[float] = []
    fold_mean_probs: list[float] = []
    fold_M: list[int] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Split train into train + val for poolGeneration.generate()
        # (it needs X_val, y_val for fitness evaluation)
        n_val = max(1, len(X_train) // 5)
        rng = np.random.default_rng(random_state + fold_idx)
        val_idx = rng.choice(len(X_train), size=n_val, replace=False)
        train_mask = np.ones(len(X_train), dtype=bool)
        train_mask[val_idx] = False
        X_val = X_train[~train_mask]
        y_val = y_train[~train_mask]
        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask]

        pool = _build_pool(X_tr, y_tr, X_val, y_val, nr_generation)
        if len(pool) == 0:
            continue

        matrix = build_correctness_matrix(pool, X_test, y_test)
        fold_curves.append(oracle_curve_array(matrix))
        fold_M.append(matrix.shape[1])
        fold_majority.append(majority_vote_accuracy(pool, X_test, y_test))
        fold_mean_probs.append(mean_probs_accuracy(pool, X_test, y_test))

    # Pad curves to same length (in case M differs; it shouldn't with same config)
    max_M = max(fold_M) if fold_M else 0
    padded = np.full((len(fold_curves), max_M), np.nan)
    for i, curve in enumerate(fold_curves):
        padded[i, : len(curve)] = curve

    return {
        "dataset": Path(dataset_path).stem,
        "n_folds": n_folds,
        "oracle_curve_mean": np.nanmean(padded, axis=0).tolist(),
        "oracle_curve_std": np.nanstd(padded, axis=0).tolist(),
        "majority_vote_mean": float(np.mean(fold_majority)),
        "majority_vote_std": float(np.std(fold_majority)),
        "mean_probs_mean": float(np.mean(fold_mean_probs)),
        "mean_probs_std": float(np.std(fold_mean_probs)),
        "n_classifiers": int(max_M),
    }
