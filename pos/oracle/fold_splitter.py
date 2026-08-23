"""Stratified validation split + protocol viability check for a dataset.

Extracted so `run_recorder` stays under the 150 LOC cap (ADR 0002).

Two concerns, both raised by the external review and fixed in ADR 0014:

1. The inner train/validation split was a plain `rng.choice` — unstratified.
   On imbalanced bases (Thyroid 12.1, Faults 12.2, Ecoli 71.5) that is an
   avoidable source of variance, and it can hand the GA a validation set
   missing whole classes.
2. Some bases in `FULL_DATASETS` cannot satisfy the protocol at all.
   Ecoli has 2 instances in its smallest class; after 10-fold CV, the 20%
   validation split and 50% bagging there is nothing left to stratify, and
   the GA used to die with `exit(0)`. Datasets are now checked up front and
   skipped *explicitly*, with the reason recorded in the run manifest.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def min_class_count(y: np.ndarray) -> int:
    """Number of instances in the least populated class."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    return int(counts.min()) if counts.size else 0


def min_instances_per_class_in_bag(
    min_count: int, n_folds: int, val_frac: float, tam_bags: float
) -> int:
    """Worst-case instances of the rarest class inside a GA bag.

    Chains the three stratified reductions of the protocol:
    k-fold train split -> validation split -> bag sub-sample.
    """
    after_cv = int(min_count * (n_folds - 1) / n_folds)
    after_val = int(after_cv * (1.0 - val_frac))
    return int(after_val * tam_bags)


def check_dataset_viability(
    y: np.ndarray,
    n_folds: int,
    modes: list[str],
    val_frac: float = 0.2,
    tam_bags: float = 0.5,
) -> tuple[bool, str]:
    """Return (ok, reason). `reason` is empty when the dataset is usable.

    The GA requires every bag to keep >= 2 instances of every class
    (`GeneticOperatorsMixin.verify_bag`); rf/bagging only need k-fold to be
    well defined, so the stricter check is applied only when "ga" is a mode.
    """
    m = min_class_count(y)
    if m < n_folds:
        return False, (
            f"min_class_count={m} < n_folds={n_folds}: StratifiedKFold cannot "
            "place the rarest class in every fold"
        )
    if "ga" in modes:
        per_bag = min_instances_per_class_in_bag(m, n_folds, val_frac, tam_bags)
        if per_bag < 2:
            return False, (
                f"min_class_count={m} leaves ~{per_bag} instance(s) of the "
                f"rarest class per GA bag (need >= 2 for verify_bag) with "
                f"n_folds={n_folds}, val_frac={val_frac}, tam_bags={tam_bags}"
            )
    return True, ""


def stratified_val_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_frac: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split (X_train, y_train) into (X_tr, y_tr, X_val, y_val), stratified.

    Falls back to an unstratified split when a class is too rare to be
    stratified — the caller has already been warned by
    `check_dataset_viability` in that case.
    """
    n_val = max(1, int(len(X_train) * val_frac))
    stratify = y_train if min_class_count(y_train) >= 2 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=n_val,
        random_state=random_state, stratify=stratify,
    )
    return X_tr, y_tr, X_val, y_val


def stratified_three_way_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_frac: float,
    random_state: int,
) -> tuple[np.ndarray, ...]:
    """Split into (X_tr, y_tr, X_val, y_val, X_dsel, y_dsel), stratified.

    ADR 0018. `X_tr` keeps exactly the share it had under the two-way split,
    so bagging/rf pools stay bit-identical to the previous run; the `val_frac`
    slice is what gets halved, into the GA's fitness set and the DSEL of the
    dynamic selection. Keeping the two disjoint is the whole point — under the
    two-way split the GA estimated local competence on data it had already
    optimised against.
    """
    X_tr, y_tr, X_rest, y_rest = stratified_val_split(
        X_train, y_train, val_frac, random_state)
    stratify = y_rest if min_class_count(y_rest) >= 2 else None
    X_val, X_dsel, y_val, y_dsel = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=random_state,
        stratify=stratify,
    )
    return X_tr, y_tr, X_val, y_val, X_dsel, y_dsel
