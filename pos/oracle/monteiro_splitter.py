"""Monteiro 2023 holdout splitter (ADR 0020).

20 independent stratified holdouts 50/25/25, new split per replication,
deterministic given a root seed. Validation and DSEL overlap by design in the
`monteiro-2023-dt-compatible-v1` profile (the published protocol uses the
validation set for both the PGDCS_A/fitness criterion and DCS/DES); this is a
recorded divergence from `pos-oracle-v2`, which separates them.

Provenance:
- 50/25/25 split: Monteiro 2023.
- stratification + new split per replication: thesis 2021 (article omits it).
- validation == DSEL: ADR 0020 §2.2 (explicit divergence from the POS profile).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


def _min_class_count(y: np.ndarray) -> int:
    _, counts = np.unique(np.asarray(y), return_counts=True)
    return int(counts.min()) if counts.size else 0


def _stratify_or_none(y: np.ndarray) -> np.ndarray | None:
    """Stratify only when every class has >= 2 instances."""
    return y if _min_class_count(y) >= 2 else None


def monteiro_holdout_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_replications: int = 20,
    root_seed: int = 86,
    train_frac: float = 0.5,
    val_frac: float = 0.25,
) -> list[dict[str, Any]]:
    """Produce `n_replications` independent 50/25/25 stratified holdouts.

    Each replication r draws from `root_seed + r`. test_frac is the
    remainder: 1 - train_frac - val_frac.

    The validation partition is also returned as DSEL (same arrays) — the
    Monteiro profile deliberately overlaps them. Callers that need disjoint
    DSEL should use the POS profile's three-way split instead.

    Returns a list of dicts, one per replication, with keys:
        X_train, y_train, X_val, y_val, X_test, y_test, X_dsel, y_dsel
    """
    test_frac = 1.0 - train_frac - val_frac
    splits: list[dict[str, Any]] = []
    for r in range(n_replications):
        seed = root_seed + r
        stratify = _stratify_or_none(y)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=(val_frac + test_frac), random_state=seed,
            stratify=stratify,
        )
        stratify_rest = _stratify_or_none(y_rest)
        rel_val = val_frac / (val_frac + test_frac)
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=(1.0 - rel_val), random_state=seed,
            stratify=stratify_rest,
        )
        splits.append({
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            # ADR 0020 §2.2: validation == DSEL in the Monteiro profile.
            "X_dsel": X_val, "y_dsel": y_val,
        })
    return splits