"""Train/test/validation splitting (extracted from pool_generation.py)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def split_data(X_data: np.ndarray, y_data: np.ndarray) -> Tuple:
    """Split data into 50% train, 25% test, 25% validation (stratified).

    Returns (X_train, y_train, X_test, y_test, X_vali, y_vali,
             id_train, id_test, id_vali).
    """
    indices = np.arange(len(X_data))
    (X_train, X_temp, y_train, y_temp, id_train, id_temp) = train_test_split(
        X_data, y_data, indices, test_size=0.5, stratify=y_data
    )
    (X_test, X_vali, y_test, y_vali, id_test, id_vali) = train_test_split(
        X_temp, y_temp, id_temp, test_size=0.5, stratify=y_temp
    )
    return (
        X_train, y_train, X_test, y_test, X_vali, y_vali,
        id_train, id_test, id_vali,
    )
