"""Min-max normalization (extracted from Cpx.py during Fase 2 refactor).

Behavior preserved exactly from the legacy Cpx.min_max_norm, including the
quirk that a constant input returns a list of int 0 (not float 0.0) and that
a 2d array iterates over rows producing per-row 1d arrays. See ADR 0004 and
the characterization tests in tests/characterization/test_cpx_pure_functions.py.
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[list, np.ndarray]


def min_max_norm(dataset: ArrayLike) -> list:
    """Normalize values to [0, 1] using global min and max.

    When min == max (constant input), returns a list of int 0 for each
    element. When dataset is a 2d ndarray, iterates over ROWS and returns
    a list of 1d arrays (one per row) — this is the legacy behavior and
    is preserved on purpose.
    """
    norm_list: list = []
    min_value = np.min(dataset)
    max_value = np.max(dataset)

    if min_value == max_value:
        for _ in dataset:
            norm_list.append(0)
        return norm_list

    for value in dataset:
        tmp = (value - min_value) / (max_value - min_value)
        norm_list.append(tmp)

    return norm_list
