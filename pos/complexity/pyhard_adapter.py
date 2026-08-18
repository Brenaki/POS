"""pyhard-based complexity adapter (Fase 3 — replaces R/ECoL).

Maps the 6 ECoL complexity groups to pyhard's instance-hardness measures,
aggregated to dataset-level scalars via mean(). For ECoL measures without a
pyhard equivalent, returns 0.0 as a documented placeholder (see ADR 0003).

ECoL → pyhard mapping:
  overlapping.F1  → feature_F1      overlapping.F1v → 0.0 (no pyhard equiv)
  overlapping.F2  → feature_F2      overlapping.F3  → feature_F3
  overlapping.F4  → feature_F4
  neighborhood.N1 → feature_N1      neighborhood.N2 → feature_N2
  neighborhood.N3 → 0.0             neighborhood.N4 → feature_kDN
  neighborhood.T1 → 0.0             neighborhood.LSC→ feature_LSC
  linearity.L1/L2/L3      → 0.0 (no pyhard equiv)
  dimensionality.T2/T3/T4 → 0.0 (no pyhard equiv)
  balance.C1 → feature_CB           balance.C2 → feature_MV
  network.Density/ClsCoef/Hubs → 0.0 (no pyhard equiv)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# ECoL measure name → pyhard feature column (None = no equivalent)
ECOL_TO_PYHARD: dict = {
    "overlapping.F1": "feature_F1", "overlapping.F1v": None, "overlapping.F2": "feature_F2",
    "overlapping.F3": "feature_F3", "overlapping.F4": "feature_F4",
    "neighborhood.N1": "feature_N1", "neighborhood.N2": "feature_N2",
    "neighborhood.N3": None, "neighborhood.N4": "feature_kDN",
    "neighborhood.T1": None, "neighborhood.LSC": "feature_LSC",
    "linearity.L1": None, "linearity.L2": None, "linearity.L3": None,
    "dimensionality.T2": None, "dimensionality.T3": None, "dimensionality.T4": None,
    "balance.C1": "feature_CB", "balance.C2": "feature_MV",
    "network.Density": None, "network.ClsCoef": None, "network.Hubs": None,
}

# Group → ordered list of ECoL measure names (matches ECoL output order)
from pos.complexity.base import HEADER, GROUP_MEASURES


def _patch_gower_distance_for_pandas_15() -> None:
    """Monkey-patch pyhard.measures.gower_distance to fix a pandas 1.5 compat bug.

    pyhard 2.2.4 uses `feature.dtypes[0]` which does LABEL access, but
    `features.dtypes` is indexed by column NAME, so it only works for the
    first column (named 0). We replace it with `.iloc[0]` (positional).
    Idempotent: no-op if already patched.
    """
    import pyhard.measures as pm

    if getattr(pm.gower_distance, "_pos_patched", False):
        return

    def _gower_distance_fixed(X: pd.DataFrame) -> np.ndarray:
        from sklearn.metrics import DistanceMetric
        N = len(X)
        n_feat = X.shape[1]
        cumsum_dist = np.zeros((N, N))
        for i in range(n_feat):
            feature = X.iloc[:, [i]]
            if feature.dtypes.iloc[0] == object:
                feature_dist = DistanceMetric.get_metric("dice").pairwise(
                    pd.get_dummies(feature)
                )
            else:
                feature_dist = DistanceMetric.get_metric("manhattan").pairwise(feature)
                feature_dist /= max(np.ptp(feature.values), 1e-8)
            cumsum_dist += feature_dist * 1 / n_feat
        return cumsum_dist

    _gower_distance_fixed._pos_patched = True  # type: ignore[attr-defined]
    pm.gower_distance = _gower_distance_fixed


_patch_gower_distance_for_pandas_15()


def _build_dataframe(X_data: np.ndarray, y_data: np.ndarray) -> pd.DataFrame:
    """Build the DataFrame pyhard expects (features + target column)."""
    df = pd.DataFrame(X_data)
    df["target"] = np.asarray(y_data)
    return df


def _aggregate(measure_values: np.ndarray) -> float:
    """Aggregate instance-level measures to a dataset-level scalar (mean)."""
    arr = np.asarray(measure_values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr))


def complexity_data3(
    X_data: np.ndarray,
    y_data: np.ndarray,
    group: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Compute complexity measures for the given groups using pyhard.

    Drop-in replacement for pos.complexity.ecol_adapter.complexity_data3.
    Returns a flat list of scalars, one per measure per group, in the same
    order ECoL would produce. Measures without a pyhard equivalent return 0.0.
    """
    from pyhard.measures import ClassificationMeasures

    df = _build_dataframe(X_data, y_data)
    cm = ClassificationMeasures(df, "target")
    all_measures = cm.calculate_all()

    result: list = []
    for grp in group:
        measure_names = GROUP_MEASURES.get(grp, [])
        for m_name in measure_names:
            ecol_key = f"{grp}.{m_name}"
            pyhard_col = ECOL_TO_PYHARD.get(ecol_key)
            if pyhard_col is None or pyhard_col not in all_measures.columns:
                result.append(0.0)
            else:
                result.append(_aggregate(all_measures[pyhard_col].values))
    return result
