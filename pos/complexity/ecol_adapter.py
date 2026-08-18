"""R/ECoL adapter (extracted from Cpx.complexity_data3 during Fase 2 refactor).

THIS MODULE REQUIRES R + ECoL + R_HOME. It is the legacy backend; Fase 3
will add a pyhard-based adapter as the default and demote this to optional.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages

from pos.complexity.base import HEADER

pandas2ri.activate()
ecol = rpackages.importr("ECoL")


def complexity_data3(
    X_data: np.ndarray,
    y_data: np.ndarray,
    group: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Compute ECoL complexity measures for the given groups.

    Mirrors Cpx.complexity_data3 exactly: iterates over the requested groups,
    calls the matching ecol.<group>(dfx, dfy, measures=..., summary='mean'),
    and concatenates the first column of each group's output.
    """
    dfx = pd.DataFrame(X_data, copy=False)
    dfy = robjects.IntVector(y_data)
    complex = np.array([])
    for i in range(0, len(group)):
        if types:
            measures = types[i]
        else:
            measures = "all"

        if group[i] == "overlapping":
            over = ecol.overlapping(dfx, dfy, measures=measures, summary="mean")
            over = np.asarray(over)
            complex = np.append(complex, over[:, 0])
        if group[i] == "neighborhood":
            nei = ecol.neighborhood(dfx, dfy, measures=measures, summary="mean")
            nei = np.asarray(nei)
            complex = np.append(complex, nei[:, 0])
        if group[i] == "linearity":
            line = ecol.linearity(dfx, dfy, measures=measures, summary="mean")
            line = np.asarray(line)
            complex = np.append(complex, line[:, 0])
        if group[i] == "dimensionality":
            dim = ecol.dimensionality(dfx, dfy, measures=measures, summary="mean")
            dim = np.asarray(dim)
            complex = np.append(complex, dim[:, 0])
        if group[i] == "balance":
            bal = ecol.balance(dfx, dfy, measures=measures, summary="mean")
            bal = np.asarray(bal)
            complex = np.append(complex, bal[:, 0])
        if group[i] == "network":
            net = ecol.network(dfx, dfy, measures=measures, summary="mean")
            net = np.asarray(net)
            complex = np.append(complex, net[:, 0])
    complex = complex.tolist()
    del dfx, dfy
    return complex
