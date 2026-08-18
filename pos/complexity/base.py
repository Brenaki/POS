"""ECoL complexity measure header constants.

The 22 measures from the R ECoL package, grouped into 6 families. Used by
both the legacy ecol_adapter (Fase 2) and the pyhard_adapter (Fase 3) to
align measure names across backends.
"""

from __future__ import annotations

HEADER: list = [
    "overlapping.F1", "overlapping.F1v", "overlapping.F2", "overlapping.F3", "overlapping.F4",
    "neighborhood.N1", "neighborhood.N2", "neighborhood.N3", "neighborhood.N4",
    "neighborhood.T1", "neighborhood.LSC",
    "linearity.L1", "linearity.L2", "linearity.L3",
    "dimensionality.T2", "dimensionality.T3", "dimensionality.T4",
    "balance.C1", "balance.C2",
    "network.Density", "network.ClsCoef", "network.Hubs",
]

GROUP_MEASURES: dict = {
    "overlapping": ["F1", "F1v", "F2", "F3", "F4"],
    "neighborhood": ["N1", "N2", "N3", "N4", "T1", "LSC"],
    "linearity": ["L1", "L2", "L3"],
    "dimensionality": ["T2", "T3", "T4"],
    "balance": ["C1", "C2"],
    "network": ["Density", "ClsCoef", "Hubs"],
}
