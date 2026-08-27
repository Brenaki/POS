"""Paper reference table access (ADR 2020 §9).

Re-exports the data tables from `_paper_tables` and provides
`get_reference_value` for the comparator. Only the Decision Tree tables
are included — the scope of `monteiro-2023-dt-compatible-v1`.
"""

from __future__ import annotations

from typing import Any

from pos.oracle._paper_tables import (
    DATASET_NAMES_PAPER,
    TABLE_4_VOTES,
    TABLE_6_GENERATIONS,
    TABLE_7_ORACLE,
    TABLE_A10_MVR,
)

__all__ = [
    "DATASET_NAMES_PAPER", "TABLE_4_VOTES", "TABLE_6_GENERATIONS",
    "TABLE_7_ORACLE", "TABLE_A10_MVR", "get_reference_value",
]

_TREATMENT_MAP = {"pgdcs_a": "pgdcs_a", "pgdcs_d": "pgdcs_d", "bagging": "bagging"}


def get_reference_value(
    dataset: str, treatment: str, metric: str
) -> dict[str, Any] | None:
    """Return published (mean, std, source_table) or None.

    For Oracle (Table 7), published_std is None (mean only).
    For MVR (Table A.10) and the selected generation (Table 6, `generation`),
    both mean and std are available. Bagging has no generation, so
    `("bagging", "generation")` returns None.
    """
    t = _TREATMENT_MAP.get(treatment)
    if t is None:
        return None
    if metric == "mvr":
        row = TABLE_A10_MVR.get(dataset)
        if row is None:
            return None
        return {
            "published_mean": row[f"{t}_mean"],
            "published_std": row[f"{t}_std"],
            "source_table": "A.10",
        }
    if metric == "generation":
        row = TABLE_6_GENERATIONS.get(dataset)
        if row is None or t == "bagging":
            return None
        return {
            "published_mean": row[f"dt_{t}_mean"],
            "published_std": row[f"dt_{t}_std"],
            "source_table": "6",
        }
    if metric == "oracle":
        row = TABLE_7_ORACLE.get(dataset)
        if row is None:
            return None
        return {
            "published_mean": row[f"dt_{t}"],
            "published_std": None,
            "source_table": "7",
        }
    return None