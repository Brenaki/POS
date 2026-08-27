"""Reference comparator (ADR 2020 §10).

Produces a descriptive comparison between POS results and the published
numbers from Monteiro 2023. Comparison is DESCRIPTIVE only: no paired test,
no automatic equivalence claim — the paper gives only mean/std, not the 20
observations or seeds.

Friedman/Nemenyi and Wilcoxon-Holm are applied only to raw POS results,
never to the published numbers.
"""

from __future__ import annotations

from typing import Any

from pos.oracle.dataset_reconciliation import comparable_statuses
from pos.oracle.paper_reference import get_reference_value

_TREATMENTS = ("pgdcs_a", "pgdcs_d", "bagging")
_METHODS = ("mvr", "oracle", "generation")


def build_comparison_row(
    dataset: str,
    treatment: str,
    method: str,
    published_mean: float,
    published_std: float | None,
    pos_mean: float,
    pos_std: float,
    dataset_conformance: str,
    protocol_compatibility: str,
    n_pos: int = 20,
) -> dict[str, Any]:
    """Build one comparison row dict."""
    delta = pos_mean - published_mean
    ci95: dict[str, float] | None = None
    if n_pos > 1 and pos_std is not None:
        se = pos_std / (n_pos ** 0.5)
        ci95 = {"low": pos_mean - 1.96 * se, "high": pos_mean + 1.96 * se}
    ref = get_reference_value(dataset, treatment, method)
    source_table = ref["source_table"] if ref else "unknown"
    return {
        "dataset": dataset,
        "treatment": treatment,
        "method": method,
        "published_mean": published_mean,
        "published_std": published_std,
        "pos_mean": pos_mean,
        "pos_std": pos_std,
        "delta_percentage_points": round(delta, 4),
        "pos_ci95": ci95,
        "dataset_conformance": dataset_conformance,
        "protocol_compatibility": protocol_compatibility,
        "source_table": source_table,
    }


def compare_reference(
    pos_results: dict[tuple[str, str, str], dict[str, float]],
    conformance: dict[str, str],
    protocol_compatibility: str = "compatible_modern",
    allowed_statuses: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Compare POS results with published reference values.

    `pos_results` keys are (dataset, treatment, method) with values
    {mean, std, n}. Only datasets whose conformance status is in
    `allowed_statuses` (default: conformant + approved_with_note, see ADR 0020
    §8) enter the report; the status travels with every row.

    Returns a list of comparison row dicts, one per (dataset, treatment,
    method) that has both a published reference and a POS result.
    """
    allowed = allowed_statuses or comparable_statuses()
    rows: list[dict[str, Any]] = []
    for (dataset, treatment, method), pos in pos_results.items():
        if conformance.get(dataset) not in allowed:
            continue
        ref = get_reference_value(dataset, treatment, method)
        if ref is None:
            continue
        rows.append(build_comparison_row(
            dataset=dataset, treatment=treatment, method=method,
            published_mean=ref["published_mean"],
            published_std=ref["published_std"],
            pos_mean=pos["mean"], pos_std=pos["std"],
            dataset_conformance=conformance.get(dataset, "unverified"),
            protocol_compatibility=protocol_compatibility,
            n_pos=int(pos.get("n", 20)),
        ))
    return rows