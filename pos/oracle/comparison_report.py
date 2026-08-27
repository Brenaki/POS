"""Turn a schema v2 summary into the descriptive comparison with the paper.

Aggregation follows what the published tables report:

- **MVR** (Table A.10): mean and standard deviation of the majority-vote
  accuracy over the run's replications, in percent.
- **Oracle** (Table 7): the classic pool oracle — at least one classifier of
  the full pool correct. That is `oracle_classic` (= `oracle_1`), which also
  equals the greedy curve's last point. Table 7 publishes no standard
  deviation, so `published_std` is null there.
- **generation** (Table 6): mean and standard deviation of the generation the
  exit criterion selected. Bagging has none.

The comparison is descriptive only: the article publishes mean/std, not the
20 observations, so no paired test and no equivalence claim can be made from
it (ADR 0020 §10). Friedman/Nemenyi and Wilcoxon-Holm stay on raw POS data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pos.oracle.comparator import compare_reference

# summary column -> (comparison method, scale factor)
_METRIC_COLUMNS = {
    "mv_acc": ("mvr", 100.0),
    "oracle_classic": ("oracle", 100.0),
    "pgdcs_best_generation": ("generation", 1.0),
}


def pos_results_from_summary(df: pd.DataFrame) -> dict[tuple[str, str, str], dict]:
    """{(dataset, treatment, method): {mean, std, n}} from summary v2 rows."""
    out: dict[tuple[str, str, str], dict] = {}
    if df.empty or "treatment" not in df.columns:
        return out
    for (dataset, treatment), grp in df.groupby(["dataset", "treatment"]):
        for col, (method, scale) in _METRIC_COLUMNS.items():
            if col not in grp.columns:
                continue
            values = grp[col].dropna()
            if values.empty:
                continue
            out[(dataset, treatment, method)] = {
                "mean": float(values.mean()) * scale,
                "std": float(values.std(ddof=1)) * scale if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }
    return out


def conformance_from_manifest(run_dir: Path) -> dict[str, str]:
    """Dataset -> conformance status, as recorded when the run was made."""
    path = Path(run_dir) / "run_manifest.json"
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text())
    return {k: v.get("conformance_status", "unverified")
            for k, v in (manifest.get("conformance") or {}).items()}


def build_report(run_dir: Path, allow_unverified: bool = False) -> list[dict[str, Any]]:
    """Comparison rows for a run directory holding a v2 summary.csv."""
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "summary.csv")
    statuses = conformance_from_manifest(run_dir)
    from pos.oracle.dataset_reconciliation import comparable_statuses
    return compare_reference(
        pos_results=pos_results_from_summary(df),
        conformance=statuses,
        protocol_compatibility="compatible_modern",
        allowed_statuses=comparable_statuses(allow_unverified),
    )
