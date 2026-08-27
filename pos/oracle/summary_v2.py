"""Summary.csv v2 schema builder (ADR 0020).

Replaces the ADR 0018 schema with:
- profile / replication identification;
- Oracle checkpoints 1, 5, 10, …, 100 for both readings of Oracle_N —
  `oracle_<n>` is the project's definition ("at least n correct", as in the
  README and schema v1) and `oracle_greedy_<n>` the greedy nested curve —
  plus both full curves as JSON and `oracle_classic` (= `oracle_1`, the point
  Table 7 publishes);
- MVR metrics (mv_acc, …) and mean-probability metrics (mp_acc, …) — with
  Decision Tree on both treatments, soft combination is a genuine
  `predict_proba` average, so the ADR 0016 margin fallback is gone;
- DCS/DES metrics for the six canonical methods;
- PGDCS fields (types, best_generation, best_score, best_metric);
- double_fault_mean, mean_individual_acc;
- **removed:** mean_probs, soft_fusion_rule, mean_decision_norm, soft_*.

The v1 loader (`load_summary_v1`) maps legacy `pgdcs` -> `pgdcs_d` and
preserves soft_* columns as-is for backwards compatibility.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from pos.oracle.des_comparison import des_columns
from pos.oracle.metrics import metric_columns
from pos.oracle.profiles import DES_METHODS_V2, ORACLE_CHECKPOINTS

ORACLE_CHECKPOINTS_V2 = ORACLE_CHECKPOINTS


def build_summary_row_v2(
    profile: str,
    dataset: str,
    fold: int,
    treatment: str,
    metrics: dict[str, Any],
    n_test: int,
) -> dict[str, Any]:
    """Build one summary.csv v2 row dict."""
    curve = metrics.get("oracle_curve", [])
    greedy = metrics.get("oracle_greedy_curve", [])
    row: dict[str, Any] = {
        "profile": profile,
        "dataset": dataset,
        "fold": fold,
        "treatment": treatment,
        "M": metrics.get("n_classifiers"),
        "n_test": int(n_test),
        "oracle_classic": curve[0] if curve else None,
        "oracle_curve_json": json.dumps(curve),
        "oracle_greedy_curve_json": json.dumps(greedy),
        "majority_vote": metrics.get("majority_vote"),
        "double_fault_mean": metrics.get("double_fault_mean"),
        "mean_individual_acc": float(np.mean(metrics.get(
            "individual_accuracies", [float("nan")]))) if metrics.get(
            "individual_accuracies") else None,
        "pgdcs_types": "|".join(metrics.get("pgdcs_types") or []),
        "pgdcs_best_generation": metrics.get("pgdcs_best_generation"),
        "pgdcs_best_score": metrics.get("pgdcs_best_score"),
        "pgdcs_best_metric": metrics.get("pgdcs_best_metric"),
    }
    # Monteiro profile uses "replication" instead of "fold" for identification
    if "monteiro" in profile:
        row["replication"] = fold
    for n in ORACLE_CHECKPOINTS_V2:
        row[f"oracle_{n}"] = curve[n - 1] if len(curve) >= n else None
        row[f"oracle_greedy_{n}"] = greedy[n - 1] if len(greedy) >= n else None
    row.update(metric_columns("mv", metrics.get("majority_vote_metrics")))
    row.update(metric_columns("mp", metrics.get("mean_probs_metrics")))
    row.update(des_columns(metrics.get("des"), methods=DES_METHODS_V2))
    return row


def empty_summary_columns_v2() -> list[str]:
    """Canonical v2 header for a run that produced no rows."""
    metrics = {
        "n_classifiers": 0, "oracle_curve": [], "oracle_greedy_curve": [],
        "majority_vote": None,
        "double_fault_mean": None, "individual_accuracies": [],
        "des": {},
    }
    return list(build_summary_row_v2("", "", 0, "", metrics, 0).keys())


def load_summary_v1(path) -> list[dict[str, Any]]:
    """Read a schema v1 summary.csv, mapping `pgdcs` -> `pgdcs_d`.

    Preserves soft_* columns as-is for historical analysis. Adds
    `profile=None` and `treatment=mode` so downstream code can treat
    both schemas uniformly.
    """
    import pandas as pd
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        row = r.to_dict()
        row["profile"] = row.get("profile")
        mode = row.get("mode", "")
        if mode == "pgdcs":
            row["treatment"] = "pgdcs_d"
        else:
            row["treatment"] = mode
        rows.append(row)
    return rows