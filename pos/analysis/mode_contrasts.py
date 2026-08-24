"""Paired contrasts between pool-generation modes (ADR 0019).

Five modes exist so that each interesting comparison moves one variable:

    pgdcs   vs ga       the per-dataset complexity-measure vote
    ga      vs randbag  the GA's search, base learner held fixed
    randbag vs bagging  the base learner, search absent from both

Until ADR 0019 only the third contrast was even expressible, and it moved two
variables at once, so "the GA pool has the largest gap" could not be separated
from "the Perceptron is the weakest base learner".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CONTRASTS = [("pgdcs", "ga"), ("ga", "randbag"), ("randbag", "bagging")]
DEFAULT_METRICS = ["oracle_1", "majority_vote", "gap_1", "recovered",
                   "mean_individual_acc", "df_ratio"]


def paired_by_dataset(df: pd.DataFrame, metric: str, mode_a: str,
                      mode_b: str) -> pd.DataFrame:
    """Fold means per dataset for both modes, inner-joined (Demsar)."""
    sub = df[df["mode"].isin([mode_a, mode_b])].dropna(subset=[metric])
    piv = sub.groupby(["dataset", "mode"])[metric].mean().unstack("mode")
    cols = [c for c in (mode_a, mode_b) if c in piv.columns]
    return piv[cols].dropna() if len(cols) == 2 else pd.DataFrame()


def contrast(df: pd.DataFrame, mode_a: str, mode_b: str,
             metrics=None) -> pd.DataFrame:
    """One row per metric: means, paired delta, wins/losses, Wilcoxon p."""
    rows = []
    for metric in (metrics or DEFAULT_METRICS):
        if metric not in df.columns:
            continue
        table = paired_by_dataset(df, metric, mode_a, mode_b)
        if table.empty or len(table) < 3:
            continue
        diff = table[mode_a] - table[mode_b]
        p = 1.0 if np.allclose(diff, 0.0) else float(
            wilcoxon(table[mode_a], table[mode_b])[1])
        rows.append({
            "metric": metric, "n": len(table),
            f"{mode_a}": float(table[mode_a].mean()),
            f"{mode_b}": float(table[mode_b].mean()),
            "delta": float(diff.mean()),
            "wins": int((diff > 1e-12).sum()),
            "losses": int((diff < -1e-12).sum()),
            "p": p,
        })
    return pd.DataFrame(rows)


def all_contrasts(df: pd.DataFrame, metrics=None) -> dict[str, pd.DataFrame]:
    """Every contrast the modes support, skipping ones a run cannot express."""
    out = {}
    present = set(df["mode"].unique())
    for a, b in CONTRASTS:
        if {a, b} <= present:
            table = contrast(df, a, b, metrics)
            if not table.empty:
                out[f"{a}_vs_{b}"] = table
    return out


def pgdcs_measure_choices(df: pd.DataFrame) -> pd.DataFrame:
    """How often the vote picked each measure pair, and how often it was F1/T1."""
    if "pgdcs_types" not in df.columns:
        return pd.DataFrame()
    sub = df[(df["mode"] == "pgdcs") & df["pgdcs_types"].notna()]
    sub = sub[sub["pgdcs_types"].astype(str).str.len() > 0]
    if sub.empty:
        return pd.DataFrame()
    counts = sub["pgdcs_types"].value_counts().rename_axis("types").reset_index(
        name="folds")
    counts["share"] = counts["folds"] / len(sub)
    counts["is_f1_t1"] = counts["types"].isin({"F1|T1", "T1|F1"})
    return counts
