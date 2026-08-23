"""Fuser-level comparison — milestone 6 of the cronograma.

`stats_tests.compare` answers "which pool is better". This module answers the
other half of objective 6: given one pool, which real combination method gets
closest to the Oracle. The columns compared are majority vote, the soft
fusion of ADR 0016, and the five DCS/DES methods of ADR 0017.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pos.analysis.loader import DES_COLS, MODES
from pos.analysis.stats_tests import compare_table

FUSERS = ["majority_vote", "mean_probs", *DES_COLS]

FUSER_LABELS = {
    "majority_vote": "MVR", "mean_probs": "fusão suave", "des_ola": "OLA",
    "des_lca": "LCA", "des_knorae": "KNORA-E", "des_knorau": "KNORA-U",
    "des_metades": "META-DES",
}


def available_fusers(df: pd.DataFrame, mode: str) -> list[str]:
    """Fuser columns that actually carry a value for this pool mode.

    A method can be inapplicable to a whole mode — META-DES needs
    `predict_proba`, so it is empty for every Perceptron pool — and including
    an all-NaN column would silently drop every dataset from the paired test.
    """
    sub = df[df["mode"] == mode]
    return [c for c in FUSERS if c in sub.columns and sub[c].notna().any()]


def fuser_table(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """dataset x fuser table for one pool mode, folds averaged."""
    sub = df[df["mode"] == mode]
    cols = available_fusers(df, mode)
    return sub.pivot_table(index="dataset", values=cols, aggfunc="mean")[cols].dropna()


def fuser_comparison(df: pd.DataFrame, mode: str) -> dict:
    """Friedman + Nemenyi over the fusers available for one pool mode."""
    return compare_table(fuser_table(df, mode), higher_is_better=True,
                         name=f"fusores [{mode}]")


def recovery_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per mode: what the pool holds, what each fuser reaches, what is left.

    `recovered` is the share of the Oracle_1 - MVR gap that the best dynamic
    selection method actually reaches. It is the number objective 8 wants to
    predict in advance, and the one that says whether DCS/DES is worth its
    cost on a given pool.
    """
    rows = []
    for mode in MODES:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        row = {
            "mode": mode,
            "oracle_1": sub["oracle_1"].mean(),
            "majority_vote": sub["majority_vote"].mean(),
            "mean_individual_acc": sub["mean_individual_acc"].mean(),
        }
        for col in available_fusers(df, mode):
            row[col] = sub[col].mean()
        if "des_best" in sub.columns:
            row["des_best"] = sub["des_best"].mean()
            row["recovered"] = sub["recovered"].mean()
            row["best_method_mode"] = (
                sub["des_best_method"].mode().iat[0]
                if sub["des_best_method"].notna().any() else None
            )
        row["gap_mv"] = row["oracle_1"] - row["majority_vote"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("mode")
    # A fuser missing for one mode (META-DES on Perceptron pools) would
    # otherwise land in a trailing column instead of next to its peers.
    order = ["oracle_1", "mean_individual_acc", *FUSERS, "des_best",
             "recovered", "best_method_mode", "gap_mv"]
    return table[[c for c in order if c in table.columns]]


def recovery_vs_redundancy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Paired (df_ratio, recovered) points, one per dataset x mode."""
    sub = df.dropna(subset=["recovered", "df_ratio"])
    grp = sub.groupby(["dataset", "mode"])[["df_ratio", "recovered"]].mean()
    return grp["df_ratio"].values, grp["recovered"].values
