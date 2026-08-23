"""Fuser-level comparison — milestone 6 of the cronograma.

`stats_tests.compare` answers "which pool is better". This module answers the
other half of objective 6: given one pool, which real combination method gets
closest to the Oracle. The columns compared are majority vote, the soft
fusion of ADR 0016, and the twelve DCS/DES/static methods of ADR 0018.

Two tiers, because Nemenyi's critical difference grows with the number of
columns: `PRIMARY_FUSERS` (the ADR 0017 set) carries the Friedman/Nemenyi
verdict and stays comparable with the previous run, while everything else is
reported descriptively with Holm-corrected Wilcoxon against majority vote.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pos.analysis.loader import DES_COLS, MODES
from pos.analysis.stats_tests import compare_table, holm_wilcoxon
from pos.oracle.des_methods import PRIMARY_METHODS

FUSERS = ["majority_vote", "mean_probs", *DES_COLS]
PRIMARY_FUSERS = ["majority_vote", "mean_probs",
                  *(f"des_{m}" for m in PRIMARY_METHODS)]

# summary.csv writes the metric suite under a short prefix per fuser, so the
# balanced-accuracy column of `majority_vote` is `mv_balanced_acc`, not
# `majority_vote_balanced_acc`.
METRIC_PREFIX = {"majority_vote": "mv", "mean_probs": "soft"}

FUSER_LABELS = {
    "majority_vote": "MVR", "mean_probs": "fusão suave", "des_ola": "OLA",
    "des_lca": "LCA", "des_mcb": "MCB", "des_rank": "Rank",
    "des_knorae": "KNORA-E", "des_knorau": "KNORA-U", "des_desp": "DES-P",
    "des_desknn": "DES-KNN", "des_metades": "META-DES", "des_knop": "KNOP",
    "des_single_best": "melhor individual", "des_static_sel": "seleção estática",
}


def available_fusers(df: pd.DataFrame, mode: str, fusers=None) -> list[str]:
    """Fuser columns that actually carry a value for this pool mode.

    A method can be inapplicable to a whole mode — META-DES and KNOP need
    `predict_proba`, so they are empty for every Perceptron pool — and
    including an all-NaN column would silently drop every dataset from the
    paired test.
    """
    sub = df[df["mode"] == mode]
    return [c for c in (fusers or FUSERS)
            if c in sub.columns and sub[c].notna().any()]


def fuser_table(df: pd.DataFrame, mode: str, fusers=None,
                value: str = "") -> pd.DataFrame:
    """dataset x fuser table for one pool mode, folds averaged.

    `value` picks the metric suffix ("" = accuracy, "balanced_acc", "f1", ...);
    milestone 7 asks the same comparison to be read on more than accuracy.
    """
    sub = df[df["mode"] == mode]
    cols = available_fusers(df, mode, fusers)
    if value:
        named = {c: f"{METRIC_PREFIX.get(c, c)}_{value}" for c in cols}
        named = {c: n for c, n in named.items() if n in sub.columns}
        if not named:
            return pd.DataFrame()
        table = sub.pivot_table(index="dataset", values=list(named.values()),
                                aggfunc="mean")
        table = table[[named[c] for c in cols if c in named]]
        return table.rename(columns={v: k for k, v in named.items()}).dropna()
    return sub.pivot_table(index="dataset", values=cols, aggfunc="mean")[cols].dropna()


def fuser_comparison(df: pd.DataFrame, mode: str, fusers=None,
                     value: str = "") -> dict:
    """Friedman + Nemenyi over the fusers available for one pool mode."""
    label = f"fusores [{mode}]" + (f" — {value}" if value else "")
    return compare_table(fuser_table(df, mode, fusers, value),
                         higher_is_better=True, name=label)


def secondary_vs_mvr(df: pd.DataFrame, mode: str, value: str = "") -> pd.DataFrame:
    """Every fuser against majority vote: delta, Wilcoxon p, Holm-corrected p.

    The descriptive tier of ADR 0018. Friedman over fourteen columns has no
    power left; a family of paired tests against the one baseline that
    matters does, provided the family is corrected.
    """
    table = fuser_table(df, mode, FUSERS, value)
    if table.empty or "majority_vote" not in table:
        return pd.DataFrame()
    others = [c for c in table.columns if c != "majority_vote"]
    res = holm_wilcoxon(table, "majority_vote", others)
    res.insert(0, "label", [FUSER_LABELS.get(c, c) for c in res.index])
    return res


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
        for col in available_fusers(df, mode, FUSERS):
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
