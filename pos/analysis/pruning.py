"""Does how much a method prunes the pool explain how well it does? (ADR 0019)

The ADR 0018 run concluded that pruning is the axis ordering the fusers, but
read it off the algorithms' definitions. With `des_<m>_sel_frac` recorded, the
claim becomes a measurement: correlate the mean selected fraction with the
method's margin over majority vote.

`routed_frac` matters for reading the numbers: DESlib decides queries whose
region of competence is unanimous without consulting the selector, so
`sel_frac` describes the queries that actually reached dynamic selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pos.oracle.des_methods import DES_METHODS


def pruning_frame(df: pd.DataFrame, methods=DES_METHODS) -> pd.DataFrame:
    """Long frame: one row per (dataset, mode, method) with sel_frac and margin."""
    rows = []
    for method in methods:
        acc_col, sel_col = f"des_{method}", f"des_{method}_sel_frac"
        if acc_col not in df.columns or sel_col not in df.columns:
            continue
        sub = df.dropna(subset=[acc_col, sel_col, "majority_vote"])
        if sub.empty:
            continue
        g = sub.groupby(["dataset", "mode"], as_index=False).agg(
            sel_frac=(sel_col, "mean"),
            routed_frac=(f"des_{method}_routed_frac", "mean")
            if f"des_{method}_routed_frac" in sub.columns else (sel_col, "size"),
            acc=(acc_col, "mean"),
            mvr=("majority_vote", "mean"),
        )
        g["method"] = method
        g["delta_vs_mvr"] = g["acc"] - g["mvr"]
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pruning_correlation(df: pd.DataFrame, methods=DES_METHODS) -> pd.DataFrame:
    """Spearman(sel_frac, delta_vs_mvr) overall and within each mode."""
    frame = pruning_frame(df, methods)
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for mode in [None, *sorted(frame["mode"].unique())]:
        sub = frame if mode is None else frame[frame["mode"] == mode]
        sub = sub.dropna(subset=["sel_frac", "delta_vs_mvr"])
        if len(sub) < 5 or sub["sel_frac"].nunique() < 2:
            continue
        rho, p = spearmanr(sub["sel_frac"], sub["delta_vs_mvr"])
        rows.append({"mode": mode or "todos", "n": len(sub),
                     "spearman": float(rho), "p": float(p)})
    return pd.DataFrame(rows)


def pruning_table(df: pd.DataFrame, methods=DES_METHODS) -> pd.DataFrame:
    """Per-method mean selected fraction and margin, ordered by pruning."""
    frame = pruning_frame(df, methods)
    if frame.empty:
        return pd.DataFrame()
    out = frame.groupby("method", as_index=False).agg(
        sel_frac=("sel_frac", "mean"),
        routed_frac=("routed_frac", "mean"),
        delta_vs_mvr=("delta_vs_mvr", "mean"),
        n=("delta_vs_mvr", "size"),
    )
    return out.sort_values("sel_frac").reset_index(drop=True)


def selected_counts(frame: pd.DataFrame, M: int = 100) -> pd.Series:
    """E[S] in classifiers rather than fractions, for the report's prose."""
    return np.round(frame["sel_frac"] * M, 2)
