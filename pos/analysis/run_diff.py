"""Difference-in-differences between the two-way and the three-way split.

ADR 0018 changed two things in one run: the GA's fitness set was halved, and
the DSEL of the dynamic selection stopped being that same fitness set. This
module separates the two effects.

`X_tr` was deliberately left untouched, so the Bagging and RF pools are
identical across the two runs and their `recovered` moves only because the
DSEL got smaller. That makes them the control group:

    effect of a smaller DSEL  = d recovered (bagging, rf)
    weakening of the GA pool  = d oracle_1 / mean_individual_acc / MVR (ga)
    DSEL bias                 = d recovered (ga) - the control delta

Every comparison is restricted to the DCS/DES methods present in *both* runs:
the new one ranks ten dynamic methods where the old one ranked five, and a
maximum over more columns is larger by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pos.analysis.loader import DYNAMIC_COLS, MODES

KEY = ["dataset", "fold", "mode"]
# These depend only on the pool and the test fold, so for Bagging and RF they
# must be bit-identical. If they are not, the decomposition below is void.
POOL_COLS = ["oracle_1", "oracle_2", "oracle_5", "oracle_M", "majority_vote",
             "mean_individual_acc", "double_fault_mean", "M"]
QUALITY_COLS = ["oracle_1", "majority_vote", "mean_individual_acc", "M"]


def common_methods(old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    """Dynamic-selection columns both runs carry."""
    return [c for c in DYNAMIC_COLS if c in old.columns and c in new.columns]


def recovered_over(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Recompute `recovered` using only `cols` as the candidate methods."""
    best = df[cols].max(axis=1)
    gap = df["oracle_1"] - df["majority_vote"]
    return pd.Series(np.where(gap > 1e-9, (best - df["majority_vote"]) / gap, np.nan),
                     index=df.index)


def pair_runs(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the two runs on (dataset, fold, mode), `recovered` realigned."""
    cols = common_methods(old, new)
    old, new = old.copy(), new.copy()
    old["rec"], new["rec"] = recovered_over(old, cols), recovered_over(new, cols)
    keep = KEY + ["rec", *POOL_COLS]
    return old[keep].merge(new[keep], on=KEY, suffixes=("_old", "_new"))


def pool_identity(merged: pd.DataFrame) -> pd.DataFrame:
    """Largest absolute discrepancy per mode on the pool-level columns."""
    rows = []
    for mode in MODES:
        sub = merged[merged["mode"] == mode]
        rows.append({"mode": mode, "folds": len(sub),
                     **{c: (float((sub[f"{c}_new"] - sub[f"{c}_old"]).abs().max())
                            if len(sub) else np.nan) for c in POOL_COLS}})
    return pd.DataFrame(rows).set_index("mode")


def dataset_wilcoxon(sub: pd.DataFrame) -> float:
    """Paired Wilcoxon on `recovered`, one observation per dataset.

    Folds are averaged first: the protocol the rest of the analysis uses
    (Demsar 2006), since the folds of a dataset are not independent.
    """
    per_ds = sub.groupby("dataset")[["rec_old", "rec_new"]].mean()
    diff = per_ds["rec_new"] - per_ds["rec_old"]
    if len(per_ds) < 3 or np.allclose(diff, 0.0):
        return float("nan")
    return float(wilcoxon(per_ds["rec_new"], per_ds["rec_old"])[1])


def did_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Per mode: the `recovered` delta and the pool-quality deltas behind it.

    Every mean is taken over the *paired* rows only. A fold where either run
    leaves `recovered` undefined (Oracle_1 already equals majority vote) drops
    from all three columns at once, so `d_rec` stays equal to the difference
    of the two means printed beside it.
    """
    rows = []
    for mode in MODES:
        sub = merged[merged["mode"] == mode].dropna(subset=["rec_old", "rec_new"])
        if sub.empty:
            continue
        rows.append({"mode": mode, "pares": len(sub),
                     "rec_old": sub["rec_old"].mean(),
                     "rec_new": sub["rec_new"].mean(),
                     "d_rec": (sub["rec_new"] - sub["rec_old"]).mean(),
                     "p": dataset_wilcoxon(sub),
                     **{f"d_{c}": (sub[f"{c}_new"] - sub[f"{c}_old"]).mean()
                        for c in QUALITY_COLS}})
    return pd.DataFrame(rows).set_index("mode")


def decompose(did: pd.DataFrame) -> dict[str, float]:
    """Split the GA's `recovered` delta into the DSEL size effect and the bias."""
    if not {"ga", "bagging", "rf"} <= set(did.index):
        return {}
    control = float(did.loc[["bagging", "rf"], "d_rec"].mean())
    ga = float(did.loc["ga", "d_rec"])
    return {"dsel_size_effect": control, "ga_delta": ga, "dsel_bias": ga - control,
            **{f"ga_{c}": float(did.loc["ga", f"d_{c}"]) for c in QUALITY_COLS}}


def format_decomposition(dec: dict[str, float]) -> str:
    """The three-line answer ADR 0018 asks for."""
    if not dec:
        return "  decomposicao impossivel — falta algum modo"
    return "\n".join([
        f"  efeito do DSEL menor (controle bagging+rf) : {dec['dsel_size_effect']:+.4f}",
        f"  delta observado no GA                      : {dec['ga_delta']:+.4f}",
        f"  vies de DSEL (GA - controle)               : {dec['dsel_bias']:+.4f}",
        f"  enfraquecimento do pool GA: d oracle_1={dec['ga_oracle_1']:+.4f} "
        f"d MVR={dec['ga_majority_vote']:+.4f} "
        f"d acc_individual={dec['ga_mean_individual_acc']:+.4f} d M={dec['ga_M']:+.2f}",
    ])
