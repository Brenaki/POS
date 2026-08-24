"""`N*` split by class cardinality, and the mechanism behind it (ADR 0019).

Objective 7 of the subproject asks whether some intermediate Oracle level is a
realistic ceiling. The answer measured so far — `N*` lands at ~M/2 in every
mode — was explained by asserting the identity `Oracle_{M/2+1} == MVR` for
binary problems. Measured on 660 binary folds, that is not an identity but a
sandwich, and the sandwich is what actually pins `N*`:

    Oracle_{M/2+1} <= MVR <= Oracle_{M/2}

Majority vote is correct whenever more than half the pool is correct, so
`MVR >= Oracle_51`; it can only additionally win 50-50 ties, so it never
exceeds `Oracle_50`. Since `N*` is the first level *strictly below* MVR, it
cannot be at or below M/2, and lands at M/2+1 or just past it (the ties make
`Oracle_51 == MVR` exactly, which is not "<", pushing `N*` to 52+).

Under plurality voting the bound does not hold, so multiclass problems are
reported separately rather than folded into a universal claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sandwich_check(df: pd.DataFrame) -> dict:
    """Verify Oracle_{M/2+1} <= MVR <= Oracle_{M/2} on the binary folds."""
    binary = df[df["is_binary"]]
    if binary.empty:
        return {"n_folds": 0}
    lower, upper, exact = [], [], []
    for curve, mv, M in zip(binary["curve"], binary["majority_vote"],
                            binary["M"], strict=True):
        half = int(M) // 2
        c_hi, c_lo = curve[half], curve[half - 1]  # Oracle_{M/2+1}, Oracle_{M/2}
        lower.append(mv >= c_hi - 1e-12)
        upper.append(mv <= c_lo + 1e-12)
        exact.append(abs(mv - c_hi) < 1e-12)
    n = len(lower)
    return {
        "n_folds": n,
        "holds_lower": int(np.sum(lower)),
        "holds_upper": int(np.sum(upper)),
        "exact_equality": int(np.sum(exact)),
        "holds_both_frac": float(np.mean(np.logical_and(lower, upper))),
    }


def nstar_by_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset mean `N*`, tagged binary/multiclass (Demsar aggregation)."""
    cols = ["dataset", "mode", "is_binary", "n_classes"]
    return df.groupby(cols, as_index=False)["nstar"].mean()


def nstar_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Median `N*` per (cardinality, mode), the table the report needs."""
    per_ds = nstar_by_cardinality(df)
    out = per_ds.groupby(["is_binary", "mode"], as_index=False).agg(
        n_datasets=("dataset", "nunique"),
        nstar_median=("nstar", "median"),
        nstar_mean=("nstar", "mean"),
    )
    out["group"] = np.where(out["is_binary"], "binaria", "multiclasse")
    return out.drop(columns="is_binary")
