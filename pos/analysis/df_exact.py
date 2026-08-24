"""Exact independence baseline for the double-fault redundancy index (ADR 0019).

`loader.df_ratio` normalises the mean double fault by `e^2`, with `e` the mean
individual error. That is the value expected under independence only when every
classifier errs at the same rate. With unequal rates the correct expectation is

    E[DF] = 2 / (M(M-1)) * sum_{i<j} e_i e_j = ((sum e)^2 - sum e^2) / (M(M-1))

which is what a reviewer of the `rho = -0.86` result asked for. No rerun is
needed: every per-classifier accuracy is already in the fold manifests, which
are versioned (the heavy .npy/.npz are not).

Jensen puts the two apart in one direction only: the mean-based denominator
`e^2` is never larger than the exact one, so `df_ratio` is biased *up* whenever
the pool's error rates disagree, and the more they disagree the more it is.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def exact_independence_df(individual_accs) -> float:
    """E[DF] under independence, given each classifier's own accuracy."""
    e = 1.0 - np.asarray(individual_accs, dtype=float)
    M = e.size
    if M < 2:
        return float("nan")
    return float((e.sum() ** 2 - (e**2).sum()) / (M * (M - 1)))


def load_manifest_table(run_dir: Path | str) -> pd.DataFrame:
    """One row per fold manifest: (dataset, fold, mode, df_indep_exact, ...).

    Uses os.walk rather than glob/find — the `find` on this machine is shimmed
    and returns wrong counts.
    """
    rows = []
    for dirpath, _, filenames in os.walk(Path(run_dir)):
        for fname in filenames:
            if not (fname.startswith("fold_manifest_") and fname.endswith(".json")):
                continue
            fm = json.loads(Path(dirpath, fname).read_text())
            accs = fm.get("individual_accuracies") or []
            rows.append({
                "dataset": fm["dataset"],
                "fold": fm["fold_idx"],
                "mode": fm["mode"],
                "df_indep_exact": exact_independence_df(accs),
                "df_indep_mean": (1.0 - float(np.mean(accs))) ** 2 if accs else np.nan,
                "acc_spread": float(np.std(accs)) if accs else np.nan,
                "pgdcs_types": "|".join(fm.get("pgdcs_types") or []),
            })
    return pd.DataFrame(rows)


def attach_df_ratio_exact(df: pd.DataFrame, run_dir: Path | str) -> pd.DataFrame:
    """Add `df_ratio_exact` (and the manifest columns it needs) to a run frame."""
    manifests = load_manifest_table(run_dir)
    if manifests.empty:
        df["df_ratio_exact"] = np.nan
        return df
    merged = df.merge(manifests, on=["dataset", "fold", "mode"], how="left",
                      suffixes=("", "_fm"))
    denom = merged["df_indep_exact"]
    merged["df_ratio_exact"] = merged["double_fault_mean"] / np.where(
        denom > 0, denom, np.nan
    )
    return merged
