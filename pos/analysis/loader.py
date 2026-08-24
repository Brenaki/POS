"""Load an experiment run's summary.csv and derive analysis columns.

Keeps every derivation in one place so the figures and the statistical
tests are guaranteed to be looking at the same numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pos.oracle.des_methods import (
    DCS_METHODS,
    DES_METHODS,
    DES_ONLY_METHODS,
)

# ADR 0019 added the PGDCS-with-vote mode and the generation control.
MODES = ["pgdcs", "ga", "randbag", "bagging", "rf"]
FOCUS_LEVELS = [1, 2, 3, 4, 5]
DES_COLS = [f"des_{m}" for m in DES_METHODS]
# `recovered` is about *dynamic* selection, so the static baselines
# (single best, static selection) must not feed the per-fold maximum.
DYNAMIC_COLS = [f"des_{m}" for m in DCS_METHODS + DES_ONLY_METHODS]


def load_run(run_dir: Path | str) -> pd.DataFrame:
    """Read summary.csv and attach the derived columns used by the analysis.

    Derived columns
    ---------------
    curve       : list[float], the full Oracle_1..M curve
    gap_N       : Oracle_N - majority_vote, for N in FOCUS_LEVELS
    nstar       : smallest N with Oracle_N < majority_vote (M if it never
                  crosses). Objective 7 of the subproject: which Oracle level
                  is as conservative as a real combination method.
    df_ratio    : double_fault_mean / e^2 with e = 1 - mean_individual_acc.
                  Double fault grows with the base learner's error rate, so
                  the raw value cannot compare pools of different strength;
                  e^2 is the value expected if the errors were independent,
                  making the ratio a scale-free redundancy index (1 = independent).
    des_best    : best *dynamic* (DCS/DES) accuracy available on that fold —
                  the static baselines are excluded on purpose (NaN if the run
                  carried no DES columns).
    oracle_1_balanced_gap : oracle_1_balanced - mv_balanced_acc, the same gap
                  read on balanced accuracy (milestone 7, ADR 0018).
    recovered   : (des_best - majority_vote) / (oracle_1 - majority_vote), the
                  share of the unexploited gap that dynamic selection actually
                  reaches. 0 = no better than majority vote, 1 = reaches the
                  Oracle. This is the quantity objective 8 asks to predict.
    """
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "summary.csv")
    df["curve"] = df["oracle_curve_json"].map(json.loads)

    for n in FOCUS_LEVELS:
        df[f"gap_{n}"] = df[f"oracle_{n}"] - df["majority_vote"]

    df["nstar"] = [
        int(np.argmax(np.asarray(c) < m) + 1) if (np.asarray(c) < m).any() else len(c)
        for c, m in zip(df["curve"], df["majority_vote"], strict=True)
    ]

    err = 1.0 - df["mean_individual_acc"]
    df["df_ratio"] = df["double_fault_mean"] / np.where(err > 0, err**2, np.nan)
    _attach_des(df)
    if {"oracle_1_balanced", "mv_balanced_acc"} <= set(df.columns):
        df["gap_1_balanced"] = df["oracle_1_balanced"] - df["mv_balanced_acc"]
    return df


def _attach_des(df: pd.DataFrame) -> None:
    """Add des_best / des_best_method / gap_des / recovered, in place."""
    present = [c for c in DYNAMIC_COLS if c in df.columns]
    if not present:
        return
    df["des_best"] = df[present].max(axis=1)
    df["des_best_method"] = df[present].apply(
        lambda r: r.idxmax()[4:] if r.notna().any() else None, axis=1)
    df["gap_des"] = df["oracle_1"] - df["des_best"]
    gap = df["oracle_1"] - df["majority_vote"]
    df["recovered"] = np.where(
        gap > 1e-9, (df["des_best"] - df["majority_vote"]) / gap, np.nan)


def mean_curve(df: pd.DataFrame, mode: str) -> np.ndarray:
    """Mean Oracle_1..M curve for one mode, averaged over every fold."""
    curves = [c for c, m in zip(df["curve"], df["mode"], strict=True) if m == mode]
    return np.mean(np.asarray(curves, dtype=float), axis=0)


def per_dataset(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot to (dataset x mode), averaging the folds. Rows = paired samples.

    Reindexed on MODES so a mode with no value for this metric comes back as
    an all-NaN column instead of vanishing (mean_probs has no GA value when
    the base learner is a Perceptron).
    """
    return df.pivot_table(index="dataset", columns="mode", values=metric,
                          aggfunc="mean").reindex(columns=MODES)
