"""Analysis of schema-v2 runs, used for the image battery (ADR 0022).

`pos.analysis.loader.load_run` reads v1 runs (`mode`, `oracle_2..5`). A v2
summary has `treatment` and only the 21 Oracle checkpoints as columns, but it
still stores the full Oracle_1..M curve as JSON, so every derivation the
report relies on is recomputed here with the same definitions:

* `nstar`     — smallest N with Oracle_N < MVR (M if it never crosses);
* `df_ratio`  — double fault over e^2, e = 1 - mean individual accuracy;
* `des_best` / `recovered` — via `loader._attach_des`, dynamic methods only.

`dfe_transfer` checks whether the DF/e^2 -> gap relation fitted on one
battery predicts another without refitting.

Class cardinality comes from the run manifest's conformance survey, so the
binary/multiclass split needs nothing outside the run directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.isotonic import IsotonicRegression

from pos.analysis.loader import _attach_des

TABLE_COLS = ["n_classes", "oracle_1", "majority_vote", "des_best", "recovered",
              "nstar", "df_ratio", "mean_individual_acc"]


def _nstar(curve: list[float], mv: float) -> int:
    below = np.asarray(curve) < mv
    return int(np.argmax(below) + 1) if below.any() else len(curve)


def load_v2_run(run_dir: Path | str) -> pd.DataFrame:
    """summary.csv of a v2 run with the analysis columns attached."""
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "summary.csv")
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    classes = {k: v.get("snapshot_classes") for k, v in manifest["conformance"].items()}
    df["mode"] = df["treatment"]
    df["curve"] = df["oracle_curve_json"].map(json.loads)
    df["n_classes"] = df["dataset"].map(classes)
    df["is_binary"] = df["n_classes"] == 2
    df["nstar"] = [_nstar(c, m) for c, m in zip(df["curve"], df["majority_vote"], strict=True)]
    err = 1.0 - df["mean_individual_acc"]
    df["df_ratio"] = df["double_fault_mean"] / np.where(err > 0, err**2, np.nan)
    _attach_des(df)
    return df


def battery_table(df: pd.DataFrame) -> pd.DataFrame:
    """dataset x mode means of the quantities the report discusses."""
    cols = [c for c in TABLE_COLS if c in df.columns]
    return df.groupby(["dataset", "mode"])[cols].mean().reset_index()


def contrast(a: pd.DataFrame, b: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Median of each metric in two groups of datasets, Mann-Whitney U p-value."""
    rows = {}
    for m in metrics:
        x, y = a[m].dropna(), b[m].dropna()
        p = mannwhitneyu(x, y).pvalue if len(x) and len(y) else np.nan
        rows[m] = {"median_a": x.median(), "median_b": y.median(),
                   "n_a": len(x), "n_b": len(y), "p_mannwhitney": p}
    return pd.DataFrame(rows).T


def dfe_transfer(ref: pd.DataFrame, new: pd.DataFrame) -> dict[str, float]:
    """Fit gap ~ DF/e^2 (decreasing isotonic on log DF/e^2) on `ref`, predict `new`.

    Both frames hold one row per dataset x mode with `df_ratio` and `gap`. The
    trivial baseline predicts the median gap of `ref` for every new point.
    """
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(np.log(ref["df_ratio"]), ref["gap"])
    pred = iso.predict(np.log(new["df_ratio"]))
    return {"spearman_ref": float(spearmanr(ref["df_ratio"], ref["gap"])[0]),
            "spearman_new": float(spearmanr(new["df_ratio"], new["gap"])[0]),
            "mae_isotonic": float(np.mean(np.abs(pred - new["gap"]))),
            "mae_trivial": float(np.mean(np.abs(ref["gap"].median() - new["gap"]))),
            "n_ref": len(ref), "n_new": len(new)}
