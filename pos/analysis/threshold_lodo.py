"""Leave-one-dataset-out validation of the DF/e^2 decision rule (ADR 0019).

The report recommends deciding whether dynamic selection is worth trying from
the normalised double-fault index alone: `>= 4` don't bother, `2-4` marginal,
`~1` strong candidate. Those cuts were both discovered and evaluated on the
same 29 datasets, which makes the reported agreement circular.

This fits the two cuts on 28 datasets and predicts the 29th, 29 times, and
scores that against the obvious baseline (always answer the most common band).
A rule that cannot beat the baseline out of sample is a description of these
datasets, not a rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The gap bands the report attaches to each DF/e^2 range.
GAP_CUTS = (0.05, 0.30)
BAND_NAMES = ("folga baixa", "folga media", "folga alta")


def gap_band(gap: float, cuts: tuple[float, float] = GAP_CUTS) -> int:
    """0 = gap below cuts[0], 1 = between, 2 = above cuts[1]."""
    return int(gap >= cuts[0]) + int(gap >= cuts[1])


def per_dataset_frame(df: pd.DataFrame, ratio_col: str = "df_ratio") -> pd.DataFrame:
    """Average folds per dataset x mode first (Demsar), then band the gap."""
    sub = df.dropna(subset=[ratio_col, "gap_1"])
    g = sub.groupby(["dataset", "mode"], as_index=False)[[ratio_col, "gap_1"]].mean()
    g["band"] = [gap_band(v) for v in g["gap_1"]]
    return g


def _fit_cuts(ratios: np.ndarray, bands: np.ndarray) -> tuple[float, float]:
    """Two thresholds on DF/e^2 maximising training agreement. High ratio =>
    low gap, so the rule is monotone decreasing: band 2 below lo, 0 above hi."""
    candidates = np.unique(np.round(ratios, 3))
    best, best_cuts = -1.0, (1.0, 4.0)
    for lo in candidates:
        for hi in candidates[candidates > lo]:
            pred = np.where(ratios >= hi, 0, np.where(ratios >= lo, 1, 2))
            score = float(np.mean(pred == bands))
            if score > best:
                best, best_cuts = score, (float(lo), float(hi))
    return best_cuts


def _predict(ratio: float, cuts: tuple[float, float]) -> int:
    lo, hi = cuts
    return 0 if ratio >= hi else (1 if ratio >= lo else 2)


def lodo_scores(df: pd.DataFrame, ratio_col: str = "df_ratio") -> dict:
    """Leave-one-dataset-out accuracy of the fitted rule vs the majority band."""
    g = per_dataset_frame(df, ratio_col)
    datasets = sorted(g["dataset"].unique())
    hits, base_hits, total, fitted = 0, 0, 0, []
    for held in datasets:
        train, test = g[g["dataset"] != held], g[g["dataset"] == held]
        cuts = _fit_cuts(train[ratio_col].to_numpy(), train["band"].to_numpy())
        fitted.append(cuts)
        majority = int(train["band"].mode().iat[0])
        for ratio, band in zip(test[ratio_col], test["band"], strict=True):
            hits += int(_predict(ratio, cuts) == band)
            base_hits += int(majority == band)
            total += 1
    lo = float(np.median([c[0] for c in fitted]))
    hi = float(np.median([c[1] for c in fitted]))
    return {
        "n_predictions": total,
        "n_datasets": len(datasets),
        "lodo_accuracy": hits / total if total else float("nan"),
        "baseline_accuracy": base_hits / total if total else float("nan"),
        "median_cuts": (lo, hi),
        "cut_spread": (float(np.std([c[0] for c in fitted])),
                       float(np.std([c[1] for c in fitted]))),
    }
