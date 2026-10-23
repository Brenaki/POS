"""Agreement between the fast complexity backend and ECoL (ADR 0023).

Input is a long frame with one row per (dataset, bag, measure) and the two
values side by side (`fast`, `ecol`). The GA only *ranks* the bags of one
dataset, so besides the pooled error and correlations this reports, per
dataset, Kendall's tau between the two orderings, whether both pick the same
least-complex bag, and the overlap of their k least-complex bags.

The GA does not rank bags by the raw value but by its dispersion
(`dispersion_linear`: mean |x_j - x_l| over the bags, min-max normalised), so
`to_dispersion` maps both columns to that quantity before the same metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from pos.dispersion import dispersion_linear


def _corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(fn(a, b)[0])


def ranking_agreement(df: pd.DataFrame, k: int = 5) -> dict[str, float]:
    """Ordering agreement for the bags of one (dataset, measure)."""
    fast, ecol = df["fast"].to_numpy(float), df["ecol"].to_numpy(float)
    best_fast = set(np.flatnonzero(np.isclose(fast, fast.min())))
    best_ecol = set(np.flatnonzero(np.isclose(ecol, ecol.min())))
    k = min(k, len(fast))
    low_fast = set(np.argsort(fast, kind="stable")[:k])
    low_ecol = set(np.argsort(ecol, kind="stable")[:k])
    return {"kendall": _corr(kendalltau, fast, ecol),
            "argmin_agree": float(bool(best_fast & best_ecol)),
            "topk_overlap": len(low_fast & low_ecol) / k}


def agreement_by_measure(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """One row per measure: pooled error/correlation + per-dataset ranking."""
    df = df.dropna(subset=["fast", "ecol"])
    rows = []
    for measure, g in df.groupby("measure", sort=False):
        a, b = g["fast"].to_numpy(float), g["ecol"].to_numpy(float)
        per = pd.DataFrame([ranking_agreement(d, k) for _, d in g.groupby("dataset")])
        rows.append({"measure": measure, "n_bags": len(g),
                     "n_datasets": g["dataset"].nunique(),
                     "mae": float(np.mean(np.abs(a - b))),
                     "max_abs": float(np.max(np.abs(a - b))),
                     "pearson": _corr(pearsonr, a, b),
                     "spearman": _corr(spearmanr, a, b),
                     "kendall_mean": float(per["kendall"].mean()),
                     "kendall_min": float(per["kendall"].min()),
                     "argmin_rate": float(per["argmin_agree"].mean()),
                     "topk_mean": float(per["topk_overlap"].mean())})
    return pd.DataFrame(rows)


def to_dispersion(df: pd.DataFrame) -> pd.DataFrame:
    """Replace `fast`/`ecol` by their per-(dataset, measure) GA dispersion."""
    df = df.dropna(subset=["fast", "ecol"]).reset_index(drop=True)
    for _, idx in df.groupby(["dataset", "measure"]).groups.items():
        for col in ("fast", "ecol"):
            vals = [[v] for v in df.loc[idx, col]]
            df.loc[idx, col] = np.asarray(dispersion_linear(vals))[:, 0]
    return df
