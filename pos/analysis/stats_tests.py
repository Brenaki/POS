"""Friedman + Nemenyi + pairwise Wilcoxon over datasets.

The protocol the reference thesis uses (Demsar 2006): each dataset is one
paired observation, the folds are averaged first, and the omnibus Friedman
test gates the post-hoc comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from pos.analysis.loader import MODES, per_dataset

# Studentised range q_0.05 for k = 3 methods, infinite df (Demsar 2006, Table 5)
Q_ALPHA_K3 = 2.343


def critical_difference(n_datasets: int, k: int = 3, q: float = Q_ALPHA_K3) -> float:
    """Nemenyi critical difference between average ranks."""
    return q * np.sqrt(k * (k + 1) / (6.0 * n_datasets))


def compare(df: pd.DataFrame, metric: str, higher_is_better: bool = True) -> dict:
    """Rank the three modes on `metric` and run Friedman + pairwise Wilcoxon.

    Returns a dict with the omnibus statistic, the average ranks (1 = best),
    the critical difference, and one entry per pair.
    """
    table = per_dataset(df, metric).dropna()
    stat, p = friedmanchisquare(*[table[m].values for m in MODES])
    ranks = table.rank(axis=1, ascending=not higher_is_better).mean()
    cd = critical_difference(len(table))

    pairs = []
    for i in range(len(MODES)):
        for j in range(i + 1, len(MODES)):
            a, b = MODES[i], MODES[j]
            _, p_w = wilcoxon(table[a], table[b])
            delta = abs(ranks[a] - ranks[b])
            pairs.append({
                "a": a, "b": b, "rank_delta": float(delta),
                "nemenyi_significant": bool(delta > cd),
                "wilcoxon_p": float(p_w),
                "mean_delta": float(table[a].mean() - table[b].mean()),
            })

    return {
        "metric": metric, "n_datasets": int(len(table)),
        "friedman_chi2": float(stat), "friedman_p": float(p),
        "ranks": {m: float(ranks[m]) for m in MODES},
        "critical_difference": float(cd),
        "means": {m: float(table[m].mean()) for m in MODES},
        "pairs": pairs,
    }


def format_comparison(res: dict) -> str:
    """One human-readable block per metric."""
    lines = [
        f"{res['metric']}  (N={res['n_datasets']} bases)",
        f"  Friedman chi2={res['friedman_chi2']:.2f} p={res['friedman_p']:.5f} "
        f"| CD={res['critical_difference']:.3f}",
        "  ranks (1=melhor): " + "  ".join(
            f"{m}={res['ranks'][m]:.2f}" for m in MODES),
        "  medias:           " + "  ".join(
            f"{m}={res['means'][m]:.4f}" for m in MODES),
    ]
    for pr in res["pairs"]:
        mark = "*" if pr["nemenyi_significant"] else " "
        lines.append(
            f"    {pr['a']:8} vs {pr['b']:8} drank={pr['rank_delta']:.2f}{mark} "
            f"wilcoxon_p={pr['wilcoxon_p']:.5f} dmedia={pr['mean_delta']:+.4f}")
    return "\n".join(lines)
