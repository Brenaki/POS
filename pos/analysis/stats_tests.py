"""Friedman + Nemenyi + pairwise Wilcoxon over datasets.

The protocol the reference thesis uses (Demsar 2006): each dataset is one
paired observation, the folds are averaged first, and the omnibus Friedman
test gates the post-hoc comparison.

`compare` ranks the three pool modes on one metric; `compare_table` takes any
dataset x method table, which is what the fuser comparison of milestone 6
needs (majority vote vs soft fusion vs the DCS/DES methods).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from pos.analysis.loader import MODES, per_dataset

# Studentised range / sqrt(2) at alpha=0.05 (Demsar 2006, Table 5). Runs to
# k=20 because ADR 0018 compares up to fourteen fusers at once — and the way
# these values grow is itself the reason that comparison is only descriptive.
Q_ALPHA = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
           7: 2.948, 8: 3.031, 9: 3.102, 10: 3.164,
           11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
           16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}
Q_ALPHA_K3 = Q_ALPHA[3]


def critical_difference(n_datasets: int, k: int = 3, q: float | None = None) -> float:
    """Nemenyi critical difference between average ranks."""
    if q is None:
        q = Q_ALPHA[k]
    return q * np.sqrt(k * (k + 1) / (6.0 * n_datasets))


def compare_table(table: pd.DataFrame, higher_is_better: bool = True,
                  name: str = "") -> dict:
    """Friedman + Nemenyi + pairwise Wilcoxon over a dataset x method table.

    Rows are datasets (paired observations), columns are the methods being
    ranked. Rank 1 is the best method.
    """
    methods = list(table.columns)
    # Friedman is only defined for k >= 3; a two-method table (a fuser against
    # majority vote) still gets ranks, the CD and the Wilcoxon test.
    if len(methods) >= 3:
        stat, p = friedmanchisquare(*[table[c].values for c in methods])
    else:
        stat = p = float("nan")
    ranks = table.rank(axis=1, ascending=not higher_is_better).mean()
    cd = critical_difference(len(table), k=len(methods))

    pairs = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            # Two fusers can agree on every dataset (soft fusion ties majority
            # vote whenever the pool is unanimous). Wilcoxon rejects an
            # all-zero difference vector; the honest reading of it is p = 1.
            p_w = (1.0 if np.allclose(table[a] - table[b], 0.0)
                   else float(wilcoxon(table[a], table[b])[1]))
            delta = abs(ranks[a] - ranks[b])
            pairs.append({
                "a": a, "b": b, "rank_delta": float(delta),
                "nemenyi_significant": bool(delta > cd),
                "wilcoxon_p": p_w,
                "mean_delta": float(table[a].mean() - table[b].mean()),
            })

    return {
        "metric": name, "methods": methods, "n_datasets": int(len(table)),
        "friedman_chi2": float(stat), "friedman_p": float(p),
        "ranks": {m: float(ranks[m]) for m in methods},
        "critical_difference": float(cd),
        "means": {m: float(table[m].mean()) for m in methods},
        "pairs": pairs,
    }


def compare(df: pd.DataFrame, metric: str, higher_is_better: bool = True) -> dict:
    """Rank the three pool modes on `metric`."""
    table = per_dataset(df, metric).dropna()[MODES]
    return compare_table(table, higher_is_better=higher_is_better, name=metric)


def format_comparison(res: dict) -> str:
    """One human-readable block per comparison."""
    methods = res["methods"]
    width = max(len(m) for m in methods)
    omnibus = ("Friedman n/a (k=2)" if np.isnan(res["friedman_chi2"]) else
               f"Friedman chi2={res['friedman_chi2']:.2f} p={res['friedman_p']:.5f}")
    lines = [
        f"{res['metric']}  (N={res['n_datasets']} bases, k={len(methods)})",
        f"  {omnibus} | CD={res['critical_difference']:.3f}",
        "  ranks (1=melhor): " + "  ".join(
            f"{m}={res['ranks'][m]:.2f}" for m in methods),
        "  medias:           " + "  ".join(
            f"{m}={res['means'][m]:.4f}" for m in methods),
    ]
    for pr in res["pairs"]:
        mark = "*" if pr["nemenyi_significant"] else " "
        lines.append(
            f"    {pr['a']:{width}} vs {pr['b']:{width}} drank={pr['rank_delta']:.2f}{mark} "
            f"wilcoxon_p={pr['wilcoxon_p']:.5f} dmedia={pr['mean_delta']:+.4f}")
    return "\n".join(lines)


def holm_wilcoxon(table: pd.DataFrame, baseline: str, others: list[str]) -> pd.DataFrame:
    """Paired Wilcoxon of each column against `baseline`, Holm-corrected.

    The descriptive tier of ADR 0018: with a dozen fusers Friedman/Nemenyi has
    no power left, but a family of paired tests against the single baseline
    that matters does — provided the family-wise error is controlled. Holm is
    used rather than Bonferroni because it is uniformly more powerful and
    needs no extra assumption.
    """
    rows = []
    for col in others:
        diff = table[col] - table[baseline]
        p = 1.0 if np.allclose(diff, 0.0) else float(wilcoxon(table[col],
                                                              table[baseline])[1])
        rows.append({"fuser": col, "mean": float(table[col].mean()),
                     "delta": float(diff.mean()),
                     "wins": int((diff > 1e-12).sum()),
                     "losses": int((diff < -1e-12).sum()),
                     "p": p})
    res = pd.DataFrame(rows).set_index("fuser").sort_values("p")
    m = len(res)
    # Holm: the k-th smallest p is scaled by (m - k), then made monotone.
    adj = [min(1.0, (m - i) * p) for i, p in enumerate(res["p"].values)]
    res["p_holm"] = np.maximum.accumulate(adj)
    return res
