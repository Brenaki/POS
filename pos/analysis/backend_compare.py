"""Compare a fast-measure run with an ECoL-measure run of the same profile (ADR 0024).

`monteiro-2023-dt-compatible-v1` and `-v2` share splits, seeds and GA; only
the complexity definitions differ. Three questions:

* does the vote pick the measures Monteiro's Table 4 shows as most voted?
* does test accuracy (MVR, `mv_acc`) move, dataset by dataset?
* does PGDCS's edge over Bagging change?
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import wilcoxon

OVERLAPPING = ("F1", "F1v", "F2", "F3", "F4")
NEIGHBORHOOD = ("N1", "N2", "N3", "N4", "T1", "LSC")


def chosen_measures(summary: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """The (overlapping, neighborhood) pair the run voted for each dataset."""
    pg = summary.dropna(subset=["pgdcs_types"])
    first = pg.groupby("dataset")["pgdcs_types"].first()
    return {d: tuple(t.split("|")) for d, t in first.items()}


def table4_winners(votes: dict[str, int]) -> tuple[set[str], set[str]]:
    """Most-voted measures per family in Table 4 (ties kept)."""
    out = []
    for fam in (OVERLAPPING, NEIGHBORHOOD):
        top = max(votes[m] for m in fam)
        out.append({m for m in fam if votes[m] == top})
    return out[0], out[1]


def table4_agreement(chosen: dict[str, tuple[str, ...]],
                     table: dict[str, dict[str, int]]) -> pd.DataFrame:
    rows = []
    for ds, (ov, nb) in sorted(chosen.items()):
        if ds not in table:
            continue
        w_ov, w_nb = table4_winners(table[ds])
        rows.append({"dataset": ds, "chosen": f"{ov}|{nb}",
                     "table4": f"{'/'.join(sorted(w_ov))}|{'/'.join(sorted(w_nb))}",
                     "overlapping_hit": ov in w_ov, "neighborhood_hit": nb in w_nb})
    return pd.DataFrame(rows)


def _per_dataset(summary: pd.DataFrame, treatment: str, col: str) -> pd.Series:
    return summary[summary["treatment"] == treatment].groupby("dataset")[col].mean()


def paired_by_dataset(a: pd.DataFrame, b: pd.DataFrame, treat_a: str, treat_b: str,
                      col: str = "mv_acc") -> dict[str, float]:
    """mean(a) - mean(b) per dataset, in pp, with a two-sided Wilcoxon."""
    sa, sb = _per_dataset(a, treat_a, col), _per_dataset(b, treat_b, col)
    common = sa.index.intersection(sb.index)
    d = (sa[common] - sb[common]) * 100
    p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
    return {"n": int(len(d)), "mean_delta_pp": float(d.mean()),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": p}
