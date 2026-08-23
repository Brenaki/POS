"""Figures for the dynamic-selection comparison (milestone 6 and 7)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from pos.analysis.figures_curves import COLORS, LABELS  # noqa: E402
from pos.analysis.fusers import (  # noqa: E402
    FUSER_LABELS,
    FUSERS,
    available_fusers,
    recovery_vs_redundancy,
)
from pos.analysis.loader import MODES  # noqa: E402


def plot_fuser_accuracy(df, out: Path) -> Path:
    """One panel per pool mode: every fuser against the Oracle_1 ceiling.

    The distance from the tallest bar to the dashed Oracle_1 line is the part
    of the pool's potential that no combination method reached.

    ADR 0018 took the comparison from seven fusers to fourteen, which is too
    many to sit side by side. The panels are stacked instead, on a shared x
    axis, so the same fuser sits in the same column in all three panels. A
    missing bar is a method that does not apply to that pool — META-DES and
    KNOP need `predict_proba`, which a Perceptron has not got.

    Each panel keeps its own y scale: the reading this figure exists for is
    within a panel (how far the best fuser stays below its own Oracle_1), and
    a shared axis driven by the weakest pool flattens the differences between
    fusers in the other two.
    """
    modes = [m for m in MODES if (df["mode"] == m).any()]
    cols = [c for c in FUSERS
            if any(c in available_fusers(df, m) for m in modes)]
    fig, axes = plt.subplots(len(modes), 1, sharex=True,
                             figsize=(max(8.0, 0.62 * len(cols) + 1.8),
                                      2.9 * len(modes) + 1.0))
    axes = np.atleast_1d(axes)
    for ax, mode in zip(axes, modes, strict=True):
        sub = df[df["mode"] == mode]
        have = available_fusers(df, mode)
        vals = [sub[c].mean() if c in have else np.nan for c in cols]
        oracle = sub["oracle_1"].mean()
        floor = min(*[v for v in vals if not np.isnan(v)],
                    sub["mean_individual_acc"].mean()) - 0.02
        ax.set_ylim(max(0.0, floor), oracle + (oracle - floor) * 0.12)
        ax.bar(range(len(cols)), vals, color=COLORS[mode], alpha=0.8)
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            # Vertical: two fusers that tie to the third decimal are exactly
            # the pair worth reading, and horizontal labels overprint there.
            ax.annotate(f"{v:.3f}", xy=(i, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        rotation=90, fontsize=7.5)
        ax.axhline(oracle, color="black", ls="--", lw=1.2)
        ax.axhline(sub["mean_individual_acc"].mean(), color="gray", ls=":", lw=1.2)
        ax.set_title(f"{LABELS[mode]} — Oracle_1 = {oracle:.3f}",
                     fontsize=10, loc="left")
        ax.grid(alpha=0.25, axis="y")
        ax.set_ylabel("acurácia média")
    axes[-1].set_xticks(range(len(cols)))
    axes[-1].set_xticklabels([FUSER_LABELS.get(c, c) for c in cols],
                             rotation=45, ha="right", fontsize=9)
    fig.suptitle("Métodos reais de combinação vs o teto Oracle_1 "
                 "(preto tracejado = Oracle_1, cinza pontilhado = acurácia "
                 "individual média)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_recovered_gap(df, out: Path) -> Path:
    """Share of the Oracle_1 - MVR gap that dynamic selection recovers.

    Left: distribution per pool mode. Right: the same value against the
    redundancy index, testing whether `DF/e^2` predicts not only how large
    the gap is but how much of it is reachable (objective 8).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    modes = [m for m in MODES if df.loc[df["mode"] == m, "recovered"].notna().any()]
    data = [df.loc[df["mode"] == m, "recovered"].dropna().values for m in modes]
    bp = ax1.boxplot(data, labels=[LABELS[m] for m in modes], patch_artist=True,
                     widths=0.55, medianprops={"color": "black", "lw": 1.6})
    for patch, m in zip(bp["boxes"], modes, strict=True):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.45)
    for i, m in enumerate(modes, start=1):
        med = float(np.median(df.loc[df["mode"] == m, "recovered"].dropna()))
        ax1.annotate(f"mediana {med:.2f}", xy=(i + 0.30, med), xytext=(3, -3),
                     textcoords="offset points", ha="left", fontsize=8.5)
    ax1.axhline(0.0, color="black", ls="--", lw=1.1)
    ax1.set_ylabel("(melhor DCS/DES − MVR) / (Oracle_1 − MVR)")
    ax1.set_title("Parcela da folga recuperada pela seleção dinâmica")
    ax1.grid(alpha=0.25, axis="y")

    x, y = recovery_vs_redundancy(df)
    for m in modes:
        sub = df[df["mode"] == m].dropna(subset=["recovered", "df_ratio"])
        grp = sub.groupby("dataset")[["df_ratio", "recovered"]].mean()
        ax2.scatter(grp["df_ratio"], grp["recovered"], color=COLORS[m], s=38,
                    alpha=0.75, edgecolor="white", linewidth=0.6, label=LABELS[m])
    rho, p_val = spearmanr(x, y)
    ax2.annotate(f"Spearman rho = {rho:+.3f}  (p = {p_val:.1e}, n = {len(x)})",
                 xy=(0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=9.5)
    ax2.axvline(1.0, color="black", ls="--", lw=1.1)
    ax2.set_xlabel("DF / e²  — redundância de erros")
    ax2.set_ylabel("folga recuperada")
    ax2.set_title("Redundância vs folga recuperada (uma marca por base)")
    ax2.legend(fontsize=9, frameon=False)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
