"""Comparison figures: gap, crossover level N*, diversity (objectives 6-8)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from pos.analysis.figures_curves import COLORS, LABELS  # noqa: E402
from pos.analysis.loader import MODES, per_dataset  # noqa: E402


def plot_nstar(df, out: Path) -> Path:
    """Distribution of N* — the Oracle level as conservative as majority vote.

    Objective 7 asks whether some intermediate Oracle level is a realistic
    upper bound. N* answers it directly: it is the level at which the bound
    stops being optimistic.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    data = [df.loc[df["mode"] == m, "nstar"].values for m in MODES]
    bp = ax.boxplot(data, labels=[LABELS[m] for m in MODES], patch_artist=True,
                    widths=0.55, medianprops={"color": "black", "lw": 1.6})
    for patch, m in zip(bp["boxes"], MODES, strict=True):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.45)
    ax.axhline(50, color="black", ls="--", lw=1.2)
    ax.annotate("M/2 = 50", xy=(3.45, 50), xytext=(-2, 5), textcoords="offset points",
                ha="right", fontsize=9)
    for i, m in enumerate(MODES, start=1):
        med = float(np.median(df.loc[df["mode"] == m, "nstar"]))
        ax.annotate(f"mediana {med:.0f}", xy=(i, med), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_ylabel("N* = menor N com Oracle_N < votação majoritária")
    ax.set_title("Onde a curva Oracle_N deixa de ser otimista (M=100)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_gap_per_dataset(df, out: Path, level: int = 1) -> Path:
    """Oracle_N - majority vote per dataset, sorted. Large gap = DCS/DES worth trying."""
    table = per_dataset(df, f"gap_{level}").sort_values("ga")
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(table) + 1.8))
    for off, mode in zip([-0.26, 0.0, 0.26], MODES, strict=True):
        ax.barh(y + off, table[mode].values, height=0.25, color=COLORS[mode],
                label=LABELS[mode])
    ax.set_yticks(y)
    ax.set_yticklabels(table.index, fontsize=8)
    ax.set_xlabel(f"Oracle_{level} − votação majoritária")
    ax.set_title(f"Folga não explorada pelo MVR (Oracle_{level} − MV), por base")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_diversity_vs_gap(df, out: Path) -> Path:
    """Redundancy index vs unexploited gap — the diversity/redundancy axis.

    x uses df_ratio (double fault normalised by the value expected under
    independent errors), so pools built from base learners of different
    strength stay comparable.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    gap = per_dataset(df, "gap_1")
    ratio = per_dataset(df, "df_ratio")
    for mode in MODES:
        ax.scatter(ratio[mode], gap[mode], color=COLORS[mode], s=38, alpha=0.75,
                   edgecolor="white", linewidth=0.6, label=LABELS[mode])
    ax.axvline(1.0, color="black", ls="--", lw=1.1)
    ax.annotate("erros independentes", xy=(1.0, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", fontsize=9)
    pooled_x = np.concatenate([ratio[m].values for m in MODES])
    pooled_y = np.concatenate([gap[m].values for m in MODES])
    rho, p_val = spearmanr(pooled_x, pooled_y)
    ax.annotate(f"Spearman rho = {rho:+.3f}  (p = {p_val:.1e}, n = {len(pooled_x)})",
                xy=(0.97, 0.72), xycoords="axes fraction", ha="right", fontsize=10)
    ax.set_xlabel("DF / e²  — redundância de erros (1 = independentes, maior = mais correlacionados)")
    ax.set_ylabel("Oracle_1 − votação majoritária")
    ax.set_title("Redundância de erros vs folga não explorada (uma marca por base)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
