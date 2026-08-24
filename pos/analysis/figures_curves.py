"""Oracle_N curve figures (subproject objectives 3, 5 and 7)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pos.analysis.loader import MODES, mean_curve  # noqa: E402

COLORS = {"pgdcs": "#8e44ad", "ga": "#c0392b", "randbag": "#e67e22",
          "bagging": "#2980b9", "rf": "#27ae60"}
# ADR 0019: `ga` is PGDCS with the complexity measures fixed at F1/T1, not
# the published method — the label says so, and `pgdcs` is the real thing.
LABELS = {"pgdcs": "PGDCS completo (Perceptron)",
          "ga": "PGDCS-F1/T1 (Perceptron)",
          "randbag": "Bags aleatorios (Perceptron)",
          "bagging": "Bagging (arvore)", "rf": "Random Forest"}


def _majority(df, mode):
    return float(df.loc[df["mode"] == mode, "majority_vote"].mean())


def plot_mean_curves(df, out: Path, zoom: int | None = None) -> Path:
    """Mean Oracle_1..M curve per mode, with each mode's majority vote line.

    `zoom` restricts the x axis to N = 1..zoom, the region the subproject
    focuses on (N = 1..5) where the traditional Oracle saturates.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    # Five majority-vote lines can land within ~0.01 of one another, so the
    # labels are staggered by rank instead of by a per-mode constant.
    ranked = sorted(MODES, key=lambda m: _majority(df, m))
    nudge = {m: (-11 if i % 2 else 5) for i, m in enumerate(ranked)}
    for mode in MODES:
        curve = mean_curve(df, mode)
        n = np.arange(1, len(curve) + 1)
        maj = _majority(df, mode)
        if zoom:
            curve, n = curve[:zoom], n[:zoom]
        label = f"{LABELS[mode]}  (Oracle_1={mean_curve(df, mode)[0]:.4f}"
        label += f", MV={maj:.4f})" if zoom else ")"
        ax.plot(n, curve, color=COLORS[mode], lw=2, marker="o" if zoom else None,
                ms=4, label=label)
        if zoom:
            # the MV lines sit ~0.15 below this window; drawing them here would
            # flatten the very differences the zoom exists to show
            continue
        ax.axhline(maj, color=COLORS[mode], ls=":", lw=1.2, alpha=0.8)
        dy = nudge[mode]
        ax.annotate(f"MV {maj:.3f}", xy=(n[-1], maj), xytext=(-4, dy),
                    textcoords="offset points", ha="right", fontsize=8,
                    color=COLORS[mode])
    ax.set_xlabel("N — número mínimo de classificadores corretos")
    ax.set_ylabel("Oracle_N (acurácia)")
    n_ds = df["dataset"].nunique()
    n_folds = df.groupby(["dataset", "mode"]).size().max()
    title = f"Curva Oracle_N — média de {n_ds} bases x {n_folds} folds"
    ax.set_title(title + (f"  |  recorte N=1..{zoom}" if zoom else ""))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    if zoom:
        ax.set_xticks(n)
        ax.legend(loc="lower left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_per_dataset_grid(df, out: Path) -> Path:
    """Small multiples: one Oracle_N curve panel per dataset.

    Shows that the aggregate mean hides very different regimes — some bases
    keep Oracle_N near 1 deep into the curve, others collapse immediately.
    """
    datasets = sorted(df["dataset"].unique())
    ncol = 6
    nrow = int(np.ceil(len(datasets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.0 * nrow),
                             sharex=True, sharey=True)
    for ax, ds in zip(axes.ravel(), datasets, strict=False):
        sub = df[df["dataset"] == ds]
        for mode in MODES:
            curves = [c for c, m in zip(sub["curve"], sub["mode"], strict=True) if m == mode]
            curve = np.mean(np.asarray(curves, dtype=float), axis=0)
            ax.plot(np.arange(1, len(curve) + 1), curve, color=COLORS[mode], lw=1.3)
            ax.axhline(sub.loc[sub["mode"] == mode, "majority_vote"].mean(),
                       color=COLORS[mode], ls=":", lw=0.8, alpha=0.7)
        ax.set_title(ds, fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_ylim(-0.02, 1.02)
    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=COLORS[m], lw=2, label=LABELS[m]) for m in MODES]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Curva Oracle_N por base (linha pontilhada = votação majoritária)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
