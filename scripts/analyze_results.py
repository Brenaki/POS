"""Analyse a finished experiment run: figures + statistical tests.

Usage:
    python scripts/analyze_results.py results/experiments/<run_dir>

Writes figures/ and analysis.txt inside the run directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from pos.analysis.figures_compare import (  # noqa: E402
    plot_diversity_vs_gap,
    plot_gap_per_dataset,
    plot_nstar,
)
from pos.analysis.figures_curves import plot_mean_curves, plot_per_dataset_grid  # noqa: E402
from pos.analysis.figures_des import plot_fuser_accuracy, plot_recovered_gap  # noqa: E402
from pos.analysis.fusers import (  # noqa: E402
    PRIMARY_FUSERS,
    fuser_comparison,
    fuser_table,
    recovery_summary,
    secondary_vs_mvr,
)
from pos.analysis.loader import MODES, load_run  # noqa: E402
from pos.analysis.stats_tests import compare, format_comparison  # noqa: E402

# (metric, higher_is_better). df_ratio and nstar are "lower is better":
# less redundancy, and a lower crossover level, are the interesting direction.
METRICS = [
    ("oracle_1", True), ("oracle_2", True), ("oracle_5", True), ("oracle_M", True),
    ("majority_vote", True), ("mean_probs", True), ("mean_individual_acc", True),
    ("gap_1", True), ("gap_5", True), ("df_ratio", False), ("nstar", False),
    ("des_best", True), ("gap_des", False), ("recovered", True),
    # Milestone 7 asks the same comparison to be read beyond accuracy.
    ("mv_balanced_acc", True), ("mv_f1", True), ("soft_balanced_acc", True),
    ("oracle_1_balanced", True), ("gap_1_balanced", True),
]

# Fuser rankings are reported on accuracy and on balanced accuracy; where the
# two disagree, the disagreement is the result (ADR 0018).
FUSER_VALUES = ["", "balanced_acc"]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run_dir = Path(sys.argv[1])
    df = load_run(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_mean_curves(df, fig_dir / "fig1_oracle_curve.png")
    plot_mean_curves(df, fig_dir / "fig2_oracle_curve_zoom.png", zoom=10)
    plot_nstar(df, fig_dir / "fig3_nstar.png")
    plot_gap_per_dataset(df, fig_dir / "fig4_gap_per_dataset.png", level=1)
    plot_diversity_vs_gap(df, fig_dir / "fig5_diversity_vs_gap.png")
    plot_per_dataset_grid(df, fig_dir / "fig6_curves_per_dataset.png")
    has_des = "recovered" in df.columns and df["recovered"].notna().any()
    if has_des:
        plot_fuser_accuracy(df, fig_dir / "fig7_fuser_accuracy.png")
        plot_recovered_gap(df, fig_dir / "fig8_recovered_gap.png")

    results, blocks = {}, []
    for metric, higher in METRICS:
        if metric not in df.columns:
            continue
        missing = [m for m in MODES if df.loc[df["mode"] == m, metric].isna().all()]
        if missing:
            blocks.append(f"{metric}: comparacao pareada impossivel — sem valor "
                          f"para {missing}")
            continue
        res = compare(df, metric, higher_is_better=higher)
        results[metric] = res
        blocks.append(format_comparison(res))

    for mode in MODES if has_des else []:
        for value in FUSER_VALUES:
            # A run predating ADR 0018 carries no per-metric columns; asking
            # for balanced accuracy there must skip, not crash.
            if fuser_table(df, mode, PRIMARY_FUSERS, value).empty:
                continue
            res = fuser_comparison(df, mode, PRIMARY_FUSERS, value)
            results[f"fusers_{mode}_{value or 'acc'}"] = res
            blocks.append(format_comparison(res))
        sec = secondary_vs_mvr(df, mode)
        if not sec.empty:
            blocks.append(f"todos os fusores vs MVR [{mode}] "
                          f"(Wilcoxon pareado, Holm)\n"
                          + sec.round(4).to_string())

    head = [
        f"run: {run_dir}",
        f"linhas: {len(df)} | bases: {df.dataset.nunique()} | modos: {sorted(df['mode'].unique())}",
        f"violacoes oracle_1 < majority: {int((df.oracle_1 < df.majority_vote - 1e-12).sum())}",
        "nao-monotonas: " + str(sum(
            1 for c in df["curve"] if any(c[i] < c[i + 1] - 1e-12 for i in range(len(c) - 1)))),
        "mean_probs ausente por modo: " + str(
            df[df.mean_probs.isna()].groupby("mode").size().to_dict()),
        "",
    ]
    if "k_eff" in df.columns and df["k_eff"].notna().any():
        head.append("DSEL por modo (tamanho e k efetivo):\n"
                    + df.groupby("mode")[["n_dsel", "k_eff"]].agg(
                        ["min", "median", "max"]).to_string())
        head.append("")
    if has_des:
        head.append("folga recuperada pela selecao dinamica:\n"
                    + recovery_summary(df).round(4).to_string())
        head.append("")
    (run_dir / "analysis.txt").write_text("\n\n".join(head + blocks) + "\n")
    (run_dir / "analysis.json").write_text(json.dumps(results, indent=2))
    print("\n\n".join(head + blocks))
    print(f"\n[ok] figuras em {fig_dir}")
    print(f"[ok] analysis.txt / analysis.json em {run_dir}")


if __name__ == "__main__":
    main()
