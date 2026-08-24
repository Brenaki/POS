"""Formatted sections answering the external review (ADR 0019).

One function per point, each returning the block of text
`scripts/analyze_review.py` writes into review_analysis.txt. Kept apart
from the script so neither file passes the 150 LOC cap (ADR 0002), and so
the formatting can be exercised by tests without running a whole analysis.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from pos.analysis.mode_contrasts import (
    all_contrasts,
    pgdcs_measure_choices,
)
from pos.analysis.nstar_split import (
    nstar_by_cardinality,
    nstar_summary,
    sandwich_check,
)
from pos.analysis.pruning import pruning_correlation, pruning_table
from pos.analysis.recoverability import (
    build_feature_frame,
    dataset_complexity_features,
    forest_model,
    lodo_regression,
    ridge_model,
)
from pos.analysis.threshold_lodo import lodo_scores

QUANTILES = [0.5, 0.75, 0.9, 0.95]


def _rho(frame, col, target="gap_1", by=None):
    keys = ["dataset"] + ([by] if by else [])
    g = frame.dropna(subset=[col, target]).groupby(keys)[[col, target]].mean()
    if len(g) < 5:
        return None
    rho, p = spearmanr(g[col], g[target])
    return f"n={len(g):3d}  rho={rho:+.4f}  p={p:.2e}"


def section_df_exact(df):
    out = ["D1 — normalizacao exata do double fault (ponto 6)",
           "denominador sob independencia com taxas de erro desiguais:",
           "  E[DF] = ((sum e)^2 - sum e^2) / (M(M-1))", ""]
    for col in ["df_ratio", "df_ratio_exact"]:
        if col not in df.columns:
            continue
        out.append(f"  {col} vs folga Oracle_1-MVR:")
        for mode in [None, *sorted(df["mode"].unique())]:
            sub = df if mode is None else df[df["mode"] == mode]
            line = _rho(sub, col, by="mode" if mode is None else None)
            if line:
                out.append(f"    {mode or 'agrupado':10s} {line}")
    if {"df_ratio", "df_ratio_exact"} <= set(df.columns):
        r = (df["df_ratio_exact"] / df["df_ratio"]).dropna()
        out.append(f"\n  razao exact/medio: mediana={r.median():.4f} max={r.max():.4f}")
        out.append(f"  desvio das acuracias individuais: media={df['acc_spread'].mean():.4f}")
    return "\n".join(out)


def section_nstar(df):
    check = sandwich_check(df)
    per = nstar_by_cardinality(df)
    b = per[per["is_binary"]]["nstar"]
    m = per[~per["is_binary"]]["nstar"]
    out = ["D2 — N* por cardinalidade de classes (ponto 3)",
           f"  sanduiche Oracle_(M/2+1) <= MVR <= Oracle_(M/2): "
           f"vale em {check.get('holds_both_frac', 0):.3f} de {check.get('n_folds', 0)} folds binarios",
           f"  igualdade exata MVR == Oracle_(M/2+1): {check.get('exact_equality', 0)}"
           f"/{check.get('n_folds', 0)} folds",
           "", nstar_summary(df).round(2).to_string(index=False)]
    if len(b) > 2 and len(m) > 2:
        p = float(mannwhitneyu(b, m)[1])
        out.append(f"\n  binaria vs multiclasse: mediana {np.median(b):.1f} vs "
                   f"{np.median(m):.1f}  Mann-Whitney p={p:.2e}")
    return "\n".join(out)


def section_thresholds(df):
    out = ["D3 — limiares DF/e^2 validados leave-one-dataset-out (ponto 5)"]
    for col in ["df_ratio", "df_ratio_exact"]:
        if col not in df.columns:
            continue
        r = lodo_scores(df, col)
        out.append(f"  {col:16s} LODO={r['lodo_accuracy']:.3f}  "
                   f"baseline={r['baseline_accuracy']:.3f}  "
                   f"cortes medianos={r['median_cuts'][0]:.2f}/{r['median_cuts'][1]:.2f}"
                   f"  n={r['n_predictions']}")
    return "\n".join(out)


def section_recovered(df):
    r = df.dropna(subset=["recovered"])
    header = "  ".join(f"Q{int(q*100)}" for q in QUANTILES)
    out = ["D4 — distribuicao de `recovered`, nao um limite superior (ponto 4)",
           f"  {'modo':10s} {'n':>4s} {'media':>8s}  {header}"]
    for mode in [*sorted(r["mode"].unique()), None]:
        s = r["recovered"] if mode is None else r[r["mode"] == mode]["recovered"]
        qs = "  ".join(f"{s.quantile(q):.3f}" for q in QUANTILES)
        out.append(f"  {mode or 'todos':10s} {len(s):4d} {s.mean():8.4f}  {qs}")
    tree = r[r["mode"].isin(["bagging", "rf"])]["recovered"]
    if len(tree) > 10:
        out.append(f"\n  arvores: fracao de folds acima de 0.15 = {(tree > 0.15).mean():.3f}"
                   f"  |  recovered <= 0 em {(tree <= 0).mean():.3f}")
        out.append(f"  cota empirica a 95%: MVR + {tree.quantile(0.95):.2f} * folga")
    return "\n".join(out)


def section_recoverability(df):
    out = ["D5 — o que preve recuperabilidade (ponto 9)"]
    frame = build_feature_frame(df, dataset_complexity_features())
    if len(frame) < 10:
        return "\n".join(out + ["  dados insuficientes"])
    for name, model in [("ridge", ridge_model()), ("floresta", forest_model())]:
        r = lodo_regression(frame, model)
        out.append(f"  {name:9s} n={r['n']:3d}  R2_LODO={r['r2_lodo']:+.4f}  "
                   f"rho={r['spearman']:+.4f} (p={r['p_value']:.2e})  "
                   f"MAE={r['mae']:.4f} vs baseline {r['baseline_mae']:.4f}")
    return "\n".join(out)


def section_pruning(df):
    table = pruning_table(df)
    if table.empty:
        return "D6 — poda medida (ponto 8)\n  run sem colunas sel_frac"
    return "\n".join(["D6 — poda medida: E[S]/M por metodo (ponto 8)",
                      table.round(4).to_string(index=False), "",
                      "  correlacao entre fracao selecionada e margem sobre o MVR:",
                      pruning_correlation(df).round(4).to_string(index=False)])


def section_modes(df):
    out = ["D7 — contrastes entre modos de geracao (ponto 2 e ponto 7)"]
    for name, table in all_contrasts(df).items():
        out += [f"\n  {name}", table.round(4).to_string(index=False)]
    choices = pgdcs_measure_choices(df)
    if not choices.empty:
        out += ["\n  medidas escolhidas pela votacao do PGDCS:",
                choices.head(12).round(4).to_string(index=False),
                f"  folds em que a escolha foi F1/T1: {int(choices.loc[choices.is_f1_t1, 'folds'].sum())}"]
    return "\n".join(out)




SECTIONS = (section_df_exact, section_nstar, section_thresholds,
            section_recovered, section_recoverability, section_pruning,
            section_modes)
