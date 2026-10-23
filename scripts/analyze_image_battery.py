"""Analyse the image battery against the tabular battery (ADR 0022).

Usage:
    python scripts/analyze_image_battery.py RUN_DIR [RUN_DIR ...] [--tabular TAB_RUN_DIR]

Each RUN_DIR is a `pos-oracle-v2` run over image snapshots (the battery may be
split across runs; their rows are concatenated and the outputs go to the
first one); TAB_RUN_DIR (by
default the canonical tabular run of the same profile) supplies the 28
tabular datasets. Writes into RUN_DIR:
  analysis_images.txt / .json  per-dataset table, binary sandwich check,
                               image-vs-tabular contrasts (Mann-Whitney),
                               DF/e^2 -> gap curve fitted on tabular, applied
                               to images
  figures/fig9_image_oracle_curves.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from pos.analysis.nstar_split import sandwich_check  # noqa: E402
from pos.analysis.v2_battery import (  # noqa: E402
    battery_table,
    contrast,
    dfe_transfer,
    load_v2_run,
)
from pos.images.embedding_dataset import IMAGE_DATASETS  # noqa: E402

TABULAR = REPO_DIR / "results" / "experiments" / "2026-08-27T23-01-32_7c5ec6d"
METRICS = ["gap", "recovered", "nstar_rel", "df_ratio", "mean_individual_acc"]


def _per_dataset(df):
    t = battery_table(df)
    t["gap"] = t["oracle_1"] - t["majority_vote"]
    t["nstar_rel"] = t["nstar"] / df["M"].iloc[0]
    return t


def _figure(df, out: Path) -> None:
    names = [n for n in IMAGE_DATASETS if n in set(df["dataset"])]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.1), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        for mode, sub in df[df["dataset"] == name].groupby("mode"):
            curve = np.mean(np.asarray(sub["curve"].tolist(), dtype=float), axis=0)
            line, = ax.plot(np.arange(1, len(curve) + 1), curve, lw=1.4, label=mode)
            ax.axhline(sub["majority_vote"].mean(), color=line.get_color(), ls=":", lw=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("N")
    np.atleast_1d(axes)[0].set_ylabel("Oracle_N (pontilhado: MVR)")
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--tabular", default=str(TABULAR))
    args = p.parse_args()
    run_dir = Path(args.run_dirs[0])
    img = pd.concat([load_v2_run(d) for d in args.run_dirs], ignore_index=True)
    img = img[img["dataset"].isin(IMAGE_DATASETS)]
    tab = load_v2_run(args.tabular)
    ti, tt = _per_dataset(img), _per_dataset(tab)
    # Contrasts run on one row per dataset (treatments averaged): the three
    # treatments of a dataset are not independent observations.
    ci = ti.groupby("dataset").mean(numeric_only=True).reset_index()
    ct = tt.groupby("dataset").mean(numeric_only=True).reset_index()

    report = {"per_dataset": ti.to_dict(orient="records"),
              "sandwich_binary": sandwich_check(img),
              "invariants": {
                  "monotone": int(sum(np.all(np.diff(c) <= 1e-12) for c in img["curve"])),
                  "oracle1_ge_mvr": int((img["oracle_1"] >= img["majority_vote"] - 1e-12).sum()),
                  "n_folds": int(len(img))},
              "dfe_transfer": dfe_transfer(tt, ti)}
    lines = ["# Bateria de imagens (ADR 0022)", "", ti.round(4).to_string(index=False), "",
             f"invariantes: {report['invariants']}",
             f"sanduiche (binarias): {report['sandwich_binary']}",
             f"DF/e2 tabular -> imagem: {report['dfe_transfer']}", ""]
    for label, mask_i, mask_t in (("todas", None, None),
                                  ("multiclasse", ci["n_classes"] > 2, ct["n_classes"] > 2),
                                  ("binarias", ci["n_classes"] == 2, ct["n_classes"] == 2)):
        a = ci if mask_i is None else ci[mask_i]
        b = ct if mask_t is None else ct[mask_t]
        c = contrast(a, b, METRICS)
        report[f"contrast_{label}"] = c.to_dict(orient="index")
        lines += [f"## imagem (a) vs tabular (b) — {label}", c.round(4).to_string(), ""]
    (run_dir / "analysis_images.txt").write_text("\n".join(lines) + "\n")
    (run_dir / "analysis_images.json").write_text(json.dumps(report, indent=1, default=float))
    _figure(img, run_dir / "figures" / "fig9_image_oracle_curves.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
