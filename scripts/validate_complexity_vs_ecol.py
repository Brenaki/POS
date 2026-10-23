"""Validate the fast complexity backend against R/ECoL on frozen bags (ADR 0023).

For every dataset: fold 0 of a stratified 10-fold split (random_state=42) is
the training set, and `generate_bags` (stratified, no replacement — the
legacy GA sampler) draws `--bags` bags with `random_state=0`. Bags are capped
at `--max-bag` instances because ECoL builds dense n x n matrices.

Each bag gets the 11 overlapping + neighborhood measures from `fast_adapter`,
from ECoL 0.3.0 (the version `Cpx.py` was written for) and, for the
neighborhood group, from ECoL 0.4.4. Output, in `--out`:
  values.csv      dataset, bag, n, measure, fast, ecol030, ecol044, seconds
  agreement.csv   per (reference, view, measure) metrics, see
                  `pos.analysis.ecol_agreement`; view = raw | dispersion
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pos.analysis.ecol_agreement import agreement_by_measure, to_dispersion  # noqa: E402
from pos.complexity.base import GROUP_MEASURES  # noqa: E402
from pos.complexity.ecol_bridge import ecol_measures  # noqa: E402
from pos.complexity.fast_adapter import complexity_data3  # noqa: E402
from pos.oracle.arff_loader import list_datasets, load_arff_dataset  # noqa: E402
from pos.pool.bag_generator import generate_bags  # noqa: E402

GROUPS = ["overlapping", "neighborhood"]
MEASURES = GROUP_MEASURES["overlapping"] + GROUP_MEASURES["neighborhood"]


def frozen_bags(X, y, n_bags: int, max_bag: int):
    tr, _ = next(StratifiedKFold(10, shuffle=True, random_state=42).split(X, y))
    X_tr, y_tr = X[tr], y[tr]
    keep = min(0.5, max_bag / len(y_tr))
    # generate_bags keeps the *train* side of train_test_split(test_size=tam).
    bags = generate_bags(X_tr, y_tr, n_bags, 1 - keep, random_state=0)
    out = [(X_tr[i], y_tr[i]) for i in map(np.asarray, bags["inst"])]
    return [b for b in out if np.unique(b[1], return_counts=True)[1].min() >= 2]


def one_dataset(path: Path, n_bags: int, max_bag: int) -> pd.DataFrame:
    X, y = load_arff_dataset(path)
    bags = frozen_bags(X, y, n_bags, max_bag)
    rows = []
    for b, (Xb, yb) in enumerate(bags):
        t0 = time.perf_counter()
        vals = complexity_data3(Xb, yb, GROUPS)
        secs = (time.perf_counter() - t0) / len(vals)
        rows += [{"bag": b, "n": len(yb), "measure": m, "fast": v, "seconds_fast": secs}
                 for m, v in zip(MEASURES, vals, strict=True)]
    df = pd.DataFrame(rows)
    for version, groups, col in (("0.3.0", tuple(GROUPS), "ecol030"),
                                 ("0.4.4", ("neighborhood",), "ecol044")):
        r = ecol_measures(bags, version, groups).rename(
            columns={"value": col, "seconds": f"seconds_{col}"})
        df = df.merge(r.drop(columns="group"), on=["bag", "measure"], how="left")
    print(f"{path.stem}: {len(bags)} bags, n={df['n'].max()}", flush=True)
    return df.assign(dataset=path.stem)


def agreement(values: pd.DataFrame, k: int) -> pd.DataFrame:
    out = []
    for ref in ("ecol030", "ecol044"):
        df = values.rename(columns={ref: "ecol"})[["dataset", "bag", "measure", "fast", "ecol"]]
        for view, d in (("raw", df), ("dispersion", to_dispersion(df))):
            out.append(agreement_by_measure(d, k).assign(reference=ref, view=view))
    cols = ["reference", "view", "measure"]
    res = pd.concat(out, ignore_index=True)
    return res[cols + [c for c in res.columns if c not in cols]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="*", help="dataset stems (default: all)")
    ap.add_argument("--bags", type=int, default=20)
    ap.add_argument("--max-bag", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    paths = [p for p in list_datasets() if not a.datasets or p.stem in a.datasets]
    a.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        frames = list(ex.map(one_dataset, paths, [a.bags] * len(paths),
                             [a.max_bag] * len(paths)))
    values = pd.concat(frames, ignore_index=True)
    values.to_csv(a.out / "values.csv", index=False)
    agree = agreement(values, a.topk)
    agree.to_csv(a.out / "agreement.csv", index=False)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    (a.out / "README.txt").write_text(
        f"commit {sha}; bags={a.bags} max_bag={a.max_bag} topk={a.topk}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(agree.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
