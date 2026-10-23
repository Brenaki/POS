"""Fast backend vs the exact ECoL ports on full-size GA bags (ADR 0023).

`validate_complexity_vs_ecol.py` caps bags at 1000 instances because R/ECoL
builds dense n x n matrices. `t1_ecol`, `n1_ecol`, `lsc_ecol`, `f4_ecol` reproduce
ECoL exactly (tests/unit/complexity/test_t1_ecol.py, test_ecol_exact.py), so
here they stand in for R on the bags the GA really sees: same frozen-bag
recipe, `tam_bags=0.5`, no cap. Output: values.csv and agreement.csv in --out.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from validate_complexity_vs_ecol import agreement, frozen_bags  # noqa: E402

from pos.complexity.ecol_exact import f4_ecol, lsc_ecol, n1_ecol  # noqa: E402
from pos.complexity.neighborhood_extra import _normalize, _t1_fast  # noqa: E402
from pos.complexity.neighborhood_fast import _lsc_fast, _n1_fast  # noqa: E402
from pos.complexity.overlapping_measures import _f4  # noqa: E402
from pos.complexity.t1_ecol import t1_ecol  # noqa: E402
from pos.oracle.arff_loader import list_datasets, load_arff_dataset  # noqa: E402


def one_dataset(path: Path, n_bags: int) -> pd.DataFrame:
    X, y = load_arff_dataset(path)
    rows = []
    for b, (Xb, yb) in enumerate(frozen_bags(X, y, n_bags, max_bag=10**9)):
        Xn = _normalize(Xb)
        for m, fast, e030, e044 in (
                ("T1", _t1_fast(Xn, yb), t1_ecol(Xb, yb, "0.3.0"), t1_ecol(Xb, yb, "0.4.4")),
                ("N1", _n1_fast(Xn, yb), n1_ecol(Xb, yb), None),
                ("LSC", _lsc_fast(Xn, yb), lsc_ecol(Xb, yb), None),
                ("F4", _f4(Xb, yb), f4_ecol(Xb, yb), None)):
            rows.append({"dataset": path.stem, "bag": b, "n": len(yb), "measure": m,
                         "fast": fast, "ecol030": e030,
                         "ecol044": e030 if e044 is None else e044})
    print(f"{path.stem}: n={len(yb)}", flush=True)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bags", type=int, default=20)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    paths = list_datasets()
    a.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        values = pd.concat(ex.map(one_dataset, paths, [a.bags] * len(paths)),
                           ignore_index=True)
    values.to_csv(a.out / "values.csv", index=False)
    agree = agreement(values, a.topk)
    agree.to_csv(a.out / "agreement.csv", index=False)
    print(agree.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
