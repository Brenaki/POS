"""v1 (fast measures) vs v2 (ECoL 0.3.0 measures) of the Monteiro profile (ADR 0024).

Usage:
    python scripts/compare_reference.py <run_v2>      # writes reference_comparison.csv
    python scripts/compare_backend_runs.py <run_v1> <run_v2>

Writes `backend_comparison.json` and `table4_agreement.csv` into <run_v2>.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from pos.analysis.backend_compare import (  # noqa: E402
    chosen_measures,
    paired_by_dataset,
    table4_agreement,
)
from pos.oracle.paper_reference import TABLE_4_VOTES  # noqa: E402


def _reference_errors(run: Path) -> dict[str, dict[str, float]]:
    path = run / "reference_comparison.csv"
    if not path.exists():
        return {}
    r = pd.read_csv(path)
    g = r.groupby(["treatment", "method"])["delta_percentage_points"]
    return {f"{t}/{m}": {"mean": float(s.mean()), "mae": float(s.abs().mean()),
                         "negatives": int((s < 0).sum()), "n": int(len(s))}
            for (t, m), s in g}


def main(v1: Path, v2: Path) -> dict:
    s1, s2 = (pd.read_csv(r / "summary.csv") for r in (v1, v2))
    agree = {name: table4_agreement(chosen_measures(s), TABLE_4_VOTES)
             for name, s in (("v1", s1), ("v2", s2))}
    both = agree["v1"].merge(agree["v2"], on=["dataset", "table4"], suffixes=("_v1", "_v2"))
    both.to_csv(v2 / "table4_agreement.csv", index=False)
    out = {
        "runs": {"v1": v1.name, "v2": v2.name},
        "table4_hits": {k: {"overlapping": int(a["overlapping_hit"].sum()),
                            "neighborhood": int(a["neighborhood_hit"].sum()),
                            "n": int(len(a))} for k, a in agree.items()},
        "same_pair": int((both["chosen_v1"] == both["chosen_v2"]).sum()),
        "v2_minus_v1": {t: paired_by_dataset(s2, s1, t, t)
                        for t in ("pgdcs_a", "pgdcs_d", "bagging")},
        "pgdcs_a_minus_bagging": {k: paired_by_dataset(s, s, "pgdcs_a", "bagging")
                                  for k, s in (("v1", s1), ("v2", s2))},
        "reference": {"v1": _reference_errors(v1), "v2": _reference_errors(v2)},
    }
    (v2 / "backend_comparison.json").write_text(json.dumps(out, indent=2))
    print(both.to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if k != "reference"}, indent=2))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
