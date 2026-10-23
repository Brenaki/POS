"""Descriptive comparison against Monteiro 2023 (ADR 0020 §10).

Usage:
    python scripts/compare_reference.py <run_dir> [--allow-unverified-data]

Reads `summary.csv` and `run_manifest.json` from a
`monteiro-2023-dt-compatible-v1` run and writes `reference_comparison.csv`
next to them: published mean/std, POS mean/std, delta in percentage points,
the dataset's conformance status and the source table of every number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import pandas as pd  # noqa: E402

from pos.oracle.comparison_report import build_report  # noqa: E402
from pos.oracle.profiles import MONTEIRO_PROFILES  # noqa: E402

SHOW = ["dataset", "treatment", "method", "published_mean", "published_std",
        "pos_mean", "pos_std", "delta_percentage_points",
        "dataset_conformance", "source_table"]


def main() -> None:
    p = argparse.ArgumentParser(description="Compare a run with the paper.")
    p.add_argument("run_dir", type=str)
    p.add_argument("--allow-unverified-data", action="store_true",
                   help="include datasets whose snapshot does not match Table 2")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "summary.csv").exists():
        raise SystemExit(f"no summary.csv in {run_dir}")

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        profile = manifest.get("profile")
        if profile not in MONTEIRO_PROFILES:
            print(f"[warn] profile is {profile!r}; only "
                  f"{' / '.join(MONTEIRO_PROFILES)} are comparable to the paper")
        if manifest.get("comparable") is False:
            print("[warn] run is marked comparable=false (smoke scale, "
                  "overrides or unverified data)")

    rows = build_report(run_dir, allow_unverified=args.allow_unverified_data)
    if not rows:
        print("[compare] no comparable rows produced")
        return

    df = pd.DataFrame(rows).sort_values(["method", "dataset", "treatment"])
    out_path = run_dir / "reference_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"[compare] wrote {len(df)} rows to {out_path}\n")
    print(df[SHOW].to_string(index=False))
    print("\n[compare] mean |delta| per method (percentage points):")
    print(df.assign(abs_delta=df["delta_percentage_points"].abs())
            .groupby("method")["abs_delta"].mean().round(2).to_string())


if __name__ == "__main__":
    main()
