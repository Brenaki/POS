"""Reconcile the POS snapshots against Table 2 (ADR 0020 §8).

Usage:
    python scripts/dataset_conformance.py [--out docs/dataset_conformance.md]

Writes a Markdown table with, per dataset: the paper's metadata, the
snapshot's metadata, the SHA-256 of the POS-approved snapshot (the article
publishes no hashes) and the conformance verdict after reconciliation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from pos.oracle.run_recorder_v2 import DEFAULT_DATASETS  # noqa: E402
from pos.oracle.run_records_v2 import survey_datasets  # noqa: E402

HEADER = ("| # | Dataset | File | Paper (inst/feat/class) | Snapshot | "
          "Status | Note | SHA-256 |\n|--:|---|---|---|---|---|---|---|")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="docs/dataset_conformance.md")
    args = p.parse_args()

    records = survey_datasets(DEFAULT_DATASETS, REPO_DIR / "Dataset")
    lines = ["# Dataset conformance — Table 2 of Monteiro 2023 (ADR 0020 §8)",
             "",
             "`conformant` matches Table 2 exactly. `approved_with_note` is a",
             "reconciled divergence: usable for the published comparison, with the",
             "reason recorded. `mismatch` is excluded unless the run passes",
             "`--allow-unverified-data`, which sets `comparable=false`.",
             "", HEADER]
    counts: dict[str, int] = {}
    for i, name in enumerate(DEFAULT_DATASETS, start=1):
        r = records[name]
        status = r["conformance_status"]
        counts[status] = counts.get(status, 0) + 1
        paper = f"{r.get('paper_instances')}/{r.get('paper_features')}/{r.get('paper_classes')}"
        snap = (f"{r.get('snapshot_instances')}/{r.get('snapshot_features')}"
                f"/{r.get('snapshot_classes')}")
        sha = (r.get("snapshot_sha256") or "-").replace("sha256:", "")[:12]
        note = (r.get("conformance_reason") or "").replace("|", "/")
        lines.append(f"| {i:02d} | {name} | {r.get('snapshot_file', '-')} | {paper} "
                     f"| {snap} | `{status}` | {note} | `{sha}` |")
    lines += ["", "## Totals", ""]
    lines += [f"- `{k}`: {v}" for k, v in sorted(counts.items())]
    out = REPO_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[conformance] wrote {out}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
