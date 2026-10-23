"""Write `Dataset/pos_catalog.json`, the POS snapshot registry (ADR 0021).

Usage:
    python scripts/build_pos_catalog.py

Registers every `Dataset/*.arff` under its run name (the paper name when an
alias applies, e.g. `Adult.arff` -> `Australian`) with its SHA-256 and shape.
A sidecar `Dataset/<stem>.json` (image embeddings, ADR 0022) is linked as the
snapshot's provenance. The `pos-oracle-v2` gate admits a dataset only when its
snapshot still hashes to the value recorded here, so re-run this script — and
commit the result — whenever a snapshot is added or deliberately replaced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from pos.oracle.dataset_gate import CATALOG_FILE  # noqa: E402
from pos.oracle.dataset_reconciliation import DATASET_ALIASES  # noqa: E402
from pos.oracle.run_records_v2 import survey_dataset  # noqa: E402


def build_catalog(dataset_dir: Path) -> dict:
    """Catalog dict for every ARFF snapshot in `dataset_dir`."""
    stem_to_name = {stem: name for name, stem in DATASET_ALIASES.items()}
    entries = {}
    for path in sorted(dataset_dir.glob("*.arff")):
        name = stem_to_name.get(path.stem, path.stem)
        r = survey_dataset(name, dataset_dir)
        entry = {"file": path.name, "sha256": r["snapshot_sha256"],
                 "instances": r["snapshot_instances"],
                 "features": r["snapshot_features"],
                 "classes": r["snapshot_classes"]}
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            entry["provenance"] = sidecar.name
        entries[name] = entry
    return {"schema_version": 1, "adr": "0021", "datasets": entries}


def main() -> None:
    dataset_dir = REPO_DIR / "Dataset"
    catalog = build_catalog(dataset_dir)
    out = dataset_dir / CATALOG_FILE
    out.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"[catalog] {len(catalog['datasets'])} snapshots -> {out}")


if __name__ == "__main__":
    main()
