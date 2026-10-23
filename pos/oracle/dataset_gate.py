"""Per-profile conformance gate (ADR 0021).

ADR 0020 gated every profile on Monteiro 2023 Table 2. That is right for
`monteiro-2023-dt-compatible-v1`, whose numbers are compared with the paper,
and wrong for `pos-oracle-v2`, whose reference scope is itself: P2 (2000
snapshot instances vs 5000 in the paper) could only enter it through
`--allow-unverified-data`, which marked the whole run `comparable=false`.

Each profile now names its dataset reference:

* `monteiro-2023-table-2` — the Table 2 verdict after reconciliation;
* `pos-catalog` — the snapshot's SHA-256 must match `Dataset/pos_catalog.json`,
  the versioned registry of the POS snapshots (tabular and image embeddings).

Unknown profiles fall back to Table 2, the stricter gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pos.oracle.dataset_reconciliation import comparable_statuses

TABLE_2 = "monteiro-2023-table-2"
POS_CATALOG = "pos-catalog"
CATALOG_FILE = "pos_catalog.json"

_PROFILE_REFERENCE = {
    "pos-oracle-v2": POS_CATALOG,
    "monteiro-2023-dt-compatible-v1": TABLE_2,
}


def dataset_reference_for(profile: str | None) -> str:
    """Dataset reference a profile is gated on (Table 2 when unknown)."""
    return _PROFILE_REFERENCE.get(profile or "", TABLE_2)


def load_pos_catalog(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    """The POS snapshot registry, `{}` when the file does not exist."""
    path = Path(dataset_dir) / CATALOG_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())["datasets"]


def pos_catalog_status(record: dict[str, Any],
                       catalog: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """(status, reason) of a surveyed snapshot against the POS catalog."""
    name, sha = record.get("name", ""), record.get("snapshot_sha256")
    entry = catalog.get(name)
    if entry is None:
        return "unverified", f"dataset {name} not in {CATALOG_FILE}"
    if sha is None:
        return "unverified", f"snapshot of {name} not found"
    if sha != entry["sha256"]:
        return "mismatch", f"sha256 {sha} != catalog {entry['sha256']}"
    return "conformant", f"sha256 matches {CATALOG_FILE}"


def gate_decision(record: dict[str, Any], profile: str | None,
                  allow_unverified: bool,
                  catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Gate fields to merge into a dataset's conformance record."""
    reference = dataset_reference_for(profile)
    if reference == POS_CATALOG:
        status, reason = pos_catalog_status(record, catalog)
    else:
        status = record.get("conformance_status", "unverified")
        reason = record.get("conformance_reason", "")
    return {
        "gate_reference": reference,
        "gate_status": status,
        "gate_reason": reason,
        "gate_passes": status in comparable_statuses(allow_unverified),
    }
