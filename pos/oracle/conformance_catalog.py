"""Conformance-gated dataset catalog v2 (ADR 2020 §8).

Maps the 28 datasets of Table 2 (Monteiro 2023) to POS snapshots, recording:
- paper metadata (instances, features, classes, source) — from the article;
- POS snapshot SHA-256 — the POS-approved snapshot hash, NOT a paper hash;
- conformance status: conformant / mismatch / unverified.

The hash is of the POS-approved snapshot; the paper does not publish SHA-256.
Preflight blocks the published comparison when shape, classes or hash diverge.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

CONFORMANCE_CONFORMANT = "conformant"
CONFORMANCE_MISMATCH = "mismatch"
CONFORMANCE_UNVERIFIED = "unverified"

# Table 2 of Monteiro 2023 (Information Fusion 89).
# Extracted verbatim from docs/Monteiro2023.pdf, page 6.
PAPER_TABLE_2: list[dict[str, Any]] = [
    {"no": 1, "name": "Australian", "instances": 690, "features": 14, "classes": 2, "source": "UCI"},
    {"no": 2, "name": "Banana", "instances": 2000, "features": 2, "classes": 2, "source": "PRTools"},
    {"no": 3, "name": "Blood", "instances": 748, "features": 4, "classes": 2, "source": "UCI"},
    {"no": 4, "name": "CTG", "instances": 2126, "features": 21, "classes": 3, "source": "UCI"},
    {"no": 5, "name": "Diabetes", "instances": 766, "features": 8, "classes": 2, "source": "UCI"},
    {"no": 6, "name": "Faults", "instances": 1941, "features": 27, "classes": 7, "source": "UCI"},
    {"no": 7, "name": "German", "instances": 1000, "features": 24, "classes": 2, "source": "STATLOG"},
    {"no": 8, "name": "Haberman", "instances": 306, "features": 3, "classes": 2, "source": "UCI"},
    {"no": 9, "name": "Heart", "instances": 270, "features": 10, "classes": 2, "source": "STATLOG"},
    {"no": 10, "name": "ILPD", "instances": 583, "features": 34, "classes": 2, "source": "UCI"},
    {"no": 11, "name": "Ionosphere", "instances": 351, "features": 16, "classes": 2, "source": "UCI"},
    {"no": 12, "name": "Laryngeal1", "instances": 213, "features": 16, "classes": 2, "source": "LKC"},
    {"no": 13, "name": "Laryngeal3", "instances": 353, "features": 2, "classes": 3, "source": "LKC"},
    {"no": 14, "name": "Lithuanian", "instances": 2000, "features": 2, "classes": 2, "source": "PRTools"},
    {"no": 15, "name": "Liver", "instances": 345, "features": 6, "classes": 2, "source": "UCI"},
    {"no": 16, "name": "Mammo", "instances": 830, "features": 5, "classes": 2, "source": "KEEL"},
    {"no": 17, "name": "Monk", "instances": 432, "features": 6, "classes": 2, "source": "KEEL"},
    {"no": 18, "name": "Phoneme", "instances": 5404, "features": 5, "classes": 2, "source": "ELENA"},
    {"no": 19, "name": "P2", "instances": 5000, "features": 2, "classes": 2, "source": "[33]"},
    {"no": 20, "name": "Segmentation", "instances": 2310, "features": 19, "classes": 7, "source": "UCI"},
    {"no": 21, "name": "Sonar", "instances": 208, "features": 60, "classes": 2, "source": "UCI"},
    {"no": 22, "name": "Thyroid", "instances": 692, "features": 16, "classes": 2, "source": "LKC"},
    {"no": 23, "name": "Vehicle", "instances": 846, "features": 18, "classes": 4, "source": "STATLOG"},
    {"no": 24, "name": "Vertebral", "instances": 300, "features": 6, "classes": 2, "source": "UCI"},
    {"no": 25, "name": "WBC", "instances": 569, "features": 30, "classes": 2, "source": "UCI"},
    {"no": 26, "name": "WDVG", "instances": 5000, "features": 21, "classes": 3, "source": "UCI"},
    {"no": 27, "name": "Weaning", "instances": 302, "features": 17, "classes": 2, "source": "LKC"},
    {"no": 28, "name": "Wine", "instances": 178, "features": 13, "classes": 3, "source": "UCI"},
]

_PAPER_BY_NAME = {d["name"]: d for d in PAPER_TABLE_2}


def check_conformance(
    paper_entry: dict[str, Any],
    actual: dict[str, int] | None,
) -> tuple[str, str]:
    """Return (status, reason). `actual` has keys instances/features/classes."""
    if actual is None:
        return CONFORMANCE_UNVERIFIED, "snapshot not found or not loaded"
    mismatches: list[str] = []
    if actual.get("instances") != paper_entry["instances"]:
        mismatches.append(
            f"instances: paper={paper_entry['instances']} actual={actual.get('instances')}"
        )
    if actual.get("features") != paper_entry["features"]:
        mismatches.append(
            f"features: paper={paper_entry['features']} actual={actual.get('features')}"
        )
    if actual.get("classes") != paper_entry["classes"]:
        mismatches.append(
            f"classes: paper={paper_entry['classes']} actual={actual.get('classes')}"
        )
    if mismatches:
        return CONFORMANCE_MISMATCH, "; ".join(mismatches)
    return CONFORMANCE_CONFORMANT, ""


def conformance_status(
    name: str,
    actual: dict[str, int] | None,
    snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a conformance report dict for one dataset."""
    paper = _PAPER_BY_NAME.get(name)
    if paper is None:
        return {
            "name": name,
            "conformance_status": CONFORMANCE_UNVERIFIED,
            "conformance_reason": f"dataset {name} not in paper Table 2",
            "snapshot_sha256": snapshot_sha256,
        }
    status, reason = check_conformance(paper, actual)
    return {
        "name": name,
        "paper_instances": paper["instances"],
        "paper_features": paper["features"],
        "paper_classes": paper["classes"],
        "paper_source": paper["source"],
        "snapshot_sha256": snapshot_sha256,
        "conformance_status": status,
        "conformance_reason": reason,
    }


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file (POS snapshot, not paper hash)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()