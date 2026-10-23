"""Reconciliation of the POS snapshots against Table 2 (ADR 0020 §8).

The conformance check compares (instances, features, classes) of each POS
snapshot with Monteiro 2023 Table 2. Five datasets diverge; ADR 0020 requires
reconciling them *before* the check becomes a hard gate, because a divergent
number in the table is not automatically a divergent dataset.

Four of the five are explained by a one-row shift in the table's *Features*
column across rows 09-13. Reading each paper value against the next row's
snapshot lines them all up:

    row  dataset      paper feat.  snapshot feat.   next row's snapshot
    09   Heart        10           13               ILPD        10
    10   ILPD         34           10               Ionosphere  34
    11   Ionosphere   16           34               Laryngeal1  16
    12   Laryngeal1   16           16               Laryngeal3  16
    13   Laryngeal3    2           16               Lithuanian   2

Row 08 (Haberman, 3) and row 14 (Lithuanian, 2) match without the shift, so
the defect is bounded to rows 09-13. Those four snapshots are therefore
`approved_with_note`: usable for the published comparison, with the
discrepancy recorded in the run manifest and the comparison report.

P2 is a genuine content divergence — the snapshot holds 2000 instances where
the paper generated 5000 — and stays `mismatch`: excluded from the published
comparison unless `--allow-unverified-data` is passed, which also sets
`comparable=false` for the whole run. This Table 2 verdict gates only the
Monteiro profile; `pos-oracle-v2` gates on its own SHA-256 catalog, where P2
is conformant (ADR 0021, `pos/oracle/dataset_gate.py`).
"""

from __future__ import annotations

from typing import Any

APPROVED_WITH_NOTE = "approved_with_note"

# Paper name -> file stem under Dataset/ when they differ.
DATASET_ALIASES: dict[str, str] = {"Australian": "Adult"}

# Paper name -> (decision, reason). Applied on top of `check_conformance`.
RECONCILIATION: dict[str, tuple[str, str]] = {
    "Heart": (APPROVED_WITH_NOTE,
              "Table 2 Features column shifted by one row (09-13): paper 10 is "
              "ILPD's attribute count; STATLOG Heart has 13"),
    "ILPD": (APPROVED_WITH_NOTE,
             "Table 2 Features column shifted by one row (09-13): paper 34 is "
             "Ionosphere's attribute count; UCI ILPD has 10"),
    "Ionosphere": (APPROVED_WITH_NOTE,
                   "Table 2 Features column shifted by one row (09-13): paper 16 "
                   "is Laryngeal1's attribute count; UCI Ionosphere has 34"),
    "Laryngeal3": (APPROVED_WITH_NOTE,
                   "Table 2 Features column shifted by one row (09-13): paper 2 "
                   "is Lithuanian's attribute count; LKC Laryngeal3 has 16"),
}


def dataset_stem(name: str) -> str:
    """File stem for a paper dataset name (`Australian` lives as `Adult`)."""
    return DATASET_ALIASES.get(name, name)


def reconcile(status: dict[str, Any]) -> dict[str, Any]:
    """Apply the recorded reconciliation to a `conformance_status` dict.

    Only a `mismatch` can be lifted, and only to `approved_with_note` — never
    silently to `conformant`, so the divergence survives into the manifest.
    """
    out = dict(status)
    decision = RECONCILIATION.get(status.get("name", ""))
    if decision is None or status.get("conformance_status") != "mismatch":
        return out
    new_status, reason = decision
    out["conformance_status"] = new_status
    out["conformance_reason"] = f"{status.get('conformance_reason', '')} | {reason}"
    out["reconciled"] = True
    return out


def comparable_statuses(allow_unverified: bool = False) -> tuple[str, ...]:
    """Statuses whose datasets may enter the published comparison."""
    if allow_unverified:
        return ("conformant", APPROVED_WITH_NOTE, "mismatch", "unverified")
    return ("conformant", APPROVED_WITH_NOTE)
