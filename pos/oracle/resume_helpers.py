"""Resume helpers: scan existing output dir for completed folds.

Extracted from run_recorder.py to satisfy the <=150 LOC file cap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def completed_folds(output_dir: Path) -> set[tuple[str, int, str]]:
    """Scan output_dir for existing fold_manifest_<mode>.json files.

    Returns a set of (dataset, fold_idx, mode) tuples already done.
    """
    done: set[tuple[str, int, str]] = set()
    if not output_dir.exists():
        return done
    for fm_path in output_dir.glob("*/fold_*/fold_manifest_*.json"):
        try:
            fm = json.loads(fm_path.read_text())
            done.add((fm["dataset"], fm["fold_idx"], fm["mode"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def load_existing_summary(output_dir: Path) -> list[dict]:
    """Load rows from an existing summary.csv for resume."""
    summary_path = output_dir / "summary.csv"
    if not summary_path.exists():
        return []
    df = pd.read_csv(summary_path)
    if df.empty:
        return []
    return df.to_dict("records")
