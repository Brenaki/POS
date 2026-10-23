"""Regenerate tests/_ecol_legacy_golden.json from R/ECoL 0.3.0 (ADR 0003, 0023).

The legacy `Cpx.complexity_data3` called ECoL 0.3.0 for the 6 groups on the
`sample.ipynb` wine split (unstratified, random_state=42). This freezes those
values, so the Python backends can be checked against them without R.
Measures that need igraph (N1, network) are null when igraph is missing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pos.complexity.base import GROUP_MEASURES  # noqa: E402
from pos.complexity.ecol_bridge import ECOL_TARBALLS, ecol_measures  # noqa: E402

OUT = REPO / "tests" / "_ecol_legacy_golden.json"
VERSION = "0.3.0"


def wine_train():
    X, y = load_wine(return_X_y=True)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.4, random_state=42)
    return X_train, y_train


def main() -> int:
    X, y = wine_train()
    df = ecol_measures([(X, y)], VERSION, tuple(GROUP_MEASURES))
    values = {f"{g}.{m}": (None if v != v else float(v))
              for g, m, v in zip(df["group"], df["measure"], df["value"], strict=True)}
    r_ver = subprocess.run(["Rscript", "-e", "cat(R.version.string)"],
                           capture_output=True, text=True).stdout
    golden = {
        "source": f"ECoL {VERSION} sourced from {ECOL_TARBALLS[VERSION][0]}",
        "tarball_sha256": ECOL_TARBALLS[VERSION][1],
        "r_version": r_ver,
        "data": "load_wine, train_test_split(test_size=0.4, random_state=42), X_train",
        "data_sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
        "summary": "mean; N4 under set.seed(0)",
        "values": values,
    }
    OUT.write_text(json.dumps(golden, indent=2) + "\n")
    print(json.dumps(values, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
