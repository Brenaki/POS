"""Difference-in-differences between the two-way and three-way split runs.

Usage:
    python scripts/compare_runs.py <run_two_way> <run_three_way>

Separates the effect of the smaller DSEL from the DSEL bias that ADR 0018
removed. See `pos.analysis.run_diff` for the design of the decomposition.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from pos.analysis.loader import load_run  # noqa: E402
from pos.analysis.run_diff import (  # noqa: E402
    common_methods,
    decompose,
    did_table,
    format_decomposition,
    pair_runs,
    pool_identity,
)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    old, new = (load_run(Path(p)) for p in sys.argv[1:3])
    merged = pair_runs(old, new)
    did = did_table(merged)
    common = common_methods(old, new)

    out = [
        f"antigo: {sys.argv[1]}  ({len(old)} linhas)",
        f"novo  : {sys.argv[2]}  ({len(new)} linhas)",
        f"folds pareados: {len(merged)}",
        f"metodos dinamicos comuns ({len(common)}): "
        + ", ".join(c[4:] for c in common),
        "",
        "identidade dos pools (max |novo - antigo|; bagging/rf devem ser 0):",
        pool_identity(merged).to_string(),
        "",
        "diferenca-em-diferencas:",
        did.round(4).to_string(),
        "",
        format_decomposition(decompose(did)),
    ]
    text = "\n".join(out)
    print(text)
    (Path(sys.argv[2]) / "did_vs_previous.txt").write_text(text + "\n")
    print(f"\n[ok] gravado em {sys.argv[2]}/did_vs_previous.txt")


if __name__ == "__main__":
    main()
