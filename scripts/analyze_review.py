"""Answer the external review's seven analysis points on a finished run.

Usage:
    python scripts/analyze_review.py results/experiments/<run_dir>

Writes review_analysis.txt inside the run directory. Complements
analyze_results.py, which keeps producing the figures and the fuser tables;
this covers only what ADR 0019 added.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from pos.analysis.dataset_meta import attach_dataset_meta  # noqa: E402
from pos.analysis.df_exact import attach_df_ratio_exact  # noqa: E402
from pos.analysis.loader import load_run  # noqa: E402
from pos.analysis.review_sections import SECTIONS  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run_dir = Path(sys.argv[1])
    df = attach_dataset_meta(attach_df_ratio_exact(load_run(run_dir), run_dir))
    blocks = [f"run: {run_dir}",
              f"linhas: {len(df)} | bases: {df.dataset.nunique()} | "
              f"modos: {sorted(df['mode'].unique())}"]
    for fn in SECTIONS:
        try:
            blocks.append(fn(df))
        except Exception as exc:  # a missing column must not lose the rest
            blocks.append(f"{fn.__name__}: FALHOU — {type(exc).__name__}: {exc}")
    text = "\n\n".join(blocks) + "\n"
    (run_dir / "review_analysis.txt").write_text(text)
    print(text)
    print(f"[ok] review_analysis.txt em {run_dir}")


if __name__ == "__main__":
    main()
