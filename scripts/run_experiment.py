"""CLI for reproducible Oracle_N experiments.

Usage:
    python scripts/run_experiment.py --smoke            # 3 datasets, 3 folds, gen=1
    python scripts/run_experiment.py --full             # 31 datasets, 10 folds, gen=20
    python scripts/run_experiment.py --config cfg.json  # custom config
    python scripts/run_experiment.py --smoke --mode rf  # only RF baseline
    python scripts/run_experiment.py --smoke --dry-run  # print manifest, no run
    python scripts/run_experiment.py --resume <dir>     # resume interrupted run

Output: results/experiments/<ISO_timestamp>_<git_sha_short>/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from pos.oracle.run_helpers import git_sha  # noqa: E402
from pos.oracle.run_recorder import record_run  # noqa: E402

SMOKE_DATASETS = ["Wine", "Banana", "Vehicle"]
FULL_DATASETS = [
    "Adult", "Banana", "Blood", "CTG", "Diabetes", "Ecoli", "Faults", "German",
    "Glass", "Haberman", "Heart", "ILPD", "Ionosphere", "Laryngeal1", "Laryngeal3",
    "Lithuanian", "Liver", "Magic", "Mammo", "Monk", "P2", "Phoneme", "Segmentation",
    "Sonar", "Thyroid", "Vehicle", "Vertebral", "WBC", "WDVG", "Weaning", "Wine",
]


def build_config(args) -> dict:
    if args.config:
        return json.loads(Path(args.config).read_text())
    if args.smoke:
        return {
            "datasets": SMOKE_DATASETS, "n_folds": 3, "nr_generation": 1,
            "random_state": 42, "modes": args.mode.split(","),
            "M": args.M, "jobs": args.jobs,
        }
    if args.full:
        return {
            "datasets": FULL_DATASETS, "n_folds": 10, "nr_generation": 20,
            "random_state": 42, "modes": args.mode.split(","),
            "M": args.M, "jobs": args.jobs,
        }
    raise SystemExit("Must pass one of --smoke / --full / --config")


def run_dir_name(repo_dir: Path) -> str:
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sha = git_sha(repo_dir)[:7]
    return f"{ts}_{sha}"


def main():
    p = argparse.ArgumentParser(description="Reproducible Oracle_N experiment runner.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="3 datasets, 3 folds, gen=1")
    g.add_argument("--full", action="store_true", help="31 datasets, 10 folds, gen=20")
    g.add_argument("--config", type=str, help="path to JSON config file")
    g.add_argument("--resume", type=str, metavar="DIR",
                   help="resume an interrupted run in this output directory")
    p.add_argument("--mode", type=str, default="ga,rf",
                    help="comma-separated modes: ga,rf,bagging (default: ga,rf)")
    p.add_argument("--M", type=int, default=100, help="pool size for RF mode (default: 100)")
    p.add_argument("--jobs", type=int, default=1,
                   help="DEAP parallel jobs for GA mode (default: 1 — safest for low-core CPUs)")
    p.add_argument("--output", type=str, default=None,
                   help="output root (default: results/experiments/)")
    p.add_argument("--dry-run", action="store_true", help="print manifest, do not run")
    args = p.parse_args()

    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            raise SystemExit(f"resume dir not found: {resume_dir}")
        manifest_path = resume_dir / "run_manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"no run_manifest.json in {resume_dir} — not a valid run dir")
        config = json.loads(manifest_path.read_text())["config"]
        print(f"[resume] dir: {resume_dir}")
        print(f"[resume] config: {json.dumps(config)}")
        manifest = record_run(config, resume_dir, resume=True)
        print(f"[done] {manifest['n_summary_rows']} rows in summary.csv")
        print(f"[done] git_sha: {manifest['git_sha']}")
        return

    config = build_config(args)
    output_root = Path(args.output) if args.output else REPO_DIR / "results" / "experiments"
    out_name = run_dir_name(REPO_DIR)
    out_dir = output_root / out_name

    if args.dry_run:
        config["_dry_run_dir"] = str(out_dir)
        print(json.dumps(config, indent=2))
        return

    print(f"[run] output: {out_dir}")
    print(f"[run] config: {json.dumps(config)}")
    manifest = record_run(config, out_dir)
    print(f"[done] {manifest['n_summary_rows']} rows in summary.csv")
    print(f"[done] git_sha: {manifest['git_sha']}")
    print(f"[done] deps: {json.dumps(manifest['deps_versions'])}")


if __name__ == "__main__":
    main()
