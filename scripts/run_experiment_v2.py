"""CLI for reproducible Oracle_N experiments (ADR 0020).

Usage:
    python scripts/run_experiment.py --profile pos-oracle-v2 --full
    python scripts/run_experiment.py --profile monteiro-2023-dt-compatible-v1 --full
    python scripts/run_experiment.py --profile <profile> --smoke
    python scripts/run_experiment.py --resume <dir>
    python scripts/compare_reference.py <run_dir>

Output: results/experiments/<ISO_timestamp>_<git_sha_short>/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from pos.oracle.profiles import PROFILE_NAMES, SCIENTIFIC_TREATMENTS, materialize_profile  # noqa: E402
from pos.oracle.run_helpers import git_sha  # noqa: E402
from pos.oracle.run_recorder_v2 import DEFAULT_DATASETS, record_run_v2  # noqa: E402


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def run_dir_name(repo_dir: Path) -> str:
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sha = git_sha(repo_dir)[:7]
    return f"{ts}_{sha}"


def main():
    p = argparse.ArgumentParser(description="ADR 0020 experiment runner.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--profile", type=str, choices=list(PROFILE_NAMES),
                   help=f"profile to run: {', '.join(PROFILE_NAMES)}")
    g.add_argument("--resume", type=str, metavar="DIR",
                   help="resume an interrupted run in this output directory")
    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--full", action="store_true", help="full scientific run")
    scale.add_argument("--smoke", action="store_true", help="reduced smoke test")
    p.add_argument("--treatments", type=str,
                   default=",".join(SCIENTIFIC_TREATMENTS),
                   help=f"comma-separated treatments (default: {','.join(SCIENTIFIC_TREATMENTS)})")
    p.add_argument("--jobs", type=int, default=default_jobs(),
                   help=f"parallel bag evaluations (default: {default_jobs()})")
    p.add_argument("--no-des", action="store_true",
                   help="skip DCS/DES baselines")
    p.add_argument("--allow-unverified-data", action="store_true",
                   help="run on non-conformant datasets (marks comparable=false)")
    p.add_argument("--datasets", type=str, default=None,
                   help="comma-separated dataset names (default: all 28 of Table 2)")
    p.add_argument("--output", type=str, default=None,
                   help="output root (default: results/experiments/)")
    p.add_argument("--dry-run", action="store_true", help="print config, do not run")
    args = p.parse_args()

    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            raise SystemExit(f"resume dir not found: {resume_dir}")
        manifest_path = resume_dir / "run_manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"no run_manifest.json in {resume_dir}")
        config = json.loads(manifest_path.read_text())["config"]
        print(f"[resume] dir: {resume_dir}")
        print(f"[resume] profile: {config.get('profile')}")
        manifest = record_run_v2(config, resume_dir, resume=True)
        print(f"[done] {manifest['n_summary_rows']} rows in summary.csv")
        return

    if not args.full and not args.smoke:
        raise SystemExit("must pass one of --full / --smoke")

    scale_val = "smoke" if args.smoke else "full"
    treatments = [t.strip() for t in args.treatments.split(",")]
    for t in treatments:
        if t not in SCIENTIFIC_TREATMENTS:
            raise SystemExit(f"invalid treatment {t!r}; choose from {SCIENTIFIC_TREATMENTS}")

    config = materialize_profile(args.profile, scale=scale_val)
    config["treatments"] = treatments
    config["datasets"] = ([d.strip() for d in args.datasets.split(",")]
                          if args.datasets else list(DEFAULT_DATASETS))
    config["jobs"] = args.jobs
    config["des_methods"] = [] if args.no_des else list(config["des_methods"])
    if args.allow_unverified_data:
        config["allow_unverified_data"] = True
        config["comparable"] = False

    output_root = Path(args.output) if args.output else REPO_DIR / "results" / "experiments"
    out_dir = output_root / run_dir_name(REPO_DIR)

    if args.dry_run:
        config["_dry_run_dir"] = str(out_dir)
        print(json.dumps(config, indent=2, default=str))
        return

    print(f"[run] profile: {args.profile}")
    print(f"[run] scale: {scale_val}")
    print(f"[run] treatments: {treatments}")
    print(f"[run] comparable: {config['comparable']}")
    print(f"[run] datasets: {len(config['datasets'])}")
    print(f"[run] output: {out_dir}")
    manifest = record_run_v2(config, out_dir)
    print(f"[done] {manifest['n_summary_rows']} rows in summary.csv")
    print(f"[done] git_sha: {manifest['git_sha']}")


if __name__ == "__main__":
    main()