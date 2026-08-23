"""Helpers for run_recorder: git info, deps versions, pool builders, hashing.

Extracted from run_recorder.py to satisfy the <=150 LOC file cap.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def git_sha(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def git_branch(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def git_dirty(repo_dir: Path) -> bool:
    """True when the run was launched from an uncommitted working tree.

    Without this the manifest's `git_sha` silently claims a provenance the
    code does not have.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


def deps_versions() -> dict[str, str]:
    mods = ["sklearn", "numpy", "deap", "deslib", "pandas", "scipy", "pyhard"]
    versions: dict[str, str] = {}
    for m in mods:
        try:
            mod = __import__(m)
            versions[m] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[m] = "not-installed"
    return versions


def indices_hash(idx: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(idx.tobytes()).hexdigest()[:16]


def build_pool_ga(X_train, y_train, X_val, y_val, nr_generation, random_state,
                  jobs=1, classifier="perc"):
    """Build a pool via poolGeneration GA (legacy API). jobs=1 = serial.

    `classifier` defaults to the linear Perceptron used by the reference
    thesis (Monteiro et al. 2022, sec. 5) and named in the subproject's
    Objective 4. Pass "tree" for the DecisionTree variant (ADR 0015).
    """
    from pool_generation import poolGeneration

    pg = poolGeneration(nr_generation=nr_generation, iteration=1, classifier=classifier,
                       types=["F1", "T1"], jobs=jobs, random_state=random_state)
    pg.generate(X_train, y_train, X_val, y_val, iteration=1)
    return pg.get_pool()


def build_pool_rf(X_train, y_train, M, random_state):
    """Build a pool via RandomForestClassifier (baseline without GA)."""
    from pos.oracle.random_forest_pool import build_rf_pool
    return build_rf_pool(X_train, y_train, M=M, random_state=random_state)


def build_pool_bagging(X_train, y_train, M, random_state):
    """Build a pool via BaggingClassifier (controlled baseline, all features)."""
    from pos.oracle.bagging_pool import build_bagging_pool
    return build_bagging_pool(X_train, y_train, M=M, random_state=random_state)


def python_version() -> str:
    return sys.version.split()[0]


def platform_str() -> str:
    return platform.platform()


def per_dataset_summary(summary_df) -> list[dict]:
    """Aggregate summary rows by (dataset, mode) → mean±std metrics."""
    if summary_df.empty or "dataset" not in summary_df.columns:
        return []
    rows: list[dict] = []
    oracle_cols = [f"oracle_{n}" for n in range(1, 6)] + ["oracle_M"]
    for (ds, mode), grp in summary_df.groupby(["dataset", "mode"]):
        row: dict = {
            "dataset": ds, "mode": mode, "n_folds": len(grp),
            "majority_vote_mean": grp["majority_vote"].mean(),
            "majority_vote_std": grp["majority_vote"].std(),
            "mean_probs_mean": grp["mean_probs"].mean(),
            "mean_probs_std": grp["mean_probs"].std(),
            "mean_individual_acc_mean": grp["mean_individual_acc"].mean(),
        }
        for col in oracle_cols:
            if col in grp.columns:
                row[f"{col}_mean"] = grp[col].mean()
                row[f"{col}_std"] = grp[col].std()
        rows.append(row)
    return rows
