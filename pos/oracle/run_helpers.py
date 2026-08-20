"""Helpers for run_recorder: git info, deps versions, pool builders, hashing.

Extracted from run_recorder.py to satisfy the <=150 LOC file cap.
"""

from __future__ import annotations

import hashlib
import json
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


def build_pool_ga(X_train, y_train, X_val, y_val, nr_generation, random_state, jobs=1):
    """Build a pool via poolGeneration GA (legacy API). jobs=1 = serial."""
    from pool_generation import poolGeneration

    pg = poolGeneration(nr_generation=nr_generation, iteration=1, classifier="tree",
                       types=["F1", "T1"], jobs=jobs)
    pg.generate(X_train, y_train, X_val, y_val, iteration=1)
    return pg.get_pool()


def build_pool_rf(X_train, y_train, M, random_state):
    """Build a pool via RandomForestClassifier (baseline without GA)."""
    from pos.oracle.random_forest_pool import build_rf_pool
    return build_rf_pool(X_train, y_train, M=M, random_state=random_state)


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


def save_fold_artifacts(fold_dir, metrics, pool, X_test, y_test) -> None:
    """Save correctness_matrix.npy + predictions.npz to fold_dir."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.save(fold_dir / "correctness_matrix.npy", metrics["correctness_matrix"])
    preds = np.array([clf.predict(X_test) for clf in pool])
    try:
        probs = np.array([clf.predict_proba(X_test) for clf in pool])
        np.savez(fold_dir / "predictions.npz", preds=preds, probs=probs, y_test=y_test)
    except (AttributeError, ValueError):
        np.savez(fold_dir / "predictions.npz", preds=preds, y_test=y_test)


def build_fold_manifest(ds_name, fold_idx, mode, metrics, random_state,
                        n_train, n_val, n_test, train_idx, test_idx) -> dict:
    """Build the fold_manifest dict."""
    return {
        "dataset": ds_name, "fold_idx": fold_idx, "mode": mode,
        "random_state": random_state, "val_seed": random_state + fold_idx,
        "n_train": int(n_train), "n_val": int(n_val), "n_test": int(n_test),
        "M": metrics["n_classifiers"],
        "individual_accuracies": metrics["individual_accuracies"],
        "oracle_curve": metrics["oracle_curve"],
        "majority_vote": metrics["majority_vote"],
        "mean_probs": metrics["mean_probs"],
        "double_fault_mean": metrics["double_fault_mean"],
        "train_indices_hash": indices_hash(train_idx),
        "test_indices_hash": indices_hash(test_idx),
    }


def build_summary_row(ds_name, fold_idx, mode, metrics, n_test) -> dict:
    """Build one summary.csv row dict. Oracle_1..5 from curve indices 0..4."""
    curve = metrics["oracle_curve"]
    row = {
        "dataset": ds_name, "fold": fold_idx, "mode": mode,
        "M": metrics["n_classifiers"], "n_test": int(n_test),
        "oracle_M": curve[-1] if curve else None,
        "oracle_curve_json": json.dumps(curve),
        "majority_vote": metrics["majority_vote"],
        "mean_probs": metrics["mean_probs"],
        "double_fault_mean": metrics["double_fault_mean"],
        "mean_individual_acc": float(np.mean(metrics["individual_accuracies"])),
    }
    for n in range(1, 6):
        row[f"oracle_{n}"] = curve[n - 1] if len(curve) >= n else None
    return row
