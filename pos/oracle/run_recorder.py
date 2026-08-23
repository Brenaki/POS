"""Run recorder: persist a reproducible Oracle_N experiment to disk.

Orchestrates 10-fold stratified CV for datasets x pool modes (ga, rf).
Per fold saves correctness_matrix.npy, predictions.npz, fold_manifest_<mode>.json.
Global: run_manifest.json, summary.csv, per_dataset_summary.csv.

ADR 0006 (protocol), ADR 0009 (reproducibility layout).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pos.oracle.arff_loader import load_arff_dataset
from pos.oracle.pool_evaluation import evaluate_pool
from pos.oracle.resume_helpers import completed_folds, load_existing_summary
from pos.oracle.run_helpers import (
    build_fold_manifest,
    build_pool_bagging,
    build_pool_ga,
    build_pool_rf,
    build_summary_row,
    deps_versions,
    git_branch,
    git_sha,
    per_dataset_summary,
    platform_str,
    python_version,
    save_fold_artifacts,
)


def _build_manifest(config: dict[str, Any], repo_dir: Path) -> dict[str, Any]:
    return {
        "timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(repo_dir), "git_branch": git_branch(repo_dir),
        "python_version": python_version(), "platform": platform_str(),
        "config": config, "deps_versions": deps_versions(),
        "protocol_ref": "docs/protocol.md",
        "adr_ref": "docs/adr/0009-experiment-reproducibility.md",
    }


def _run_fold(X_tr, y_tr, X_val, y_val, X_test, y_test, mode, M, nr_gen, rs, jobs=1):
    """Build pool and evaluate. Returns (metrics, pool) or (None, None)."""
    if mode == "ga":
        pool = build_pool_ga(X_tr, y_tr, X_val, y_val, nr_gen, rs, jobs=jobs)
    elif mode == "rf":
        pool = build_pool_rf(X_tr, y_tr, M, rs)
    elif mode == "bagging":
        pool = build_pool_bagging(X_tr, y_tr, M, rs)
    else:
        return None, None
    if len(pool) == 0:
        return None, None
    metrics = evaluate_pool(pool, X_test, y_test)
    return metrics, pool


def record_run(config: dict[str, Any], output_dir: Path | str,
               resume: bool = False) -> dict[str, Any]:
    """Run experiments per config and persist artifacts. Returns run_manifest.

    If resume=True, skips (dataset, fold, mode) tuples that already have a
    fold_manifest_<mode>.json, and appends to any existing summary.csv.
    """
    repo_dir = Path(__file__).resolve().parents[2]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[str] = config["datasets"]
    n_folds: int = config["n_folds"]
    nr_generation: int = config["nr_generation"]
    random_state: int = config["random_state"]
    modes: list[str] = config["modes"]
    M: int = config.get("M", 100)
    jobs: int = config.get("jobs", 1)
    dataset_dir = Path(config.get("dataset_dir", repo_dir / "Dataset"))

    manifest = _build_manifest(config, repo_dir)
    done = completed_folds(output_dir) if resume else set()
    summary_rows = load_existing_summary(output_dir) if resume else []
    if resume and done:
        print(f"[resume] {len(done)} folds already completed — skipping")

    for ds_name in datasets:
        ds_path = dataset_dir / f"{ds_name}.arff"
        if not ds_path.exists():
            print(f"[skip] {ds_name}: not found at {ds_path}")
            continue
        X, y = load_arff_dataset(ds_path)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            n_val = max(1, len(X_train) // 5)
            rng = np.random.default_rng(random_state + fold_idx)
            val_idx = rng.choice(len(X_train), size=n_val, replace=False)
            mask = np.ones(len(X_train), dtype=bool)
            mask[val_idx] = False
            X_val, y_val = X_train[~mask], y_train[~mask]
            X_tr, y_tr = X_train[mask], y_train[mask]

            for mode in modes:
                if (ds_name, fold_idx, mode) in done:
                    continue
                try:
                    metrics, pool = _run_fold(X_tr, y_tr, X_val, y_val, X_test, y_test,
                                              mode, M, nr_generation, random_state, jobs=jobs)
                except Exception as exc:
                    print(f"[error] {ds_name} fold={fold_idx} mode={mode}: {type(exc).__name__}: {exc}")
                    manifest.setdefault("errors", []).append({
                        "dataset": ds_name, "fold": fold_idx, "mode": mode,
                        "error_type": type(exc).__name__, "error_msg": str(exc)[:500],
                    })
                    continue
                if metrics is None:
                    continue
                fold_dir = output_dir / ds_name / f"fold_{fold_idx}"
                save_fold_artifacts(fold_dir, metrics, pool, X_test, y_test)

                fm = build_fold_manifest(
                    ds_name, fold_idx, mode, metrics, random_state,
                    len(X_tr), len(X_val), len(y_test), train_idx, test_idx,
                )
                (fold_dir / f"fold_manifest_{mode}.json").write_text(json.dumps(fm, indent=2))
                summary_rows.append(build_summary_row(ds_name, fold_idx, mode, metrics, len(y_test)))

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        (output_dir / "summary.csv").write_text(
            "dataset,fold,mode,M,n_test,oracle_1,oracle_2,oracle_3,oracle_4,"
            "oracle_5,oracle_M,oracle_curve_json,majority_vote,mean_probs,"
            "double_fault_mean,mean_individual_acc\n")
    else:
        summary_df.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(per_dataset_summary(summary_df)).to_csv(
        output_dir / "per_dataset_summary.csv", index=False
    )
    manifest["n_summary_rows"] = len(summary_rows)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
