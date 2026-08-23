"""Run recorder: persist a reproducible Oracle_N experiment to disk.

Orchestrates 10-fold stratified CV for datasets x pool modes (ga, rf).
Per fold saves correctness_matrix_<mode>.npy, predictions_<mode>.npz and
fold_manifest_<mode>.json.
Global: run_manifest.json, summary.csv, per_dataset_summary.csv.

ADR 0006 (protocol), ADR 0009 (reproducibility layout).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pos.oracle.arff_loader import load_arff_dataset
from pos.oracle.des_comparison import effective_k, evaluate_des
from pos.oracle.des_methods import DES_METHODS
from pos.oracle.fold_splitter import (
    check_dataset_viability,
    stratified_three_way_split,
)
from pos.oracle.pool_evaluation import evaluate_pool
from pos.oracle.resume_helpers import completed_folds, load_existing_summary
from pos.oracle.run_helpers import (
    build_pool_bagging,
    build_pool_ga,
    build_pool_rf,
    per_dataset_summary,
)
from pos.oracle.run_records import (
    build_fold_manifest,
    build_run_manifest,
    build_summary_row,
    empty_summary_columns,
    save_fold_artifacts,
)

VAL_FRAC = 0.2


def _run_fold(X_tr, y_tr, X_val, y_val, X_dsel, y_dsel, X_test, y_test, mode,
              M, nr_gen, rs, jobs=1, base_classifier="perc", des_methods=()):
    """Build pool and evaluate. Returns (metrics, pool) or (None, None)."""
    if mode == "ga":
        pool = build_pool_ga(X_tr, y_tr, X_val, y_val, nr_gen, rs, jobs=jobs,
                             classifier=base_classifier)
    elif mode == "rf":
        pool = build_pool_rf(X_tr, y_tr, M, rs)
    elif mode == "bagging":
        pool = build_pool_bagging(X_tr, y_tr, M, rs)
    else:
        return None, None
    if len(pool) == 0:
        return None, None
    metrics = evaluate_pool(pool, X_test, y_test)
    if des_methods:
        # DSEL is disjoint from both pool training and the GA's fitness set
        # (ADR 0018) — under the old two-way split it was the fitness set.
        des, preds, notes = evaluate_des(pool, X_dsel, y_dsel, X_test, y_test,
                                         random_state=rs,
                                         methods=tuple(des_methods))
        metrics["des"], metrics["des_preds"], metrics["des_notes"] = des, preds, notes
        metrics["k_eff"] = effective_k(len(y_dsel))
        metrics["n_dsel"] = int(len(y_dsel))
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
    base_classifier: str = config.get("base_classifier", "perc")
    des_methods: list[str] = config.get("des_methods", list(DES_METHODS))
    dataset_dir = Path(config.get("dataset_dir", repo_dir / "Dataset"))

    manifest = build_run_manifest(config, repo_dir)
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
        ok, reason = check_dataset_viability(y, n_folds, modes, val_frac=VAL_FRAC)
        if not ok:
            print(f"[skip] {ds_name}: {reason}")
            manifest.setdefault("skipped_datasets", []).append(
                {"dataset": ds_name, "reason": reason})
            continue
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_tr, y_tr, X_val, y_val, X_dsel, y_dsel = stratified_three_way_split(
                X_train, y_train, VAL_FRAC, random_state + fold_idx)

            for mode in modes:
                if (ds_name, fold_idx, mode) in done:
                    continue
                try:
                    metrics, pool = _run_fold(X_tr, y_tr, X_val, y_val,
                                              X_dsel, y_dsel, X_test, y_test,
                                              mode, M, nr_generation, random_state,
                                              jobs=jobs, base_classifier=base_classifier,
                                              des_methods=des_methods)
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
                save_fold_artifacts(fold_dir, metrics, pool, X_test, y_test, mode)

                fm = build_fold_manifest(
                    ds_name, fold_idx, mode, metrics, random_state,
                    len(X_tr), len(X_val), len(y_test), train_idx, test_idx,
                )
                (fold_dir / f"fold_manifest_{mode}.json").write_text(json.dumps(fm, indent=2))
                summary_rows.append(build_summary_row(ds_name, fold_idx, mode, metrics, len(y_test)))

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        (output_dir / "summary.csv").write_text(
            ",".join(empty_summary_columns()) + "\n")
    else:
        summary_df.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(per_dataset_summary(summary_df)).to_csv(
        output_dir / "per_dataset_summary.csv", index=False
    )
    manifest["n_summary_rows"] = len(summary_rows)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
