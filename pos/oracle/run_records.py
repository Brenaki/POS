"""Per-fold and per-run record builders: artifacts, manifest, summary row.

Split out of `run_helpers` in ADR 0018 — giving every fuser five metrics
instead of one pushed that module past the 150 LOC cap (ADR 0002).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pos.oracle.des_comparison import des_columns
from pos.oracle.metrics import metric_columns
from pos.oracle.run_helpers import (
    deps_versions,
    git_branch,
    git_dirty,
    git_sha,
    indices_hash,
    platform_str,
    python_version,
)


def save_fold_artifacts(fold_dir, metrics, pool, X_test, y_test, mode) -> None:
    """Save correctness_matrix_<mode>.npy + predictions_<mode>.npz to fold_dir.

    The filenames carry the mode: without it, running `--mode ga,bagging,rf`
    made every mode overwrite the previous one's artifacts in the same fold
    directory, so only the last mode survived on disk (ADR 0014).
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.save(fold_dir / f"correctness_matrix_{mode}.npy", metrics["correctness_matrix"])
    preds = np.array([clf.predict(X_test) for clf in pool])
    # Storing only the DES accuracy scalar is what forced the ADR 0018 rerun;
    # the predictions make any future metric an offline computation.
    arrays = {f"des_{k}": v for k, v in (metrics.get("des_preds") or {}).items()}
    arrays["preds"] = preds
    arrays["y_test"] = y_test
    try:
        arrays["probs"] = np.array([clf.predict_proba(X_test) for clf in pool])
    except (AttributeError, ValueError):
        pass
    np.savez(fold_dir / f"predictions_{mode}.npz", **arrays)


def build_fold_manifest(ds_name, fold_idx, mode, metrics, random_state,
                        n_train, n_val, n_test, train_idx, test_idx) -> dict:
    """Build the fold_manifest dict."""
    return {
        "dataset": ds_name, "fold_idx": fold_idx, "mode": mode,
        "random_state": random_state, "val_seed": random_state + fold_idx,
        "n_train": int(n_train), "n_val": int(n_val), "n_test": int(n_test),
        "n_dsel": metrics.get("n_dsel"), "k_eff": metrics.get("k_eff"),
        "M": metrics["n_classifiers"],
        "individual_accuracies": metrics["individual_accuracies"],
        "oracle_curve": metrics["oracle_curve"],
        "majority_vote": metrics["majority_vote"],
        "mean_probs": metrics["mean_probs"],
        "soft_fusion_rule": metrics.get("soft_fusion_rule", "none"),
        "double_fault_mean": metrics["double_fault_mean"],
        "oracle_1_balanced": metrics.get("oracle_1_balanced"),
        "majority_vote_metrics": metrics.get("majority_vote_metrics", {}),
        "mean_probs_metrics": metrics.get("mean_probs_metrics"),
        "des": metrics.get("des", {}),
        "des_notes": metrics.get("des_notes", {}),
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
        "soft_fusion_rule": metrics.get("soft_fusion_rule", "none"),
        "double_fault_mean": metrics["double_fault_mean"],
        "mean_individual_acc": float(np.mean(metrics["individual_accuracies"])),
        "oracle_1_balanced": metrics.get("oracle_1_balanced"),
        "n_dsel": metrics.get("n_dsel"), "k_eff": metrics.get("k_eff"),
    }
    for n in range(1, 6):
        row[f"oracle_{n}"] = curve[n - 1] if len(curve) >= n else None
    row.update(metric_columns("mv", metrics.get("majority_vote_metrics")))
    row.update(metric_columns("soft", metrics.get("mean_probs_metrics")))
    row.update(des_columns(metrics.get("des")))
    return row


def empty_summary_columns() -> list[str]:
    """Canonical summary.csv header, for the run that produced no rows.

    Derived from the same builders as a real row — hand-maintaining it went
    from awkward to unworkable once every fuser gained five metrics.
    """
    metrics = {
        "n_classifiers": 0, "oracle_curve": [], "majority_vote": None,
        "mean_probs": None, "double_fault_mean": None,
        "individual_accuracies": [float("nan")], "des": {},
    }
    return list(build_summary_row("", 0, "", metrics, 0))


def build_run_manifest(config: dict[str, Any], repo_dir: Path) -> dict[str, Any]:
    return {
        "timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(repo_dir), "git_branch": git_branch(repo_dir),
        "git_dirty": git_dirty(repo_dir),
        "python_version": python_version(), "platform": platform_str(),
        "config": config, "deps_versions": deps_versions(),
        "protocol_ref": "docs/protocol.md",
        "adr_ref": "docs/adr/0009-experiment-reproducibility.md",
    }
