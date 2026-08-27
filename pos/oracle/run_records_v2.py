"""Manifest and conformance records for schema v2 runs (ADR 0020).

`survey_datasets` produces the conformance-gated catalog the run manifest
carries: paper metadata from Table 2, the SHA-256 of the POS-approved
snapshot (the paper publishes no hashes), the conformance verdict and the
reconciliation note when one applies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pos.oracle.arff_loader import load_arff_dataset
from pos.oracle.conformance_catalog import conformance_status, file_sha256
from pos.oracle.dataset_reconciliation import dataset_stem, reconcile
from pos.oracle.run_helpers import (
    deps_versions,
    git_branch,
    git_dirty,
    git_sha,
    platform_str,
    python_version,
)


def dataset_path(name: str, dataset_dir: Path) -> Path:
    """Snapshot path for a paper dataset name (handles the Adult alias)."""
    return Path(dataset_dir) / f"{dataset_stem(name)}.arff"


def survey_dataset(name: str, dataset_dir: Path) -> dict[str, Any]:
    """Conformance record for one dataset, reconciliation applied."""
    path = dataset_path(name, dataset_dir)
    if not path.exists():
        return reconcile(conformance_status(name, None, None))
    X, y = load_arff_dataset(path)
    actual = {"instances": int(len(y)), "features": int(X.shape[1]),
              "classes": int(len(np.unique(y)))}
    record = conformance_status(name, actual, file_sha256(path))
    record.update({f"snapshot_{k}": v for k, v in actual.items()})
    record["snapshot_file"] = path.name
    return reconcile(record)


def survey_datasets(names, dataset_dir: Path) -> dict[str, dict[str, Any]]:
    """Conformance records for every requested dataset, keyed by paper name."""
    return {n: survey_dataset(n, dataset_dir) for n in names}


def build_run_manifest_v2(
    config: dict[str, Any], repo_dir: Path, conformance: dict[str, Any]
) -> dict[str, Any]:
    """Run manifest for a v2 run: config, provenance, conformance catalog."""
    return {
        "schema_version": 2,
        "profile": config.get("profile"),
        "scale": config.get("scale"),
        "comparable": config.get("comparable"),
        "fingerprint": config.get("fingerprint"),
        "config": config,
        "git_sha": git_sha(repo_dir),
        "git_branch": git_branch(repo_dir),
        "git_dirty": git_dirty(repo_dir),
        "python": python_version(),
        "platform": platform_str(),
        "deps": deps_versions(),
        "conformance": conformance,
    }


def fold_manifest_v2(
    dataset: str, index: int, treatment: str, metrics: dict[str, Any],
    split: dict[str, Any], profile: str,
) -> dict[str, Any]:
    """Per-(dataset, split, treatment) manifest. `mode` mirrors `treatment`
    so the v1 resume scanner keeps working unchanged."""
    return {
        "schema_version": 2,
        "profile": profile,
        "dataset": dataset,
        "fold_idx": index,
        "treatment": treatment,
        "mode": treatment,
        "seed": split.get("seed"),
        "n_train": int(len(split["y_train"])),
        "n_val": int(len(split["y_val"])),
        "n_dsel": int(len(split["y_dsel"])),
        "n_test": int(len(split["y_test"])),
        "M": metrics.get("n_classifiers"),
        "pgdcs_types": metrics.get("pgdcs_types"),
        "pgdcs_best_generation": metrics.get("pgdcs_best_generation"),
        "pgdcs_best_metric": metrics.get("pgdcs_best_metric"),
        "pgdcs_best_score": metrics.get("pgdcs_best_score"),
        "k_eff": metrics.get("k_eff"),
    }


def per_dataset_summary_v2(summary_df) -> list[dict[str, Any]]:
    """Aggregate v2 rows by (dataset, treatment) into mean/std per metric.

    This is the level the published tables live at: A.10 is mean/std of MVR
    over the replications, Table 6 is the mean/std of the selected
    generation, Table 7 is the mean Oracle of the pool.
    """
    if summary_df.empty or "treatment" not in summary_df.columns:
        return []
    from pos.oracle.profiles import ORACLE_CHECKPOINTS

    base = ["mv_acc", "mv_f1", "mv_balanced_acc", "mp_acc",
            "mean_individual_acc", "double_fault_mean",
            "pgdcs_best_generation", "pgdcs_best_score"]
    oracle = ["oracle_classic"]
    oracle += [f"oracle_{n}" for n in ORACLE_CHECKPOINTS]
    oracle += [f"oracle_greedy_{n}" for n in ORACLE_CHECKPOINTS]
    des = [c for c in summary_df.columns
           if c.startswith("des_") and c.count("_") == 1]
    rows: list[dict[str, Any]] = []
    for (ds, treatment), grp in summary_df.groupby(["dataset", "treatment"]):
        row: dict[str, Any] = {"dataset": ds, "treatment": treatment,
                               "n_splits": len(grp)}
        for col in base + oracle + des:
            if col in grp.columns:
                row[f"{col}_mean"] = grp[col].mean()
                row[f"{col}_std"] = grp[col].std()
        rows.append(row)
    return rows
