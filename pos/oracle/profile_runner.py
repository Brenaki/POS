"""Profile-aware experiment runner (ADR 0020 T13).

Dispatches a profile config to its splitter and treatment builders, replacing
the old `--mode` dispatch:

- `pos-oracle-v2`: 10-fold StratifiedKFold, then a disjoint three-way split of
  each training fold (pool training / PGDCS fitness / DSEL);
- `monteiro-2023-dt-compatible-v1`: 20 independent stratified holdouts
  50/25/25, a new split per replication.

Treatments are exactly `pgdcs_a`, `pgdcs_d`, `bagging`. PGDCS_A and PGDCS_D
share one evolution trajectory per split (`shared_evolution_trajectory`) and
diverge only at the exit criterion, which is where Monteiro 2023 separates
them too.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from pos.oracle.bagging_pool import build_bagging_pool_v2
from pos.oracle.fold_splitter import stratified_three_way_split
from pos.oracle.monteiro_splitter import monteiro_holdout_splits
from pos.oracle.pgdcs_treatment import build_pgdcs_result
from pos.oracle.profiles import MONTEIRO_PROFILES
from pos.oracle.summary_v2 import build_summary_row_v2
from pos.oracle.treatment_eval import evaluate_treatment


def profile_splits(config: dict[str, Any], X, y) -> list[dict[str, Any]]:
    """Per-profile list of splits, each with its own seed and index."""
    profile = config["profile"]
    if profile == "pos-oracle-v2":
        rs = config.get("random_state", 42)
        skf = StratifiedKFold(n_splits=config.get("n_folds", 10),
                              shuffle=True, random_state=rs)
        out = []
        for i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            X_tr0, y_tr0 = X[tr_idx], y[tr_idx]
            a, b, c, d, e, f = stratified_three_way_split(
                X_tr0, y_tr0, config.get("val_frac", 0.2), rs + i)
            out.append({"index": i, "seed": rs + i,
                        "X_train": a, "y_train": b, "X_val": c, "y_val": d,
                        "X_dsel": e, "y_dsel": f,
                        "X_test": X[te_idx], "y_test": y[te_idx],
                        "train_idx": tr_idx, "test_idx": te_idx})
        return out
    if profile in MONTEIRO_PROFILES:
        root = config.get("reproducibility_root_seed", 86)
        splits = monteiro_holdout_splits(
            X, y, n_replications=config.get("n_replications", 20),
            root_seed=root, train_frac=config.get("train_frac", 0.5),
            val_frac=config.get("val_frac", 0.25))
        for i, s in enumerate(splits):
            s["index"] = i
            s["seed"] = root + i
        return splits
    raise ValueError(f"unknown profile: {config['profile']!r}")


def build_treatment_pool(treatment, X_tr, y_tr, cfg, pgdcs=None):
    """Return (fitted pool, extra summary fields) for one treatment."""
    if treatment == "bagging":
        rs = cfg.get("bagging_random_state", cfg.get("random_state", 86))
        return build_bagging_pool_v2(X_tr, y_tr, M=cfg["M"], random_state=rs), {}
    if treatment in ("pgdcs_a", "pgdcs_d"):
        if pgdcs is None:
            raise ValueError(f"{treatment} needs a PGDCS trajectory")
        key, metric = ("best_a", "mvr") if treatment == "pgdcs_a" else ("best_d", "gdisp")
        best = pgdcs[key]
        return best["pool"], {
            "pgdcs_types": pgdcs.get("selected_measures", []),
            "pgdcs_best_generation": best["gen"],
            "pgdcs_best_score": best[metric],
            "pgdcs_best_metric": metric,
        }
    raise ValueError(f"unknown treatment: {treatment}")


def run_profile_experiment(
    config: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    build_pgdcs_fn=build_pgdcs_result,
    on_row=None,
    skip_fn=None,
) -> list[dict[str, Any]]:
    """Run one dataset through the profile and return summary v2 rows.

    `build_pgdcs_fn(X_tr, y_tr, X_val, y_val, cfg) -> {best_a, best_d,
    selected_measures}`; None skips the PGDCS treatments. `on_row(row, split,
    treatment, metrics, pool)` is called after every evaluated treatment so a
    recorder can persist per-fold artifacts without re-running anything.
    `skip_fn(index, treatment) -> bool` lets a resuming recorder drop work
    that is already on disk; a split whose treatments are all skipped never
    runs its PGDCS trajectory.
    """
    profile = config["profile"]
    treatments = list(config["treatments"])
    des_methods = config.get("des_methods", [])
    needs_pgdcs = any(t.startswith("pgdcs") for t in treatments)
    rows: list[dict[str, Any]] = []

    for split in profile_splits(config, X, y):
        todo = [t for t in treatments
                if skip_fn is None or not skip_fn(split["index"], t)]
        if not todo:
            continue
        cfg = dict(config, _seed=split["seed"])
        pgdcs = None
        if needs_pgdcs and any(t.startswith("pgdcs") for t in todo) \
                and build_pgdcs_fn is not None:
            pgdcs = build_pgdcs_fn(split["X_train"], split["y_train"],
                                   split["X_val"], split["y_val"], cfg)
        for treatment in todo:
            if treatment.startswith("pgdcs") and pgdcs is None:
                continue
            pool, extra = build_treatment_pool(
                treatment, split["X_train"], split["y_train"], cfg, pgdcs)
            metrics = evaluate_treatment(
                pool, split["X_test"], split["y_test"],
                split["X_dsel"], split["y_dsel"], des_methods, split["seed"])
            metrics.update(extra)
            row = build_summary_row_v2(
                profile=profile, dataset=dataset_name, fold=split["index"],
                treatment=treatment, metrics=metrics,
                n_test=len(split["y_test"]))
            rows.append(row)
            if on_row is not None:
                on_row(row, split, treatment, metrics, pool)
    return rows
