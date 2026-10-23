"""Run recorder for the ADR 0020 profiles (schema v2).

Replaces the `--mode` recorder for the two scientific profiles. Per dataset:

1. gate the snapshot on the profile's dataset reference (ADR 0021): Table 2
   with the recorded reconciliation for the Monteiro profile, the SHA-256
   catalog `Dataset/pos_catalog.json` for `pos-oracle-v2`. A dataset that
   fails is skipped unless `allow_unverified_data` is set, which also marks
   the whole run `comparable=false`;
2. vote the complexity measures once per dataset (Monteiro profile, whose
   Table 4 is per dataset) or per fold (POS profile);
3. run every split x treatment, persisting per-split artifacts as it goes.

Resume skips any (dataset, split, treatment) that already has a fold
manifest on disk, and never re-runs the PGDCS trajectory of a split whose
treatments are all done. `summary.csv` is flushed after every dataset, so an
interrupted run keeps the rows it already produced — resume detects finished
work from the fold manifests and would otherwise skip it without being able
to recover its rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pos.oracle.arff_loader import load_arff_dataset
from pos.oracle.conformance_catalog import PAPER_TABLE_2
from pos.oracle.dataset_gate import gate_decision, load_pos_catalog
from pos.oracle.pgdcs_treatment import select_measures
from pos.oracle.profile_runner import profile_splits, run_profile_experiment
from pos.oracle.resume_helpers import completed_folds, load_existing_summary
from pos.oracle.run_records import save_fold_artifacts
from pos.oracle.run_records_v2 import (
    build_run_manifest_v2,
    dataset_path,
    fold_manifest_v2,
    per_dataset_summary_v2,
    survey_datasets,
)
from pos.oracle.summary_v2 import empty_summary_columns_v2

DEFAULT_DATASETS = tuple(d["name"] for d in PAPER_TABLE_2)


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]],
                   manifest: dict[str, Any]) -> None:
    """Flush summary, per-dataset aggregation and manifest to disk."""
    df = pd.DataFrame(rows)
    if df.empty:
        (output_dir / "summary.csv").write_text(
            ",".join(empty_summary_columns_v2()) + "\n")
    else:
        df.to_csv(output_dir / "summary.csv", index=False)
        pd.DataFrame(per_dataset_summary_v2(df)).to_csv(
            output_dir / "per_dataset_summary.csv", index=False)
    manifest["n_summary_rows"] = len(rows)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))


def _dataset_measures(config, X, y):
    """Vote the two measures once, on the first split's training partition."""
    first = profile_splits(config, X, y)[0]
    return select_measures(first["X_train"], first["y_train"], config,
                           first["seed"])


def record_run_v2(config: dict[str, Any], output_dir: Path | str,
                  resume: bool = False) -> dict[str, Any]:
    """Run every dataset under the profile in `config`; return the manifest."""
    repo_dir = Path(__file__).resolve().parents[2]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(config.get("datasets") or DEFAULT_DATASETS)
    dataset_dir = Path(config.get("dataset_dir", repo_dir / "Dataset"))
    allow_unverified = bool(config.get("allow_unverified_data", False))
    catalog = load_pos_catalog(dataset_dir)

    conformance = survey_datasets(datasets, dataset_dir)
    for record in conformance.values():
        record.update(gate_decision(record, config.get("profile"),
                                    allow_unverified, catalog))
    manifest = build_run_manifest_v2(config, repo_dir, conformance)
    done = completed_folds(output_dir) if resume else set()
    rows = load_existing_summary(output_dir) if resume else []
    if resume and done:
        print(f"[resume] {len(done)} (dataset, split, treatment) already done")

    for name in datasets:
        gate = conformance[name]
        if not gate["gate_passes"]:
            print(f"[skip] {name}: {gate['gate_reference']}={gate['gate_status']}")
            manifest.setdefault("skipped_datasets", []).append(
                {"dataset": name, "reason": gate["gate_reason"],
                 "conformance_status": gate["gate_status"],
                 "gate_reference": gate["gate_reference"]})
            continue
        path = dataset_path(name, dataset_dir)
        X, y = load_arff_dataset(path)
        cfg = dict(config)
        n_splits = cfg.get("n_folds") or cfg.get("n_replications") or 0
        pending_pgdcs = any(
            (name, i, t) not in done
            for i in range(n_splits)
            for t in cfg["treatments"] if t.startswith("pgdcs")
        )
        if cfg.get("measure_selection") == "once_per_dataset" and pending_pgdcs:
            cfg["_selected_measures"] = _dataset_measures(cfg, X, y)
            print(f"[{name}] measures: {cfg['_selected_measures']}")

        def _on_row(row, split, treatment, metrics, pool, _ds=name, _p=cfg["profile"]):
            fold_dir = output_dir / _ds / f"fold_{split['index']}"
            save_fold_artifacts(fold_dir, metrics, pool,
                                split["X_test"], split["y_test"], treatment)
            (fold_dir / f"fold_manifest_{treatment}.json").write_text(json.dumps(
                fold_manifest_v2(_ds, split["index"], treatment, metrics,
                                 split, _p), indent=2))

        try:
            rows.extend(run_profile_experiment(
                cfg, X, y, name, on_row=_on_row,
                skip_fn=lambda i, t, _ds=name: (_ds, i, t) in done))
        except Exception as exc:  # one bad dataset must not lose the run
            print(f"[error] {name}: {type(exc).__name__}: {exc}")
            manifest.setdefault("errors", []).append({
                "dataset": name, "error_type": type(exc).__name__,
                "error_msg": str(exc)[:500]})
            continue
        _write_outputs(output_dir, rows, manifest)
        print(f"[done] {name}: {len(rows)} rows so far")

    _write_outputs(output_dir, rows, manifest)
    return manifest
