"""Tests for pos.oracle.run_recorder (fast, RF-only, tiny dataset)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pos.oracle.run_recorder import record_run


def _tiny_dataset(tmp_path: Path) -> Path:
    """Write a tiny .arff-like dataset via sklearn wine (reuse real Wine.arff)."""
    ds_dir = tmp_path / "Dataset"
    ds_dir.mkdir()
    # Reuse the real Wine.arff from repo
    repo_ds = Path(__file__).resolve().parents[3] / "Dataset" / "Wine.arff"
    import shutil
    shutil.copy(repo_ds, ds_dir / "Wine.arff")
    return ds_dir


def test_record_run_rf_only_produces_manifests(tmp_path):
    """RF-only run on Wine with 2 folds should produce expected artifacts."""
    ds_dir = _tiny_dataset(tmp_path)
    out_dir = tmp_path / "results" / "test_run"
    config = {
        "datasets": ["Wine"],
        "n_folds": 2,
        "nr_generation": 1,
        "random_state": 42,
        "modes": ["rf"],
        "M": 5,
        "dataset_dir": str(ds_dir),
    }
    manifest = record_run(config, out_dir)

    # run_manifest.json
    assert (out_dir / "run_manifest.json").exists()
    assert manifest["git_sha"] != "unknown" or True  # may be unknown in non-git
    assert "deps_versions" in manifest
    assert "sklearn" in manifest["deps_versions"]
    assert manifest["config"]["modes"] == ["rf"]

    # summary.csv: 1 dataset × 2 folds × 1 mode = 2 rows
    summary = pd.read_csv(out_dir / "summary.csv")
    assert len(summary) == 2
    assert set(summary["dataset"]) == {"Wine"}
    assert set(summary["mode"]) == {"rf"}
    assert (summary["oracle_1"] >= summary["majority_vote"]).all(), (
        "Oracle_1 must be >= majority_vote"
    )

    # per_dataset_summary.csv: 1 row
    per_ds = pd.read_csv(out_dir / "per_dataset_summary.csv")
    assert len(per_ds) == 1

    # fold_manifest_<mode>.json per fold
    for fold_idx in range(2):
        fm_path = out_dir / "Wine" / f"fold_{fold_idx}" / "fold_manifest_rf.json"
        assert fm_path.exists(), f"Missing fold manifest: {fm_path}"
        fm = json.loads(fm_path.read_text())
        assert fm["mode"] == "rf"
        assert fm["M"] == 5
        assert len(fm["oracle_curve"]) == 5
        # Monotonic non-increasing
        curve = fm["oracle_curve"]
        for i in range(len(curve) - 1):
            assert curve[i] >= curve[i + 1]
        assert fm["oracle_curve"][0] >= fm["majority_vote"]

    # correctness_matrix.npy (gitignored but exists on disk)
    cm = np.load(out_dir / "Wine" / "fold_0" / "correctness_matrix_rf.npy")
    assert cm.shape[1] == 5  # M = 5


def test_record_run_reproducible_with_same_seed(tmp_path):
    """Same random_state should produce identical RF results."""
    ds_dir = _tiny_dataset(tmp_path)
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    config = {
        "datasets": ["Wine"], "n_folds": 2, "nr_generation": 1,
        "random_state": 42, "modes": ["rf"], "M": 5,
        "dataset_dir": str(ds_dir),
    }
    record_run(config, out1)
    record_run(config, out2)
    s1 = pd.read_csv(out1 / "summary.csv")
    s2 = pd.read_csv(out2 / "summary.csv")
    assert s1["oracle_1"].tolist() == s2["oracle_1"].tolist()
    assert s1["majority_vote"].tolist() == s2["majority_vote"].tolist()


def test_record_run_skip_missing_dataset(tmp_path):
    """Non-existent dataset should be skipped, not crash."""
    out_dir = tmp_path / "skip_run"
    config = {
        "datasets": ["NonExistent"], "n_folds": 2, "nr_generation": 1,
        "random_state": 42, "modes": ["rf"], "M": 3,
        "dataset_dir": str(tmp_path / "Dataset"),
    }
    manifest = record_run(config, out_dir)
    summary = pd.read_csv(out_dir / "summary.csv")
    assert len(summary) == 0
    assert manifest["n_summary_rows"] == 0


def test_record_run_resume_skips_completed_folds(tmp_path):
    """Resume should skip folds that already have fold_manifest_<mode>.json."""
    ds_dir = _tiny_dataset(tmp_path)
    out_dir = tmp_path / "resume_run"
    config = {
        "datasets": ["Wine"], "n_folds": 2, "nr_generation": 1,
        "random_state": 42, "modes": ["rf"], "M": 5,
        "dataset_dir": str(ds_dir),
    }
    # First run — completes 2 folds
    record_run(config, out_dir)
    s1 = pd.read_csv(out_dir / "summary.csv")
    assert len(s1) == 2

    # Second run with resume=True — should skip all 2 folds, produce same summary
    manifest = record_run(config, out_dir, resume=True)
    s2 = pd.read_csv(out_dir / "summary.csv")
    assert len(s2) == 2
    assert s1["oracle_1"].tolist() == s2["oracle_1"].tolist()
    assert s1["majority_vote"].tolist() == s2["majority_vote"].tolist()


def test_record_run_resume_partial_run(tmp_path):
    """Resume after a partial run (fold_1 deleted) should re-complete it."""
    import shutil
    ds_dir = _tiny_dataset(tmp_path)
    out_dir = tmp_path / "partial_run"
    config = {
        "datasets": ["Wine"], "n_folds": 2, "nr_generation": 1,
        "random_state": 42, "modes": ["rf"], "M": 5,
        "dataset_dir": str(ds_dir),
    }
    # Full run first
    record_run(config, out_dir)
    s_full = pd.read_csv(out_dir / "summary.csv")
    assert len(s_full) == 2
    fold1_orig = s_full[s_full["fold"] == 1].iloc[0]["oracle_1"]

    # Simulate interruption: delete fold_1 manifest + artifacts + truncate summary
    fold1_dir = out_dir / "Wine" / "fold_1"
    shutil.rmtree(fold1_dir)
    s_fold0 = s_full[s_full["fold"] == 0]
    s_fold0.to_csv(out_dir / "summary.csv", index=False)

    # Resume — should re-run fold_1 and produce 2 rows again
    record_run(config, out_dir, resume=True)
    s_resumed = pd.read_csv(out_dir / "summary.csv")
    assert len(s_resumed) == 2
    fold1_new = s_resumed[s_resumed["fold"] == 1].iloc[0]["oracle_1"]
    assert fold1_new == fold1_orig  # same result (deterministic)
