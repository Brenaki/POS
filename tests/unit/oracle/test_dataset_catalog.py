"""Tests for pos.oracle.dataset_catalog.

TDD — tests written BEFORE/AFTER implementation. Validates that every
ARFF dataset in POS/Dataset/ loads cleanly and has sane metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pos.oracle.dataset_catalog import (
    build_catalog,
    default_catalog_path,
    load_catalog,
    save_catalog,
)

DATASET_DIR = Path(__file__).resolve().parents[3] / "Dataset"


@pytest.fixture(scope="module")
def catalog():
    """Build the full catalog once per module (slow: loads 31 datasets)."""
    return build_catalog(DATASET_DIR)


def test_catalog_has_one_row_per_arff(catalog):
    """Catalog should have exactly as many rows as .arff files."""
    arff_count = len(list(DATASET_DIR.glob("*.arff")))
    assert len(catalog) == arff_count, (
        f"Expected {arff_count} rows, got {len(catalog)}"
    )


def test_catalog_columns(catalog):
    """Catalog must have the expected columns."""
    expected = {
        "name", "n_samples", "n_features", "n_classes",
        "imbalance_ratio", "minority_class_fraction", "file_size_bytes",
    }
    assert expected.issubset(set(catalog.columns))


def test_catalog_sorted_by_name(catalog):
    """Catalog rows should be sorted by dataset name."""
    names = catalog["name"].tolist()
    assert names == sorted(names), f"Catalog not sorted: {names[:5]}..."


def test_wine_metadata(catalog):
    """Wine dataset: 178 samples, 3 classes, 13 features (UCI standard)."""
    wine = catalog[catalog["name"] == "Wine"].iloc[0]
    assert wine["n_samples"] == 178
    assert wine["n_classes"] == 3
    assert wine["n_features"] == 13


def test_all_datasets_have_at_least_2_classes(catalog):
    """Every dataset must be a classification problem with >= 2 classes."""
    bad = catalog[catalog["n_classes"] < 2]
    assert len(bad) == 0, f"Datasets with < 2 classes: {bad['name'].tolist()}"


def test_imbalance_ratio_at_least_1(catalog):
    """Imbalance ratio must be >= 1.0 (majority >= minority by definition)."""
    bad = catalog[catalog["imbalance_ratio"] < 1.0]
    assert len(bad) == 0, f"imbalance_ratio < 1.0: {bad['name'].tolist()}"


def test_minority_fraction_in_valid_range(catalog):
    """Minority class fraction must be in (0, 0.5]."""
    bad = catalog[
        (catalog["minority_class_fraction"] <= 0.0)
        | (catalog["minority_class_fraction"] > 0.5)
    ]
    assert len(bad) == 0, f"Invalid minority fraction: {bad['name'].tolist()}"


def test_all_datasets_have_samples_and_features(catalog):
    """No dataset should have 0 samples or 0 features."""
    assert (catalog["n_samples"] > 0).all(), "Dataset with 0 samples"
    assert (catalog["n_features"] > 0).all(), "Dataset with 0 features"


def test_save_and_load_catalog_roundtrip(catalog, tmp_path):
    """save_catalog -> load_catalog should preserve data."""
    path = tmp_path / "catalog.csv"
    save_catalog(catalog, path)
    loaded = load_catalog(path)
    assert len(loaded) == len(catalog)
    assert list(loaded["name"]) == list(catalog["name"])
    # Numeric columns should be preserved
    assert loaded["n_samples"].tolist() == catalog["n_samples"].tolist()


def test_default_catalog_path_under_results():
    """default_catalog_path() should point to results/datasets/catalog.csv."""
    p = default_catalog_path()
    assert p.name == "catalog.csv"
    assert p.parent.name == "datasets"
    assert p.parent.parent.name == "results"
