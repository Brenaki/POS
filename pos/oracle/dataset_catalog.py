"""Dataset catalog: metadata for the 31 ARFF datasets in POS/Dataset/.

Builds a tabular catalog with one row per .arff file: name, n_samples,
n_features, n_classes, imbalance ratio, minority class fraction, file size.
Used to document the dataset selection (cronograma Atividade 2) and to
sanity-check that every dataset loads cleanly before experiments.

ADR 0006 (protocol), ADR 0009 (reproducibility).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pos.oracle.arff_loader import list_datasets, load_arff_dataset


def _imbalance_ratio(y: np.ndarray) -> tuple[float, float]:
    """Return (imbalance_ratio, minority_class_fraction) for a label vector.

    imbalance_ratio = n_majority / n_minority (>= 1.0).
    minority_class_fraction = n_minority / n_total (in (0, 0.5]).
    """
    _, counts = np.unique(y, return_counts=True)
    n_total = int(counts.sum())
    n_min = int(counts.min())
    n_maj = int(counts.max())
    imb = n_maj / n_min if n_min > 0 else float("inf")
    min_frac = n_min / n_total if n_total > 0 else 0.0
    return float(imb), float(min_frac)


def build_catalog(dataset_dir: Path | str | None = None) -> pd.DataFrame:
    """Build a catalog DataFrame with one row per .arff dataset.

    Parameters
    ----------
    dataset_dir : Path or None
        Directory containing .arff files. Defaults to POS/Dataset/.

    Returns
    -------
    df : pd.DataFrame with columns:
        name, n_samples, n_features, n_classes, imbalance_ratio,
        minority_class_fraction, file_size_bytes
    """
    paths = list_datasets(dataset_dir)
    rows: list[dict] = []
    for p in paths:
        X, y = load_arff_dataset(p)
        imb, min_frac = _imbalance_ratio(y)
        rows.append({
            "name": p.stem,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(np.unique(y).size),
            "imbalance_ratio": imb,
            "minority_class_fraction": min_frac,
            "file_size_bytes": int(p.stat().st_size),
        })
    return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)


def save_catalog(df: pd.DataFrame, path: Path | str) -> None:
    """Save catalog to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_catalog(path: Path | str) -> pd.DataFrame:
    """Load catalog from CSV."""
    return pd.read_csv(path)


def default_catalog_path() -> Path:
    """Return the default catalog path: results/datasets/catalog.csv."""
    # pos/oracle/dataset_catalog.py -> pos/oracle -> pos -> POS -> results
    return Path(__file__).resolve().parents[2] / "results" / "datasets" / "catalog.csv"
