"""ARFF dataset loader for Oracle_N experiments.

Loads .arff files from POS/Dataset/ using scipy.io.arff (via sklearn-like
parsing), returns (X, y) as numpy arrays with integer-encoded labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder


def load_arff_dataset(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load an ARFF file and return (X, y) as numpy arrays.

    Parameters
    ----------
    path : Path or str
        Path to the .arff file.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features), dtype float
    y : np.ndarray of shape (n_samples,), dtype int (label-encoded)

    Raises
    ------
    FileNotFoundError : if path does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ARFF file not found: {path}")

    # scipy.io.arff.loadarff returns a structured np.ndarray
    data, _meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    # Convention: last column is the class label (ARFF @attribute Class ...)
    # Find the last column or one named 'class'/'Class'
    label_col = None
    for candidate in ("Class", "class", "target", "label"):
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        label_col = df.columns[-1]

    y_raw = df[label_col].astype(str)
    X = df.drop(columns=[label_col]).to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(y_raw)
    return X, y


def list_datasets(dataset_dir: Path | str | None = None) -> list[Path]:
    """Return sorted list of .arff files in the dataset directory.

    Defaults to POS/Dataset/ relative to this package.
    """
    if dataset_dir is None:
        # pos/oracle/arff_loader.py → pos/oracle/ → pos/ → POS/ → POS/Dataset/
        dataset_dir = Path(__file__).resolve().parents[2] / "Dataset"
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        return []
    return sorted(dataset_dir.glob("*.arff"))
