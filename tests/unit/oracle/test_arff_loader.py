"""Unit tests for pos.oracle.arff_loader.

TDD — tests written BEFORE implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

DATASET_DIR = Path(__file__).resolve().parents[3] / "Dataset"


def test_load_arff_wine_returns_X_y():
    """load_arff_dataset(Wine.arff) retorna (X, y) arrays numpy."""
    from pos.oracle.arff_loader import load_arff_dataset

    path = DATASET_DIR / "Wine.arff"
    if not path.exists():
        pytest.skip(f"Dataset não encontrado: {path}")
    X, y = load_arff_dataset(path)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.ndim == 2
    assert X.shape[0] == y.shape[0]


def test_load_arff_wine_known_shape():
    """Wine tem 178 amostras, 13 features, 3 classes."""
    from pos.oracle.arff_loader import load_arff_dataset

    path = DATASET_DIR / "Wine.arff"
    if not path.exists():
        pytest.skip(f"Dataset não encontrado: {path}")
    X, y = load_arff_dataset(path)
    assert X.shape == (178, 13)
    assert len(np.unique(y)) == 3


def test_load_arff_banana_returns_valid():
    """Banana é binário, 2 features."""
    from pos.oracle.arff_loader import load_arff_dataset

    path = DATASET_DIR / "Banana.arff"
    if not path.exists():
        pytest.skip(f"Dataset não encontrado: {path}")
    X, y = load_arff_dataset(path)
    assert X.ndim == 2
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 2
    assert len(np.unique(y)) == 2


def test_load_arff_invalid_path_raises():
    """Caminho inexistente levanta FileNotFoundError."""
    from pos.oracle.arff_loader import load_arff_dataset

    with pytest.raises(FileNotFoundError):
        load_arff_dataset(Path("/tmp/inexistente_nunca_existiu.arff"))


def test_load_arff_labels_are_encoded():
    """y deve ser array numérico (labels string → int via LabelEncoder)."""
    from pos.oracle.arff_loader import load_arff_dataset

    path = DATASET_DIR / "Wine.arff"
    if not path.exists():
        pytest.skip(f"Dataset não encontrado: {path}")
    _, y = load_arff_dataset(path)
    assert y.dtype.kind in ("i", "u", "f")  # int/uint/float, não string/object
