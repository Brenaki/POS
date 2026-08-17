"""Shared pytest fixtures and R-availability detection for the POS test suite.

The characterization tests import the legacy `Cpx` and `pool_generation` modules,
which call `rpy2.robjects.packages.importr('ECoL')` at import time. To allow the
Python-pure functions of those modules to be tested without R installed, we
inject lightweight mocks for `rpy2` and `ECoL` into `sys.modules` before the
first import. Tests that actually need R are marked `@pytest.mark.requires_r`
and are skipped automatically when R is not on PATH.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# R availability detection
# ---------------------------------------------------------------------------

def _r_is_available() -> bool:
    """Return True iff the R executable is on PATH (and thus R_HOME can be resolved)."""
    return shutil.which("R") is not None


def pytest_collection_modifyitems(config, items):
    """Auto-skip `requires_r` tests when R is not installed."""
    if _r_is_available():
        return
    skip_r = pytest.mark.skip(reason="R not installed on PATH (run `sudo pacman -S r tk`)")
    for item in items:
        if "requires_r" in item.keywords:
            item.add_marker(skip_r)


# ---------------------------------------------------------------------------
# rpy2/ECoL mocking so Python-pure functions in Cpx.py can be imported without R
# ---------------------------------------------------------------------------

def _install_rpy2_mock() -> None:
    """Inject minimal mocks for rpy2 + ECoL so `import Cpx` works without R.

    Only the symbols referenced at module top-level of Cpx.py / pool_generation.py
    are faked: `pandas2ri.activate`, `rpackages.importr`, `robjects.IntVector`.
    Functions that actually call R (complexity_data3) will fail at runtime if
    called without a real R — those tests must be marked `requires_r`.
    """
    if "rpy2" in sys.modules and getattr(sys.modules["rpy2"], "_pos_mock", False):
        return  # already mocked

    fake_rpy2 = types.ModuleType("rpy2")
    fake_robjects = types.ModuleType("rpy2.robjects")
    fake_packages = types.ModuleType("rpy2.robjects.packages")
    fake_pandas2ri = types.ModuleType("rpy2.robjects.pandas2ri")

    class _FakeIntVector:
        def __init__(self, values):
            self.values = list(values)

    class _FakeRPackages:
        @staticmethod
        def importr(name):
            class _FakePackage:
                def __getattr__(self, measure):
                    def _call(*args, **kwargs):
                        raise RuntimeError(
                            f"Mocked ECoL.{measure}() called — R is not available. "
                            "Mark this test with @pytest.mark.requires_r and run with R installed."
                        )
                    return _call
            return _FakePackage()

    fake_pandas2ri.activate = lambda: None
    fake_robjects.IntVector = _FakeIntVector
    fake_robjects.pandas2ri = fake_pandas2ri  # expose as attribute too
    fake_packages.importr = _FakeRPackages().importr
    fake_rpy2.robjects = fake_robjects
    fake_rpy2._pos_mock = True  # type: ignore[attr-defined]

    sys.modules["rpy2"] = fake_rpy2
    sys.modules["rpy2.robjects"] = fake_robjects
    sys.modules["rpy2.robjects.packages"] = fake_packages
    sys.modules["rpy2.robjects.pandas2ri"] = fake_pandas2ri


_install_rpy2_mock()


# Make sure the repo root is on sys.path so `import Cpx` / `import pool_generation`
# resolve to the legacy script files.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Reproducible dataset fixture (mirrors sample.ipynb)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def wine_split():
    """Reproduce the load_wine split used in sample.ipynb with random_state=42.

    Returns a dict with keys: X_train, y_train, X_test, y_test, X_valid, y_valid.
    The split sizes match the notebook:
      X_train, X_test = train_test_split(X, y, test_size=0.4, random_state=42)
      _, X_valid, _, y_valid = train_test_split(X_train, y_train, test_size=0.4, random_state=42)
    """
    X, y = load_wine(return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    _, X_valid, _, y_valid = train_test_split(
        X_train, y_train, test_size=0.4, random_state=42, stratify=y_train
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_valid": X_valid,
        "y_valid": y_valid,
    }


@pytest.fixture(scope="session")
def wine_split_no_stratify():
    """Reproduce the EXACT sample.ipynb split (which does NOT stratify).

    sample.ipynb calls train_test_split without stratify=, so we keep an
    unstratified variant for byte-exact characterization of the notebook.
    """
    X, y = load_wine(return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    _, X_valid, _, y_valid = train_test_split(
        X_train, y_train, test_size=0.4, random_state=42
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_valid": X_valid,
        "y_valid": y_valid,
    }


@pytest.fixture
def small_complexity_matrix():
    """A deterministic 3-bag × 2-measure complexity matrix for dispersion tests."""
    return [[0.1, 0.2], [0.3, 0.5], [0.6, 0.9]]


@pytest.fixture
def dummy_predictions():
    """Deterministic predictions for diversity/double-fault characterization."""
    y_test = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    preds = np.array([
        [0, 0, 1, 1, 0, 0, 1, 0],  # correct on 7/8
        [0, 1, 1, 1, 1, 0, 0, 0],  # correct on 6/8
        [0, 0, 0, 1, 1, 0, 1, 1],  # correct on 6/8
    ])
    return y_test, preds
