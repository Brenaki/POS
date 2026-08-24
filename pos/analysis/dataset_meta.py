"""Attach dataset-level metadata to a run frame (ADR 0019).

Two of the questions the external review raised need columns that the run does
not record because they are properties of the dataset, not of the fold:

* `N*` sits at ~M/2 in the binary problems for a structural reason — with M
  classifiers, `Oracle_{M/2+1} == MVR` exactly, because 51 correct votes out of
  100 already win a two-class majority. Stating "no useful intermediate Oracle
  level exists" without splitting binary from multiclass overstates the result:
  under plurality voting the identity does not hold.
* predicting *recoverability* (how much of the gap dynamic selection converts)
  needs class count, dimensionality and imbalance as candidate features.

`dataset_catalog.build_catalog` already computes all of it from the ARFF files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pos.oracle.dataset_catalog import build_catalog

META_COLS = ["n_classes", "n_samples", "n_features", "imbalance_ratio",
             "minority_class_fraction"]


def dataset_meta(dataset_dir: Path | str | None = None) -> pd.DataFrame:
    """One row per dataset: name + the metadata columns above."""
    catalog = build_catalog(dataset_dir)
    meta = catalog[["name", *META_COLS]].rename(columns={"name": "dataset"})
    meta["is_binary"] = meta["n_classes"] == 2
    return meta


def attach_dataset_meta(df: pd.DataFrame,
                        dataset_dir: Path | str | None = None) -> pd.DataFrame:
    """Left-join the dataset metadata onto a per-fold run frame."""
    return df.merge(dataset_meta(dataset_dir), on="dataset", how="left")
