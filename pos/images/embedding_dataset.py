"""Torch-free half of the image battery (ADR 0022).

The CNN forward pass runs in an isolated torch environment
(`scripts/extract_image_embeddings.py`); everything that decides *which*
images and *how* their embeddings become a tabular snapshot lives here, so it
is tested in the pinned main environment:

* `IMAGE_DATASETS` — the five snapshots, three 10-class and two binary;
* `stratified_sample` — the experiment sample, drawn from the official test
  split, and the disjoint PCA-fit sample, drawn from the official train split;
* `project` — PCA fit on the disjoint sample only, so no fold ever leaks
  into the projection;
* `write_arff` — numeric attributes plus a nominal `class`, readable by
  `pos.oracle.arff_loader` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class ImageSpec:
    """One image snapshot: torchvision source and optional class subset."""

    source: str
    classes: tuple[int, ...] = ()
    n_experiment: int = 2000
    n_pca_fit: int = 2000
    n_components: int = 32


IMAGE_DATASETS: dict[str, ImageSpec] = {
    "MNIST": ImageSpec("MNIST"),
    "FashionMNIST": ImageSpec("FashionMNIST"),
    "CIFAR10": ImageSpec("CIFAR10"),
    # cat (3) vs dog (5): the canonical hard pair of CIFAR-10
    "CIFAR10CatDog": ImageSpec("CIFAR10", classes=(3, 5)),
    # T-shirt/top (0) vs shirt (6): the most confused Fashion-MNIST pair
    "FashionTeeShirt": ImageSpec("FashionMNIST", classes=(0, 6)),
}


def stratified_sample(y: np.ndarray, n: int, seed: int,
                      classes: tuple[int, ...] = ()) -> np.ndarray:
    """Sorted indices of a class-proportional sample of size `n`, no replacement.

    Restricted to `classes` when given; takes every eligible index when `n`
    exceeds them. Per-class quotas use largest-remainder rounding.
    """
    y = np.asarray(y)
    eligible = np.flatnonzero(np.isin(y, classes)) if classes else np.arange(len(y))
    if n >= len(eligible):
        return np.sort(eligible)
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y[eligible], return_counts=True)
    exact = counts * n / counts.sum()
    quota = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - quota))[: n - quota.sum()]:
        quota[i] += 1
    picked = [rng.choice(eligible[y[eligible] == lab], size=q, replace=False)
              for lab, q in zip(labels, quota, strict=True)]
    return np.sort(np.concatenate(picked))


def project(fit_features: np.ndarray, features: np.ndarray, n_components: int,
            seed: int) -> tuple[np.ndarray, PCA]:
    """PCA fit on `fit_features` only, applied to `features`."""
    if n_components > fit_features.shape[1]:
        raise ValueError(f"n_components={n_components} > {fit_features.shape[1]} features")
    pca = PCA(n_components=n_components, random_state=seed).fit(fit_features)
    return pca.transform(features), pca


def write_arff(path: Path, X: np.ndarray, y: np.ndarray, relation: str) -> None:
    """Numeric ARFF with a nominal `class` attribute (original label values)."""
    labels = sorted({int(v) for v in y})
    lines = [f"@relation {relation}", ""]
    lines += [f"@attribute f{i:02d} numeric" for i in range(X.shape[1])]
    lines += ["@attribute class {" + ",".join(str(v) for v in labels) + "}", "",
              "@data"]
    lines += [",".join(f"{v:.6g}" for v in row) + f",{int(c)}" for row, c in zip(X, y, strict=True)]
    Path(path).write_text("\n".join(lines) + "\n")
