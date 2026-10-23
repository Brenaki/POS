"""Tests for the torch-free half of the image battery (ADR 0022).

Sampling, the leakage-free PCA projection and the ARFF writer are pure
numpy/sklearn, so they are tested here with synthetic features; the CNN
forward pass lives in `scripts/extract_image_embeddings.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from pos.images.embedding_dataset import (
    IMAGE_DATASETS,
    project,
    stratified_sample,
    write_arff,
)
from pos.oracle.arff_loader import load_arff_dataset


class TestSpecs:
    def test_five_datasets_two_of_them_binary(self):
        assert len(IMAGE_DATASETS) == 5
        binary = [s for s in IMAGE_DATASETS.values() if s.classes]
        assert len(binary) == 2 and all(len(s.classes) == 2 for s in binary)

    def test_sources_are_torchvision_names(self):
        assert {s.source for s in IMAGE_DATASETS.values()} == {
            "MNIST", "FashionMNIST", "CIFAR10"}


class TestStratifiedSample:
    def test_preserves_class_proportions(self):
        y = np.repeat([0, 1, 2], [600, 300, 100])
        idx = stratified_sample(y, 100, seed=0)
        assert len(idx) == 100
        assert np.bincount(y[idx]).tolist() == [60, 30, 10]

    def test_is_deterministic_and_without_replacement(self):
        y = np.repeat([0, 1], 500)
        a, b = stratified_sample(y, 200, seed=7), stratified_sample(y, 200, seed=7)
        assert np.array_equal(a, b) and len(set(a.tolist())) == 200

    def test_class_subset_keeps_only_those_classes(self):
        y = np.repeat(np.arange(10), 100)
        idx = stratified_sample(y, 120, seed=0, classes=(3, 5))
        assert set(y[idx].tolist()) == {3, 5}

    def test_asking_for_more_than_available_takes_everything(self):
        y = np.repeat([0, 1], 10)
        assert len(stratified_sample(y, 50, seed=0)) == 20


class TestProject:
    def test_pca_is_fit_only_on_the_disjoint_set(self):
        rng = np.random.default_rng(0)
        fit, exp = rng.normal(size=(300, 64)), rng.normal(size=(200, 64)) + 100
        out, pca = project(fit, exp, n_components=8, seed=0)
        assert out.shape == (200, 8)
        assert np.allclose(pca.mean_, fit.mean(axis=0))

    def test_rejects_more_components_than_features(self):
        with pytest.raises(ValueError):
            project(np.zeros((10, 4)), np.zeros((5, 4)), n_components=8, seed=0)


class TestWriteArff:
    def test_round_trips_through_the_project_loader(self, tmp_path):
        X = np.arange(12, dtype=float).reshape(6, 2) / 7
        y = np.array([3, 5, 3, 5, 3, 5])
        path = tmp_path / "Toy.arff"
        write_arff(path, X, y, relation="Toy")
        X2, y2 = load_arff_dataset(path)
        assert np.allclose(X2, X, atol=1e-6)
        assert y2.tolist() == [0, 1, 0, 1, 0, 1]
