"""Tests for bag sampling modes (ADR 0020 T4).

The PGDCS protocol (Monteiro 2023) uses random sampling with replacement,
NOT stratified. Class coverage is enforced by the genetic operators
(crossover/mutation), not by the sampler. The legacy stratified path stays
available for backwards compatibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from pos.pool.bag_generator import (
    generate_bags,
    generate_bags_random_replacement,
)


@pytest.fixture
def balanced():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    y = np.array([0] * 100 + [1] * 100)
    return X, y


@pytest.fixture
def imbalanced():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    y = np.array([0] * 150 + [1] * 50)
    return X, y


class TestRandomReplacementSampling:
    def test_produces_requested_number_of_bags(self, balanced):
        X, y = balanced
        bags = generate_bags_random_replacement(X, y, nr_bags=10, tam_bags=0.5,
                                                 random_state=42)
        assert len(bags["name"]) == 10
        assert len(bags["inst"]) == 10

    def test_bag_size_is_fraction_of_train(self, balanced):
        X, y = balanced
        bags = generate_bags_random_replacement(X, y, nr_bags=5, tam_bags=0.5,
                                                 random_state=42)
        expected = int(len(y) * 0.5)
        for inst in bags["inst"]:
            assert len(inst) == expected

    def test_samples_with_replacement(self, balanced):
        """A bag of 100 from 200 with replacement should almost always have duplicates."""
        X, y = balanced
        bags = generate_bags_random_replacement(X, y, nr_bags=20, tam_bags=0.5,
                                                 random_state=42)
        any_duplicate = any(len(set(inst)) < len(inst) for inst in bags["inst"])
        assert any_duplicate, "no bag had duplicate indices — sampling without replacement?"

    def test_is_not_stratified(self, imbalanced):
        """Random sampling should not preserve the 75/25 class ratio exactly."""
        X, y = imbalanced
        bags = generate_bags_random_replacement(X, y, nr_bags=50, tam_bags=0.5,
                                                 random_state=42)
        ratios = []
        for inst in bags["inst"]:
            y_bag = y[inst]
            _, counts = np.unique(y_bag, return_counts=True)
            ratios.append(counts.min() / counts.sum())
        # stratified would always give 0.25; random should vary
        assert len(set(ratios)) > 1, "all bags have identical class ratio — stratified?"

    def test_is_deterministic_with_seed(self, balanced):
        X, y = balanced
        a = generate_bags_random_replacement(X, y, nr_bags=3, tam_bags=0.5,
                                              random_state=42)
        b = generate_bags_random_replacement(X, y, nr_bags=3, tam_bags=0.5,
                                              random_state=42)
        for ia, ib in zip(a["inst"], b["inst"]):
            assert ia == ib

    def test_different_seeds_produce_different_bags(self, balanced):
        X, y = balanced
        a = generate_bags_random_replacement(X, y, nr_bags=1, tam_bags=0.5,
                                              random_state=42)
        b = generate_bags_random_replacement(X, y, nr_bags=1, tam_bags=0.5,
                                              random_state=43)
        assert a["inst"][0] != b["inst"][0]

    def test_per_bag_seed_is_derived(self, balanced):
        """bag i uses random_state + i; bags 0 and 1 differ."""
        X, y = balanced
        bags = generate_bags_random_replacement(X, y, nr_bags=2, tam_bags=0.5,
                                                 random_state=42)
        assert bags["inst"][0] != bags["inst"][1]


class TestLegacyStratifiedSampler:
    def test_legacy_generate_bags_still_stratified(self, imbalanced):
        """Backwards compat: generate_bags (stratified, no replacement) unchanged."""
        X, y = imbalanced
        bags = generate_bags(X, y, nr_bags=3, tam_bags=0.5, random_state=42)
        for inst in bags["inst"]:
            y_bag = y[inst]
            _, counts = np.unique(y_bag, return_counts=True)
            ratio = counts.min() / counts.sum()
            # stratified 75/25 -> 0.25 exactly
            assert abs(ratio - 0.25) < 0.02

    def test_legacy_generate_bags_no_replacement(self, balanced):
        """Legacy path has no duplicate indices (train_test_split)."""
        X, y = balanced
        bags = generate_bags(X, y, nr_bags=5, tam_bags=0.5, random_state=42)
        for inst in bags["inst"]:
            assert len(set(inst)) == len(inst), "legacy path sampled with replacement"