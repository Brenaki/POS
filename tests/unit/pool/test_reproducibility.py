"""TDD tests for P0 #1: GA reproducibility — random_state must be respected."""

from __future__ import annotations

import numpy as np
import pytest

from pos.pool.bag_generator import generate_bags


class TestBagReproducibility:
    """generate_bags must be reproducible with the same random_state."""

    def test_same_seed_same_bags(self):
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        X = np.random.RandomState(99).randn(200, 5)
        y = np.random.RandomState(99).randint(0, 2, size=200)
        bags1 = generate_bags(X, y, nr_bags=5, tam_bags=0.5, random_state=rng1)
        bags2 = generate_bags(X, y, nr_bags=5, tam_bags=0.5, random_state=rng2)
        assert bags1["inst"] == bags2["inst"]

    def test_different_seed_different_bags(self):
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(43)
        X = np.random.RandomState(99).randn(200, 5)
        y = np.random.RandomState(99).randint(0, 2, size=200)
        bags1 = generate_bags(X, y, nr_bags=5, tam_bags=0.5, random_state=rng1)
        bags2 = generate_bags(X, y, nr_bags=5, tam_bags=0.5, random_state=rng2)
        assert bags1["inst"] != bags2["inst"]

    def test_int_random_state(self):
        """Passing int random_state should work like RandomState(int)."""
        X = np.random.RandomState(99).randn(200, 5)
        y = np.random.RandomState(99).randint(0, 2, size=200)
        bags1 = generate_bags(X, y, nr_bags=5, tam_bags=0.5, random_state=42)
        bags2 = generate_bags(X, y, nr_bags=5, tam_bags=0.5, random_state=42)
        assert bags1["inst"] == bags2["inst"]