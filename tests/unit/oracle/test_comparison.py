"""Unit tests for pos.oracle.comparison (majority vote, mean probs).

TDD — tests written BEFORE implementation.
"""

from __future__ import annotations

import numpy as np


def test_majority_vote_accuracy_unanimous():
    """Pool unânime: majority vote = própria acurácia de cada clf."""
    from pos.oracle.comparison import majority_vote_accuracy

    class Clf:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.array([0, 1, 0, 1])

    pool = [Clf(), Clf(), Clf()]
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 1, 0, 1])
    acc = majority_vote_accuracy(pool, X, y)
    assert acc == 1.0


def test_majority_vote_accuracy_majority_wrong():
    """2 de 3 classificadores erram em todas as amostras → majority vote erra."""
    from pos.oracle.comparison import majority_vote_accuracy

    class RightClf:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.array([0, 1])

    class WrongClf:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.array([1, 0])

    pool = [RightClf(), WrongClf(), WrongClf()]
    X = np.array([[1], [2]])
    y = np.array([0, 1])
    # votes: sample 0 → [0, 1, 1] → majority=1 (wrong, true=0)
    # votes: sample 1 → [1, 0, 0] → majority=0 (wrong, true=1)
    acc = majority_vote_accuracy(pool, X, y)
    assert acc == 0.0


def test_majority_vote_accuracy_majority_right():
    """2 de 3 classificadores acertam → majority vote acerta."""
    from pos.oracle.comparison import majority_vote_accuracy

    class RightClf:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.array([0, 1, 0])

    class WrongClf:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.array([1, 0, 1])

    pool = [RightClf(), RightClf(), WrongClf()]
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])
    # votes: [0,0,1]→0 (right), [1,1,0]→1 (right), [0,0,1]→0 (right)
    acc = majority_vote_accuracy(pool, X, y)
    assert acc == 1.0


def test_majority_vote_accuracy_tie():
    """Empate 2-2 em amostra binária: scipy/scikit usa o que tiver predict."""
    from pos.oracle.comparison import majority_vote_accuracy

    class ClfA:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.array([0, 1])

    class ClfB:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.array([1, 0])

    pool = [ClfA(), ClfA(), ClfB(), ClfB()]
    X = np.array([[1], [2]])
    y = np.array([0, 1])
    # votes: sample0 → [0,0,1,1] → tie → behaviour depends on implementation
    # Document expected: scipy mode picks first sorted → 0 (right for sample0)
    # sample1 → [1,1,0,0] → tie → 0 (wrong for sample1, true=1)
    acc = majority_vote_accuracy(pool, X, y)
    # Accept either 0.5 or 1.0 depending on tie-breaking
    assert acc in (0.5, 1.0)


def test_mean_probs_accuracy_all_correct():
    """Pool onde mean of probs argmax = true label em tudo."""
    from pos.oracle.comparison import mean_probs_accuracy

    class Clf:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])

    pool = [Clf(), Clf(), Clf()]
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])
    acc = mean_probs_accuracy(pool, X, y)
    assert acc == 1.0


def test_mean_probs_accuracy_all_wrong():
    """Pool onde mean of probs argmax != true label em tudo."""
    from pos.oracle.comparison import mean_probs_accuracy

    class Clf:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.1, 0.9], [0.9, 0.1]])

    pool = [Clf(), Clf()]
    X = np.array([[1], [2]])
    y = np.array([0, 1])
    # mean prob: sample0 → [0.1, 0.9] → argmax=1 (wrong, true=0)
    # sample1 → [0.9, 0.1] → argmax=0 (wrong, true=1)
    acc = mean_probs_accuracy(pool, X, y)
    assert acc == 0.0
