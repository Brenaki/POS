"""Exceptions raised by the GA pool generator.

Replaces the legacy `exit(0)` in `genetic_operators.crossover`, which raised
`SystemExit` — a `BaseException` that `except Exception:` in
`pos.oracle.run_recorder` does NOT catch, so a single unusable dataset
silently killed the whole experiment process with exit code 0 (ADR 0014).
"""

from __future__ import annotations


class StratificationError(Exception):
    """A bag could not be built keeping every class with >= 2 instances.

    Raised when the training split is too small / too imbalanced for the
    configured `tam_bags`. Caught per-fold by the experiment runner.
    """
