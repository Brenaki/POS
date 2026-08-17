"""Enforces the <150 LOC cap per file (ADR 0002).

Excludes documentation, notebooks, config, legacy facades, and the test
suite itself. Only `.py` files under `pos/` (and the legacy scripts until
Fase 2 completes) are checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # tests/unit/ -> tests/ -> POS/

MAX_LOC = 150

# Directories whose .py files are exempt from the cap
EXEMPT_DIRS = {
    ".venv",
    "__pycache__",
    ".git",
    "tests",  # tests can be longer; characterization tests need full context
    "docs",
}

# Specific files exempt from the cap (legacy facades are thin but may exceed
# temporarily during the transition)
EXEMPT_FILES = {
    "cpx_legacy.py",
    "pool_generation_legacy.py",
    "Cpx.py",  # legacy, will be removed in Fase 2
    "pool_generation.py",  # legacy, will be removed in Fase 2
    "__init__.py",
}


def _count_loc(path: Path) -> int:
    """Count non-blank lines (LOC). Comments count toward the cap."""
    with path.open(encoding="utf-8") as f:
        lines = f.readlines()
    return sum(1 for line in lines if line.strip())


def _python_files_to_check():
    for py in REPO_ROOT.rglob("*.py"):
        rel = py.relative_to(REPO_ROOT)
        parts = rel.parts
        if any(part in EXEMPT_DIRS for part in parts):
            continue
        if rel.name in EXEMPT_FILES:
            continue
        yield py


def test_every_python_file_is_under_150_loc():
    offenders = []
    for py in _python_files_to_check():
        loc = _count_loc(py)
        if loc > MAX_LOC:
            offenders.append((py.relative_to(REPO_ROOT), loc))
    assert not offenders, (
        f"Files exceeding {MAX_LOC} LOC cap (ADR 0002):\n"
        + "\n".join(f"  {p}: {n} LOC" for p, n in offenders)
    )


def test_legacy_files_known_sizes():
    """Document the current size of legacy files that will be split in Fase 2."""
    legacy = {
        "Cpx.py": 262,
        "pool_generation.py": 657,
    }
    for name, expected in legacy.items():
        path = REPO_ROOT / name
        if path.exists():
            loc = _count_loc(path)
            # We don't assert exact equality (comments may shift), just that
            # they're still roughly the size we documented in ADR 0002.
            assert loc <= expected + 20, f"{name} grew unexpectedly: {loc} > {expected + 20}"


if __name__ == "__main__":
    # Manual inspection helper
    for py in _python_files_to_check():
        loc = _count_loc(py)
        marker = "OK" if loc <= MAX_LOC else "OVER"
        print(f"  {marker:4s} {loc:4d}  {py.relative_to(REPO_ROOT)}")
