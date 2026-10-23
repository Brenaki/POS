"""Complexity backend selection.

Default backend is `fast_adapter` (numpy/KD-tree; ADR 0023, which closes
ADR 0003): it is what the GA and the voter already used (ADR 0019), and the
measures it reproduces exactly against R/ECoL 0.3.0 are pinned by
tests/characterization/test_ecol_golden.py. `pyhard_adapter` stays importable
but is no longer the default. R is development-only: `ecol_bridge` runs the
reference ECoL through Rscript to validate and regenerate golden values.
"""

from pos.complexity.fast_adapter import complexity_data3  # noqa: F401
from pos.complexity.base import HEADER, GROUP_MEASURES  # noqa: F401
