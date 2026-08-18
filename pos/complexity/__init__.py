"""Complexity backend selection.

Default backend is the R/ECoL adapter. The pyhard adapter will replace this
in Fase 3 (ADR 0003).
"""

from pos.complexity.ecol_adapter import complexity_data3  # noqa: F401
from pos.complexity.base import HEADER, GROUP_MEASURES  # noqa: F401
