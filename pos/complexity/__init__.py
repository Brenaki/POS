"""Complexity backend selection.

Default backend is pyhard (Fase 3, ADR 0003). The legacy R/ECoL adapter is
available as `pos.complexity.ecol_adapter` for regenerating golden values.
"""

from pos.complexity.pyhard_adapter import complexity_data3  # noqa: F401
from pos.complexity.base import HEADER, GROUP_MEASURES  # noqa: F401
