"""Legacy pool_generation module — now a facade re-exporting from pos.pool.

This file preserves backwards compatibility for sample.ipynb and any code
that does `from pool_generation import poolGeneration`. The actual
implementation lives in pos/pool/ (see ADR 0005). Do not add new code here.
"""

from pos.pool.pool_generation import poolGeneration

__all__ = ["poolGeneration"]
