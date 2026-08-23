"""Restore the numpy aliases DESlib 0.3.5 still uses, then import it.

DESlib 0.3.5 (the last release, 2021) calls `np.float`, `np.int` and
`np.bool` in `dcs/base.py`, `des/knora_e.py`, `des/knora_u.py`,
`des/meta_des.py` and `util/aggregation.py`. NumPy removed those aliases in
1.24, so on this environment (numpy 1.24.1) OLA and LCA happen to work while
KNORA-E, KNORA-U and META-DES die with

    AttributeError: module 'numpy' has no attribute 'float'

The aliases were plain references to the Python builtins, so restoring them
is exactly the behaviour DESlib was written against — it changes no numeric
result. Kept in one module, applied once, so the day DESlib is upgraded the
fix is a single deletion (ADR 0017).
"""

from __future__ import annotations

import numpy as np

# Only the two DESlib actually reaches on the DCS/DES paths used here.
# `np.bool`/`np.object` still exist in 1.24 behind a FutureWarning, and
# touching them would emit that warning on every import for nothing.
_ALIASES = {"float": float, "int": int}


def install_numpy_aliases() -> list[str]:
    """Re-add the removed aliases. Returns the names that were missing."""
    restored = []
    for name, builtin in _ALIASES.items():
        if not hasattr(np, name):
            setattr(np, name, builtin)
            restored.append(name)
    return restored
