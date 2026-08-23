"""Registry of the DCS/DES/static methods the runs evaluate (ADR 0018).

Kept apart from `des_comparison` so the list can grow without pushing the
evaluation loop over the 150 LOC cap, and so the ordering that the summary
CSV header depends on lives in exactly one place.

The list is ordered by how aggressively a method prunes the pool, because
that is the axis the ADR 0017 run turned up: OLA and LCA keep a single
classifier and lose to plain majority vote on tree pools, while KNORA-U
keeps everyone that is locally right and does not.
"""

from __future__ import annotations

from pos.oracle.deslib_compat import install_numpy_aliases

# DCS: pick one classifier. DES: pick a subset. static: no local competence.
DCS_METHODS = ("ola", "lca", "mcb", "rank")
DES_ONLY_METHODS = ("knorae", "knorau", "desp", "desknn", "metades", "knop")
STATIC_METHODS = ("single_best", "static_sel")
DES_METHODS = DCS_METHODS + DES_ONLY_METHODS + STATIC_METHODS

# The ADR 0017 set. Friedman/Nemenyi runs on this, so the critical difference
# stays usable and the previous run stays comparable column by column; the
# rest are reported descriptively with Wilcoxon vs MVR.
PRIMARY_METHODS = ("ola", "lca", "knorae", "knorau", "metades")

# Need predict_proba, so they sit out the GA's Perceptron pool.
NEEDS_PROBA = ("metades", "knop")

DEFAULT_K = 7


def method_classes() -> dict:
    """Lazy import: deslib is heavy and only needed when a run asks for it."""
    install_numpy_aliases()
    from deslib.dcs import LCA, MCB, OLA, Rank
    from deslib.des import DESKNN, DESP, KNOP, KNORAE, KNORAU, METADES
    from deslib.static import SingleBest, StaticSelection

    return {
        "ola": OLA, "lca": LCA, "mcb": MCB, "rank": Rank,
        "knorae": KNORAE, "knorau": KNORAU, "desp": DESP,
        "desknn": DESKNN, "metades": METADES, "knop": KNOP,
        "single_best": SingleBest, "static_sel": StaticSelection,
    }


def takes_k(name: str) -> bool:
    """Static methods have no region of competence, so no `k`."""
    return name not in STATIC_METHODS
