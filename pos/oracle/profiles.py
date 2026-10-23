"""Scientific profile registry (ADR 0020).

Two profiles share a PGDCS core but differ in protocol, splitting and
comparison target. Each profile is a frozen dict of protocol parameters;
`materialize_profile` turns one into a run config the recorder consumes.

* `pos-oracle-v2` — POS main experiment, 10-fold CV, Oracle_N.
* `monteiro-2023-dt-compatible-v1` — Decision-Tree slice of Monteiro 2023,
  20× holdout 50/25/25, for comparison with the published tables.
* `monteiro-2023-dt-compatible-v2` — v1 with the ECoL 0.3.0 complexity
  measures (`complexity_backend`, ADR 0024).

The profile dict is the single source of truth; nothing in the recorder
hardcodes protocol numbers. A config fingerprint lets resume refuse to
continue a run whose configuration changed under it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

SCIENTIFIC_TREATMENTS = ("pgdcs_a", "pgdcs_d", "bagging")

# ADR 0020 (revised): 21 checkpoints, 1 then every 5 up to M=100.
ORACLE_CHECKPOINTS = (1,) + tuple(range(5, 101, 5))

DES_METHODS_V2 = ("lca", "ola", "rank", "knorae", "knorau", "metades")

PGDCS_CORE: dict[str, Any] = {
    "nr_generation": 20,
    "nr_individual": 100,
    "nr_pop": 100,
    "nr_child": 100,
    "proba_crossover": 0.9,
    "proba_mutation": 0.1,
    "tam_bags": 0.5,
    "classifier": "tree",
}

POS_ORACLE_V2: dict[str, Any] = {
    "profile": "pos-oracle-v2",
    "split": "three_way_disjoint",
    "n_folds": 10,
    "cv_shuffle": True,
    "random_state": 42,
    "M": 100,
    "oracle_checkpoints": ORACLE_CHECKPOINTS,
    "des_methods": DES_METHODS_V2,
    "reference_scope": "pos-oracle-v2",
    "validation_dsel_overlap": False,
    "measure_selection": "per_fold",
    "measure_fraction": 0.5,
    "bag_sampling": "random_with_replacement",
    "shared_evolution_trajectory": True,
    "modernization": True,
    "keep_fitted_classifiers": True,
    **PGDCS_CORE,
}

MONTIERO_2023_DT: dict[str, Any] = {
    "profile": "monteiro-2023-dt-compatible-v1",
    "split": "holdout_50_25_25",
    "n_replications": 20,
    "train_frac": 0.5,
    "val_frac": 0.25,
    "test_frac": 0.25,
    "stratified_splits": True,
    "new_split_per_replication": True,
    "reproducibility_root_seed": 86,
    "bagging_random_state": 86,
    "M": 100,
    "oracle_checkpoints": ORACLE_CHECKPOINTS,
    "des_methods": DES_METHODS_V2,
    "reference_scope": "decision-tree / pgdcs-a / pgdcs-d / bagging",
    "validation_dsel_overlap": True,
    "measure_selection": "once_per_dataset",
    "measure_fraction": 0.4,
    "measure_ambiguity": (
        "paper says u=0.4 for the step-1 sub-samples and 'a maximum of 50%' "
        "in the same sentence; u=0.4 adopted, GA bags stay at 0.5 "
        "('data subsets with just 50% of the training dataset size')"
    ),
    "bag_sampling": "random_with_replacement",
    "shared_evolution_trajectory": True,
    "modernization": True,
    "keep_fitted_classifiers": True,
    "random_state": 86,
    **PGDCS_CORE,
}

# v1 with T1/N1/LSC/F4 computed as ECoL 0.3.0 does (ADR 0024). v1 keeps the
# fast definitions, and its fingerprint, so the published run stays reproducible.
MONTEIRO_2023_DT_ECOL: dict[str, Any] = {
    **MONTIERO_2023_DT,
    "profile": "monteiro-2023-dt-compatible-v2",
    "complexity_backend": "ecol-0.3.0",
}

_PROFILES = {
    "pos-oracle-v2": POS_ORACLE_V2,
    "monteiro-2023-dt-compatible-v1": MONTIERO_2023_DT,
    "monteiro-2023-dt-compatible-v2": MONTEIRO_2023_DT_ECOL,
}

MONTEIRO_PROFILES = ("monteiro-2023-dt-compatible-v1", "monteiro-2023-dt-compatible-v2")

PROFILE_NAMES = tuple(_PROFILES.keys())


def profile_for(name: str) -> dict[str, Any]:
    """Return a deep copy of the profile dict for `name`. KeyError if unknown."""
    if name not in _PROFILES:
        raise KeyError(f"unknown profile: {name!r} (known: {PROFILE_NAMES})")
    return copy.deepcopy(_PROFILES[name])


_MONTEIRO_SMOKE = {
    "n_replications": 3, "nr_generation": 3, "M": 12, "nr_individual": 12,
    "nr_pop": 12, "nr_child": 12,
}

_SMOKE_OVERRIDES = {
    "pos-oracle-v2": {
        "n_folds": 3, "nr_generation": 3, "M": 12, "nr_individual": 12,
        "nr_pop": 12, "nr_child": 12,
    },
    **dict.fromkeys(MONTEIRO_PROFILES, _MONTEIRO_SMOKE),
}


def _fingerprint(cfg: dict[str, Any]) -> str:
    """Stable hash of the protocol-relevant subset of the config."""
    payload = {k: v for k, v in sorted(cfg.items()) if k != "fingerprint"}
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def materialize_profile(
    name: str, scale: str = "full", overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a run config from a profile.

    scale: "full" (scientific) or "smoke" (reduced, non-comparable).
    overrides: optional per-run parameter changes; any non-empty override
    marks the resulting config as `comparable=False` (ADR 0020 §6).
    """
    if scale not in ("full", "smoke"):
        raise ValueError(f"scale must be 'full' or 'smoke', got {scale!r}")
    cfg = profile_for(name)
    cfg["treatments"] = list(SCIENTIFIC_TREATMENTS)
    cfg["scale"] = scale
    cfg["comparable"] = scale == "full"
    if scale == "smoke":
        cfg.update(_SMOKE_OVERRIDES[name])
        cfg["comparable"] = False
    if overrides:
        cfg.update(overrides)
        cfg["comparable"] = False
    cfg["fingerprint"] = _fingerprint(cfg)
    return cfg
