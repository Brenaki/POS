"""Legacy Cpx module — now a facade re-exporting from the `pos` package.

This file preserves backwards compatibility for `sample.ipynb` and any code
that does `import Cpx` or `from Cpx import ...`. The actual implementations
live in the `pos/` package (see ADR 0004). Do not add new code here.

As of Fase 3 (ADR 0003), complexity_data3 uses pyhard by default — R/ECoL is
no longer required. The `ecol` symbol is kept as None for compat (it was an
R package handle; pyhard has no equivalent).
"""

from pos.normalization import min_max_norm
from pos.dispersion import dispersion, dispersion_linear
from pos.diversity import diversitys
from pos.voting import voting_classifier
from pos.classifiers import biuld_classifier, biuld_classifier_tree
from pos.io_csv import save_bag
from pos.complexity.base import HEADER as header
from pos.complexity import complexity_data3

# `ecol` was the R ECoL package handle imported via rpy2. With the pyhard
# backend (Fase 3), there is no R handle. We expose `None` for backwards
# compat so `hasattr(Cpx, 'ecol')` still works without crashing on import.
ecol = None

__all__ = [
    "min_max_norm",
    "dispersion_linear",
    "dispersion",
    "diversitys",
    "voting_classifier",
    "biuld_classifier",
    "biuld_classifier_tree",
    "save_bag",
    "header",
    "complexity_data3",
    "ecol",
]
