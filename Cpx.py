"""Legacy Cpx module — now a facade re-exporting from the `pos` package.

This file preserves backwards compatibility for `sample.ipynb` and any code
that does `import Cpx` or `from Cpx import ...`. The actual implementations
live in the `pos/` package (see ADR 0004). Do not add new code here.
"""

from pos.normalization import min_max_norm
from pos.dispersion import dispersion, dispersion_linear
from pos.diversity import diversitys
from pos.voting import voting_classifier
from pos.classifiers import biuld_classifier, biuld_classifier_tree
from pos.io_csv import save_bag
from pos.complexity.base import HEADER as header
from pos.complexity import complexity_data3

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
]
