"""CSV bag persistence (extracted from Cpx.py during Fase 2 refactor).

Preserves the legacy path/mkdir logic verbatim, including the use of
os.system('mkdir -p ...') and the slightly inconsistent path construction.
"""

from __future__ import annotations

import csv
import os
from typing import List


def save_bag(
    inds: List,
    types: str,
    local: str,
    base_name: str,
    iteration: int,
) -> None:
    """Persist a bag's instance indices to a CSV file.

    Behavior is preserved from Cpx.save_bag: writes `base_name + '.csv'` in
    the current working directory (the `local`/`iteration` path arguments
    feed into the mkdir call but the actual file write uses only base_name).
    """
    if types == "validation":
        if os.path.exists(local + "Validacao/" + str(iteration)) == False:
            os.system("mkdir -p " + local + "/" + str(iteration) + "/" + base_name + ".csv")
        with open(base_name + ".csv", "w") as f:
            w = csv.writer(f)
            w.writerow(inds)

    if types == "test":
        if os.path.exists(local + "Teste/" + str(iteration)) == False:
            os.system("mkdir -p " + local + "/" + str(iteration) + "/" + base_name + ".csv")
        with open(base_name + ".csv", "w") as f:
            w = csv.writer(f)
            w.writerow(inds)

    if types == "train":
        if os.path.exists(local + "Treino/" + str(iteration)) == False:
            os.system("mkdir -p " + local + "/" + str(iteration) + "/" + base_name + ".csv")
        with open(base_name + ".csv", "w") as f:
            w = csv.writer(f)
            w.writerow(inds)

    if types == "bags":
        if os.path.exists(local + "Bags/" + str(iteration)) == False:
            os.system("mkdir -p " + local + "/" + str(iteration) + "/" + base_name + ".csv")
        with open(base_name + ".csv", "a") as f:
            w = csv.writer(f)
            w.writerow(inds)
