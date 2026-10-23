"""Subprocess bridge to the reference R/ECoL measures (ADR 0023).

Development-only: used to validate `fast_adapter` and to regenerate golden
values, never by the pipeline. It talks to `Rscript` through CSV files, not
rpy2, and sources a pinned ECoL *source tarball* (`scripts/r/ecol_measures.R`)
so that ECoL 0.3.0 — the version whose names (`overlapping`, `T1`, `LSC`) the
legacy `Cpx.py` and `base.HEADER` use — and the current 0.4.4 can run side by
side. Needs `Rscript` plus the R packages cluster, e1071, MASS (and igraph for
N1/network). Tarballs are cached under `~/.cache/pos/ecol` after a SHA-256
check.
"""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
R_SCRIPT = REPO_ROOT / "scripts" / "r" / "ecol_measures.R"
CACHE = Path.home() / ".cache" / "pos" / "ecol"

ECOL_TARBALLS = {
    "0.3.0": ("https://cran.r-project.org/src/contrib/Archive/ECoL/ECoL_0.3.0.tar.gz",
              "8e5837c18a4843e94a7ac2330be2155e73f4d503d6e4244a356ea0080df5440a"),
    "0.4.4": ("https://cran.r-project.org/src/contrib/ECoL_0.4.4.tar.gz",
              "4d5d0785108c382143c41853ed45276362baeee9eb30271806c9f7011e5004a5"),
}

# ECoL 0.4 renamed the overlapping group and two neighborhood measures.
_GROUP_ALIASES = {"0.4.4": {"overlapping": "featurebased"}}
_MEASURE_ALIASES = {"0.4.4": {"N5": "T1", "N6": "LSC"}}


def canonical_names(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """Map a version's group/measure names back to the `base.HEADER` names."""
    groups = {v: k for k, v in _GROUP_ALIASES.get(version, {}).items()}
    measures = _MEASURE_ALIASES.get(version, {})
    return df.assign(group=df["group"].replace(groups),
                     measure=df["measure"].replace(measures))


def ensure_ecol_source(version: str, cache: Path = CACHE) -> Path:
    """Download (once), verify and extract the ECoL tarball; return its dir."""
    url, sha = ECOL_TARBALLS[version]
    target = cache / version / "ECoL"
    if target.is_dir():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    tarball = cache / f"ECoL_{version}.tar.gz"
    if not tarball.exists():
        urllib.request.urlretrieve(url, tarball)  # noqa: S310 - pinned CRAN URL
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if digest != sha:
        tarball.unlink()
        raise ValueError(f"ECoL {version}: SHA-256 {digest} != pinned {sha}")
    with tarfile.open(tarball) as tf:
        tf.extractall(cache / version)  # noqa: S202 - hash-verified archive
    return target


def ecol_measures(bags: list[tuple[np.ndarray, np.ndarray]], version: str = "0.3.0",
                  groups: tuple[str, ...] = ("overlapping", "neighborhood"),
                  workdir: Path | None = None) -> pd.DataFrame:
    """Run ECoL `version` on each (X, y) bag.

    Returns one row per (bag, group, measure) with `value` and the per-measure
    wall time `seconds`, names canonicalised to `base.HEADER`.
    """
    src = ensure_ecol_source(version)
    aliases = _GROUP_ALIASES.get(version, {})
    r_groups = ",".join(aliases.get(g, g) for g in groups)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        tmp_path = Path(tmp)
        for b, (X, y) in enumerate(bags):
            cols = {f"x{j}": X[:, j] for j in range(X.shape[1])}
            pd.DataFrame({**cols, "class": np.asarray(y)}).to_csv(
                tmp_path / f"bag_{b:05d}.csv", index=False)
        out = tmp_path / "out.csv"
        subprocess.run(["Rscript", str(R_SCRIPT), str(src), str(tmp_path),
                        str(out), r_groups], check=True, capture_output=True)
        df = pd.read_csv(out)
    df["bag"] = df.pop("file").str.extract(r"bag_(\d+)")[0].astype(int)
    return canonical_names(df, version)[["bag", "group", "measure", "value", "seconds"]]
