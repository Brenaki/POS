"""Extract ResNet-18 embeddings for the image battery (ADR 0022).

Runs in the isolated torch environment, never in the pinned main venv:

    ~/.local/share/pos/venv-embed/bin/python scripts/extract_image_embeddings.py \
        [--datasets MNIST,CIFAR10CatDog] [--cache ~/.cache/pos-images]

Per dataset in `pos.images.embedding_dataset.IMAGE_DATASETS`:

1. stratified experiment sample from the official *test* split and a disjoint
   PCA-fit sample from the official *train* split (fixed seeds);
2. 512-d penultimate-layer features of torchvision ResNet-18 with ImageNet
   weights (`IMAGENET1K_V1`), images resized to 224x224, grey replicated to RGB;
3. PCA to `n_components`, fit on the train-split sample only;
4. `Dataset/<name>.arff` plus `Dataset/<name>.json` with the full provenance
   (indices, versions, weights, explained variance, SHA-256).

Afterwards run `scripts/build_pos_catalog.py` so the gate admits them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from pos.images.embedding_dataset import (  # noqa: E402
    IMAGE_DATASETS,
    project,
    stratified_sample,
    write_arff,
)

SEED = 42
WEIGHTS = models.ResNet18_Weights.IMAGENET1K_V1
GREY = {"MNIST", "FashionMNIST"}


def _transform(source: str) -> transforms.Compose:
    steps = [transforms.Grayscale(num_output_channels=3)] if source in GREY else []
    steps += [transforms.Resize((224, 224)), transforms.ToTensor(),
              transforms.Normalize(WEIGHTS.meta.get("mean", (0.485, 0.456, 0.406)),
                                   WEIGHTS.meta.get("std", (0.229, 0.224, 0.225)))]
    return transforms.Compose(steps)


def _load(source: str, train: bool, cache: Path):
    cls = getattr(datasets, source)
    return cls(str(cache), train=train, download=True, transform=_transform(source))


def _backbone() -> torch.nn.Module:
    model = models.resnet18(weights=WEIGHTS)
    model.fc = torch.nn.Identity()
    return model.eval()


@torch.no_grad()
def _embed(ds, idx: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx.tolist()), batch_size=128, num_workers=0)
    return np.concatenate([model(x).numpy() for x, _ in loader])


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def extract(name: str, cache: Path, model: torch.nn.Module) -> dict:
    """Write one snapshot and its provenance; return the provenance dict."""
    spec = IMAGE_DATASETS[name]
    t0 = time.time()
    test, train = _load(spec.source, False, cache), _load(spec.source, True, cache)
    y_test, y_train = np.asarray(test.targets), np.asarray(train.targets)
    exp_idx = stratified_sample(y_test, spec.n_experiment, SEED, spec.classes)
    fit_idx = stratified_sample(y_train, spec.n_pca_fit, SEED + 1, spec.classes)
    X, pca = project(_embed(train, fit_idx, model), _embed(test, exp_idx, model),
                     spec.n_components, SEED)
    arff = REPO_DIR / "Dataset" / f"{name}.arff"
    write_arff(arff, X, y_test[exp_idx], relation=name)
    prov = {
        "adr": "0022", "dataset": name, "source": f"torchvision.datasets.{spec.source}",
        "classes": list(spec.classes) or "all", "seed": SEED,
        "experiment_split": "official test", "experiment_indices": exp_idx.tolist(),
        "pca_fit_split": "official train", "pca_fit_indices": fit_idx.tolist(),
        "backbone": "resnet18", "weights": str(WEIGHTS), "embedding_dim": 512,
        "input_size": [224, 224], "grey_to_rgb": spec.source in GREY,
        "n_components": spec.n_components,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "instances": int(len(exp_idx)), "n_classes": int(len(np.unique(y_test[exp_idx]))),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "python": platform.python_version(), "seconds": round(time.time() - t0, 1),
        "arff_sha256": _sha256(arff),
    }
    arff.with_suffix(".json").write_text(json.dumps(prov, indent=1) + "\n")
    return prov


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(IMAGE_DATASETS))
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "pos-images"))
    args = p.parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = _backbone()
    for name in args.datasets.split(","):
        prov = extract(name, Path(args.cache), model)
        print(f"[{name}] {prov['instances']} x {prov['n_components']}, "
              f"{prov['n_classes']} classes, PCA var {prov['pca_explained_variance']:.3f}, "
              f"{prov['seconds']} s")


if __name__ == "__main__":
    main()
