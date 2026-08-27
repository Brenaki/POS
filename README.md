# Oracle_N — Analyzing the Impact of Multi-Level Oracles on Multiple Classifier Systems

> Research subproject extending the doctoral thesis *"Classifier Pool Generation based on a Two-level Diversity Approach"* ([Monteiro et al., 2022](https://doi.org/10.1016/j.inffus.2022.09.001)) with a generalized Oracle upper bound.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-105%20passing-brightgreen)](#testing)
[![ADR](https://img.shields.io/badge/ADRs-10-blueviolet)](docs/adr/)

---

## Overview

Multiple Classifier Systems (MCS) combine diverse models to improve generalization.
The **Oracle** is an abstract reference model that considers a sample correctly
classified when **at least one** classifier in the pool predicts its true label.
While widely used as a theoretical upper bound for Dynamic Classifier Selection (DCS)
and Dynamic Ensemble Selection (DES), the traditional Oracle is often **overly
optimistic**: real selection methods may fail to identify the correct classifier even
when one exists in the pool.

This project investigates a **generalization of the Oracle** to multiple levels of
stringency. Instead of requiring only one correct classifier, we define:

> **Oracle_N(x) = 1** when **at least N** classifiers in the pool correctly predict
> the label of sample **x**, and **0** otherwise.

This produces an **Oracle_N curve** (Oracle_1 ≥ Oracle_2 ≥ ... ≥ Oracle_M) that
transitions from the optimistic traditional Oracle (N=1) to progressively more
conservative upper bounds. The research focus is on **N = 1, 2, 3, 4, 5** —
intermediate levels that reveal the transition between optimistic and realistic
performance ceilings.

### Key questions

1. How does the Oracle upper bound behave as the minimum number of correct
   classifiers increases?
2. Are intermediate Oracle levels (N=2..5) closer to real combination methods
   (majority vote, mean probabilities, DCS/DES) than the traditional Oracle (N=1)?
3. Can the Oracle_N curve serve as a diagnostic tool to assess whether a pool's
   diversity is exploitable by real methods, or merely superficial?

### Pool generation methods compared

Since **ADR 0020** the comparison isolates one variable — *how the pool was
generated* — by using a Decision Tree pool of 100 classifiers everywhere:

| Treatment | Description | Source |
|------|-------------|--------|
| `pgdcs_a` | PGDCS: complexity-measure vote + NSGA-II, exit at the generation with the highest majority-vote accuracy | Monteiro 2023 |
| `pgdcs_d` | Same trajectory, exit at the generation with the highest global dispersion (`Gdisp`) | Monteiro 2023 |
| `bagging` | `BaggingClassifier(n_estimators=100, max_samples=0.5, random_state=86)` | Monteiro 2023 (published seed) |

`ga` (fixed F1/T1), `randbag` and `rf` are no longer scientific modes, and the
Perceptron leaves the canonical path: with a tree on both sides, a difference
cannot be blamed on the base classifier.

---

## Repository structure

```
POS/
├── pos/
│   ├── complexity/          # Data complexity measures
│   │   ├── fast_adapter.py      # Drop-in numpy-only adapter (105x faster)
│   │   ├── overlapping_measures.py  # F1, F2, F3, F4
│   │   ├── neighborhood_measures.py # N1, N2, kDN, LSC
│   │   ├── pyhard_adapter.py    # Legacy pyhard backend (not in hot path)
│   │   └── base.py              # Group/measure mappings
│   ├── pool/                # Classifier pool generation
│   │   ├── pool_generation.py   # poolGeneration facade (GA via DEAP)
│   │   ├── fitness_evaluator.py # Complexity-guided fitness + ThreadPool
│   │   ├── ea_loop.py           # DEAP evolutionary loop
│   │   └── ...                  # bag_generator, genetic_operators, etc.
│   ├── oracle/              # Oracle_N implementation
│   │   ├── oracle_n.py          # oracle_n_accuracy(matrix, n)
│   │   ├── oracle_curve.py     # Oracle_1..M curve
│   │   ├── correctness_matrix.py # Hit matrix from pool
│   │   ├── pool_evaluation.py  # evaluate_pool() — all metrics
│   │   ├── run_recorder.py     # Experiment orchestrator + --resume
│   │   ├── run_helpers.py      # Git/deps/hash helpers, manifests
│   │   ├── resume_helpers.py   # Resume from completed folds
│   │   ├── arff_loader.py     # Load .arff datasets
│   │   ├── dataset_catalog.py # 31-dataset catalog
│   │   └── ...
│   └── ...                  # normalization, diversity, voting, etc.
├── scripts/
│   └── run_experiment.py        # CLI: --smoke / --full / --resume
├── tests/                      # 105 tests (unit + integration + characterization)
├── docs/
│   ├── adr/                    # 10 Architecture Decision Records
│   ├── protocol.md             # Formal experimental protocol
│   └── SubProjeto...md         # Subproject specification (PT-BR)
├── results/
│   ├── datasets/catalog.csv    # 31 datasets metadata
│   └── experiments/            # Run outputs (manifests + summaries)
├── ROADMAP.md                  # Progress tracker
└── requirements.txt            # Pinned deps (UTF-16 BOM)
```

---

## Experimental protocol

Full protocol in [`docs/protocol.md`](docs/protocol.md); the decision record is
[ADR 0020](docs/adr/0020-monteiro-2023-compatible-profiles.md). A run is always
**one explicit profile**, never a mix:

| | `pos-oracle-v2` | `monteiro-2023-dt-compatible-v1` |
|---|---|---|
| Purpose | the POS Oracle_N experiment | reproducible slice of Monteiro 2023 |
| Partition | 10-fold stratified + disjoint three-way split | 20 independent 50/25/25 holdouts |
| Root seed | 42 | 86 |
| Measure vote | per fold | once per dataset |
| Comparable to the paper | no | yes (Tables 6, 7, A.10) |

### Datasets

The **28 datasets of Table 2** of Monteiro 2023. Every snapshot is checked
against the table and stamped with the SHA-256 of the POS-approved snapshot
(the paper publishes no hashes). Current state — see
[`docs/dataset_conformance.md`](docs/dataset_conformance.md): **23
`conformant`, 4 `approved_with_note`, 1 `mismatch`** (P2, 2000 vs 5000
instances, excluded from the published comparison).

### Pool configuration

| Parameter | Value |
|-----------|-------|
| Pool size (M) | 100 |
| GA generations | 20 (generation 0 evaluated, never selected; ties keep the oldest) |
| Population / children | 100 / 100, crossover 0.9, mutation 0.1 |
| Base classifier | DecisionTreeClassifier (all defaults) |
| Bag sampling | random **with replacement**, 50% of the training set, not stratified |
| Measure vote | 10 rounds × 100 sub-samples; one Overlapping + one Neighborhood measure |
| Exit criterion | `pgdcs_a` = max MVR accuracy, `pgdcs_d` = max `Gdisp` |

### Metrics

- **Oracle_N** (`oracle_<n>`): at least `n` classifiers correct — the project's
  definition, decreasing. `oracle_classic` = `oracle_1` is the only point
  comparable to the paper's Table 7.
- **Greedy Oracle curve** (`oracle_greedy_<n>`): nested construction, increasing;
  a POS contribution that depends on its ordering.
- Checkpoints `1, 5, 10, …, 100` (21 points) for both; full 100-point curves in
  `oracle_curve_json` and `oracle_greedy_curve_json`.
- **Majority vote** (`mv_*`) and **mean probability** (`mp_*`), each with
  accuracy, macro precision/recall/F1 and balanced accuracy.
- **DCS/DES**: LCA, OLA, Rank, KNORA-E, KNORA-U, META-DES (DESlib).
- **Double-fault mean** and individual accuracies.

### Reproducibility

Only Bagging's `random_state=86` is a published seed; every other seed is
derived deterministically from the profile's root seed and recorded as a
modernization. Each run persists `run_manifest.json` (config, fingerprint, git
SHA, dep versions and the full conformance catalog), `summary.csv` (schema v2),
`per_dataset_summary.csv`, per-split manifests and the correctness matrices.

---

## Quick start

### Prerequisites

- Python 3.10
- Dependencies: `pip install -r requirements.txt`
- Datasets: included in `Dataset/` (31 `.arff` files)

> **Note:** R/ECoL is no longer required for the experiment pipeline.
> The `fast_adapter` (ADR 0010) replaces pyhard with pure numpy.
> R remains optional for regenerating legacy golden values only.

### Smoke test (validates the pipeline, ~1 min)

```bash
python scripts/run_experiment_v2.py --profile pos-oracle-v2 --smoke --datasets Wine,Banana
```

Reduced folds, generations and M. Marked `comparable=false` — it validates the
pipeline, never a scientific result.

### Full experiment

```bash
# the POS experiment
python scripts/run_experiment_v2.py --profile pos-oracle-v2 --full

# the slice comparable to the paper
python scripts/run_experiment_v2.py --profile monteiro-2023-dt-compatible-v1 --full
```

Output in `results/experiments/<timestamp>_<git_sha>/`. `--jobs` defaults to
cores − 1; it does not change results, since every bag carries its own seed.

### Resume, compare, audit the data

```bash
python scripts/run_experiment_v2.py --resume results/experiments/<dir>
python scripts/compare_reference.py results/experiments/<dir>
python scripts/dataset_conformance.py
```

Resume skips any (dataset, split, treatment) that already has a manifest on
disk, and never re-runs the PGDCS trajectory of a finished split.

---

## Architecture decisions (ADRs)

| # | Title | Summary |
|---|-------|---------|
| 0001 | Python version and deps | Pin Python 3.10, sklearn 1.2.0, numpy 1.24.1, deap 1.3.3 |
| 0002 | File size cap 150 LOC | All files ≤150 lines (except README, ROADMAP) |
| 0003 | Remove R via pyhard | Replace R/ECoL with pyhard for complexity measures |
| 0004 | Split Cpx.py | Decompose monolithic Cpx.py into `pos/` package |
| 0005 | Split pool_generation.py | Decompose 657-line file into `pos/pool/` modules |
| 0006 | Oracle_N architecture | Define Oracle_N formula, 10-fold CV protocol |
| 0007 | DEAP generation function bug | Document pre-existing DEAP loop bug |
| 0008 | Fix DEAP generation function | Patch DEAP loop + fallback stop-criteria |
| 0009 | Experiment reproducibility | Artifact layout, manifests, seeds, git SHA tracking |
| 0010 | Fast complexity (numpy) | Reimplement 8 measures in pure numpy — 105× speedup |

---

## Engineering constraints

This project follows strict engineering rules (per `AGENTS.md`):

- **SOLID, TDD (Kent Beck), Refactoring (Martin Fowler), DDD (Eric Evans)**
- **≥80% test coverage** target (currently 105 tests, coverage in progress)
- **≤150 lines of code per file** (except README, ROADMAP)
- **ADR per architectural decision** — 10 ADRs to date
- **ROADMAP.md** updated every session

---

## Testing

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=pos --cov-fail-under=80

# Run only Oracle_N tests
python -m pytest tests/unit/oracle/
```

105 tests across three suites:
- **Unit tests** (`tests/unit/`): Oracle_N, correctness matrix, arff loader, etc.
- **Integration tests** (`tests/integration/`): end-to-end experiment pipeline.
- **Characterization tests** (`tests/characterization/`): golden values from
  original codebase before refactoring.

---

## Performance optimization

The experiment pipeline was optimized from an estimated **~58 days** to **~3 hours**
(a ~460× speedup) through four changes documented in [ADR 0010](docs/adr/0010-fast-complexity-numpy.md):

| Optimization | Impact |
|---|---|
| Pure-numpy complexity measures (no pyhard) | 105× per call |
| Eliminate 3 unused classifiers + 11 unused measures | included above |
| Pool reuse in `_run_fold` | 2× per fold |
| `--jobs N` ThreadPoolExecutor parallelism | ~2× on multi-core |

Results are preserved: CV splits, GA algorithm, and base classifiers are identical.
The only difference is fitness values (≤0.04 on N1 due to distance matrix), which
does not significantly affect the Oracle curve.

---

## Timeline

| # | Activity | Months | Status |
|---|----------|--------|--------|
| 1 | Literature review (MCS, DCS/DES, diversity, Oracle) | Sep–Nov 2026 | In progress |
| 2 | Dataset selection & experimental protocol | Oct–Nov 2026 | Done |
| 3 | Pool implementation & individual evaluation | Nov 2026–Jan 2027 | Done |
| 4 | Oracle (traditional + multi-level) implementation | Dec 2026–Feb 2027 | Done |
| 5 | Semestral report (PROPESP, by Mar 15 2027) | Mar 2027 | Pending |
| 6 | Comparative experiments (majority vote, prob. mean, DCS/DES) | Feb–Jun 2027 | Pending |
| 7 | Results analysis, charts, recommendations | Aug–Nov 2027 | Pending |
| 8 | Final report (by Sep 11 2027) | Jul–Aug 2027 | Pending |

---

## References

- Monteiro, R. M. et al. "Exploring diversity in data complexity and classifier
  decision spaces for pool generation." *Information Fusion*, 2022.
  DOI: [10.1016/j.inffus.2022.09.001](https://doi.org/10.1016/j.inffus.2022.09.001)
- Kuncheva, L. I.; Whitaker, C. J. "Measures of Diversity in Classifier Ensembles."
  *Machine Learning*, 51, 181–207, 2003.
- Cruz, R. M. O.; Sabourin, R.; Cavalcanti, G. D. C. "Dynamic classifier selection:
  recent advances and perspectives." *Information Fusion*, 41, 195–216, 2018.
- Souza, M. A.; Cavalcanti, G. D. C. "On the Characterization of the Oracle for
  Dynamic Classifier Selection." *IJCNN*, 2017.

---

## License

MIT. See the original thesis code for upstream licensing.