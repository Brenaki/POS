# ROADMAP — Oracle_N Research Project

> Tracks progress against the cronograma in
> `docs/SubProjeto - ANÁLISE DO IMPACTO DE ORACLES EM DIFERENTES NÍVEIS NA ACURÁCIA DE SISTEMAS DE MÚLTIPLOS CLASSIFICADORES.md`.
> Update this file every session that advances a milestone.

## Legend

- [ ] not started
- [~] in progress
- [x] done
- [!] blocked

## Cronograma (Set/2026 → Ago/2027)

| #  | Atividade                                                              | Meses              | Status |
|----|------------------------------------------------------------------------|--------------------|--------|
| 1  | Revisão bibliográfica sobre MCS, DCS/DES, diversidade e Oracle         | Set–Nov/2026       | [~]    |
| 2  | Seleção das bases de dados e definição do protocolo experimental       | Out–Nov/2026       | [x]    |
| 3  | Implementação do pool de classificadores e avaliação individual         | Nov/2026–Jan/2027  | [x]    |
| 4  | Implementação do Oracle tradicional e dos Oracles em diferentes níveis | Dez/2026–Fev/2027  | [x]    |
| 5  | Relatório Semestral conforme modelo PROPESP (até 15 mar 2027)          | Mar/2027           | [ ]    |
| 6  | Experimentos comparativos (votação majoritária, combinação de probs, DCS/DES) | Fev–Jun/2027 | [ ]    |
| 7  | Análise dos resultados, gráficos e recomendações                       | Ago–Nov/2027       | [ ]    |
| 8  | Relatório final (até 11 set 2027)                                      | Jul–Ago/2027       | [ ]    |

## Engineering milestones (enabling the cronograma)

These are prerequisites mandated by `AGENTS.md` (TDD, Refactoring, ≤150 LOC/file,
ADR-per-decision, ≥80% coverage). They are sequenced before the science milestones.

### Fase 0 — Infraestrutura
- [x] Python 3.10 venv via `uv` (`.venv/`)
- [x] Deps Python não-R instaladas (sklearn, numpy, pandas, deap, deslib, mlxtend)
- [x] `pyproject.toml` (pytest + ruff + mypy + cov≥80)
- [x] `ROADMAP.md` (this file)
- [x] ADRs 0001–0003 criados
- [~] R + ECoL instalados (pendente: `sudo pacman -S r tk` + `R -e 'install.packages("ECoL")'`)
- [~] `rpy2==3.5.7` instalado (bloqueado por R)
- [ ] `pyhard` validado funcional com deps pinadas

### Fase 1 — Suite de caracterização (Fowler, antes de mudar nada)
- [ ] `tests/conftest.py` com fixtures `load_wine_split`, `requires_r` skip
- [ ] `tests/characterization/cpx_characterization_test.py` — golden values de
      `min_max_norm`, `dispersion_linear`, `dispersion`, `diversitys`,
      `voting_classifier`, `complexity_data3` (R), `biuld_classifier[_tree]`
- [ ] `tests/characterization/pool_generation_characterization_test.py` —
      `poolGeneration.generate → get_bags → get_pool` end-to-end (load_wine, rs=42)
- [ ] `tests/unit/` para funções isoladas
- [ ] Cobertura ≥80% das partes não-R; verdes

### Fase 2 — Refatoração estrutural (preserva comportamento)
- [ ] Pacote `pos/` criado
- [ ] `Cpx.py` → `pos/{normalization,dispersion,diversity,voting,complexity/*,classifiers,io_csv}.py` (cada <150 LOC)
- [ ] `pool_generation.py` → `pos/pool/{bag_generator,data_splitter,complexity_voter,fitness_evaluator,genetic_operators,stop_criteria,pool_builder,pool_generation}.py` (<150 LOC cada)
- [ ] Facades `cpx_legacy.py` + `pool_generation_legacy.py` re-exportam API pública
- [ ] `sample.ipynb` continua rodando sem alteração
- [ ] ADRs 0004–0005

### Fase 3 — Remoção do R via pyhard (modifica comportamento)
- [ ] `pos/complexity/pyhard_adapter.py` com `ClassificationMeasures`
- [ ] Substituir `ecol.overlapping/neighborhood/...` em `complexity_data3`
- [ ] Golden values do ECoL preservados em `tests/_ecol_legacy_golden.json`
- [ ] Testes de caracterização atualizados para assertar propriedades (shape, range)
- [ ] ADR 0003 atualizado com decisão pyhard final

### Fase 4 — Oracle_N (TDD, novo)
- [x] `pos/oracle/correctness_matrix.py` — matriz de acertos do pool
- [x] `pos/oracle/oracle_n.py` — `oracle_n_accuracy(matrix, n)`
- [x] `pos/oracle/oracle_curve.py` — curva Oracle_1..M
- [x] `pos/oracle/comparison.py` — vs majority/mean (DCS/DES na Fase 5)
- [x] `pos/oracle/arff_loader.py` — carrega 31 datasets .arff
- [x] `pos/oracle/experiment.py` — 10-fold stratified CV, random_state=42
- [x] `tests/unit/oracle/` com casos triviais (N=1, N=M, N intermediário, monotonicidade)
- [x] ADR 0006 — arquitetura Oracle_N, protocolo 10-fold, fórmula
- [x] ADR 0008 — fix DEAP generation_function (pré-requisito Q1=B)

### Fase 5 — Infraestrutura de experimentos (GA pool vs RF baseline)
- [x] `pos/oracle/dataset_catalog.py` — catálogo de 31 datasets (build/save/load)
- [x] `pos/oracle/pool_evaluation.py` — `evaluate_pool()` (acurácias individuais,
      matriz de acertos, curva Oracle, majority vote, mean probs, double-fault)
- [x] `pos/oracle/random_forest_pool.py` — `build_rf_pool()` via RandomForestClassifier
- [x] `pos/oracle/run_recorder.py` — `record_run()` orquestra experimentos com try/except por fold
- [x] `pos/oracle/run_helpers.py` — helpers de git/deps/hash, `build_pool_ga`/`build_pool_rf`,
      `save_fold_artifacts`, `build_fold_manifest`, `build_summary_row`, `per_dataset_summary`
- [x] `scripts/run_experiment.py` — CLI `--smoke`/`--full`/`--config`/`--mode`/`--jobs`/`--M`/`--dry-run`
- [x] `docs/protocol.md` — protocolo experimental formal
- [x] `results/datasets/catalog.csv` — 31 datasets (178–19020 samples, 2–8 classes, IMB 1.0–71.5)
- [x] Smoke test validado: 3 datasets (Wine/Banana/Vehicle) × 2 modes (ga/rf) × 3 folds = 18 rows
      — oracle_1 ≥ majority_vote (0 violações), curvas monótonas 18/18, M=100, manifest+artefatos OK
- [x] ADR 0009 — layout de reprodutibilidade de experimentos

## Decision log (ADRs)

- `docs/adr/0001-python-version-and-deps.md` — pin Python 3.10, deps de requirements.txt
- `docs/adr/0002-file-size-cap-150.md` — aperta o cap de 300 → 150 LOC/arquivo
- `docs/adr/0003-remove-r-via-pyhard.md` — substituir R/ECoL por pyhard
- `docs/adr/0004-split-cpx.md` — split Cpx.py em pacote pos/
- `docs/adr/0005-split-pool-generation.md` — split pool_generation.py via mixins
- `docs/adr/0006-oracle-n-architecture.md` — arquitetura Oracle_N, protocolo 10-fold
- `docs/adr/0007-deap-generation-function-bug.md` — bug pre-existente DEAP (corrigido por 0008)
- `docs/adr/0008-fix-deap-generation-function.md` — fix DEAP loop + fallback stop-criteria
- `docs/adr/0009-experiment-reproducibility.md` — layout de reprodutibilidade de experimentos

## Notes

- **R install pending**: rode `sudo pacman -S r tk` no Arch, depois
  `R -e 'install.packages("ECoL", repos="https://cloud.r-project.org")'` e
  `export R_HOME="$(R RHOME)"` antes de instalar `rpy2`.
- **pyhard vs deps pinadas**: pyhard 2.2.4 requer numpy~=1.23, sklearn~=1.5,
  scipy~=1.13 — conflita com as versões pinadas do POS. Decisão adiada para a
  Fase 3 (talvez venv separado ou downgrade do pyhard).
