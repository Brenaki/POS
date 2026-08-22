# Oracle_N — Análise do Impacto de Oracles em Diferentes Níveis na Acurácia de Sistemas de Múltiplos Classificadores

> Subprojeto de pesquisa que estende a tese de doutorado *"Classifier Pool Generation based on a Two-level Diversity Approach"* ([Monteiro et al., 2022](https://doi.org/10.1016/j.inffus.2022.09.001)) com um limite superior Oracle generalizado.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![Licença](https://img.shields.io/badge/licença-MIT-green)](LICENSE)
[![Testes](https://img.shields.io/badge/testes-105%20passando-brightgreen)](#testes)
[![ADR](https://img.shields.io/badge/ADRs-10-blueviolet)](docs/adr/)

---

## Visão Geral

Sistemas de Múltiplos Classificadores (MCS) combinam modelos distintos para melhorar
a generalização. O **Oracle** é um modelo abstrato de referência que considera uma
amostra corretamente classificada quando **pelo menos um** classificador do pool
prediz seu rótulo verdadeiro. Embora amplamente utilizado como limite superior
teórico para Seleção Dinâmica de Classificadores (DCS) e Seleção Dinâmica de
Ensembles (DES), o Oracle tradicional é frequentemente **excessivamente otimista**:
métodos reais de seleção podem não conseguir identificar o classificador correto
mesmo quando ele existe no pool.

Este projeto investiga uma **generalização do Oracle** para múltiplos níveis de
exigência. Em vez de requerer apenas um classificador correto, definimos:

> **Oracle_N(x) = 1** quando **pelo menos N** classificadores do pool predizem
> corretamente o rótulo da amostra **x**, e **0** caso contrário.

Isso produz uma **curva Oracle_N** (Oracle_1 ≥ Oracle_2 ≥ ... ≥ Oracle_M) que
transita do Oracle tradicional otimista (N=1) para limites progressivamente mais
conservadores. O foco da pesquisa é **N = 1, 2, 3, 4, 5** — níveis intermediários
que revelam a transição entre limites otimistas e realistas de desempenho.

### Perguntas-chave

1. Como o limite superior Oracle se comporta à medida que aumenta o número mínimo
   de classificadores corretos?
2. Os níveis intermediários de Oracle (N=2..5) estão mais próximos dos métodos reais
   de combinação (votação majoritária, média de probabilidades, DCS/DES) do que o
   Oracle tradicional (N=1)?
3. A curva Oracle_N pode servir como ferramenta de diagnóstico para avaliar se a
   diversidade de um pool é explorável por métodos reais ou meramente superficial?

### Métodos de geração de pool comparados

| Modo | Descrição | Origem |
|------|-----------|--------|
| `ga` | Geração de pool via GA com fitness guiado por complexidade (NSGA-II, DEAP) | Monteiro et al., 2022 |
| `rf` | Baseline RandomForest (sem GA, config default do sklearn) | Novo baseline |

---

## Estrutura do repositório

```
POS/
├── pos/
│   ├── complexity/          # Medidas de complexidade de dados
│   │   ├── fast_adapter.py      # Adapter numpy puro (105x mais rápido)
│   │   ├── overlapping_measures.py  # F1, F2, F3, F4
│   │   ├── neighborhood_measures.py # N1, N2, kDN, LSC
│   │   ├── pyhard_adapter.py    # Backend pyhard legado (fora do caminho quente)
│   │   └── base.py              # Mapeamentos grupo/medida
│   ├── pool/                # Geração de pool de classificadores
│   │   ├── pool_generation.py   # Facade poolGeneration (GA via DEAP)
│   │   ├── fitness_evaluator.py # Fitness guiado por complexidade + ThreadPool
│   │   ├── ea_loop.py           # Loop evolutivo DEAP
│   │   └── ...                  # bag_generator, genetic_operators, etc.
│   ├── oracle/              # Implementação Oracle_N
│   │   ├── oracle_n.py          # oracle_n_accuracy(matrix, n)
│   │   ├── oracle_curve.py     # Curva Oracle_1..M
│   │   ├── correctness_matrix.py # Matriz de acertos do pool
│   │   ├── pool_evaluation.py  # evaluate_pool() — todas as métricas
│   │   ├── run_recorder.py     # Orquestrador de experimentos + --resume
│   │   ├── run_helpers.py      # Helpers de git/deps/hash, manifests
│   │   ├── resume_helpers.py   # Retomar de folds completos
│   │   ├── arff_loader.py     # Carrega datasets .arff
│   │   ├── dataset_catalog.py # Catálogo de 31 datasets
│   │   └── ...
│   └── ...                  # normalization, diversity, voting, etc.
├── scripts/
│   └── run_experiment.py        # CLI: --smoke / --full / --resume
├── tests/                      # 105 testes (unit + integration + characterization)
├── docs/
│   ├── adr/                    # 10 Architecture Decision Records
│   ├── protocol.md             # Protocolo experimental formal
│   └── SubProjeto...md         # Especificação do subprojeto
├── results/
│   ├── datasets/catalog.csv    # Metadados dos 31 datasets
│   └── experiments/            # Saídas das execuções (manifests + summaries)
├── ROADMAP.md                  # Acompanhamento de progresso
└── requirements.txt            # Deps pinadas (UTF-16 BOM)
```

---

## Protocolo experimental

### Datasets

**31 datasets de classificação** em formato `.arff` dos repositórios UCI, KEEL e
OpenML. Faixa: 178–19.020 amostras, 2–8 classes, 2–60 features, razão de
desbalanceamento 1.0–71.5. Metadados em `results/datasets/catalog.csv`.

### Validação

- **10-fold stratified cross-validation** (`StratifiedKFold`, `random_state=42`).
- Split interno treino/validação: 20% do treino via
  `np.random.default_rng(42 + fold_idx)` — determinístico, independente por fold.
- Resultados reportados como média ± desvio sobre 10 folds.

### Configuração do pool

| Parâmetro | Valor |
|-----------|-------|
| Tamanho do pool (M) | 100 |
| Gerações do GA | 20 |
| Classificador base | DecisionTreeClassifier |
| Algoritmo GA | NSGA-II (DEAP) |
| Baseline RF | RandomForestClassifier(n_estimators=100, bootstrap=True) |
| Modos | `ga` (complexidade guiada), `rf` (baseline) |
| Medidas de complexidade | F1, F2, F3, F4 (overlapping), N1, N2, kDN, LSC (neighborhood) |

### Métricas

- **Curva Oracle** (Oracle_1..M): monotônica não-crescente.
- **Oracle_1..5**: níveis de foco para análise.
- **Votação majoritária**, **média de probabilidades**.
- **Acurácias individuais** de cada classificador.
- **Matriz de double-fault** (diversidade par-a-par).
- DCS/DES (OLA, LCA, Rank, META-DES): planejado para o marco 6.

### Reprodutibilidade

Cada execução persiste:
- `run_manifest.json` — config, git SHA, versões de deps, timestamps.
- `fold_manifest.json` — seeds, hashes dos índices (SHA256), métricas por fold.
- `summary.csv` — uma linha por (dataset, fold, mode), incluindo `oracle_1..5`.
- `per_dataset_summary.csv` — média ± desvio por (dataset, mode).
- `correctness_matrix.npy`, `predictions.npz` — arrays pesados (gitignored, recuperáveis via manifest).

---

## Início rápido

### Pré-requisitos

- Python 3.10
- Dependências: `pip install -r requirements.txt`
- Datasets: incluídos em `Dataset/` (31 arquivos `.arff`)

> **Nota:** R/ECoL não é mais necessário para o pipeline de experimentos.
> O `fast_adapter` (ADR 0010) substitui pyhard por numpy puro.
> R permanece opcional apenas para regenerar golden values legados.

### Smoke test (valida o pipeline, ~2 min)

```bash
python scripts/run_experiment.py --smoke
```

Executa 3 datasets (Wine, Banana, Vehicle) × 2 modos (ga, rf) × 3 folds = 18 linhas.
Valida: geração de manifest, monotonicidade do Oracle, Oracle_1 ≥ votação majoritária.

### Experimento completo (~3 horas com --jobs 4)

```bash
python scripts/run_experiment.py --full --jobs 4
```

Executa 31 datasets × 10 folds × 2 modos = 620 folds. Saída em
`results/experiments/<timestamp>_<git_sha>/`.

### Retomar execução interrompida

```bash
python scripts/run_experiment.py --resume results/experiments/<dir> --jobs 4
```

Pula folds já completados (detectados via `fold_manifest_*.json`) e continua
de onde parou.

---

## Decisões arquiteturais (ADRs)

| # | Título | Resumo |
|---|-------|--------|
| 0001 | Versão Python e deps | Pin Python 3.10, sklearn 1.2.0, numpy 1.24.1, deap 1.3.3 |
| 0002 | Limite de 150 LOC por arquivo | Todos arquivos ≤150 linhas (exceto README, ROADMAP) |
| 0003 | Remover R via pyhard | Substituir R/ECoL por pyhard para medidas de complexidade |
| 0004 | Split Cpx.py | Decompor Cpx.py monolítico em pacote `pos/` |
| 0005 | Split pool_generation.py | Decompor arquivo de 657 linhas em módulos `pos/pool/` |
| 0006 | Arquitetura Oracle_N | Definir fórmula Oracle_N, protocolo 10-fold CV |
| 0007 | Bug DEAP generation_function | Documentar bug pre-existente do loop DEAP |
| 0008 | Fix DEAP generation_function | Corrigir loop DEAP + fallback stop-criteria |
| 0009 | Reprodutibilidade de experimentos | Layout de artefatos, manifests, seeds, rastreamento git SHA |
| 0010 | Complexidade rápida (numpy) | Reimplementar 8 medidas em numpy puro — 105× speedup |

---

## Restrições de engenharia

Este projeto segue regras de engenharia rigorosas (conforme `AGENTS.md`):

- **SOLID, TDD (Kent Beck), Refactoring (Martin Fowler), DDD (Eric Evans)**
- **≥80% de cobertura de testes** (atualmente 105 testes, cobertura em progresso)
- **≤150 linhas de código por arquivo** (exceto README, ROADMAP)
- **ADR por decisão arquitetural** — 10 ADRs até o momento
- **ROADMAP.md** atualizado a cada sessão

---

## Testes

```bash
# Executar todos os testes
python -m pytest

# Executar com cobertura
python -m pytest --cov=pos --cov-fail-under=80

# Apenas testes do Oracle_N
python -m pytest tests/unit/oracle/
```

105 testes em três suítes:
- **Testes unitários** (`tests/unit/`): Oracle_N, matriz de acertos, arff loader, etc.
- **Testes de integração** (`tests/integration/`): pipeline de experimento ponta-a-ponta.
- **Testes de caracterização** (`tests/characterization/`): golden values do
  codebase original antes da refatoração.

---

## Otimização de performance

O pipeline de experimentos foi otimizado de uma estimativa de **~58 dias** para
**~3 horas** (speedup de ~460×) através de quatro mudanças documentadas no
[ADR 0010](docs/adr/0010-fast-complexity-numpy.md):

| Otimização | Impacto |
|---|---|
| Medidas de complexidade em numpy puro (sem pyhard) | 105× por chamada |
| Eliminar 3 classificadores não-usados + 11 medidas não-usadas | incluído acima |
| Reuso de pool em `_run_fold` | 2× por fold |
| `--jobs N` ThreadPoolExecutor | ~2× em multi-core |

Os resultados são preservados: splits de CV, algoritmo GA e classificadores base
são idênticos. A única diferença é no valor do fitness (≤0.04 na medida N1 devido
à matriz de distância), o que não afeta significativamente a curva Oracle.

---

## Cronograma

| # | Atividade | Meses | Status |
|---|-----------|-------|--------|
| 1 | Revisão bibliográfica (MCS, DCS/DES, diversidade, Oracle) | Set–Nov/2026 | Em andamento |
| 2 | Seleção de bases e protocolo experimental | Out–Nov/2026 | Concluído |
| 3 | Implementação do pool e avaliação individual | Nov/2026–Jan/2027 | Concluído |
| 4 | Implementação do Oracle tradicional e multi-nível | Dez/2026–Fev/2027 | Concluído |
| 5 | Relatório Semestral (PROPESP, até 15 mar 2027) | Mar/2027 | Pendente |
| 6 | Experimentos comparativos (votação, média probs, DCS/DES) | Fev–Jun/2027 | Pendente |
| 7 | Análise dos resultados, gráficos e recomendações | Ago–Nov/2027 | Pendente |
| 8 | Relatório final (até 11 set 2027) | Jul–Ago/2027 | Pendente |

---

## Referências

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

## Licença

MIT. Veja o código da tese original para licenciamento upstream.