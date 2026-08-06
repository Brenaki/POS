# Análise dos Projetos: POS vs ComplexityGuidedEnsemble

> Documento de referência interna. Descreve o que cada repositório realiza, à luz da tese
> "Classifier Pool Generation based on a Two-level Diversity Approach"
> (doi:10.1016/j.inffus.2022.09.001), e como eles se relacionam com o subprojeto Oracle_N.

## Repositórios

- **POS** — https://github.com/marcosmonteirojr/_PGCDS
  Implementação de referência da tese. É o repositório que modificamos neste projeto.
- **ComplexityGuidedEnsemble (CGE)** — https://github.com/matheusMrs07/ComplexityGuidedEnsemble/tree/branch-de-teste
  Reinterpretação independente da ideia "complexidade-guiada". Somente leitura para nós; fonte
  de ideias/código a portar, não alvo de edições.

## A tese base (contexto)

O artigo **"Exploring diversity in data complexity and classifier decision spaces for pool
generation"** propõe **geração de pool de classificadores via estratégia em dois níveis**:

- **Nível 1 — Diversidade no espaço de complexidade dos dados:** particiona o treino em *bags*
  que representam subproblemas com dificuldades diferentes, medidas por métricas de complexidade
  do dataset (overlapping, neighborhood, linearity, dimensionality, balance, network).
- **Nível 2 — Diversidade no espaço de decisão dos classificadores:** usa *double fault* entre
  classificadores para garantir que erram em amostras diferentes.
- **Otimização:** algoritmo evolutivo (DEAP, NSGA-II) com fitness multiobjetivo ponderado por
  `fit_value` (dispersão das complexidades + divergência de decisões − acurácia do conjunto),
  e dois critérios de parada: `maxdistance` (máx. dispersão) ou `maxacc` (máx. acurácia do voting).
- **Resultado relatado:** melhor acurácia em 327/336 experimentos (97,3%) contra métodos clássicos
  de geração de pool, com e sem seleção dinâmica (DCS/DES via DESlib).

## POS — implementação fiel da tese

**O que realiza:** é a implementação de referência do artigo, em Python, com acoplamento direto
à tese.

- `Cpx.py` (262 linhas) — **camada de complexidade e diversidade**:
  - `complexity_data3()` chama o pacote R **ECoL** via **rpy2** para calcular 22 medidas de
    complexidade (6 grupos: overlapping, neighborhood, linearity, dimensionality, balance,
    network).
  - `diversitys()` calcula *double fault* par a par via `deslib.util.diversity`.
  - `dispersion_linear()` / `dispersion()` — distância média par a par entre bags no espaço de
    complexidade, normalizada min-max.
  - `biuld_classifier` (Perceptron) e `biuld_classifier_tree` (DecisionTree) — construtores de
    base learners. (Typo histórico `biuld_*` preservado até ADR de rename.)
  - `voting_classifier` — *hard voting* via `mlxtend.EnsembleVoteClassifier`.

- `pool_generation.py` (657 linhas) — **núcleo evolutivo (DEAP)**:
  - `poolGeneration` — classe única que orquestra tudo.
  - `generate_bags()` — cria `nr_bags` subamostras estratificadas do treino (`tam_bags`).
  - `get_complexity()` — avalia cada bag: complexidade (R), acurácia individual, predição em
    validação, *double fault* → vetor fitness 3D.
  - `crossover()` / `mutation()` — operadores genéticos sobre **índices de instâncias** (não
    sobre features), preservando estratificação mínima (`verify_bag`).
  - `the_function()` — callback por geração que aplica `max_distance` ou `max_acc` e persiste o
    melhor conjunto.
  - `generate()` — entry point: split treino/val, votação de complexidades para escolher
    `types`, loop de `iteration` execuções do `eaMuPlusLambda` do DEAP.
  - `get_best_types()` — *voting* sobre 100 subamostras para selecionar as 2 medidas de
    complexidade mais variantes → alimenta o GA.

- `sample.ipynb` — demonstra geração de pool + avaliação com **DESlib** (OLA, LCA, Rank) em
  `load_wine`.

**Características marcantes:**
- Dependência pesada de **R + ECoL** (não há fallback Python puro).
- `requirements.txt` em **UTF-16** (gotcha: leitura textual precisa de `encoding='utf-16'`).
- `R_HOME` hardcoded para caminho Windows no notebook.
- Sem testes, sem lint, sem CI; `pool_generation.py` já excede o limite de 300 linhas.
- Nomenclatura com typos históricos (`biuld_*`) preservada intencionalmente.

## ComplexityGuidedEnsemble (CGE) — reformulação independente

**O que realiza:** é uma **reinterpretação** da ideia "complexidade-guiada", não uma continuação
direta da tese. Substitui o GA por amostragem determinística e elimina R.

- `complexity_sampler.py` — **ComplexityGuidedSampler**:
  - Calcula complexidade **em Python puro** via **`pyhard`** (overlap, error rate,
    neighborhood) ou função customizada — sem R/rpy2.
  - Amostragem por **peso Gaussiano** centrado em `μ` (alvo de dificuldade) com `σ`
    (seletividade): cada bag focaliza um nível de dificuldade.
  - Balanceamento de classes por ressampling independente por classe.
  - Suporte multiclasse nativo.

- `complexity_guided_ensemble.py` — **ComplexityGuidedEnsemble**:
  - Gera `n_estimators` bags varrendo `μ ∈ [0,1]`, treina um base learner por bag, agrega por
    *soft voting* (probabilidades alinhadas) ou *hard voting*.
  - Sem GA, sem NSGA-II, sem *double fault* explícito — a diversidade emerge da variação
    sistemática de `μ`.
  - `n_jobs` para paralelismo.

- `complexity_guided_ensemble_v2.py` — variante com *active learning* e otimização de fitness
  opcional.
- `autorank.ipynb`, `experimental_CG-Ensemble.ipynb`, CSVs/PNGs
  (`algorithm_ranking.png`, `critical_difference_diagram.png`,
  `f1_comparison_ir_groups_sidebyside.png`) — **benchmarking estatístico** comparando CGE vs
  Random Forest vs Bagging via *Autorank* (Friedman/Nemenyi), com foco em **datasets
  desbalanceados**.

**Características marcantes:**
- **Sem R** — removida a maior barreira de infraestrutura.
- **Determinístico** (não-estocástico) por construção: `μ` varre [0,1] uniformemente, o que dá
  diversidade sem GA.
- **Interpretável**: cada `μ` documenta a dificuldade-alvo do membro.
- Tem testes (`test_complexity_sampler.py`, `test_ensemble.py`, `test_ensemble_v2.py`) e demos
  — embora não cobrem 80%.
- Repositório de terceiros; para nós é **somente leitura**.

## Comparação direta

| Dimensão | POS (tese) | ComplexityGuidedEnsemble |
|---|---|---|
| **Mecanismo de diversidade** | GA multiobjetivo (DEAP/NSGA-II) sobre índices de instâncias | Varredura determinística de `μ` (peso Gaussiano) |
| **Complexidade dos dados** | R/ECoL via rpy2 (22 medidas, 6 grupos) | `pyhard` Python puro (3 medidas + custom) |
| **Diversidade de decisão** | *double fault* explícito (Nível 2 da tese) | Implícita pela variação de `μ` |
| **Base learners** | Perceptron / DecisionTree | Configurável (padrão DecisionTree) |
| **Agregação** | Hard voting (mlxtend) | Soft voting (recomendado) ou hard |
| **Multiclasse** | Via R/ECoL (indireto) | Nativo |
| **Reprodutibilidade** | Sementes espalhadas; sem fixture de experimentos | `random_state` centralizado |
| **Dependência externa** | R instalado + `ECoL` + `R_HOME` | Apenas `pip install pyhard` |
| **Testes** | Nenhum | `test_*.py` (parcial) |
| **Tamanho dos arquivos** | `pool_generation.py` 657 linhas (viola regra ≤300) | `complexity_sampler.py` 1044 linhas (também excede) |
| **Alinhamento à tese** | Implementação literal | Generalização que descarta o GA |

## Síntese para o subprojeto Oracle_N

- **POS** é o ponto de partida obrigatório: implementa fielmente os dois níveis da tese e já
  integra DESlib (necessário para os experimentos de comparação previstos no cronograma).
- **ComplexityGuidedEnsemble** oferece duas **portabilidades candidatas** (decidir via ADR):
  1. **Substituir R/ECoL por `pyhard`** — remove a dependência de R, simplificando
     drasticamente a infraestrutura.
  2. **Adotar o regime determinístico `μ`-sweep** como *baseline* alternativo de geração de
     pool, contra o qual o método evolutivo da tese será comparado nas fases de
     experimentação.

O ponto de tensão metodológico: a tese fundamenta-se no *double fault* como medida de
diversidade de decisão (Nível 2); o CGE abandona esse sinal explícito. Se o subprojeto Oracle_N
depende da caracterização precisa da diversidade do pool — o que é provável, dado que Oracle_N
é função direta da concordância entre classificadores — o *double fault* provavelmente deve ser
**mantido**, e o CGE serve apenas como *baseline* ou como fonte da simplificação `pyhard`.