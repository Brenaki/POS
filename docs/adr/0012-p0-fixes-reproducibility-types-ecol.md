# ADR 0012: Correções P0 — reprodutibilidade do GA, tipos de medidas e definições ECoL

## Date
2026-08-22

## Status
Accepted

## Context

Uma revisão externa identificou três bugs críticos (P0) que invalidavam
qualquer resultado científico do modo `ga`:

### P0 #1: GA não era reprodutível

`build_pool_ga()` recebia `random_state` mas **não passava para
`poolGeneration`**. Além disso:
- `generate_bags()` chamava `train_test_split` sem `random_state`
- `genetic_operators.py` usava `random.randint`/`random.sample` sem
  `random.seed`

Duas runs com `random_state=42` produziam pools diferentes. Toda a
reprodutibilidade do modo `ga` era falsa.

### P0 #2: `types=["F1","T1"]` era ignorado pelo fast_adapter

O `fast_adapter.complexity_data3()` recebia `types` mas **ignorava
completamente** — iterava todas as medidas de cada grupo. Com
`group=["overlapping","neighborhood"]`, o vetor de complexidade tinha 11
valores (F1,F1v,F2,F3,F4,N1,N2,N3,N4,T1,LSC) em vez de 2 (F1,T1).

O fitness `evaluate_linear_dispersion` usava `dist[0]` e `dist[1]`:
- `dist[0]` = F1 (correto)
- `dist[1]` = F1v = **0.0** (placeholder não implementado)

Então o **segundo objetivo do NSGA-II estava degenerado** — sempre 0.
O GA otimizava F1 + 0.0 + diversity em vez de F1 + T1 + diversity.

### P0 #3: F1–F4 usavam definições do pyhard, não do ECoL

As medidas em `overlapping_measures.py` eram versões **instance-level**
do pyhard (fraction of features in overlap per instance), não as medidas
**dataset-level** do ECoL R (Maximum Fisher's Discriminant Ratio, Volume
of Overlap Region, etc.) que o PGDCS usa.

T1 (neighborhood) retornava `0.0` como placeholder.

O PGDCS depende das medidas de complexidade para construir o espaço onde
o GA procura diversidade. Métricas diferentes → fitness diferente →
bags diferentes → pool diferente → Oracle_N diferente.

## Decision

### 1. Reprodutibilidade do GA (P0 #1)

- `poolGeneration.__init__` recebe `random_state` (novo parâmetro)
- `generate()` chama `random.seed(rs)` e `np.random.seed(rs)` quando
  `random_state` é int
- `generate_bags()` recebe `random_state` e passa para
  `train_test_split(random_state=rs_i)` onde `rs_i = rs + i` (offset
  por bag para variar sem perder reprodutibilidade)
- `build_pool_ga()` passa `random_state` para `poolGeneration`

### 2. fast_adapter respeita `types` (P0 #2)

`complexity_data3()` agora tem dois modos:
- `types=None`: computa todas as medidas de cada grupo (legacy mode)
- `types=["F1","T1"]`: computa apenas `types[i]` para `group[i]`
  (one-to-one correspondence, matching ECoL R behavior)

Retorna um escalar por elemento de `group` (não 11).

### 3. Reimplementar F1–F4 e T1 com definições ECoL (P0 #3)

**overlapping_measures.py** + **fisher_measures.py** (split para LOC cap):
- **F1**: Maximum Fisher's Discriminant Ratio → 1/(Fisher+1), mean over features
- **F1v**: Directional Fisher via LDA projection (one-vs-one, mean)
- **F2**: Volume of Overlap Region → product of per-feature overlap/range
- **F3**: Max Individual Feature Efficiency → 1 - max(non-overlap fraction)
- **F4**: Collective Feature Efficiency → fraction remaining after
  iterative best-F3 elimination

**neighborhood_fast.py**:
- **T1**: 1-NN leave-one-out error rate (ECoL N3, same as original
  complexity library T1). KDTree para 1-NN queries.

### 4. P1: ADR 0010 corrigido

Removida a afirmação contraditória "os valores de Oracle são preservados".
Substituída por: "a composição do pool e, consequentemente, os valores
experimentais de Oracle_N podem mudar."

### 5. P2: Modo `bagging` adicionado

- `pos/oracle/bagging_pool.py`: `build_bagging_pool()` via
  `BaggingClassifier(max_features=1.0)` — usa todas as features,
  isolando o efeito do GA do efeito de feature subsampling do RF
- `run_recorder` e `run_experiment.py` suportam `--mode ga,rf,bagging`

## Consequences

- **+** GA agora é reprodutível: mesma seed → mesmo pool
- **+** GA otimiza F1 + T1 + diversity (não F1 + 0.0 + diversity)
- **+** Medidas de complexidade correspondem ao ECoL/PGDCS
- **+** Bagging baseline isola efeito do GA
- **−** Runs anteriores (commit fcae405, 30 folds) são inválidas —
  usavam F1v=0.0 como 2º objetivo e medidas pyhard em vez de ECoL
- **−** Novo arquivo `fisher_measures.py` adicionado (split de LOC)

## Tests

- 3 testes de reprodutibilidade (`test_reproducibility.py`)
- 5 testes de fast_adapter types (`test_fast_adapter_types.py`)
- 20 testes de medidas ECoL (`test_ecol_measures.py`)
- 3 testes de bagging pool (`test_bagging_pool.py`)
- 158 testes totais passando, 0 falhas

## Referências

- ECoL R source: github.com/lpfgarcia/ECoL, R/feature-based.R, R/neighborhood.R
- ADR 0010 — fast complexity numpy (corrigido)
- ADR 0009 — reprodutibilidade de experimentos
- ADR 0011 — KD-tree neighborhood measures