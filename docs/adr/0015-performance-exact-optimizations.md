# ADR 0015: Otimizações exatas do GA, Perceptron da tese e paralelismo

## Date
2026-08-22

## Status
Accepted

## Context

O objetivo era reduzir o custo do experimento **sem alterar a base científica**.
A referência é dupla:

- `docs/SubProjeto - ANÁLISE DO IMPACTO DE ORACLES...md` — o objeto de estudo é
  a curva **Oracle_N**. O Objetivo 4 pede pools construídos com "algoritmos
  simples e amplamente conhecidos, como **Random Forest e Perceptron**".
- `docs/Classifier Pool Generation based on a Two-level Diversity Approach.pdf`
  (Monteiro et al., 2022) — o PGDCS, base de código de partida.

Medição inicial (`--full`, modo GA, serial), por chamada de `get_complexity`
numa dobra de Magic (M=100, n_val=3423, bag=6847):

| componente          | tempo   | share |
|---------------------|---------|-------|
| complexidade (T1)   | 36,9 s  | 53%   |
| `diversitys`        | 18,0 s  | 26%   |
| fit das árvores     |  9,9 s  | 14%   |
| `voting_classifier` |  3,9 s  |  6%   |
| resto               |  0,3 s  |  1%   |
| **total**           | **69,0 s** | | 

Extrapolado: **8,05 h só para Magic**.

## Decision

Todas as mudanças de desempenho abaixo são **exatas**: verificadas
bit-a-bit ou dentro do épsilon de ponto flutuante.

### 1. `diversitys` vetorizado — 8.877x

`deslib.util.diversity.double_fault` percorre **cada amostra em Python puro**,
e `diversitys` o chamava M*(M-1) vezes: ~34 milhões de iterações
interpretadas por chamada.

A Eq. 2 da tese é `DDV[C_i] = Σ_j DF(C_i, C_j) / (N-1)`, e
`DF = N00/n = |{s : ambos erram}| / n`. Com `W[i,s] = (p_i[s] != y[s])`,
isso é `N00 = (W @ W.T) / n` — um único matmul.

Magic (M=100, n=3423): **24,40 s → 0,0027 s**. Diferença máxima 2,2e-16
(ordem de somatório).

O mesmo laço existia em `pos/oracle/pool_evaluation._double_fault_matrix`
(4950 chamadas por dobra, por modo) e foi substituído pela mesma função.

### 2. Memoização da avaliação de bags — ~1,75x no GA inteiro

Bags são *append-only* e imutáveis; a semente da árvore é
`random_state + posição`, também estável. Logo
`(complexidade, score, predições)` é função pura da identidade do bag.

Isso importa porque a seleção mu+lambda reapresenta sobreviventes a cada
geração: `the_function` chama `get_complexity(population=...)` sobre 100
indivíduos **já avaliados**. Metade de todo o trabalho de cada geração era
recomputação.

Fingerprint bit-a-bit (predições do pool + curva Oracle, seed 42):

| base    | antes   | depois  | idêntico |
|---------|---------|---------|----------|
| Wine    |  5,18 s |  3,26 s | sim      |
| Vehicle |  8,73 s |  4,91 s | sim      |
| CTG     | 20,88 s | 11,97 s | sim      |

Estimadores treinados só são retidos sob `maxacc` — o único caminho que os
consome. Guardar ~2100 árvores treinadas custaria mais memória do que o
cache economiza.

### 3. `voting_classifier` sob demanda

`EnsembleVoteClassifier` usa `refit=True` por padrão, então cada chamada
**retreinava os 100 classificadores** sobre X_val. O resultado (`score_g`) só
é lido pelo critério `maxacc`; com o padrão `maxdistance` ninguém olha para
ele. Passa a ser calculado apenas quando `stop_criteria == "maxacc"`.

### 4. T1: árvores por classe reaproveitadas

Para `r_i > 0`, todo ponto dentro de `r_i` é necessariamente da mesma classe:
um inimigo está a `d >= 2*r_i`, logo `d <= r_i` exigiria `r_i <= 0`. Então a
consulta de raio pode ir à árvore da própria classe (~n/k pontos) em vez da
árvore completa. `r_i == 0` é a única exceção (duplicata com rótulo
diferente, a distância 0) e cai na árvore completa.

Uma KD-tree por classe agora serve às duas etapas (inimigo mais próximo e
consulta de raio). Verificado em 62 bags de todas as 31 bases: **0
divergências**. Magic: 0,219 s → 0,139 s na consulta de raio.

### 5. Micro-otimizações exatas

- `dispersion_linear` (Eq. 1 da tese) vetorizada: 16x, diferença 0.
- `dispersion` não usa mais `n_jobs=6` numa matriz 100x3: 18x, diferença 0.
- `build_bags` usa indexação avançada em vez de montar lista de views.

### 6. Perceptron linear como classificador-base do GA

A seção 5 da tese: *"we used a linear Perceptron as base classifier"*. O
Objetivo 4 do SubProjeto também nomeia Perceptron explicitamente. O código
usava `DecisionTreeClassifier`, e o caminho `classifier="perc"` estava
**quebrado**: `biuld_classifier` fazia `X_test != None`, que levanta
`ValueError: The truth value of an array ... is ambiguous` para qualquer
X_test numpy.

Corrigido para `is not None`; `n_jobs=4` removido (o GA já paraleliza sobre
bags, o pool aninhado só disputava CPU); `random_state` passa a ser honrado
(o Perceptron embaralha). `PoolBuilderMixin.get_pool()` construía o pool
final com `Perceptron(tol=1.0)` — hiperparâmetros **diferentes** dos usados
no fitness; agora são os mesmos.

`build_pool_ga(classifier=...)` e `--base-classifier {perc,tree}` tornam a
escolha explícita, com `perc` como padrão.

**Consequência que precisa constar no relatório**: o Perceptron não tem
`predict_proba`, então a "média das probabilidades preditas" (Objetivo 6 do
SubProjeto) **não é definida para o pool GA-Perceptron**. `evaluate_pool` e
`run_experiment` já registram `mean_probs = None` nesse caso. A comparação
com média de probabilidades continua disponível para os modos `rf` e
`bagging`. Se ela for exigida também para o pool GA, as saídas são: usar
`tree` como base, ou calibrar o Perceptron, ou definir formalmente uma
"média de decision_function" — nenhuma foi adotada aqui.

### 7. `--jobs` passa a usar a máquina

Escalonamento medido em 4 núcleos: jobs=1 3,85 s → jobs=4 1,60 s (**2,4x**).
O default do CLI vira `nº de núcleos - 1`. Isso é seguro para
reprodutibilidade porque cada bag tem semente própria
(`random_state + posição`), não dependente da ordem de execução.

## Consequences

Estimativa do `--full` no modo GA (21 rodadas x 100 bags x 10 dobras):

| base  | antes  | depois (serial) |
|-------|--------|-----------------|
| Magic | 8,05 h | 2,35 h          |
| WDVG  | 3,32 h | 0,94 h          |

Mais o paralelismo (2,4x) e a troca de DecisionTree por Perceptron.

A suíte completa de testes caiu de 158 s para ~80 s pelas mesmas correções.

**Bases**: Magic, Ecoli e Glass são as três bases do nosso `FULL_DATASETS`
que **não** estão na Tabela 1 da tese (28 bases). Ecoli e Glass já são
reprovadas pelo portão de elegibilidade do ADR 0014. Magic foi mantido
deliberadamente como base extra de escalabilidade, fora do conjunto de
comparação direta com o paper — isso precisa ser dito no relatório.

**Desvios de protocolo ainda em aberto** (documentados, não corrigidos aqui):

1. A tese usa divisão 50/25/25 com 20 replicações; usamos 10-fold com 20% de
   validação (ADR 0006).
2. `G_disp` (Eq. 3 da tese) divide por `N-1`; `dispersion()` divide por `N`
   (inclui a diagonal zero). É um fator constante `N/(N-1)`, que não altera
   qual geração é o argmax — logo não muda a seleção, só a escala do valor.
3. `get_best_types()` (etapa 1 do PGDCS) segue bypassado — ADR 0014.

## Tests

- `tests/unit/test_diversity_vectorized.py` — equivalência com deslib.
- `tests/unit/complexity/test_t1_hyperspheres.py` — equivalência do T1.
- `tests/unit/pool/test_eval_cache.py` — cache não altera resultados e evita
  reavaliação de sobreviventes.
- `tests/characterization/test_cpx_pure_functions.py` — os dois testes que
  fixavam o bug do Perceptron foram convertidos: agora exigem que o caminho
  funcione e seja reprodutível.
