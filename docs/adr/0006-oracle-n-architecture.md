# ADR 0006 — Arquitetura Oracle_N (generalização do Oracle)

- **Status**: Accepted
- **Date**: 2026-08-18

## Context

O subprojeto (`docs/SubProjeto...md`) propõe uma generalização do Oracle
tradicional para múltiplos níveis de exigência. O Oracle tradicional
(Oracle_1) considera uma amostra corretamente classificada se **≥1
classificador** do pool acertar. A generalização Oracle_N exige que **≥N
classificadores** acertem simultaneamente, produzindo uma curva
monotônica não-crescente `Oracle_1 ≥ Oracle_2 ≥ ... ≥ Oracle_M`.

## Definição formal

Para um pool de M classificadores e uma amostra x com rótulo verdadeiro y:

```
correctness_matrix[i, j] = 1  se clf_j prediz y_i corretamente, senão 0

Oracle_N(x_i) = 1  se sum(correctness_matrix[i, :]) >= N, senão 0

oracle_n_accuracy(matrix, N) = mean(Oracle_N(x_i) para todo i)
                             = (nº de amostras com >= N acertos) / (nº total de amostras)
```

Propriedades:
- `oracle_n_accuracy(M) <= oracle_n_accuracy(N)` para `N <= M` (monotônica)
- `oracle_n_accuracy(1)` = Oracle tradicional
- `oracle_n_accuracy(M)` = unanimidade (todos devem acertar)
- `oracle_n_accuracy(0)` = 1.0 (trivialmente, todo mundo "passa")

## Protocolo experimental (Q3 = 10-fold)

- **10-fold stratified cross-validation** com `random_state=42` fixo
- Para cada fold: treinar pool (via `poolGeneration.generate()`), gerar
  matriz de acertos no fold de teste, calcular curva Oracle_1..M
- Reportar média ± desvio da curva Oracle_N sobre os 10 folds
- Datasets: 31 arquivos `.arff` em `POS/Dataset/` (Wine, Banana, Glass, etc.)

## Arquitetura (todos <150 LOC)

```
pos/oracle/
  __init__.py                  # re-exports públicos
  correctness_matrix.py        # build_correctness_matrix(pool, X, y) -> np.ndarray (n, M)
  oracle_n.py                  # oracle_n_accuracy(matrix, n) -> float
  oracle_curve.py              # oracle_curve(matrix) -> dict[int, float]
  comparison.py                # majority_vote_accuracy, mean_probs_accuracy
  arff_loader.py               # load_arff_dataset(path) -> (X, y)
  experiment.py                # run_experiment(dataset_path, M, ...) -> results dict

tests/unit/oracle/
  test_correctness_matrix.py   # matriz trivial hand-crafted
  test_oracle_n.py             # Oracle_1 tradicional, Oracle_M unanimidade, N=0
  test_oracle_curve.py         # monotonicidade Oracle_1 >= ... >= Oracle_M
  test_comparison.py           # majority vote com maioria conhecida
  test_arff_loader.py          # carrega Wine.arff, valida shape/classes
  test_experiment.py           # mock pool, valida estrutura do resultado
```

## Comparadores (Q2 = básicos nesta fase)

Implementados em `pos/oracle/comparison.py`:
- `majority_vote_accuracy(pool, X, y)` — votação majoritária hard
- `mean_probs_accuracy(pool, X, y)` — média das probabilidades preditas

DCS/DES (OLA, LCA, Rank, META-DES via DESlib) ficam para Fase 5 — exigem
regiões de competência e splits mais elaborados.

## Origem do pool (Q1 = poolGeneration.get_pool())

Usar `poolGeneration.generate() → get_pool()` (agora funcional após ADR 0008).
Não criar pool manual — aproveitar o GA de diversidade/complexidade do
trabalho original.

## Consequências

- **+** Oracle_N é puramente funcional sobre a matriz de acertos — não
  depende do GA, do DEAP, nem do pyhard. Testes unitários são triviais.
- **+** A curva Oracle_N é uma propriedade do pool+dataset, independente
  do protocolo experimental.
- **−** O experimento completo (10-fold × 31 datasets × pool GA) é
  computacionalmente caro (~270s por pool × 10 folds × 31 datasets).
  Paralelização fica para milestone de experimentos.
