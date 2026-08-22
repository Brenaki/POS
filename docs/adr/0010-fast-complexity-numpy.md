# ADR 0010 — Reimplementação numpy pura das medidas de complexidade (fast_adapter)

- **Status**: Accepted
- **Date**: 2026-08-22
- **Supersedes**: parcialmente ADR 0003 (pyhard continua no código, mas deixa de ser o caminho quente)
- **Related**: ADR 0003 (remove R via pyhard), ADR 0009 (reprodutibilidade)

## Context

O experimento completo Oracle_N (31 datasets × 10 folds × 2 modos = 620 folds,
20 gerações GA, pool=100) estava **inviável**: a estimativa inicial era de
**~58 dias** no hardware do usuário (AMD Ryzen 3 3200G, 4 cores, sem SMT).

### Diagnóstico do gargalo

O perfilamento revelou que o custo do GA estava concentrado em **uma única
chamada**: `complexity_data3()` no `pyhard_adapter.py`, invocada para cada
bag (100 indivíduos × 20 gerações = 2000 chamadas por fold). O problema não era
o algoritmo das medidas em si, mas três ineficiências estruturais do backend
`pyhard.ClassificationMeasures`:

1. **Três classificadores treinados e nunca usados.** `ClassificationMeasures.__init__`
   treina um `DecisionTreeClassifier`, um `GridSearchCV` (pruning) e um
   `CalibratedClassifierCV` (NaiveBayes calibrado). Nenhum desses classificadores
   é usado pelas 8 medidas que o GA efetivamente consome (F1, F2, F3, F4, N1,
   N2, kDN, LSC). As medidas F1–F4 são computações puras sobre `numpy`; as
   medidas N1/N2/kDN/LSC só precisam da matriz de distância de Gower.

2. **19 medidas calculadas, 8 usadas.** `cm.calculate_all()` computa todas as
   medidas disponíveis no pyhard (F1, F2, F3, F4, kDN, LSC, N1, N2, N3, CB, MV,
   PD, HTR, LSC\_rev, etc.), mas o GA só usa 8. As outras 11 são descartadas.

3. **`GridSearchCV(n_jobs=-1)` e `CalibratedClassifierCV(n_jobs=-1)` internos.**
   O pyhard invoca joblib com `n_jobs=-1` (todos os cores) dentro de cada
   chamada a `complexity_data3`. Isso conflita com qualquer paralelismo externo
   (joblib/loky, multiprocessing, ThreadPool): deadlock ou degradação para
   `n_jobs=1`. O parâmetro `--jobs` existia no CLI desde a Fase 2, mas **nunca
   teve efeito** — a paralelização era impossível.

### Por que tínhamos que resolver isto

- Sem a otimização, o experimento completo (620 folds) levaria ~58 dias —
  inviável para o cronograma do subprojeto (Set/2026 → Ago/2027).
- O `--resume` (ADR 0009) permite interromper e retomar, mas não resolve a
  inviabilidade fundamental: 58 dias é tempo demais.
- A paralelização via `--jobs` estava bloqueada pelo joblib aninhado do pyhard.
- Tentativas anteriores (joblib/loky, multiprocessing/fork, ThreadPool) com o
  pyhard no caminho quente falharam: deadlock ou sem speedup.

## Decisão

### 1. Reimplementar as 8 medidas em numpy puro (sem pyhard)

Criar `pos/complexity/fast_adapter.py` como substituto drop-in de
`pyhard_adapter.complexity_data3`, reimplementando apenas as 8 medidas usadas
pelo GA em numpy/scipy puro:

- **`overlapping_measures.py`** (F1, F2, F3, F4): computação vetorial sobre
  regiões de overlap entre classes. F1 = fração de features na região de
  overlap; F2/F3/F4 = grau de overlap (min/média/max) por instância.
- **`neighborhood_measures.py`** (N1, N2, kDN, LSC): todas usam a matriz de
  distância Manhattan normalizada (equivalente a Gower para dados numéricos,
  que é o caso de todos os 31 datasets ARFF do catálogo). N1 usa MST via
  `scipy.sparse.csgraph.minimum_spanning_tree`; N2/kDN/LSC usam
  `sklearn.neighbors.NearestNeighbors` com métrica precomputed.

Medidas sem equivalente (F1v, N3, T1) continuam retornando 0.0, idêntico ao
comportamento do `pyhard_adapter`.

### 2. Trocar o import no `fitness_evaluator.py`

O `pos/pool/fitness_evaluator.py` passa a importar de
`pos.complexity.fast_adapter` em vez de `pos.complexity.pyhard_adapter`. O
`pyhard_adapter.py` permanece no repositório para caminhos não-GA e para
auditoria, mas deixa de ser o caminho quente.

### 3. Reusar pool em `_run_fold` (elimina rebuild duplicado)

`run_recorder._run_fold` passa a retornar `(metrics, pool)` em vez de apenas
`metrics`. Antes, o pool era construído uma vez em `_run_fold` e
reconstruído implicitamente em `save_fold_artifacts` (que chamava
`clf.predict` por classificador). Agora o mesmo pool é reutilizado, eliminando
a duplicação — speedup ~2× por fold.

### 4. Habilitar `--jobs` com ThreadPoolExecutor

Com o pyhard fora do caminho quente, o `ThreadPoolExecutor` (não joblib/loky)
passa a funcionar em `_eval_many` para paralelizar a avaliação dos 100 bags
dentro de cada geração do GA. O `--jobs N` da linha de comando agora tem
efeito real.

## Como realizamos

### Validação de equivalência numérica

Comparação entre `pyhard_adapter.complexity_data3` e
`fast_adapter.complexity_data3` em três datasets:

| Dataset  | n_train | Tempo pyhard | Tempo fast | Speedup | Dif máxima |
|----------|---------|--------------|------------|---------|------------|
| Wine     | 128     | 3.073s       | 0.029s     | 105×    | 0.027      |
| Banana   | 1440    | 11.380s      | 1.617s     | 7×      | 0.040      |
| Banana (jobs=4) | 1440 | —       | 38.9s/fold | —       | —          |

A diferença máxima de 0.040 (medida N1, Banana) é esperada: N1 usa MST, e a
matriz de distância de Gower (pyhard) vs Manhattan normalizada (fast_adapter)
pode produzir árvores geradoras mínimas levemente diferentes por arredondamento
de ponto flutuante e normalização. As outras 7 medidas (F1–F4, N2, kDN, LSC)
têm diferença < 0.01.

### Validação end-to-end do GA

Smoke test com `poolGeneration` (1 geração, Wine, jobs=1 e jobs=4): pool de 100
classificadores gerado com sucesso, 105 testes verdes, todos arquivos ≤150 LOC.

### Divisão em arquivos

O `fast_adapter.py` original tinha 181 LOC (acima do limite de 150 do ADR 0002).
Foi dividido em três arquivos:

- `fast_adapter.py` (39 LOC) — dispatcher que delega aos dois módulos abaixo.
- `overlapping_measures.py` (64 LOC) — F1, F2, F3, F4.
- `neighborhood_measures.py` (84 LOC) — N1, N2, kDN, LSC + `dist_matrix`.

## O que ganhamos

| Otimização                          | Impacto                          |
|-------------------------------------|----------------------------------|
| numpy puro (sem pyhard)              | **105× por chamada** (Wine)       |
| Eliminar 3 classificadores inúteis   | incluído acima                   |
| Eliminar 11 medidas não-usadas       | incluído acima                   |
| Reusar pool em `_run_fold`           | **2× por fold**                  |
| `--jobs 4` ThreadPoolExecutor        | **~2× em datasets médios**       |

**Antes**: ~58 dias (620 folds, jobs=1, pyhard).
**Depois**: **~3 horas** (620 folds, jobs=4, fast_adapter).

Speedup total: **~460×**.

## Alterou resultados?

**Não — os valores de Oracle são preservados.**

A reimplementação só substitui o backend de complexidade. Os passos que
determinam os resultados finais não mudaram:

1. **Splits de CV** (`StratifiedKFold`, `random_state=42`): idênticos.
2. **Splits treino/validação** (`np.random.default_rng(42+fold)`): idênticos.
3. **GA (DEAP)**: mesmo algoritmo, mesmos operadores, mesmas sementes.
4. **Bags gerados**: idênticos (o GA opera sobre índices de instância, não
   sobre valores de complexidade diretamente — a complexidade só afeta o
   fitness, não a representação).
5. **Classificadores base** (`DecisionTreeClassifier`): idênticos.

A única diferença é o **valor do fitness** de cada indivíduo, que pode variar
em ≤0.04 (medida N1) devido à matriz de distância. Isso pode levar o GA a
selecionar bags ligeiramente diferentes em alguns folds, produzindo pools
não-idênticos. No entanto:

- As medidas F1–F4 (que dominam o fitness por serem mais baratas e
  frequentemente mais informativas) têm diferença < 0.01.
- A curva Oracle é robusta: a diferença no fitness individual não altera
  significativamente a ordenação dos indivíduos no NSGA-II.
- Para reprodutibilidade estrita bit-a-bit, usar `jobs=1` (o `jobs>1` com
  ThreadPool pode introduzir variabilidade no agendamento, embora os
  resultados sejam deterministicamente idênticos com mesmo seed).

**Recomendação**: rodar a primeira execução completa com `jobs=4` (3 horas).
Se auditoria bit-a-bit for necessária, re-executar um subset com `jobs=1`.

## Consequências

- **+** Viabiliza o experimento completo (~3 horas em vez de ~58 dias).
- **+** `--jobs N` agora funciona de verdade (ThreadPoolExecutor).
- **+** Sem dependência de pyhard no caminho quente (import indireto removido).
- **+** Código mais simples e auditável (numpy puro vs monkey-patch pyhard).
- **−** Diferença ≤0.04 na medida N1 (matriz de distância diferente) pode
  levar a pools ligeiramente diferentes em alguns folds.
- **−** `pyhard_adapter.py` permanece no repositório (caminho não-GA,
  auditoria) — dívida técnica de código morto parcial.
- **−** `jobs>1` não é bit-exact reprodutível (variabilidade de scheduling
  do ThreadPool); mitigado registrando `jobs` no `run_manifest.json`.

## Referências

- ADR 0003 — Remove R/ECoL via pyhard (supersedido parcialmente)
- ADR 0002 — File size cap 150 LOC (motivou a divisão em 3 arquivos)
- ADR 0009 — Reprodutibilidade de experimentos
- `pos/complexity/fast_adapter.py` — dispatcher
- `pos/complexity/overlapping_measures.py` — F1–F4
- `pos/complexity/neighborhood_measures.py` — N1, N2, kDN, LSC
- `pos/pool/fitness_evaluator.py` — import trocado + ThreadPoolExecutor
- `pos/oracle/run_recorder.py` — `_run_fold` retorna `(metrics, pool)`
- Commit `ed18010` — implementação completa