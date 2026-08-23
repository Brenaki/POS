# Protocolo Experimental — Oracle_N

> Protocolo formal para os experimentos do subprojeto
> `docs/SubProjeto - ANÁLISE DO IMPACTO DE ORACLES EM DIFERENTES NÍVEIS NA ACURÁCIA DE SISTEMAS DE MÚLTIPLOS CLASSIFICADORES.md`.
> Referências: ADR 0006 (arquitetura Oracle_N), ADR 0009 (reprodutibilidade).

## 1. Datasets

31 datasets de classificação em formato `.arff` em `POS/Dataset/`:

- **Fonte**: repositórios públicos (UCI, KEEL, OpenML) — versão congelada no
  commit `275f795` do repo.
- **Metadados**: `results/datasets/catalog.csv` (gerado por
  `pos.oracle.dataset_catalog.build_catalog`).
- **Faixa**: 178–19020 amostras, 2–8 classes, 2–60 features, imbalance ratio
  1.0–71.5 (ver catálogo para detalhes por dataset).
- **Convenção**: última coluna do `.arff` é o rótulo (`Class`/`class`/`target`/
  `label` ou última por default). Labels string são label-encoded via
  `sklearn.LabelEncoder`.

## 2. Protocolo de validação cruzada

- **10-fold stratified cross-validation** (`sklearn.model_selection.StratifiedKFold`).
- `shuffle=True`, `random_state=42` fixo (splits externos idênticos em toda
  re-execução).
- Para cada fold `i` ∈ {0, ..., 9}:
  - **Treino** (9 folds): usado para treinar o pool de classificadores.
  - **Teste** (1 fold): usado para gerar a matriz de acertos e calcular
    Oracle_N, majority vote, mean probs.
- **Split interno treino/validação** (para `poolGeneration.generate()`, que
  precisa de `X_val`/`y_val` para avaliar fitness):
  - Do treino do fold, 20% é reservado para validação via
    `pos.oracle.fold_splitter.stratified_val_split()`, que usa
    `train_test_split(..., stratify=y_train)`.
  - Semente `random_state + fold_idx` = 42+i — determinística e independente
    por fold.
  - **Estratificado desde o ADR 0014.** Antes era `rng.choice(...)` puro, o
    que em bases desbalanceadas (Thyroid 12.1, Faults 12.2) podia entregar ao
    GA uma validação com classes ausentes.
- **Métricas reportadas**: média ± desvio sobre os 10 folds.

### 2.1. Elegibilidade da base (ADR 0014)

Antes de rodar, cada base passa por `check_dataset_viability()`:

1. `min_class_count >= n_folds` — sem isso o `StratifiedKFold` não coloca a
   classe mais rara em todos os folds.
2. Para o modo `ga`, a classe mais rara precisa sobreviver à cadeia
   k-fold → validação → bag com **≥ 2 instâncias** (exigência de
   `GeneticOperatorsMixin.verify_bag`).

Bases reprovadas são **puladas explicitamente** e registradas em
`run_manifest.json` sob `skipped_datasets` (com o motivo). Elas não são
removidas de `FULL_DATASETS`, para que a exclusão fique auditável.

Com o catálogo atual, **Ecoli** (2 instâncias na menor classe) e **Glass**
(9 < 10 folds) são reprovadas: o protocolo científico roda com **29 das 31
bases**. Rodá-las exigiria outro protocolo (menos folds ou `tam_bags` maior),
documentado à parte.

## 3. Pool de classificadores — três modos

### 3.1. Modo `ga` (pool via algoritmo genético)

- `poolGeneration.generate() → get_pool()` (tese original, ADR 0008).
- `classifier="perc"` → **Perceptron linear** em bags gerados pelo GA.
  É o classificador-base da tese (Monteiro et al. 2022, seç. 5) e está
  nomeado no Objetivo 4 do SubProjeto. Até o ADR 0015 o código usava
  `DecisionTreeClassifier` e o caminho `perc` estava quebrado.
  `--base-classifier tree` mantém a variante com árvore.
- **Limitação a reportar**: o Perceptron não expõe `predict_proba`, logo a
  *média das probabilidades preditas* (Objetivo 6) **não é definida** para o
  pool GA-Perceptron — `mean_probs` fica `None`. A comparação continua
  disponível nos modos `rf` e `bagging`.
- `types=["F1", "T1"]` hard-coded (bypassa `get_best_types`, que chamaria
  pyhard 100×12 = caro — ver `pos/pool/complexity_voter.py`).
- **F1** = Maximum Fisher's Discriminant Ratio (ECoL dataset-level)
- **T1** = Fraction of Hyper-spheres Covering Data (ECoL N5)
- `random_state` propagado para `random.seed()`, `np.random.seed()`,
  `train_test_split` e `DecisionTreeClassifier(random_state=rs+i)`.
- `nr_generation`: controla gerações do GA. Smoke test usa 3 (mínimo para
  exercitar a avaliação de prole além da 1ª geração — ADR 0014); execução
  científica usa 20.
- **Gdisp**: `maxdistance` agora seleciona a geração com maior dispersão
  global (matriz Nx3 de fitness), não a última geração (ADR 0013). A geração
  registrada em `gen_temp` é a real desde o ADR 0014.
- **Escopo**: este é um GA com `types=["F1","T1"]` **fixo**. O estágio do
  PGDCS que vota as medidas de complexidade mais dispersas
  (`get_best_types`) existe mas está bypassado, portanto o modo `ga` é uma
  **variante GA-F1/T1** do PGDCS, não sua reprodução integral. Ver ADR 0014.

### 3.2. Modo `bagging` (baseline controlado — Bagging)

- `BaggingClassifier(n_estimators=M, random_state=rs, bootstrap=True,
  max_features=1.0)` — usa TODAS as features por árvore.
- Isola o efeito do GA do feature subsampling do Random Forest.
- Retorna `bag.estimators_` — lista de `DecisionTreeClassifier` fitted.

### 3.3. Modo `rf` (baseline forte — RandomForest)

- `RandomForestClassifier(n_estimators=M, random_state=rs, bootstrap=True)`
  com **config default** do sklearn (`max_features='sqrt'`).
- Retorna `forest.estimators_` — mesma interface do pool GA.
- **Decisão (ADR 0009)**: manter defaults. `max_features='sqrt'` introduz
  diversidade de features que o GA não explora — diferença *esperada e
  informativa*.

## 4. Métricas

Para cada (dataset, fold, modo), calcular:

- **Curva Oracle completa**: `Oracle_1, Oracle_2, ..., Oracle_M` onde M =
  tamanho do pool. Monotônica não-crescente.
  - `Oracle_1` = Oracle tradicional (≥1 clf correto).
  - `Oracle_M` = unanimidade (todos M corretos).
- **Majority vote**: `scipy.stats.mode` sobre predições hard do pool.
- **Mean probs**: argmax da média de `predict_proba` sobre o pool.
- **Acurácias individuais**: acurácia de cada classificador do pool no
  fold de teste.
- **Double-fault matrix** (M×M): diversidade par-a-par via
  `pos.diversity.double_fault` — fração de amostras onde ambos erram.
- **DCS/DES** (OLA, LCA, Rank, META-DES via DESlib): **futuro** —
  cronograma Ativ.6 (Fev–Jun/2027), fora desta rodada.

## 5. Reprodutibilidade

- `random_state=42` fixo para `StratifiedKFold` (splits externos).
- `random_state + fold_idx` para splits internos treino/val.
- `RandomForestClassifier(random_state=rs)` determinístico.
- `poolGeneration` herda aleatoriedade do DEAP (ver limitação §3.1).
- **Manifest por execução** (`run_manifest.json`): timestamp ISO, git SHA,
  branch, config completa, versões de todas as deps relevantes
  (`scikit-learn`, `numpy`, `deap`, `deslib`, `pandas`, `scipy`, `pyhard`).
- **Manifest por fold** (`fold_manifest.json`): seeds, hashes SHA256 dos
  índices train/test/val, shapes, M, acurácias individuais, oracle_curve,
  majority, mean_probs, double_fault_mean.
- Reexecutar o mesmo `config` + mesmo `git_sha` reproduz os mesmos
  `oracle_curve_mean` (com ressalva DEAP paralelo).

## 6. Artefatos salvos

Ver ADR 0009 para layout completo. Resumo:

| Artefato | Versionado | Descrição |
|---|---|---|
| `run_manifest.json` | sim | config + git sha + versões deps |
| `summary.csv` | sim | uma linha por (dataset, fold, mode) |
| `per_dataset_summary.csv` | sim | mean±std por (dataset, mode) |
| `fold_manifest.json` | sim | seeds, hashes, métricas por fold |
| `correctness_matrix_<mode>.npy` | não | (n_test × M) — recuperável via manifest |
| `predictions_<mode>.npz` | não | preds + probs por clf — recuperável via manifest |

O sufixo `<mode>` foi acrescentado no ADR 0014: sem ele, rodar
`--mode ga,bagging,rf` fazia cada modo sobrescrever os artefatos do anterior
no mesmo diretório de fold, e só o último sobrevivia em disco.

## 7. Configurações de execução

| Profile | Datasets | Folds | `nr_generation` | Modos | Uso |
|---|---|---|---|---|---|
| `--smoke` | Wine, Banana, Vehicle | 3 | 3 | ga + bagging + rf | validar fluxo |
| `--full` | 31 listadas, 29 elegíveis | 10 | 20 | ga + bagging + rf | execução científica |

`--jobs` usa por padrão `nº de núcleos - 1` (ADR 0015). Isso é seguro para
reprodutibilidade porque cada bag carrega semente própria
(`random_state + posição`), independente da ordem de execução.

`--mode` default = `ga,bagging,rf`. Bagging entrou justamente como baseline
controlado (ADR 0012), então faz parte da configuração oficial.

Smoke test **não** valida qualidade científica — apenas o pipeline de
reprodução: manifest, arquivos, monotonicidade da curva Oracle,
`Oracle_1 ≥ majority_vote`, reprodutibilidade com seed. Desde o ADR 0014 o
smoke usa `nr_generation=3` e não 1: com uma única geração o GA nunca executa
o caminho de avaliação de prole a partir da segunda geração, que foi
exatamente onde estava o bug P0 mais grave do projeto.

### 7.1. Conjunto de bases vs. a tese (ADR 0015)

A Tabela 1 da tese usa **28 bases**, que são exatamente o nosso
`FULL_DATASETS` menos `{Ecoli, Glass, Magic}` (o `Adult.arff` corresponde ao
"Australian" da tabela: 690 × 14 × 2).

- **Ecoli** e **Glass** são reprovadas pelo portão de elegibilidade do
  ADR 0014 (classe minoritária < 10 folds).
- **Magic** (19.020 instâncias) foi **mantida deliberadamente** como base
  extra de escalabilidade. Ela está fora do conjunto de comparação direta
  com o paper, e isso deve ser explicitado no relatório.

Ou seja: as 28 bases da tese estão todas cobertas, e Magic é um acréscimo
nosso, não uma substituição.

## 8. Foco científico — Oracle_1..5

O subprojeto investiga **Oracles em diferentes níveis** (Oracle_N onde N = 1..M).
O foco da análise é **N = 1, 2, 3, 4, 5** — níveis intermediários que revelam a
transição entre o Oracle tradicional (otimista) e limites mais conservadores.

- `oracle_1` = Oracle tradicional (≥1 classificador correto)
- `oracle_2` = ≥2 classificadores corretos simultaneamente
- `oracle_3` = ≥3 classificadores corretos
- `oracle_4` = ≥4 classificadores corretos
- `oracle_5` = ≥5 classificadores corretos
- `oracle_M` = unanimidade (todos M corretos) — registrado para referência

A curva completa (Oracle_1..M) é preservada em `oracle_curve_json` no
`summary.csv` e em `oracle_curve` no `fold_manifest.json` para auditoria.
As colunas `oracle_1`..`oracle_5` no `summary.csv` e
`per_dataset_summary.csv` facilitam a análise comparativa com majority vote,
mean probs e (futuro) DCS/DES.

**Referência experimental**: a tese base (Monteiro et al., 2022) usou 28
datasets, 20 replicações, pool=100, 20 gerações do GA, 7 métodos de fusão
(MVR + 6 DCS/DES). Nosso protocolo usa 31 datasets, 10-fold CV (10 folds),
pool=100, 20 gerações do GA, 3 modos (ga + bagging + rf), focando Oracle_1..5.