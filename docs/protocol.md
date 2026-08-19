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
    `np.random.default_rng(random_state + fold_idx).choice(...)`.
  - Semente `random_state + fold_idx` = 42+i — determinística e independente
    por fold.
- **Métricas reportadas**: média ± desvio sobre os 10 folds.

## 3. Pool de classificadores — dois modos

### 3.1. Modo `ga` (pool via algoritmo genético)

- `poolGeneration.generate() → get_pool()` (tese original, ADR 0008).
- `classifier="tree"` → `DecisionTreeClassifier` em bags gerados pelo GA.
- `types=["F1", "T1"]` hard-coded (bypassa `get_best_types`, que chamaria
  pyhard 100×12 = caro — ver `pos/pool/complexity_voter.py`).
- `nr_generation`: controla gerações do GA. Smoke test usa 1 (degenerado,
  valida fluxo); execução científica usa ≥20.
- **Limitação conhecida**: DEAP com `jobs=8` paralelo pode não ser
  bit-exact reprodutível; `jobs=1` para reprodutibilidade estrita (lento).
  Registrar `jobs` no manifest.

### 3.2. Modo `rf` (baseline sem GA — RandomForest)

- `RandomForestClassifier(n_estimators=M, random_state=rs, bootstrap=True)`
  com **config default** do sklearn (`max_features='sqrt'`,
  `max_samples=None` se bootstrap=True → tamanho do treino).
- Retorna `forest.estimators_` — lista de `DecisionTreeClassifier` fitted,
  mesma interface `.predict`/`.predict_proba` do pool GA.
- **Decisão (ADR 0009)**: manter defaults. `max_features='sqrt'` introduz
  diversidade de features que o GA não explora — diferença *esperada e
  informativa* entre as duas abordagens, não confounder. Para igualar
  condições no futuro: `max_features=1.0` + novo ADR derivado.

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
| `correctness_matrix.npy` | não | (n_test × M) — recuperável via manifest |
| `predictions.npz` | não | preds + probs por clf — recuperável via manifest |

## 7. Configurações de execução

| Profile | Datasets | Folds | `nr_generation` | Modos | Uso |
|---|---|---|---|---|---|
| `--smoke` | Wine, Banana, Glass | 3 | 1 | ga + rf | validar fluxo |
| `--full` | 31 | 10 | 20 | ga + rf | execução científica |

Smoke test **não** valida qualidade científica (pool GA degenerado com
`nr_generation=1`) — apenas o pipeline de reprodução: manifest, arquivos,
monotonicidade da curva Oracle, `Oracle_1 ≥ majority_vote`, reprodutibilidade
com seed.