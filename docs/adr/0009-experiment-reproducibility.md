# ADR 0009 — Reprodutibilidade de experimentos Oracle_N

- **Status**: Accepted
- **Date**: 2026-08-19

## Context

O subprojeto (`docs/SubProjeto...md`) exige que todos os experimentos sejam
**cientificamente reproduzíveis**: sementes fixas, protocolos versionados,
datasets versionados, e o máximo de informação salva sobre cada execução.
O ADR 0006 definiu o protocolo (10-fold stratified CV, `random_state=42`),
mas não definiu *onde* nem *como* os artefatos de cada execução são
persistidos para auditoria e replicação futura.

O usuário pediu explicitamente: "máximo de informação salva sobre os testes
usando a pool com níveis variados de oracles e níveis variados de oracles
sem a pool". Isso requer:

1. Persistir a matriz de acertos e as predições brutas de cada fold.
2. Distinguir duas fontes de pool: (a) o pool GA via `poolGeneration`
   (diversidade/complexidade guiada) e (b) uma baseline **sem o GA** —
   `RandomForestClassifier` com config default.
3. Salvar a curva Oracle completa N=1..M (não apenas níveis fixos).
4. Garantir que uma re-execução com o mesmo config produza os mesmos números.

## Decisão

### Layout de artefatos

Toda execução de experimento grava sob:

```
results/experiments/<ISO_timestamp>_<git_sha_short>/
  run_manifest.json              # config + git sha + versões deps (versionado)
  summary.csv                    # (dataset, fold, mode, oracle_1..M, maj, meanp) (versionado)
  per_dataset_summary.csv        # mean±std por (dataset, mode) (versionado)
  <Dataset>/
    fold_<i>/
      correctness_matrix.npy     # (n_test × M) — gitignored, pesado
      predictions.npz            # preds por clf, y_test, probs — gitignored
      fold_manifest.json         # seeds, índices, shapes, métricas (versionado)
```

- **Versionado no git**: `run_manifest.json`, `summary.csv`,
  `per_dataset_summary.csv`, `fold_manifest.json` (pequenos, texto).
- **Gitignored** (locais apenas): `*.npy`, `*.npz` (arrays pesados, recuperáveis
  via re-execução usando o manifest).
- `results/datasets/catalog.csv` (metadados dos 31 datasets) também versionado.

### Conteúdo do `run_manifest.json`

```json
{
  "timestamp_iso": "2026-08-19T14:30:00",
  "git_sha": "275f795...",
  "git_branch": "master",
  "config": {
    "n_folds": 10,
    "nr_generation": 20,
    "random_state": 42,
    "classifier": "tree",
    "modes": ["ga", "rf"],
    "datasets": ["Wine", "Banana", "..."]
  },
  "deps_versions": {
    "scikit-learn": "1.2.0",
    "numpy": "1.24.1",
    "deap": "1.3.3",
    "deslib": "0.3.5",
    "pandas": "1.5.3",
    "scipy": "...",
    "pyhard": "2.2.4"
  },
  "protocol_ref": "docs/protocol.md",
  "adr_ref": "docs/adr/0006-oracle-n-architecture.md"
}
```

### Conteúdo do `fold_manifest.json`

```json
{
  "dataset": "Wine",
  "fold_idx": 0,
  "mode": "ga",
  "random_state": 42,
  "val_seed": 42,
  "n_train": 106, "n_val": 27, "n_test": 45,
  "M": 100,
  "individual_accuracies": [0.91, 0.88, ...],
  "oracle_curve": [0.98, 0.95, ..., 0.45],
  "majority_vote": 0.93,
  "mean_probs": 0.94,
  "double_fault_mean": 0.12,
  "train_indices_hash": "sha256:...",
  "test_indices_hash": "sha256:..."
}
```

Os hashes dos índices permitem verificar que duas execuções com o mesmo
`random_state` produziram os mesmos splits sem precisar versionar os arrays
de índices.

### Modos de pool

- **`ga`**: `poolGeneration.generate() → get_pool()` — pool via GA de
  diversidade/complexidade (tese original). `classifier="tree"` ->
  `DecisionTreeClassifier` em bags gerados pelo GA.
- **`rf`**: `RandomForestClassifier(n_estimators=M, random_state=rs,
  bootstrap=True)` com **config default** (`max_features='sqrt'`,
  `max_samples=None` se bootstrap=True usa tamanho do treino). Retorna os
  `estimators_` individuais — mesma interface `.predict`/`.predict_proba`
  do pool GA. **Baseline sem o GA de diversidade guiada.**

**Decisão sobre RF config**: manter defaults do sklearn. `max_features='sqrt'`
introduz diversidade de features que o GA não explora — isso é uma diferença
*esperada e informativa* entre as duas abordagens, não um confounder. Se
futuramente quiser igualar condições, usar `max_features=1.0` e registrar
num ADR derivado.

### Reprodutibilidade

- `random_state=42` fixo para o `StratifiedKFold` (splits externos).
- Para cada fold, o split interno treino/val usa
  `np.random.default_rng(random_state + fold_idx)` — determinístico e
  independente por fold.
- `poolGeneration` recebe `random_state` via sua configuração interna (DEAP
  não é totalmente determinístico em paralelo; `jobs=8` pode introduzir
  variabilidade — registrar isso no manifest como limitação conhecida se
  `jobs>1`).
- `RandomForestClassifier(random_state=rs)` é determinístico.
- Reexecutar o mesmo `config` + mesmo `git_sha` deve reproduzir os mesmos
  `oracle_curve_mean` (com ressalva DEAP paralelo).

### Curva Oracle

- Salvar a **curva completa N=1..M** para cada fold/dataset/modo.
- `summary.csv` armazena a curva como JSON em uma coluna `oracle_curve_json`
  + colunas derivadas `oracle_1`, `oracle_M` para leitura rápida.
- Níveis fixos intermediários podem ser derivados post-hoc sem re-executar.

## Consequências

- **+** Auditoria completa: qualquer número em um paper pode ser rastreado
  até `fold_manifest.json` + `run_manifest.json` + git SHA.
- **+** Arrays pesados recuperáveis via re-execução (manifest basta).
- **+** Comparações GA vs RF na mesma estrutura de diretório.
- **−** Disco local: ~estimativa 1-5 MB por fold em `.npy`/`.npz`; 31×10×2
  = 620 folds ≈ 0.6-3 GB para execução completa (aceitável local).
- **−** DEAP paralelo (`jobs=8`) pode não ser bit-exact reprodutível;
  mitigado registrando `jobs` no manifest e recomendando `jobs=1` para
  reprodutibilidade estrita (lento).
- **−** RF default `max_features='sqrt'` não iguala condições do GA;
  registrada como decisão consciente (diferença informativa).

## Referências

- ADR 0006 — arquitetura Oracle_N, protocolo 10-fold
- `docs/protocol.md` — protocolo experimental formal
- `pos/oracle/run_recorder.py` — implementação
- `scripts/run_experiment.py` — CLI