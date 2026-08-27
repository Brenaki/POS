# Protocolo Experimental — Oracle_N

> Protocolo formal dos experimentos do subprojeto
> `docs/SubProjeto - ANÁLISE DO IMPACTO DE ORACLES EM DIFERENTES NÍVEIS NA ACURÁCIA DE SISTEMAS DE MÚLTIPLOS CLASSIFICADORES.md`.
> Vigente: **ADR 0020** (dois perfis científicos). Referências anteriores:
> ADR 0006 (arquitetura Oracle_N), ADR 0009 (reprodutibilidade),
> ADR 0018 (split de três vias), ADR 0019 (run canônico).

Desde o ADR 0020 não existe "o protocolo do POS": existem **dois perfis**,
escolhidos explicitamente na linha de comando. Um run nunca é uma mistura dos
dois, e o perfil viaja no `run_manifest.json` e em toda linha do `summary.csv`.

| | `pos-oracle-v2` | `monteiro-2023-dt-compatible-v1` |
|---|---|---|
| Objetivo | experimento principal do POS (Oracle_N) | fatia reproduzível de Monteiro 2023 |
| Partição | 10-fold estratificado + split de três vias | 20 holdouts 50/25/25 independentes |
| Identificador da linha | `fold` | `replication` |
| Seed raiz | 42 | 86 (`reproducibility_root_seed`) |
| Seleção de medidas | por fold (só no treino do fold) | uma vez por dataset (Tabela 4 é por dataset) |
| DSEL | disjunto da validação | **igual** à validação (como no artigo) |
| Comparável ao artigo | não | sim (Tabelas 6, 7 e A.10) |

## 1. Datasets

- 28 datasets da **Tabela 2 de Monteiro 2023**, em `.arff` sob `POS/Dataset/`.
- `Australian` está no disco como `Adult.arff` (690/14/2) — alias registrado
  em `pos/oracle/dataset_reconciliation.py`.
- Cada snapshot é conferido contra a Tabela 2 (instâncias/atributos/classes) e
  carimbado com SHA-256 **do snapshot aprovado pelo POS** — o artigo não
  publica hashes.
- Estado atual (`docs/dataset_conformance.md`, gerado por
  `scripts/dataset_conformance.py`): **23 `conformant`, 4
  `approved_with_note`, 1 `mismatch`**.
  - `approved_with_note`: Heart, ILPD, Ionosphere, Laryngeal3 — a coluna
    *Features* da Tabela 2 está deslocada em uma linha nas linhas 09–13.
  - `mismatch`: P2 (2000 instâncias no snapshot, 5000 no artigo). Fica fora
    da comparação publicada, salvo `--allow-unverified-data`, que marca o run
    inteiro como `comparable=false`.
- Ecoli, Glass e Magic não fazem parte da Tabela 2 e ficam fora dos perfis.

## 2. Partição dos dados

### 2.1 `pos-oracle-v2`

- `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
- Do treino de cada fold, split estratificado de três vias e **disjunto**:
  treino do pool / fitness do PGDCS / DSEL para DCS-DES (ADR 0018).
- Semente interna `42 + fold`.

### 2.2 `monteiro-2023-dt-compatible-v1`

- 20 replicações independentes, **novo holdout a cada replicação**:
  50% treino / 25% validação / 25% teste, estratificado.
- Semente da replicação `r` = `86 + r`.
- Proveniência: 50/25/25 vem do artigo; estratificação e novo split por
  replicação vêm da tese de 2021 (o artigo omite); validação = DSEL é uma
  divergência explícita registrada no ADR 0020 §2.2.

## 3. Tratamentos — três, todos com árvore de decisão

A única variável entre eles é **como o pool foi gerado**. Todos entregam 100
`DecisionTreeClassifier` e passam pelos mesmos Oracles, fusores e métodos de
seleção dinâmica.

### 3.1 `pgdcs_a` e `pgdcs_d`

Uma única trajetória evolutiva por split (`shared_evolution_trajectory`), da
qual saem os dois pools — é exatamente onde Monteiro 2023 separa as variantes:

1. **Votação das medidas** (`pos/pool/monteiro_voter.py`): `T=10` rodadas de
   `n=100` subamostras aleatórias com reposição (1000 no total). Vence uma
   medida de *Overlapping* e uma de *Neighborhood*; empate é desfeito pela
   dispersão média nas 10 rodadas.
2. **NSGA-II** (`pos/pool/pgdcs_evolution.py`): população 100, 100 filhos,
   crossover 0.9, mutação 0.1, **20 gerações**, bags aleatórios **com
   reposição** (não estratificados — a cobertura de classes é responsabilidade
   dos operadores: o crossover só aceita indivíduo com todas as classes, a
   mutação troca instância preservando o rótulo).
3. **Critério de saída**, avaliado em **todas** as gerações 1..20:
   - `pgdcs_a` = geração de maior acurácia de voto majoritário na validação;
   - `pgdcs_d` = geração de maior `Gdisp` (dispersão global no espaço 3D de
     fitness).
   - A **geração 0 é avaliada mas nunca selecionada**; empate mantém a
     geração **mais antiga** (atualização com `>` estrito).
4. O pool final são **os classificadores que o próprio GA treinou** naquela
   geração (Algoritmo 2 do artigo), não árvores reconstruídas a partir dos
   bags.

### 3.2 `bagging`

`BaggingClassifier(n_estimators=100, max_samples=0.5, bootstrap=True,
random_state=86)` com árvore de decisão default — os únicos hiperparâmetros
que o artigo publica literalmente.

### 3.3 O que saiu

`ga` (F1/T1 fixo), `randbag` e `rf` deixam de ser modos científicos, e o
Perceptron sai do caminho canônico: com árvore nos dois lados, a comparação
não confunde *gerador do pool* com *classificador-base*.

## 4. Métricas

### 4.1 Oracle_N — duas leituras, nomes distintos

| Coluna | Definição | Forma |
|---|---|---|
| `oracle_<n>` | **pelo menos `n`** dos M classificadores corretos — definição do projeto (README, subprojeto, schema v1) | decrescente, independente de ordem |
| `oracle_greedy_<n>` | construção gulosa aninhada: a cada passo entra o classificador que cobre mais amostras novas | crescente |
| `oracle_classic` | `oracle_1` = último ponto da curva gulosa = Oracle clássico do pool | escalar |

- Checkpoints reportados: `1, 5, 10, 15, …, 100` (21 pontos), para as duas
  leituras. As curvas completas de 100 pontos ficam em `oracle_curve_json` e
  `oracle_greedy_curve_json`.
- **Só `oracle_classic` é comparável à Tabela 7.** O resto de qualquer das
  duas curvas é contribuição do POS; a gulosa ainda depende da ordenação.

### 4.2 Fusão estática

- **MVR** (`mv_*`): voto majoritário — a regra do artigo.
- **Mean probability** (`mp_*`): média de `predict_proba`. Com árvore nos dois
  tratamentos, a combinação suave é uma média de probabilidades de verdade; o
  substituto de margem normalizada do ADR 0016 (que existia por causa do
  Perceptron) sai do fluxo. Em árvores não podadas as duas regras coincidem
  por construção, e reportar as duas deixa isso visível.
- Cada fusor reporta acurácia, precisão/revocação/F1 macro e acurácia
  balanceada.

### 4.3 Seleção dinâmica

Seis métodos canônicos via DESlib: **LCA, OLA, Rank** (DCS) e **KNORA-E,
KNORA-U, META-DES** (DES). Região de competência estimada no DSEL do perfil.

### 4.4 Diversidade

`double_fault_mean` (média fora da diagonal da matriz M×M) e as acurácias
individuais de cada classificador do pool.

## 5. Reprodutibilidade

- Toda semente não publicada é derivada deterministicamente da semente raiz do
  perfil e registrada como **modernização** — só o `random_state=86` do
  Bagging é número publicado.
- Bags carregam semente própria (`random_state + posição`), então `--jobs` não
  afeta o resultado.
- `run_manifest.json` (schema v2): perfil, escala, `comparable`, fingerprint da
  config, git SHA/branch/dirty, versões das deps, plataforma e o **catálogo de
  conformance** completo.
- Manifesto por split e tratamento: sementes, tamanhos das partições, M,
  medidas escolhidas, geração selecionada e critério.
- `--resume` retoma pelo manifesto em disco e não re-executa a trajetória
  PGDCS de um split já concluído.

## 6. Artefatos

| Artefato | Descrição |
|---|---|
| `run_manifest.json` | config, proveniência, conformance dos 28 datasets |
| `summary.csv` | uma linha por (dataset, split, tratamento) — schema v2 |
| `per_dataset_summary.csv` | mean/std por (dataset, tratamento) — nível das tabelas publicadas |
| `reference_comparison.csv` | comparação descritiva com o artigo |
| `fold_manifest_<tratamento>.json` | sementes, tamanhos, geração escolhida |
| `correctness_matrix_<tratamento>.npy` | (n_test × M) |
| `predictions_<tratamento>.npz` | predições por classificador |

O schema v1 continua legível via `pos.oracle.summary_v2.load_summary_v1`, que
mapeia o antigo modo `pgdcs` para `pgdcs_d` e preserva as colunas `soft_*`.

## 7. Execução

```bash
# experimento principal do POS
python scripts/run_experiment_v2.py --profile pos-oracle-v2 --full

# fatia comparável ao artigo
python scripts/run_experiment_v2.py --profile monteiro-2023-dt-compatible-v1 --full

# subconjunto, retomada, comparação e conformance
python scripts/run_experiment_v2.py --profile <perfil> --full --datasets Wine,Banana
python scripts/run_experiment_v2.py --resume results/experiments/<dir>
python scripts/compare_reference.py results/experiments/<dir>
python scripts/dataset_conformance.py
```

- `--smoke` reduz folds, gerações e M: valida o pipeline, **nunca** é
  resultado científico (`comparable=false`).
- Qualquer override de M, gerações, splits ou métodos marca
  `comparable=false`.
- `--jobs` default = núcleos − 1.

## 8. Comparação com o artigo

`scripts/compare_reference.py` produz, por (dataset × tratamento × método):
média e desvio publicados, média e desvio do POS, delta em pontos
percentuais, IC 95% do POS, status de conformance do dataset e a tabela de
origem.

| Método | Tabela | Observação |
|---|---|---|
| `mvr` | A.10 | média e desvio publicados |
| `oracle` | 7 | o artigo publica só a média → `published_std = null` |
| `generation` | 6 | geração média selecionada; Bagging não tem |

A comparação é **descritiva**: o artigo publica média/desvio, não as 20
observações nem as sementes, então não cabe teste pareado nem alegação
automática de equivalência. Friedman/Nemenyi e Wilcoxon-Holm continuam
restritos aos resultados brutos gerados pelo POS.
