# ADR 0020 — POS standardization and Monteiro 2023 comparison profiles

**Status:** proposed  
**Date:** 2026-08-27  
**Supersedes:** ADR 0019 (canonical rerun & generation control) — keeps its
PGDCS core, replaces its single-protocol / five-mode surface.

## Context

The POS project started from the doctoral thesis "Classifier Pool Generation
based on a Two-level Diversity Approach" (Monteiro 2021) and has since
diverged in protocol, treatments and metrics from both the thesis and the
follow-up journal article *Exploring diversity in data complexity and
classifier decision spaces for pool generation* (Monteiro 2023, *Information
Fusion* 89). The current `--mode` CLI exposes five modes (`pgdcs, ga,
randbag, bagging, rf`), uses a single 10-fold protocol, mixes `soft_fusion`
into the schema, samples PGDCS bags with stratification the paper does not
prescribe, and cannot be compared to the published tables because datasets
have drifted (P2 2000 vs 5000, Heart/ILPD/Ionosphere/Laryngeal3 attributes).

This ADR defines two explicit profiles, a shared PGDCS core aligned with
Monteiro 2023, a conformance-gated dataset catalog, and a descriptive
comparator — turning "the POS run" into either the project's own
`pos-oracle-v2` experiment or a `monteiro-2023-dt-compatible-v1` reproduction
slice, never an ambiguous mix.

## Decision

### 1. Sources and normative precedence

- **Normative source:** Monteiro 2023 (`docs/Monteiro2023.pdf`).
- **Complementary source:** thesis 2021 (`docs/Marcos_Monteiro_Junior_2021.pdf`),
  used to fill details the article omits (split stratification, new split per
  replication).
- **Internal contradictions / ambiguities** in the article are explicitly
  recorded in this ADR and resolved by a justified operational decision.
- **Reproduction class:** *compatible modern*. Uses current dependencies
  (recent sklearn/scipy) and the NumPy complexity backend, not the pinned
  legacy stack (ECoL 0.3.0 / sklearn 0.24.2 / DESlib 0.3). Bit-for-bit
  reproduction is neither possible nor a goal.

### 2. Profiles

#### 2.1 `pos-oracle-v2` (POS main experiment)

- `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
- Per outer fold, disjoint partitions for pool training, PGDCS fitness, DSEL
  and test.
- Measure selection inside each fold's training data only.
- Treatments: `pgdcs_a`, `pgdcs_d`, `bagging`.
- MVR is the only static fusion rule. `mean_probs`, `soft_fusion_rule`,
  `mean_decision_norm`, `soft_*` are **removed** from the v2 schema and flow.
- Canonical DCS/DES: LCA, OLA, Rank, KNORA-E, KNORA-U, META-DES.
- Oracle curve `Oracle_1..Oracle_100` via **greedy construction** (§5),
  reported at checkpoints `1, 5, 10, 20, …, 100`.

#### 2.2 `monteiro-2023-dt-compatible-v1` (comparison slice)

- **Name** makes the slice explicit: Decision Tree + PGDCS_A/D + Bagging
  only — **not** the full 336-comparison protocol (Perceptron, AdaBoost, RF,
  DSOC are out of scope). `reference_scope` field in the manifest records it.
- **20 independent external replications.**
- **Holdout 50% train / 25% validation-DSEL / 25% test**, stratified
  (*source: thesis*; the article does not state it), new split per replication
  (*source: thesis*). Provenance field per split.
- **Validation/DSEL overlap:** the validation set is used both by the
  PGDCS_A/fitness criterion and by DCS/DES, reproducing the published design.
  This overlap (different from `pos-oracle-v2`, which separates them) is
  recorded as a divergence from the POS protocol.
- **Measure selection:** once per dataset, reused across the 20 replications
  (*source: thesis*).
- **Reproducibility:** `reproducibility_root_seed=86`. Only
  `BaggingClassifier(random_state=86)` is published; split, PGDCS and
  NSGA-II seeds derive deterministically from the root and are marked as
  **modernization** in the manifest.
- **Comparison targets:** only the Decision Tree tables of the paper —
  Tables 4, 6, 7, A.10, A.12, A.14.

### 3. Shared PGDCS core

#### 3.1 MOGA (paper Table 5)

| Parameter | Value |
|---|---|
| Inducer | Decision Tree, `all defaults` |
| Generations | 20 |
| Population | 100 |
| Offspring size | 100 (children = 100) — both stored in the manifest |
| Crossover | 0.9 |
| Mutation | 0.1 |
| NSGA-II | yes, for front ranking |

#### 3.2 Per-generation selection

- **20 candidate generations** (`gen=1..20`); each post-selection population
  has MVR validation accuracy and Gdisp computed.
- **`gen=0`** is evaluated to initialise NSGA-II but **never selected**
  (Algorithm 1).
- **`PGDCS_A`** = argmax MVR validation accuracy; **`PGDCS_D`** = argmax Gdisp.
- **Tie-break:** keep the oldest generation (update only when
  `temp < current_score`, never `<=`). Covered by test.
- **Shared A/D trajectory** (*explicit modernization*): one evolution per
  fold/replication, extracting both pools. The exit criterion diverges only
  after `P(t+1)` is produced by NSGA-II, so this is algorithmically
  consistent and roughly halves the cost. Manifest records
  `shared_evolution_trajectory=true`, `modernization=true`.

#### 3.3 Bag sampling

- **Measure voting (first stage):**
  - `n = 100` subsets, `T = 10` iterations, 1000 subsets total.
  - **Random with replacement, not stratified** (follows the paper).
  - **Fraction 0.5** of the train set (decision justified by the thesis and
    consistency with the second stage).
  - **Documented ambiguity:** the article reports `u=0.4` but also describes a
    "maximum 50%" size. We adopt 0.5 and record the divergence here.
- **MOGA (second stage):** each initial individual contains 50% of the train
  set, random sampling with replacement.
- **Class coverage** is guaranteed by the genetic operators, not by sampling:
  - crossover accepts an individual only if it covers every class;
  - mutation swaps instances preserving the label (donor and receiver share
    the label).
- **No `stratified_bag_sampling`** in the published protocol. If stratified
  bags are forced, it must appear as `modernization: stratified_bag_sampling`,
  not as reproduced behaviour.

#### 3.4 Measure selection (Equations 10–12)

- Voting over the 10 iterations × 100 subsets.
- Select **exactly one** Overlapping measure + **one** Neighborhood measure.
- Vote tie → highest mean dispersion across the `T` iterations.

#### 3.5 Final pool classifiers

- **Keep the fitted classifiers** of the best generation (consistent with
  Algorithm 2: the ensemble `E` carries the trained classifiers). Avoids
  reconstruction and divergence between fitness and final pool.
- If reconstruction is technically necessary, using a `seed_per_bag` derived
  from the bag's stable identifier is an acceptable modernization — but
  keeping fitted is preferred.

### 4. Bagging baseline

- `BaggingClassifier(n_estimators=100, max_samples=0.5, bootstrap=True,
  random_state=86)` with default Decision Tree.
- `random_state=86` is the only seed published; all others derive from the
  root seed (modernization).

### 5. Oracle_N — definition and comparison

Two readings of "Oracle_N" exist in this project and both are recorded, under
names that cannot be confused (revised during implementation, see §11.2):

| Column | Definition | Shape | Comparable to Table 7 |
|---|---|---|---|
| `oracle_<n>` | at least `n` of the M classifiers correct — the project's definition (README, subproject document, schema v1) | decreasing, order-free | `oracle_1` yes |
| `oracle_greedy_<n>` | greedy nested construction: at each step add the classifier that covers the most new samples | increasing | last point yes |
| `oracle_classic` | `oracle_1` = greedy last point = classic pool oracle | scalar | **yes** |

- Only the classic pool oracle (`oracle_classic`) is **directly comparable**
  to Table 7. The rest of either curve is a POS contribution; the greedy one
  additionally depends on its ordering, so no published number matches it.
- Report checkpoints `1, 5, 10, 15, …, 100` (21 points, revised from 12 —
  §11.1) for both readings, plus `oracle_curve_json` and
  `oracle_greedy_curve_json` (100 values each).
- **`published_std` for Table 7 = `null`** (the article publishes a mean, not
  a std).

### 6. CLI

```bash
python scripts/run_experiment.py --profile pos-oracle-v2 --full
python scripts/run_experiment.py --profile monteiro-2023-dt-compatible-v1 --full
python scripts/run_experiment.py --profile <profile> --smoke
python scripts/compare_reference.py <run_dir>
```

- Replace `--mode` with `--treatments`, accepting only
  `pgdcs_a,pgdcs_d,bagging`.
- Remove `--base-classifier`; scientific profiles are tree-only.
- `ga`, `randbag`, `rf` leave the canonical register (RF may remain as an
  isolated legacy utility).
- Any override of M, generations, splits or methods → `comparable=false`.
- `--smoke` uses reduced values, never a scientific result.
- Resume only proceeds when the configuration fingerprint is identical.

### 7. Structured results

#### 7.1 PGDCS result (per experimental unit)

```
selected_measures
measure_votes
generation_trace[0..20]
pgdcs_a.best_generation / best_validation_accuracy / pool
pgdcs_d.best_generation / best_gdisp / pool
```

#### 7.2 `summary.csv` v2 schema

- Identification: profile, dataset, fold/replication, treatment.
- Oracle: `oracle_1, oracle_5, oracle_10, …, oracle_100` + `oracle_curve_json`.
- MVR: accuracy + macro/balanced metrics.
- DCS/DES: metrics for the six methods.
- Diversity: mean individual accuracy, double fault.
- PGDCS: chosen measures, `best_generation`, `best_score`, `best_metric`.
- **Removed columns:** `mean_probs`, `soft_fusion_rule`,
  `mean_decision_norm`, `soft_*`.

#### 7.3 Artifacts per experimental unit

- Hashes and sizes of train, validation, DSEL and test.
- Derived seeds.
- Full Oracle curve (100 values).
- Individual and DCS/DES predictions.
- Selected bags of PGDCS_A and PGDCS_D.
- 20-generation trajectory.
- Complexity backend config + dependency versions.

#### 7.4 Backwards compatibility

Loaders accept schema v1 runs; legacy `pgdcs` maps to `pgdcs_d` on read only.

### 8. Dataset conformance

#### 8.1 Versioned catalog (28 datasets of paper Table 2)

Fields:
```
paper_instances / paper_features / paper_classes / paper_source
snapshot_sha256 / snapshot_provenance   # POS approved snapshot, NOT a paper hash
conformance_status: conformant | mismatch | unverified
conformance_reason
```

The hash is of the **POS-approved snapshot**, not the paper (the paper does
not publish SHA-256).

#### 8.2 Gates

- Preflight blocks the published comparison when shape, classes or hash
  diverge.
- `--allow-unverified-data` permits diagnostic runs, recorded as
  `comparable=false`, excluded from the reference report.
- Covers known divergences: P2 (2000 vs 5000 instances),
  Heart/ILPD/Ionosphere/Laryngeal3 (attributes).
- **Reconciliation of the 28 snapshots happens before SHA becomes a
  definitive gate** (acceptance step 2).

### 9. Paper reference tables

Versioned with source table/page:
- Table 4 — measures and votes;
- Table 6 — selected generations;
- Table 7 — Oracle;
- Table A.10 — Decision Tree MVR;
- Table A.12 — Decision Tree DCS;
- Table A.14 — Decision Tree DES.

### 10. Comparator (`scripts/compare_reference.py`)

Per dataset × treatment × method:
```
published_mean, published_std (null for Table 7), pos_mean, pos_std,
delta_percentage_points, pos_ci95,
dataset_conformance, protocol_compatibility, source_table
```

- Comparison with published numbers is **descriptive** (no paired test, no
  automatic equivalence claim — the paper gives only mean/std, not the 20
  observations or seeds).
- **Friedman/Nemenyi and Wilcoxon-Holm** are applied only to raw results
  generated by POS.

## Consequences

- `ga`, `randbag`, `rf` stop being first-class scientific modes; `rf`
  survives as a legacy utility only.
- The ADR 0016 `soft_fusion` / `mean_decision_norm` machinery is removed from
  the v2 schema and flow; the genuine `predict_proba` average returns as
  `mp_*` (§11.3). Historical v1 runs remain readable via the loader fallback.
- `--mode` and `--base-classifier` CLI flags are removed; replaced by
  `--profile` and `--treatments`.
- The single 10-fold protocol is replaced by profile-driven dispatch; the
  recorder gains a profile-aware path.
- PGDCS bags drop stratification in favour of random-with-replacement
  sampling; class coverage is enforced by the genetic operators instead.
- The Oracle curve gains a greedy construction definition alongside the
  project's "at least N" definition; only the classic pool oracle
  (`oracle_classic`) is comparable to Table 7.
- A conformance gate is introduced; non-conforming datasets cannot carry the
  "comparable to reference" label.

## 11. Amendments made during implementation

### 11.1 Oracle checkpoints: 12 → 21

`1, 5, 10, 20, …, 100` skipped the region where a 100-tree pool actually
separates. Checkpoints are now `1` then every 5 up to 100 (21 points), for
both readings. The full 100-point curves are stored regardless, so the
reported checkpoints can change without re-running any evolution.

### 11.2 Oracle column naming

The first implementation gave the greedy curve the plain `oracle_<n>` name.
That collides with the project's own definition — the README, the subproject
document and schema v1 all define `Oracle_N` as "at least N classifiers
correct" — and would have made `oracle_50` mean two different things across
schema versions. The plain name now carries the project definition and the
greedy curve is explicitly `oracle_greedy_<n>` (§5).

### 11.3 Mean probability is restored as a first-class fusion rule

ADR 0016's "soft fusion" existed because the Perceptron has no
`predict_proba` and needed a normalised-margin substitute. With Decision Tree
on **both** treatments the substitute is unnecessary and the two pools can be
combined identically: `mp_*` columns hold a genuine `predict_proba` average.
The ambiguous `soft_*` / `mean_decision_norm` names do not come back.

Empirically the two static rules coincide on unpruned trees: a fully grown
tree's leaf probabilities are 0/1, so the probability average is the vote
fraction and its argmax is the majority vote (up to tie-breaking). Reporting
both makes that degeneracy visible instead of implied.

### 11.4 Dataset reconciliation (closes acceptance step 2)

`docs/dataset_conformance.md`, generated by
`scripts/dataset_conformance.py`. Result: **23 conformant, 4
`approved_with_note`, 1 `mismatch`**.

`Australian` is stored as `Adult.arff` (690/14/2, matching Table 2 exactly);
the alias lives in `pos/oracle/dataset_reconciliation.py`.

Heart, ILPD, Ionosphere and Laryngeal3 diverge only because Table 2's
*Features* column is shifted by one row across rows 09–13 — each paper value
is the next row's snapshot value, and rows 08 and 14 match without the shift.
They are `approved_with_note`: usable, with the divergence carried into every
run manifest and comparison row. A `mismatch` is never silently lifted to
`conformant`.

P2 stays `mismatch` (snapshot 2000 instances, paper 5000 — a content
divergence, not a typesetting one) and is excluded from the published
comparison unless `--allow-unverified-data` marks the run `comparable=false`.

### 11.5 PGDCS engine wiring

`PgdcsTracker` was written before anything produced trajectories for it.
`pos/pool/pgdcs_evolution.py` now subclasses `poolGeneration` to draw bags
with replacement, record `(gen, MVR, Gdisp, fitted pool)` for gen 0..20, and
keep the estimators the GA itself scored (Algorithm 2) rather than rebuilding
them from bags. MVR is read from the cached per-bag validation predictions,
which costs nothing, instead of refitting an ensemble every generation.
`pos/oracle/pgdcs_treatment.py` adapts it to the runner, and
`pos/oracle/run_recorder_v2.py` is the schema v2 recorder the CLI drives.

### 11.6 Table 6 enters the comparator

The selected-generation mean/std is a published number
(`get_reference_value(dataset, treatment, "generation")`), so the comparator
reports `mvr`, `oracle` **and** `generation`.

### 11.7 Investigação do gap de acurácia do PGDCS (fecha o ponto aberto)

O ponto aberto perguntava se a diferença entre o PGDCS medido aqui e o
publicado na Tabela A.10 é *protocol drift* ou diferença real da evolução. A
investigação (trajetórias completas gravadas em Wine, Heart e Blood, 20
replicações cada, com acurácia medida em **validação** e em **teste** a cada
geração) responde: é **viés de seleção na medida publicada**, e não uma falha
da evolução.

**Evidência.** Para cada base, `VAL@sel` é a acurácia MVR no conjunto de
validação da geração escolhida — a mesma quantidade que o Algoritmo 1
(linha 16, `acurácia(E(t+1), V)`) maximiza — e `TEST@sel` é a acurácia da
mesma geração em teste:

| Base  | pub A | VAL@A | TEST@A | pub D | VAL@D | TEST@D | pub Bag | Bag aqui |
|-------|-------|-------|--------|-------|-------|--------|---------|----------|
| Wine  | 99.3  | 99.9  | 96.3   | 96.4  | 99.4  | 95.3   | 95.2    | 95.2     |
| Heart | 90.1  | 92.5  | 81.7   | 88.4  | 89.6  | 81.7   | 80.4    | 80.4     |
| Blood | 83.6  | 81.9  | 76.4   | 81.7  | 80.4  | 76.6   | 77.4    | —        |
| Sonar | 84.7  | 93.6  | 75.7   | 85.5  | —     | 74.9   | 74.3    | 75.5     |

O erro médio absoluto do PGDCS publicado contra `VAL@sel` é ≈1.7 pp; contra
`TEST@sel` é ≈6.3 pp. O Bagging — que não passa por seleção de geração —
reproduz em teste com erro ≤1.2 pp (Wine e Heart batem na primeira casa
decimal). A magnitude do gap também acompanha o quanto a MVR de validação é
otimizável: Wine satura (VAL≈100, gap 3 pp), Sonar e Heart não saturam
(gap 9–10 pp).

**Confirmação sobre as 27 bases comparáveis** (run
`2026-08-27T19-47-09_7c5ec6d`, perfil completo):

| Tratamento | erro médio \|Δ\| | média com sinal | negativos |
|------------|----------------|-----------------|-----------|
| PGDCS_A    | 6.19 pp        | **−6.19 pp**    | **27/27** |
| PGDCS_D    | 5.25 pp        | **−5.25 pp**    | **27/27** |
| Bagging    | 0.75 pp        | −0.16 pp        | 14/27     |

O que decide a questão é o **sinal**, não a magnitude. O Bagging erra 0.75 pp
em módulo com média com sinal de −0.16 pp e 14 negativos de 27 — ruído
simétrico, como se espera de uma reimplementação fiel. O PGDCS erra sempre
para baixo: **54 comparações consecutivas** (A e D em 27 bases) sem uma única
exceção. Um defeito de implementação no PGDCS produziria dispersão nos dois
sentidos, ainda mais com o Bagging — que compartilha split, semente, indutor,
M e protocolo de avaliação — reproduzindo sem viés. O que distingue os dois
braços é exatamente a seleção de geração sobre `V`.

Oracle e geração escolhida, no mesmo run: erro médio de 0.44 pp contra a
Tabela 7 e 3.19 gerações contra a Tabela 6. A Tabela 7 reproduz porque o
Oracle não depende do critério de saída — todas as gerações têm cobertura
quase total.

Ou seja: o número do PGDCS na Tabela A.10 é o máximo de 20 gerações medido no
**mesmo** conjunto de 25% usado para escolher a geração, enquanto o do Bagging
é uma medida honesta em teste. A comparação publicada PGDCS × Bagging é entre
uma quantidade otimizada e uma não otimizada; a vantagem de 4–10 pp é o
tamanho esperado desse viés, não um ganho generalizável.

**Confirmação independente.** A acurácia de teste ao longo das gerações sobe
até a geração ~4–8 e depois **cai**, enquanto a MVR de validação continua
subindo monotonicamente até a geração 20 (Wine: VAL 95.0 → 99.8, TEST 95.3 →
95.0). É o formato clássico de sobreajuste ao conjunto de seleção.

**O tamanho honesto do efeito.** Com os dois perfis executados, dá para medir
o que o PGDCS realmente ganha do Bagging quando as duas pontas são medidas em
teste:

| Perfil | validação | PGDCS_A − Bagging | vence em |
|--------|-----------|-------------------|----------|
| `monteiro-2023-dt-compatible-v1` | 25% (= DSEL) | **+0.39 pp** | 22/27 |
| `pos-oracle-v2` | 10% (disjunta) | **−0.13 pp** | 12/27 |

No perfil compatível com o artigo o ganho existe, é consistente (22 de 27) e
vale ~0.4 pp — uma ordem de grandeza abaixo dos 4–6 pp da Tabela A.10. No
perfil próprio do POS, cuja validação é menor e disjunta do DSEL, o ganho
desaparece: 12 vitórias em 27 é sorteio. A seleção de geração precisa de um
conjunto de validação grande para não escolher ruído, e o que ela entrega,
quando entrega, é marginal.

**Consequência para este projeto.** O POS mantém `mv_acc` como acurácia em
teste. `pgdcs_best_score` (MVR de validação da geração vencedora) fica
registrado no manifesto de fold e é a coluna comparável à Tabela A.10; o
comparador continua a apontar o delta contra `mv_acc`, agora com esta nota.
Nenhuma mudança de protocolo é feita para "fechar" o gap: fechá-lo exigiria
reportar a métrica enviesada.

### 11.8 Correção na votação de medidas (Eq. 10-11)

A votação acumulava os σ de todas as rodadas e votava na **média corrente**,
não no σ da rodada. A Eq. (10) é `ind = argmax_j σ_j^t`, por iteração `t`, e a
Eq. (11) soma um voto por família a cada `t`. O efeito do bug era concentrar os
votos de uma família em uma única medida (9–10 votos de 10), incompatível com o
espalhamento das Tabelas 3 e 4 (Wine: F3 8, F1 1, F1v 1). Corrigido em
`votes_from_sigmas`; o desempate por dispersão média ao longo das `T` iterações
(Eq. 12) foi mantido.

Impacto medido: em Heart, trocar as medidas votadas pelas da Tabela 4 (F4+N4)
move a acurácia de teste de 80.1 para 81.7 — ~1.6 pp de um gap de 10 pp. A
escolha de medidas **não** é a causa do gap (§11.7), mas a votação agora segue
o paper.

A votação continua instável entre sementes e entre partições de treino: dos 5
conjuntos comparados, só Wine reproduz os vencedores da Tabela 4. Isso é
inerente ao critério (as dispersões das medidas ficam a menos de 0.01 umas das
outras — ver a Tabela 3 do paper, onde o vencedor de cada iteração muda 7 vezes
em 10), não um defeito adicional.

### 11.9 `u = 0.4` deixa de ser ambíguo

§5.1 do paper diz "considering `n = 100`, and `u = 0.4`, we randomly sampled 100
data subsets considering each a maximum of 50% of the original training size" —
duas frações na mesma frase. §5.2 resolve a segunda: "The first generation of
our MOGA consists of data subsets with **just 50% of the training dataset
size**". São passos diferentes. Adotado: `measure_fraction = 0.4` (passo 1,
votação) e `tam_bags = 0.5` (passo 2, bags do AG). Antes ambos eram 0.5.

### 11.10 Phoneme: o único outlier do braço Bagging não é bug

O Bagging reproduz a Tabela A.10 com erro médio de 0,75 pp e média com sinal de
−0,16 pp (14/27 negativos — ruído simétrico). Phoneme é a única base fora da
faixa: −4,86 pp, contra máximo de 2,27 pp em qualquer outra. Investigado.

**Não é a base.** O snapshot bate exatamente com a Tabela 2 (5404 × 5, 2
classes) e com o Phoneme canônico da ELENA: 3818/1586, taxa da classe
majoritária 0,7065.

**Não é `max_samples`.** O baseline usa `max_samples=0.5`. Rodando as 20
replicações do perfil com `max_samples` em {0,5; 1,0} sobre Phoneme e nove
controles, Phoneme fica em −4,86 e −4,17 pp respectivamente — o parâmetro move
0,7 pp, não 4,9. E `0.5` é a escolha certa para o conjunto: Haberman cai 2,7 pp
com `1.0`, o resto é indiferente.

**O número publicado está acima do teto da base.** Dando toda vantagem que o
protocolo nega — 90% de treino (10-fold) em vez de 50%, e indutores mais fortes:

| método (Phoneme, 10-fold, 90% treino) | acurácia |
|---|---|
| árvore única | 87,36 ± 1,66 |
| Bagging `max_samples=0.5`, M=100 | 90,23 ± 1,47 |
| Bagging `max_samples=1.0`, M=100 | 90,86 ± 1,09 |
| Random Forest M=100 | 91,10 ± 1,19 |
| 1-NN | 90,58 ± 1,57 |
| **Tabela A.10, Bagging (50% treino)** | **93,1 ± 1,0** |
| **Tabela A.10, PGDCS_A (50% treino)** | **96,8 ± 1,2** |

O melhor alcançável neste snapshot, com quase o dobro dos dados de treino e um
indutor mais forte, é 91,1%. O Bagging publicado está 2,0 pp acima disso e o
PGDCS_A publicado, 5,7 pp. Os valores medidos batem com a literatura de Phoneme
(árvore ≈87%, RF/1-NN ≈90–91%); os publicados, não.

Conclusão: nenhuma configuração do protocolo alcança o número publicado a partir
do Phoneme canônico. Ou a base do paper difere da ELENA, ou a linha está
inflada. Registrado como limite de reprodutibilidade por base — não há correção
de implementação a fazer. Sem Phoneme, o erro do Bagging cai para 0,59 pp médio
com máximo de 2,27 pp.

### 11.11 P2 entra no `pos-oracle-v2`

O portão de conformance em `run_recorder_v2.py` é global: aplica a Tabela 2 a
qualquer perfil. O `pos-oracle-v2` não replica ninguém — seu `reference_scope`
é ele mesmo — e ainda assim perdia P2 por `mismatch` (paper 5000 instâncias,
snapshot 2000). P2 foi acrescentado ao run existente por `--resume` com
`allow_unverified_data`, o que marca `comparable=false` no manifesto conforme a
semântica já documentada em §8.2. O run vai de 810 para 840 linhas (28 bases).

Medido: PGDCS_A 95,05 · PGDCS_D 95,05 · Bagging 94,60 · Oracle₁ 100,00 nas três.
Não comparável com a Tabela A.10 (91,9 / 91,7 / 88,0), que foi medida sobre
5000 instâncias geradas com o mesmo procedimento, não sobre estas 2000.

Fazer o portão consultar o escopo do perfil, em vez de aplicar a Tabela 2
sempre, foi resolvido no **ADR 0021** (portão por perfil): o `pos-oracle-v2`
passa a verificar cada snapshot contra o catálogo SHA-256 do próprio POS
(`Dataset/pos_catalog.json`), e P2 entra sem `--allow-unverified-data`.

## Open points

- ~~Portão de conformance por perfil~~ — resolvido no ADR 0021.
- Reproduzir os vencedores da Tabela 4 exige saber em qual partição de treino o
  paper rodou a votação (§11.8); o paper não informa. Fica registrado como
  limite de reprodutibilidade, não como pendência de implementação.
- Phoneme (§11.10) e P2 (§11.11) ficam fora de qualquer comparação publicada com
  a Tabela A.10, cada um pelo motivo registrado acima.

## Test plan

TDD, ≤150 LOC per file, ≥80% coverage. Mandatory tests (T1–T18) listed in
the implementation plan.