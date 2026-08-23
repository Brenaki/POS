# ADR 0014: Fitness da prole após a 1ª geração, portões de protocolo e T1 escalável

## Date
2026-08-22

## Status
Accepted

## Context

Terceira revisão externa do `master` no commit `55c2da6`. As correções do ADR
0013 (T1 = N5, Gdisp com população inteira, `random_state` nas árvores de
fitness) foram confirmadas. A revisão apontou um P0 novo no GA e uma lista de
pendências metodológicas. Todos os itens abaixo foram **verificados
empiricamente** antes de qualquer alteração.

### P0 #1 — a prole era avaliada com os bags errados a partir da 2ª geração

`FitnessEvaluatorMixin.get_complexity(first_evaluate=False, population=None)`
montava os **nomes** corretos da prole:

```python
begin = self.name_individual - self.nr_individual
for i in range(begin, self.name_individual):
    dist["name"].append([i])
```

mas avaliava sempre as **mesmas posições** de `self.bags`:

```python
self._eval_many(range(100, self.nr_individual + 100))
```

`range(100, 200)` é constante. Instrumentando uma execução real (Wine,
`nr_generation=3`, `random_state=42`):

```
gc 2: nomes 100..199  ->  bags avaliados 100..199   (correto por coincidência)
gc 4: nomes 200..299  ->  bags avaliados 100..199   ERRADO
gc 6: nomes 300..399  ->  bags avaliados 100..199   ERRADO
```

Ou seja: numa execução de 20 gerações, apenas a geração 1 avaliava a própria
prole. Da geração 2 em diante o NSGA-II selecionava indivíduos usando o
fitness de outros indivíduos — o comportamento evolutivo do GA era inválido.

O teste de caracterização já chamava o `range(100, nr_individual+100)` de
"known legacy bug", mas subestimava o impacto. Os testes de reprodutibilidade
do ADR 0012 não pegam isso: um bug determinístico continua determinístico.

### P0 #2 — `exit(0)` matava a execução inteira, com código de saída 0

`GeneticOperatorsMixin.crossover` fazia `print("Stratification error"); exit(0)`
quando 30 tentativas de cruzamento não produziam um bag estratificado.
`exit(0)` levanta `SystemExit`, que herda de `BaseException` e **não** é
capturado pelo `except Exception` de `pos/oracle/run_recorder.py`.

Verificado em Ecoli, fold 2 do protocolo `--full`:

```
Interation -  0
gen  nevals
0    100
Stratification error
*** NOT caught by 'except Exception': SystemExit: 0 ***
```

Como `summary.csv`, `per_dataset_summary.csv` e `run_manifest.json` só são
escritos **depois** do laço de datasets, o `--full` morreria em Ecoli (a 6ª
das 31 bases), com **exit code 0** e sem nenhum arquivo de resumo — parecendo
sucesso.

O laço `while y_data[inst] != y2_data[inst2 - 1]` em `mutation` tem o mesmo
problema em forma pior: se o bag perdeu a classe do doador, ele não termina.

### P0 #3 — Ecoli e Glass são incompatíveis com 10-fold

Do `results/datasets/catalog.csv`:

| base  | n   | classes | menor classe |
|-------|-----|---------|--------------|
| Ecoli | 336 | 8       | **2**        |
| Glass | 214 | 6       | **9**        |

Ambas abaixo de `n_splits=10`. Medido nos 10 folds de Ecoli: 8 folds ficam com
1 instância da classe rara no treino (o `train_test_split(stratify=)` de
`generate_bags` levanta `ValueError`, capturado e o fold é perdido) e 2 folds
ficam com 2 (caem no `exit(0)` acima).

O `try/except` silencioso levaria a um `n` diferente por base/modo sem aviso.

### P0 #4 — `_t1_fast` não escala para as bases grandes

O ADR 0013 dizia "aceitável para bags de ~250 amostras". Um bag de Magic tem
~6,8 mil (19.020 → 90% outer → 80% pós-validação → 50% por bag). O laço guloso
reconstruía `set(np.where(covered)[0])` uma vez **por candidato, por rodada**.
Medido (`_t1_fast` original):

| n     | tempo   |
|-------|---------|
| 250   | 0,29 s  |
| 500   | 1,48 s  |
| 1.000 | 10,44 s |
| 2.500 | 154,6 s |

≈ O(n^2,8). Extrapolando para um bag de Magic: dezenas de minutos **por bag**,
× 100 bags × 21 gerações × 10 folds. A estimativa de ~3 h para o experimento
completo era inviável por várias ordens de grandeza.

### Pendências menores confirmadas

- `the_function` passava `generation=self.generation`, que ficava fixo em 0 —
  `gen_temp` registrava a geração errada.
- `PoolBuilderMixin.get_pool()` reconstruía o pool final com
  `DecisionTreeClassifier()` sem `random_state`.
- O split interno treino/validação usava `rng.choice` — não estratificado.
- `save_fold_artifacts` gravava `correctness_matrix.npy` sem o modo no nome:
  com `--mode ga,bagging,rf`, cada modo sobrescrevia o anterior.
- Documentação: `docs/protocol.md` dizia "três modos" mas listava
  `ga + rf` nas tabelas e "2 modos" no fim; o CLI tinha `default="ga,rf"`.

## Decision

### 1. Resolver a prole por ID, não por posição presumida

```python
begin = self.name_individual - self.nr_individual
names = list(range(begin, self.name_individual))
indices = [self.bags["name"].index(i) for i in names]
c, score, pred, pool = self._eval_many(indices)
```

Mesma estratégia já usada no ramo `population is not None`. Não depende mais
da coincidência nome == posição nem de `nr_individual == 100`.

### 2. `StratificationError` em vez de `exit(0)`

Novo `pos/pool/errors.py`. `crossover` levanta após 30 tentativas; `mutation`
ganhou um teto de 1000 tentativas com o mesmo erro. Ambos são `Exception`,
portanto o `run_recorder` registra o fold em `manifest["errors"]` e segue.

### 3. Portão de elegibilidade por base

Novo `pos/oracle/fold_splitter.py` com `check_dataset_viability()`:
`min_class_count >= n_folds`, e para o modo `ga` a classe mais rara precisa
sobreviver à cadeia k-fold → validação → bag com ≥ 2 instâncias.

Bases reprovadas são puladas **explicitamente** e registradas em
`run_manifest.json` sob `skipped_datasets`, com o motivo. Ecoli e Glass ficam
em `FULL_DATASETS` para que a exclusão seja auditável: o `--full` roda 29
bases elegíveis das 31 listadas.

### 4. Greedy lazy (CELF) em T1

`_greedy_cover_fraction()` substitui o rescan O(n³) por um heap com chave
`(-ganho, índice)`. A cobertura marginal é submodular, então um candidato
cujo ganho recalculado ainda é ≤ o topo do heap é o máximo verdadeiro. A
chave lexicográfica reproduz **exatamente** o desempate do laço ingênuo
(maior ganho, menor índice).

Equivalência verificada em 12 configurações (classes separadas, sobrepostas e
dados com duplicatas — o caso onde o desempate importa): 0 divergências.

| n     | antes    | depois  |
|-------|----------|---------|
| 250   | 0,29 s   | 0,004 s |
| 1.000 | 10,44 s  | 0,033 s |
| 2.500 | 154,6 s  | 0,146 s |
| 7.000 | (horas)  | 0,92 s  |

Um bag de Magic passa a custar menos de 1 s.

### 5. Correções pontuais

- `the_function`: `self.generation = gen` e `generation=gen` nas chamadas de
  `max_distance`/`max_acc`.
- `get_pool()`: `DecisionTreeClassifier(random_state=self.random_state + i)`.
- `stratified_val_split()` (`train_test_split(..., stratify=y_train)`) usado
  por `run_recorder` e por `experiment.py`.
- `save_fold_artifacts` grava `correctness_matrix_<mode>.npy` e
  `predictions_<mode>.npz`.
- CLI: `--mode` default `ga,bagging,rf`; `--smoke` passa a `nr_generation=3`.
- `docs/protocol.md` alinhado.

### 6. `--smoke` com 3 gerações

Um smoke com `nr_generation=1` jamais executaria o caminho onde estava o P0
#1. Três gerações é o mínimo que exercita duas avaliações de prole.

## Consequences

- **Resultados de GA anteriores a este commit não são válidos** para
  `nr_generation > 1`. Não há execução versionada em `results/`, então nada
  precisa ser retratado.
- O `--full` passa a cobrir 29 bases, não 31. Isso precisa constar no
  relatório: não é uma falha silenciosa, é uma exclusão documentada.
- T1 continua a aproximação `raio = d(nearest enemy)/2` sobre features
  normalizadas com `KDTree(metric="manhattan")`. O ECoL calcula os raios
  recursivamente sobre a sua própria matriz de distâncias. A otimização deste
  ADR **não** aproxima nem afasta a implementação do ECoL — os valores são
  bit-a-bit os mesmos de antes, só mais rápidos. A validação de equivalência
  contra o ECoL em R continua **pendente**.
- O modo `ga` permanece uma variante **GA-F1/T1** do PGDCS. `get_best_types()`
  segue bypassado. Isso deve ser apresentado como tal, não como reprodução
  integral do PGDCS.

## Tests

- `tests/unit/pool/test_offspring_generation_indices.py` — a geração *g*
  precisa avaliar os nomes `100g..100g+99` **nos bags desses mesmos nomes**;
  teste específico para o sintoma legado (geração 2 lendo posições 100..199);
  `gen_temp` não fica preso em 0; `StratificationError` não é `SystemExit`.
- `tests/unit/oracle/test_fold_splitter.py` — aritmética de bag, portão de
  elegibilidade, estratificação e reprodutibilidade do split de validação.
- `tests/unit/complexity/test_t1_hyperspheres.py` — equivalência do greedy
  lazy contra o ingênuo (incluindo empates) e teto de tempo para n = 7.000.

## Pendências deixadas em aberto (não resolvidas aqui)

1. Validar `T1_fast` contra o `T1`/`N5` do ECoL em R sobre bags congelados
   (erro absoluto, Pearson/Spearman e, principalmente, o *ranking* dos bags).
2. Decidir formalmente entre `GA-F1/T1` e o PGDCS completo (reativar
   `get_best_types` com Overlapping + Neighborhood → NSGA-II).
3. Protocolo alternativo documentado para Ecoli e Glass, se elas forem
   necessárias ao estudo.
