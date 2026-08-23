# ADR 0013: T1 = Fraction of Hyper-spheres (N5), maxdistance/Gdisp fix

## Date
2026-08-22

## Status
Accepted

## Context

Uma segunda revisão externa identificou dois P0 científicos restantes:

### P0 #1: T1 estava implementado como N3 (1-NN error), não N5 (Hyper-spheres)

O PGDCS usa `types=["F1", "T1"]` onde T1 é "Fraction of Hyper-spheres Covering
Data" conforme o paper (p. 4): hiperesferas centradas em instâncias, expandidas
até tocar instâncias de outra classe. T1 = #esferas / #instâncias.

O ECoL atual chama isso de N5. O mapeamento PGDCS→ECoL:
- PGDCS T1 = ECoL N5 (Fraction of Hyper-spheres)
- PGDCS N3 = ECoL N3 (1-NN leave-one-out error) — não implementado

Nossa implementação anterior fazia T1 = N3 (1-NN error rate), que é a medida
errada. O GA estava otimizando F1 + 1-NN-error + DoubleFault em vez de
F1 + Hyper-spheres + DoubleFault.

### P0 #2: maxdistance/Gdisp sempre retornava 0

O callback `generation_function` recebia `population[0].fitness.values` —
uma única tupla (3 valores). `max_distance` chamava `dispersion()` com uma
matriz 1×3, cuja distância média é sempre 0. O critério nunca melhorava,
e o código caía no fallback: salvar a última população.

O PGDCS seleciona a geração com maior dispersão global (Gdisp) — a geração
onde a população é mais espalhada no espaço de fitness 3D. Sem essa seleção,
o GA está efetivamente usando a última geração, não a melhor.

### Reprodutibilidade do DecisionTreeClassifier

`DecisionTreeClassifier()` era criado sem `random_state` explícito. Com
`jobs=1` e `np.random.seed()` global, era provavelmente determinístico,
mas não garantido com threads.

## Decision

### 1. T1 = Fraction of Hyper-spheres Covering Data (N5)

Criado `pos/complexity/neighborhood_extra.py` com:
- `_t1_fast(Xn, y)`: N5 via KDTree (nearest enemy distance → radius =
  de/2 → greedy set cover)
- `_n3_fast(Xn, y)`: N3 = 1-NN leave-one-out error rate (movido de
  neighborhood_fast.py, agora corretamente nomeado)

`neighborhood_fast.py` delega T1 e N3 para `neighborhood_extra.py`.

### 2. maxdistance/Gdisp corrigido

`ea_loop.py` agora extrai a matriz Nx3 de fitness de TODA a população:
```python
_population_fitness_matrix(population)  # Nx3 array
```

`max_distance()` recebe a matriz completa e calcula `dispersion(Nx3)` —
a distância média real entre todos os indivíduos no espaço de fitness.

O critério agora seleciona a geração com maior Gdisp, como o PGDCS.

### 3. random_state no DecisionTreeClassifier

`biuld_classifier_tree()` recebe `random_state` explícito.
`fitness_evaluator._eval_one()` e `parallel_distance2()` propagam
`random_state + i` (offset por bag) para cada árvore.

## Consequences

- **+** T1 agora corresponde ao PGDCS (Fraction of Hyper-spheres)
- **+** Gdisp seleciona a geração mais dispersa, não a última
- **+** Reprodutibilidade end-to-end comprovada (3 testes E2E passando)
- **+** DecisionTree com seed explícita por bag
- **−** T1 (N5) é O(n²) no pior caso (greedy set cover) — aceitável
  para bags de ~250 amostras (tam_bags=0.5)
- **−** N4 ainda é kDN, não Non-Linearity of 1-NN — divergência
  documentada, não afeta o experimento atual (types=["F1","T1"])

## Tests

- 5 testes T1 (test_ecol_measures.py::TestT1Measure) — inclui test_t1_is_not_n3
- 3 testes E2E (test_ga_reproducibility.py) — same seed → same pool → same Oracle_N
- 159 testes totais passando, 0 falhas

## Referências

- PGDCS paper: "Fraction of Hyper-spheres Covering Data (T1) is calculated
  after the creation of hyper-spheres that are centered in a randomly picked
  instance of a class. These hyper-spheres are increased until they touch
  instances from other classes."
- ECoL R source: github.com/lpfgarcia/ECoL, R/neighborhood.R (class.N5)
- ADR 0012 — correções P0 anteriores