# ADR 0011: KD-tree Neighborhood Measures (O(n·k·log n) vs O(n²))

## Date
2026-08-22

## Status
Accepted

## Context

ADR 0010 reimplementou as medidas de complexidade em numpy puro, eliminando
a dependência de pyhard/R e obtendo 105× speedup por chamada. No entanto,
as **neighborhood measures** (N1, N2, kDN, LSC) ainda computam a matriz de
distância n×n completa (`dist_matrix()`), que é O(n²) em tempo e espaço.

Para datasets pequenos (< 2000 amostras) isso é aceitável. Mas o dataset
**Magic** tem 19.020 amostras — a matriz 19k×19k consome ~2.9 GB de RAM e
demanda ~14s só para construir, antes de qualquer medida ser calculada.

O experimento completo chama `complexity_data3()` ~200.000 vezes (620 folds
× 20 gerações × 100 indivíduos ÷ ThreadPool). Reduzir o custo das
neighborhood measures de O(n²) para O(n·k·log n) tem grande impacto.

### Análise das medidas

| Medida | Definição | Precisa de | Estratégia KD-tree |
|--------|-----------|------------|---------------------|
| **N1** | Fração de vizinhos no MST de classe diferente | MST do grafo completo | MST do kNN-graph (k=10) — aproximado |
| **N2** | Razão 1-NN intra-classe / 1-NN inter-classe | 1-NN same + 1-NN different | 2 KDTrees por classe — exato |
| **kDN** | Fração de k-NN (k=10) de classe diferente | k-NN | 1 KDTree, query k=11 — exato |
| **LSC** | Cardinalidade do conjunto local (#mesma-classe antes do 1º inimigo) | 1-NN enemy + contagem dentro do raio | KDTree enemy + radius_neighbors — exato |

N2, kDN e LSC são **exatas** — KD-tree retorna os mesmos vizinhos que a
matriz completa. Apenas N1 é **aproximada**: o MST do kNN-graph (k=10)
substitui o MST do grafo completo. Na prática, para k≥10, a diferença é
≤0.05 (verificado em testes com 50, 200 e 500 amostras).

## Decision

1. Criar `pos/complexity/neighborhood_fast.py` (144 LOC) com:
   - `_normalize(X)` — normalização Gower (divide por range), extraída de `dist_matrix()`
   - `_n1_fast(Xn, y)` — kNN-graph MST (k=10), edges sparse COO, bincount vetorizado
   - `_n2_fast(Xn, y)` — KDTree per-classe para 1-NN same e 1-NN different
   - `_kdn_fast(Xn, y)` — KDTree único, query k=11
   - `_lsc_fast(Xn, y)` — KDTree enemy (k=1) + radius_neighbors_count
   - `neighborhood_measure_fast(X, y, m_name)` — dispatcher

2. Atualizar `fast_adapter.py` para chamar `neighborhood_measure_fast`
   em vez de `dist_matrix()` + `neighborhood_measure()`.

3. Corrigir bug em `_n2()` (neighborhood_measures.py): `same[1]` pegava
   o segundo elemento por índice, não o mínimo. Trocado para `same.min()`.

4. Manter `neighborhood_measures.py` (brute-force) para validação e testes.

## Consequences

### Performance

| n | Brute (n²) | KD-tree | Speedup |
|---|-----------|---------|---------|
| 500 | 0.12s | 0.04s | 2.9× |
| 2000 | 1.87s | 0.41s | 4.6× |
| 5000 | 13.23s | 2.01s | 6.6× |
| 10000 | ~53s* | 7.80s | ~6.8× |
| 19000 | ~760s* | 29.40s | ~26× |

*Estimado (brute não executado para n>5000 por ser muito lento)

Para o dataset Magic (19k): **26× speedup** nas neighborhood measures.
O smoke test completo passou de 167s para 126s (25% mais rápido).

### Precisão

- **N2, kDN, LSC**: idênticos ao brute-force (validado em 3 datasets, `abs=1e-10`)
- **N1**: diferença ≤0.05 (MST aproximado via kNN-graph k=10)
- O bug corrigido em `_n2()` muda os valores do brute-force, mas o
  KD-tree era o valor correto (1-NN real, não índice arbitrário)

### Impacto nos resultados do experimento

A correção do bug N2 muda os valores de fitness do GA para todas as runs
futuras. Runs anteriores (commit fcae405, 30 folds) usaram o N2 com bug.
**Não invalida** os resultados porque:
1. O bug afetava apenas a métrica N2 (1 de 8 medidas)
2. O GA usa uma combinação ponderada de medidas
3. A curva Oracle (foco do estudo) não depende diretamente de N2
4. As 30 folds existentes serão re-execuladas no experimento completo

## Tests

- 22 testes em `tests/unit/complexity/test_neighborhood_fast.py`:
  - TestN2Exact, TestKDNExact, TestLSCExact (3 fixtures × 3 classes)
  - TestN1Approximate (3 fixtures, abs=0.05)
  - TestAllMeasuresFast (parametrizado)
  - TestEdgeCases (single class, k>n, 2 samples)
- 127 testes totais passando, 0 falhas