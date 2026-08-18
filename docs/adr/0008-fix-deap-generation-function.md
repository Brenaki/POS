# ADR 0008 — Fix DEAP generation_function e fallback do stop-criteria

- **Status**: Accepted
- **Date**: 2026-08-18
- **Supersedes**: ADR 0007 (parcialmente — o bug está corrigido, mas ADR 0007 permanece como registro histórico do bug)

## Context

ADR 0007 documentou que `poolGeneration.generate()` crasha com
`TypeError: eaMuPlusLambda() got an unexpected keyword argument 'generation_function'`
porque DEAP 1.3.3 (pinado em `requirements.txt`) não aceita esse parâmetro.
A Fase 4 (Oracle_N) precisa de `get_pool()` retornando classificadores fitted,
o que exige que `generate()` funcione end-to-end.

## Decisão

1. **Re-implementar o loop mu+lambda** em `pos/pool/ea_loop.py` com suporte
   a callback por geração. A função `ea_mu_plus_lambda_with_callback` é
   idêntica a `deap.algorithms.eaMuPlusLambda` exceto por invocar
   `generation_function(population, gen, population[0].fitness.values)`
   após cada seleção de geração (e após a avaliação inicial em gen=0).

2. **`pool_generation.py`** passa a importar e usar
   `ea_mu_plus_lambda_with_callback` em vez de `algorithms.eaMuPlusLambda`.

3. **Inicializar `pop_temp`, `bags_temp`, `gen_temp`** no `__init__` de
   `poolGeneration` (antes não eram inicializados — bug latente exposto
   quando o loop DEAP passa a rodar).

4. **Fallback no `the_function`**: se `stop_criteria in ("maxdistance","maxacc")`
   mas `pop_temp` está vazio (nenhuma geração melhorou `dist_temp`/`acc_temp`),
   salvar a última população (`self.off`) em vez de crashar com
   `AttributeError`. Isso garante que `get_bags()`/`get_pool()` sempre
   retornem resultados não-vazios após `generate()`.

## Interpretação do callback

O contrato original do `generation_function` é ambíguo porque o DEAP
upstream nunca suportou esse parâmetro — era uma versão modificada usada
pelo autor original (Marcos Monteiro), cujo código-fonte não está disponível.

A interpretação adotada: `fitness = population[0].fitness.values` (tupla
de 3 escalares = fitness do primeiro indivíduo da população pós-seleção,
representante do first-front do NSGA-II).

**Bug conceitual residual em `max_distance`**: `dispersion(np.column_stack([f0, f1, f2]))`
com `f0,f1,f2` escalares cria uma matriz 1x3, e `pairwise_distances` de 1 linha
= `[[0.0]]`, então `dist_dist_media = 0.0` sempre. Isso significa que
`max_distance` nunca atualiza `pop_temp` com essa interpretação de callback.

O fallback (decisão 4) contorna isso: se `pop_temp` está vazio, usamos a
última população. Uma correção mais profunda exigiria passar a matriz Nx3
de fitness de toda a população para `max_distance`, mas isso mudaria a
semântica do critério de parada e precisa de seu próprio ADR futuro.

## Consequências

- **+** `poolGeneration.generate()` funciona end-to-end (validado em
  load_wine, 100 bags, 100 DecisionTrees fitted, ~270s com nr_generation=1).
- **+** `get_pool()` retorna classificadores fitted → habilita Fase 4.
- **+** `sample.ipynb` pode rodar (necessário ajustar `R_HOME` no Linux).
- **−** O critério de parada `maxdistance`/`maxacc` é efetivamente
  desabilitado (sempre cai no fallback) até que o bug conceitual de
  `max_distance` seja corrigido em ADR futuro.
- **−** A interpretação do callback é uma aproximação do contrato original
  perdido; sem o DEAP modificado original, não há como verificar fidelidade.
