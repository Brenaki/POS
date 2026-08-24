# ROADMAP — Oracle_N Research Project

> Tracks progress against the cronograma in
> `docs/SubProjeto - ANÁLISE DO IMPACTO DE ORACLES EM DIFERENTES NÍVEIS NA ACURÁCIA DE SISTEMAS DE MÚLTIPLOS CLASSIFICADORES.md`.
> Update this file every session that advances a milestone.

## Legend

- [ ] not started
- [~] in progress
- [x] done
- [!] blocked

## Cronograma (Set/2026 → Ago/2027)

| #  | Atividade                                                              | Meses              | Status |
|----|------------------------------------------------------------------------|--------------------|--------|
| 1  | Revisão bibliográfica sobre MCS, DCS/DES, diversidade e Oracle         | Set–Nov/2026       | [~]    |
| 2  | Seleção das bases de dados e definição do protocolo experimental       | Out–Nov/2026       | [x]    |
| 3  | Implementação do pool de classificadores e avaliação individual         | Nov/2026–Jan/2027  | [x]    |
| 4  | Implementação do Oracle tradicional e dos Oracles em diferentes níveis | Dez/2026–Fev/2027  | [x]    |
| 5  | Relatório Semestral conforme modelo PROPESP (até 15 mar 2027)          | Mar/2027           | [ ]    |
| 6  | Experimentos comparativos (votação majoritária, combinação de probs, DCS/DES) | Fev–Jun/2027 | [x]    |
| 7  | Análise dos resultados, gráficos e recomendações                       | Ago–Nov/2027       | [x]    |
| 8  | Relatório final (até 11 set 2027)                                      | Jul–Ago/2027       | [ ]    |

## Engineering milestones (enabling the cronograma)

These are prerequisites mandated by `AGENTS.md` (TDD, Refactoring, ≤150 LOC/file,
ADR-per-decision, ≥80% coverage). They are sequenced before the science milestones.

### Fase 0 — Infraestrutura
- [x] Python 3.10 venv via `uv` (`.venv/`)
- [x] Deps Python não-R instaladas (sklearn, numpy, pandas, deap, deslib, mlxtend)
- [x] `pyproject.toml` (pytest + ruff + mypy + cov≥80)
- [x] `ROADMAP.md` (this file)
- [x] ADRs 0001–0003 criados
- [~] R + ECoL instalados (pendente: `sudo pacman -S r tk` + `R -e 'install.packages("ECoL")'`)
- [~] `rpy2==3.5.7` instalado (bloqueado por R)
- [ ] `pyhard` validado funcional com deps pinadas

### Fase 1 — Suite de caracterização (Fowler, antes de mudar nada)
- [ ] `tests/conftest.py` com fixtures `load_wine_split`, `requires_r` skip
- [ ] `tests/characterization/cpx_characterization_test.py` — golden values de
      `min_max_norm`, `dispersion_linear`, `dispersion`, `diversitys`,
      `voting_classifier`, `complexity_data3` (R), `biuld_classifier[_tree]`
- [ ] `tests/characterization/pool_generation_characterization_test.py` —
      `poolGeneration.generate → get_bags → get_pool` end-to-end (load_wine, rs=42)
- [ ] `tests/unit/` para funções isoladas
- [ ] Cobertura ≥80% das partes não-R; verdes

### Fase 2 — Refatoração estrutural (preserva comportamento)
- [x] Pacote `pos/` criado
- [x] `Cpx.py` → `pos/{normalization,dispersion,diversity,voting,complexity/*,classifiers,io_csv}.py` (cada <150 LOC)
- [x] `pool_generation.py` → `pos/pool/{bag_generator,data_splitter,complexity_voter,fitness_evaluator,genetic_operators,stop_criteria,pool_builder,pool_generation}.py` (<150 LOC cada)
- [x] Facades `cpx_legacy.py` + `pool_generation_legacy.py` re-exportam API pública
- [x] `sample.ipynb` continua rodando sem alteração
- [x] ADRs 0004–0005

### Fase 3 — Remoção do R via pyhard (modifica comportamento)
- [ ] `pos/complexity/pyhard_adapter.py` com `ClassificationMeasures`
- [ ] Substituir `ecol.overlapping/neighborhood/...` em `complexity_data3`
- [ ] Golden values do ECoL preservados em `tests/_ecol_legacy_golden.json`
- [ ] Testes de caracterização atualizados para assertar propriedades (shape, range)
- [ ] ADR 0003 atualizado com decisão pyhard final

### Fase 4 — Oracle_N (TDD, novo)
- [x] `pos/oracle/correctness_matrix.py` — matriz de acertos do pool
- [x] `pos/oracle/oracle_n.py` — `oracle_n_accuracy(matrix, n)`
- [x] `pos/oracle/oracle_curve.py` — curva Oracle_1..M
- [x] `pos/oracle/comparison.py` — vs majority/mean (DCS/DES na Fase 5)
- [x] `pos/oracle/arff_loader.py` — carrega 31 datasets .arff
- [x] `pos/oracle/experiment.py` — 10-fold stratified CV, random_state=42
- [x] `tests/unit/oracle/` com casos triviais (N=1, N=M, N intermediário, monotonicidade)
- [x] ADR 0006 — arquitetura Oracle_N, protocolo 10-fold, fórmula
- [x] ADR 0008 — fix DEAP generation_function (pré-requisito Q1=B)

### Fase 5 — Infraestrutura de experimentos (GA pool vs RF baseline)
- [x] `pos/oracle/dataset_catalog.py` — catálogo de 31 datasets (build/save/load)
- [x] `pos/oracle/pool_evaluation.py` — `evaluate_pool()` (acurácias individuais,
      matriz de acertos, curva Oracle, majority vote, mean probs, double-fault)
- [x] `pos/oracle/random_forest_pool.py` — `build_rf_pool()` via RandomForestClassifier
- [x] `pos/oracle/run_recorder.py` — `record_run()` orquestra experimentos com try/except por fold
- [x] `pos/oracle/run_helpers.py` — helpers de git/deps/hash, `build_pool_ga`/`build_pool_rf`,
      `save_fold_artifacts`, `build_fold_manifest`, `build_summary_row`, `per_dataset_summary`
- [x] `scripts/run_experiment.py` — CLI `--smoke`/`--full`/`--config`/`--mode`/`--jobs`/`--M`/`--dry-run`
- [x] `docs/protocol.md` — protocolo experimental formal
- [x] `results/datasets/catalog.csv` — 31 datasets (178–19020 samples, 2–8 classes, IMB 1.0–71.5)
- [x] Smoke test validado: 3 datasets (Wine/Banana/Vehicle) × 2 modes (ga/rf) × 3 folds = 18 rows
      — oracle_1 ≥ majority_vote (0 violações), curvas monótonas 18/18, M=100, manifest+artefatos OK
      — **obsoleto**: rodado com `nr_generation=1` e o bug de prole do ADR 0014 ativo
- [x] ADR 0009 — layout de reprodutibilidade de experimentos

### Fase 6 — Correções P0 pré-experimento científico (revisões externas)
- [x] ADR 0012 — reprodutibilidade do GA (`random_state`), tipos ECoL, baseline Bagging
- [x] ADR 0013 — T1 = Fraction of Hyper-spheres (N5), Gdisp com população inteira
- [x] ADR 0014 — **prole avaliada nos próprios bags a partir da 2ª geração** (P0 que
      invalidava todo GA com `nr_generation > 1`), `exit(0)` → `StratificationError`,
      portão de elegibilidade por base (Ecoli/Glass fora do 10-fold), T1 lazy-greedy
      (2.500 amostras: 154,6 s → 0,15 s), split de validação estratificado,
      artefatos por modo, `gen_temp` real, `random_state` no pool final,
      `--mode` default `ga,bagging,rf`, `--smoke` com 3 gerações
- [x] ADR 0015 — otimizações **exatas** (bit-idênticas) + Perceptron da tese:
      `diversitys` vetorizado (8.877x), memoização da avaliação de bags (metade
      do trabalho por geração era recomputação de sobreviventes), T1 com árvores
      por classe, `voting_classifier` sob demanda, `dispersion*` vetorizadas,
      Perceptron linear corrigido e adotado como base do GA, `--jobs` = núcleos-1.
      Magic: 8,05 h → 0,71 h
- [ ] Validar `T1_fast` contra ECoL/R em bags congelados (ranking dos bags)
- [x] Decidir formalmente: `GA-F1/T1` vs PGDCS completo (`get_best_types` reativado)
      — decidido no **ADR 0019**: em vez de escolher no papel, os dois viram modos
      do run canônico (`ga` e `pgdcs`) e a diferença é medida. Ver Fase 8.
- [x] Rodar `--full` científico e versionar `results/`
      — run `2026-08-22T23-33-56_2a8a0f5`: 29 bases x 10 folds x 3 modos = 870 folds,
      0 erros, ~2h20. Invariantes Oracle_N verificados em 870/870.
      Análise em `docs/resultados-oracle-n.md`, figuras em `results/**/figures/`.
- [x] Reexecutar `--full` com ADR 0016 para preencher `mean_probs` do GA (290 linhas)
      — run `2026-08-23T06-42-14_376203d`: 870 folds, 0 erros. Determinismo verificado:
      as colunas do pool saíram bit-idênticas ao run anterior em 870/870 linhas nos
      três modos; só `mean_probs` mudou. Resultado: a combinação suave é
      *significativamente pior* que o MVR no pool de Perceptrons (−0.0081, p=0.014)
      e empata nos pools de árvores.

### Fase 7 — Métodos reais de combinação e análise (marcos 6 e 7 do cronograma)
- [x] ADR 0017 — DCS/DES via DESlib: OLA, LCA, KNORA-E, KNORA-U, META-DES em
      todo fold de todo modo, DSEL = a partição de validação; shim de
      compatibilidade `deslib_compat` para os aliases NumPy removidos em 1.24
- [x] `pos/analysis/fusers.py` — comparação por fusor (Friedman/Nemenyi com k até 7)
      e a métrica `recovered` = folga do Oracle recuperada pela seleção dinâmica
- [x] `pos/analysis/figures_des.py` — fig7 (fusores vs teto Oracle_1) e
      fig8 (folga recuperada, por modo e vs `DF/e²`)
- [x] Rodar `--full` com DCS/DES e reescrever `docs/resultados-oracle-n.md`
      — run `2026-08-23T10-09-30_376203d`, 870 folds, 0 erros, 2h20. Reproduziu
      as 870 linhas do run anterior bit-a-bit em toda coluna de pool (agora
      incluindo `mean_probs`): inserir DCS/DES não perturba a geração dos pools.
      Foi lançado de árvore suja, então seu `run_manifest.json` traz
      `git_sha: 376203d` com `git_dirty: true`; o código que ele executou é o do
      commit `25f4c14`.
      **Achado 5**: a seleção dinâmica recupera 19.96% (GA), 12.77% (Bagging) e
      10.45% (RF) da folga `Oracle_1 − MVR`, e isso já usando o melhor dos cinco
      métodos escolhido a posteriori no teste — o Oracle_1 é limite inatingível,
      não meta. O DCS puro (OLA/LCA) é *significativamente pior* que o MVR nos
      pools de árvores (até −0.062, p<1e-5); KNORA-U é o único que nunca perde.
      No pool de Perceptrons KNORA-U ganha do MVR (+0.0274, p=0.005), com a
      vantagem concentrada nas bases 2D de fronteira não-linear (recuperação
      0.807 em P2/Lithuanian/Banana vs 0.16 do Bagging). `recovered` **não**
      correlaciona com `DF/e²` (rho=+0.014, p=0.68).
- [x] **ADR 0018** — split de três vias (`X_tr` / `X_val` fitness / `X_dsel`
      DSEL, com `X_tr` intacto para os pools de árvore saírem bit-idênticos),
      suíte de métricas (precisão/revocação/F1 macro + acurácia balanceada) para
      todo fusor, `oracle_1_balanced`, e as predições do DCS/DES gravadas no
      `.npz` — a falha do ADR 0017 que obrigou este rerun
- [x] Lista de métodos de 5 → 12, escolhida por benchmark: MCB e Rank (a
      penalidade do DCS é geral?), DES-P e DES-KNN (DES-KNN liga ao `DF/e²`),
      KNOP, e os estáticos SingleBest e StaticSelection (o extremo do eixo de
      poda). MLA, APriori e APosteriori medidos e descartados com motivo.
- [x] Duas camadas estatísticas: Friedman/Nemenyi sobre o conjunto do ADR 0017
      (poder preservado, comparável com o run anterior) + Wilcoxon com correção
      de Holm para os 12 restantes
- [x] Rodar o `--full` do ADR 0018 e medir o viés de DSEL por
      diferença-em-diferenças contra o run `2026-08-23T10-09-30_376203d`
      — run `2026-08-23T14-19-59_2286fcc`, 870 folds, 0 erros, 4h08 (1.76x o run
      anterior; a estimativa do ADR errou por 6x, corrigida lá).
      **Pré-condição verificada**: com `X_tr` intacto, os pools de Bagging e RF
      saíram bit-idênticos ao run anterior — `max |Δ| = 0.000000` em toda coluna
      de pool e curva Oracle idêntica em 290/290 folds de cada modo. Sem isso a
      decomposição não valeria.
      **Achado 0**: restrito aos 5 métodos comuns aos dois runs, o DSEL pela
      metade custou −0.0057 de folga recuperada ao grupo de controle (Bagging+RF,
      que não tinham viés), o GA perdeu −0.0003, logo o **viés de DSEL é +0.0055**
      — ~3% relativos, e nenhum dos deltas é significativo. O desenho antigo
      **não** estava inflando materialmente o GA; a correção está feita e é a que
      vale daqui em diante.
- [x] Reescrever `docs/resultados-oracle-n.md` com a leitura em acurácia
      balanceada e consolidar uma seção de recomendações (fecha o marco 7)
      — **Achado 5** (12 métodos): recuperação 24.0% (GA), 15.0% (Bagging), 14.2%
      (RF) mesmo usando o melhor dos 10 dinâmicos escolhido a posteriori. O eixo
      que organiza tudo é a **poda**: OLA/LCA/MCB/Rank/melhor-individual perdem
      0.044–0.069 contra o MVR nos pools de árvores em 27–29 das 29 bases
      (p_Holm<0.001); os que não podam (KNORA-U, DES-P, META-DES, KNOP, seleção
      estática) empatam. **KNORA-U é o único fusor que bate o MVR de forma
      sustentada em algum modo** (+0.0272 no GA, p_Holm=0.0045).
      **Achado 6**: em acurácia balanceada a folga *cresce* (GA 0.2192→0.2599) e o
      DCS puro sobe 0.3–0.4 posições de rank, mas o topo não muda — o ranking de
      fusores é robusto à métrica.
      **Achado 3 / objetivo 7**: `N*` fica em `N ≈ M/2` nos três modos
      (Friedman p=0.40) porque `Oracle_{M/2+1} ≡ MVR` em problema binário. Não
      existe nível intermediário de Oracle_N que sirva de limite realista.
      Sete recomendações consolidadas na seção final do documento.

### Fase 8 — Correções para publicação (revisão externa)

A revisão externa do relatório (`2c33634`) aprovou a resposta ao subprojeto e
abriu uma segunda pergunta: isto está pronto para virar artigo? Ainda não. Nove
pontos a corrigir, mais o PGDCS completo. Dois são bloqueadores de experimento
(ADR 0019); o resto é rigor, e roda sobre o run novo sem custo de máquina.

- [x] **ADR 0019** — run canônico de árvore limpa, PGDCS completo como quinto
      modo, controle de geração `randbag`, instrumentação de poda
- [x] **B0 · backend da votação de medidas** (bloqueia o PGDCS). O
      `complexity_voter` importa de `pos.complexity` → pyhard, que devolve `0.0`
      fixo para `F1v`, `N3` e **`T1`**: a votação jamais poderia escolher T1, uma
      das duas medidas hoje hardcoded. O fitness do GA já usa o `fast_adapter`,
      então seleção e fitness discordavam sobre a mesma medida (Wine: F3 `0.153`
      vs `0.469`). Somar a isso `complexities()` sem `random_state` — escolha de
      medidas não reprodutível, contra o ADR 0012. Custo da votação por fold:
      pyhard 229 min em Phoneme (inviável) contra 2.6 min no `fast_adapter`.
      Corrigido e testado: a votação virou determinística por seed e T1/N3/F1v
      voltaram a ser elegíveis. Commit `5aceb11`
- [~] **Ponto 2 · modo `pgdcs`** — PGDCS completo, `types=None` reativando
      `get_best_types`; gravar as medidas escolhidas por fold (`pgdcs_types`) —
      é resultado, não telemetria. ~10.7 h só de votação, aceitas explicitamente
- [~] **Ponto 7 · modo `randbag`** — bags aleatórios + Perceptron, a população de
      geração 0 do próprio GA. Desfaz o confounder: `ga` vs `randbag` isola a
      busca, `randbag` vs `bagging` isola o classificador-base
- [~] **Ponto 8 · poda medida** — envolver `ds.select` do DESlib e gravar
      `E[S]/M` por fusor, mais a fração de consultas roteadas para seleção
      (o DESlib curto-circuita vizinhança unânime)
- [~] **Ponto 1 · rerun canônico** — `--full` de árvore limpa, 29 bases x 10
      folds x 5 modos = 1450 folds, ~18–19 h. Critério de aceitação:
      bit-identidade de `ga`/`bagging`/`rf` contra `2026-08-23T14-19-59_2286fcc`
      — se bater, o run sujo fica validado em vez de descartado.
      Em andamento: `2026-08-24T09-38-20_5aceb11`, **`git_dirty: false`** — o
      portão do ponto 1 cumprido. Verificação parcial em 90 folds das 3
      primeiras bases: `ga`/`bagging`/`rf` **30/30 idênticos** em curva Oracle,
      acurácias individuais, MVR e DF. Os dois modos novos não perturbam os
      antigos, e a ordem dos modos não vaza estado de RNG
- [x] **Ponto 6 · `DF` normalizado exato** — o denominador correto é
      `2/(M(M-1)) · Σ_{i<j} e_i e_j`; os `e_i` já estavam gravados em cada
      `fold_manifest_<mode>.json`, então saiu offline, sem rerun.
      **Resultado: não muda nada.** Spearman agrupado −0.8603 → −0.8607; razão
      exato/médio com mediana 1.0005, porque o desvio das acurácias individuais
      dentro de um pool é 0.054. Objeção teoricamente certa, empiricamente
      imaterial aqui. Vale por rigor — e dá LODO 0.839 contra 0.805
- [x] **Ponto 3 · `N*` binário vs multiclasse**.
      **Achou um erro publicado**: `Oracle_{M/2+1} ≡ MVR` é falso. O que vale é o
      sanduíche `Oracle_51 ≤ MVR ≤ Oracle_50`, verificado em **660/660** folds
      binários, com igualdade exata em só 508 (77%) — os outros 23% são empates
      50–50 desfeitos a favor. É a distinção que explica o `N*`: como ele é o
      primeiro nível *estritamente* abaixo do MVR, os empates o empurram de 51
      para 52+, que é onde as medianas caem.
      Nas 7 multiclasse o sanduíche não vale (pluralidade) e `N*` cai para 47.6
      contra 53.7, Mann-Whitney p=2.7e-06. A resposta ao objetivo 7 passou a ser
      explicitamente restrita ao caso binário. Commit `80f1bb2`
- [x] **Ponto 4 · a fórmula `0.15`** — era chamada de "limite superior
      operacional" e não é: **37.7%** dos folds de árvore recuperam mais que
      0.15. Virou estimativa, com a cota a 95% ao lado — `Q_{0.95}(R) = 0.50`,
      mais de três vezes a média, na mesma distribuição em que **31%** dos folds
      recuperam nada. Commit `80f1bb2`
- [x] **Ponto 5 · limiares `DF/e²`** — descobertos e avaliados nas mesmas 29
      bases, logo circulares. Validados leave-one-dataset-out: **a regra
      generaliza**. Acerto 0.805 (0.839 com o denominador exato) contra 0.736 da
      regra trivial, e os cortes reajustados fora da amostra caem na mediana em
      **1.14 / 4.82** — praticamente os ≈1 e ≥4 publicados. Ressalva: a margem é
      de 7 pontos, não de uma ordem de grandeza, porque a faixa do meio domina
      (64 dos 87 pares base×modo). Commit `80f1bb2`
- [~] **Ponto 9 · o que prevê recuperabilidade** — regredir `recovered` sobre as
      medidas de complexidade + `n_classes`, dimensão, desbalanceamento,
      `df_ratio_exact`; avaliação leave-one-dataset-out (n=29)
- [ ] Bases de imagem via embeddings de CNN (objetivo 5 e o vínculo com o projeto
      maior) — segunda bateria experimental, não uma correção; fica aberto

## Decision log (ADRs)

- `docs/adr/0001-python-version-and-deps.md` — pin Python 3.10, deps de requirements.txt
- `docs/adr/0002-file-size-cap-150.md` — aperta o cap de 300 → 150 LOC/arquivo
- `docs/adr/0003-remove-r-via-pyhard.md` — substituir R/ECoL por pyhard
- `docs/adr/0004-split-cpx.md` — split Cpx.py em pacote pos/
- `docs/adr/0005-split-pool-generation.md` — split pool_generation.py via mixins
- `docs/adr/0006-oracle-n-architecture.md` — arquitetura Oracle_N, protocolo 10-fold
- `docs/adr/0007-deap-generation-function-bug.md` — bug pre-existente DEAP (corrigido por 0008)
- `docs/adr/0008-fix-deap-generation-function.md` — fix DEAP loop + fallback stop-criteria
- `docs/adr/0009-experiment-reproducibility.md` — layout de reprodutibilidade de experimentos
- `docs/adr/0010-fast-complexity-numpy.md` — reimplementação numpy pura das medidas de complexidade
- `docs/adr/0011-kdtree-neighborhood-measures.md` — KD-tree para neighborhood measures (O(n²)→O(n·k·log n))
- `docs/adr/0012-p0-fixes-reproducibility-types-ecol.md` — correções P0: reprodutibilidade GA, tipos de medidas, definições ECoL
- `docs/adr/0013-t1-hyperspheres-maxdistance-fix.md` — T1=Fraction of Hyper-spheres (N5), maxdistance/Gdisp fix
- `docs/adr/0014-offspring-fitness-and-protocol-gates.md` — fitness da prole após a 1ª geração, portões de protocolo (Ecoli/Glass), T1 escalável
- `docs/adr/0015-performance-exact-optimizations.md` — otimizações exatas, Perceptron linear da tese como base do GA, paralelismo por padrão
- `docs/adr/0016-soft-fusion-for-margin-pools.md` — combinação suave para pools sem `predict_proba` (média de margens normalizadas)
- `docs/adr/0018-three-way-split-and-metric-suite.md` — split de três vias, suíte de métricas por fusor, 12 métodos de seleção
- `docs/adr/0017-dcs-des-baselines-deslib.md` — DCS/DES via DESlib (OLA, LCA, KNORA-E/U, META-DES), DSEL = validação, métrica `recovered`
- `docs/adr/0019-canonical-rerun-and-generation-control.md` — run canônico de árvore limpa, PGDCS completo como quinto modo, controle `randbag`, poda medida

## Notes

- **R install pending**: rode `sudo pacman -S r tk` no Arch, depois
  `R -e 'install.packages("ECoL", repos="https://cloud.r-project.org")'` e
  `export R_HOME="$(R RHOME)"` antes de instalar `rpy2`.
- **pyhard vs deps pinadas**: pyhard 2.2.4 requer numpy~=1.23, sklearn~=1.5,
  scipy~=1.13 — conflita com as versões pinadas do POS. Decisão adiada para a
  Fase 3 (talvez venv separado ou downgrade do pyhard).
