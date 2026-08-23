# Resultados — Oracle_N em diferentes níveis

- **Run**: `results/experiments/2026-08-23T06-42-14_376203d/` (ADR 0016; substitui o
  run `2026-08-22T23-33-56_2a8a0f5`, idêntico exceto pela coluna `mean_probs`)
- **Commit**: `376203d` (branch `master`)
- **Protocolo**: 29 bases × 10-fold estratificado × 3 modos, M=100, GA com 20 gerações,
  `random_state=42`. 870 folds, 0 erros. Ecoli e Glass excluídas pelo portão de
  viabilidade (menor classe com 2 e 9 instâncias, < 10 folds).
- **Reprodução**: `python scripts/run_experiment.py --full` →
  `python scripts/analyze_results.py results/experiments/<run>`

Este documento mapeia os resultados aos objetivos específicos do subprojeto
(`docs/SubProjeto - ANÁLISE DO IMPACTO DE ORACLES...md`).

---

## Verificação dos invariantes (objetivos 2 e 3)

A relação `Oracle_1 ≥ Oracle_2 ≥ ... ≥ Oracle_M` é a definição da medida
generalizada. Verificada em **870/870 folds**, sem exceção. `Oracle_1 ≥ votação
majoritária` também vale em 870/870 — o Oracle tradicional nunca subestimou um
método real de combinação.

Identidade adicional, verificada com erro máximo `5.6e-16`:

```
mean(Oracle_1 .. Oracle_M) ≡ acurácia individual média do pool
```

É consequência de `Σ_N P(≥N corretos) / M = E[fração de classificadores corretos]`.
Serve como teste de consistência gratuito entre a matriz de acertos e a curva, e
implica que "área sob a curva Oracle_N" **não é uma métrica nova** — é a acurácia
média dos classificadores-base sob outro nome.

---

## Resultado global (objetivos 5 e 6)

| modo | Oracle_1 | Oracle_2 | Oracle_5 | Oracle_M | votação maj. | média probs | acc. individual | DF |
|---|---|---|---|---|---|---|---|---|
| GA (PGDCS, Perceptron) | **0.9979** | 0.9956 | 0.9905 | 0.1136 | 0.7824 | 0.7743 | 0.7052 | 0.1564 |
| Bagging | 0.9951 | 0.9935 | 0.9884 | 0.3052 | 0.8467 | 0.8466 | 0.7931 | 0.1094 |
| Random Forest | 0.9970 | **0.9958** | **0.9920** | 0.1725 | **0.8549** | 0.8546 | 0.7806 | 0.1069 |

Teste de Friedman com pós-teste de Nemenyi (N=29 bases, k=3, CD=0.615) e Wilcoxon
pareado — ver `analysis.txt` do run para o relatório completo.

A coluna `média probs` não usa a mesma regra nos três modos, e a tabela registra
qual foi usada (`soft_fusion_rule`): Bagging e RF usam a média das probabilidades
preditas; o GA usa a média das distâncias normalizadas ao hiperplano, porque o
Perceptron linear da tese não expõe `predict_proba` (ADR 0016). O que é comparável
entre as três colunas é o desempenho da combinação suave, não a escala do escore.

**Verificação de reprodutibilidade.** O ADR 0016 exigiu reexecutar o `--full` para
reconstruir os pools. Como `random_state=42` é propagado, o run novo tinha de
reproduzir os pools antigos exatamente. Verificado em **870/870 linhas nos três
modos**: `oracle_1..5`, `oracle_M`, `oracle_curve_json`, `majority_vote`,
`double_fault_mean` e `mean_individual_acc` saíram bit-idênticos ao run anterior.
A única coluna que mudou foi `mean_probs`, que é o que o ADR 0016 alterou.

### Figuras

| arquivo | conteúdo |
|---|---|
| `figures/fig1_oracle_curve.png` | curva Oracle_1..M média, com a linha da votação majoritária de cada modo |
| `figures/fig2_oracle_curve_zoom.png` | recorte N=1..10 — a faixa de foco do subprojeto |
| `figures/fig3_nstar.png` | distribuição de N*, o nível em que o Oracle deixa de ser otimista |
| `figures/fig4_gap_per_dataset.png` | folga `Oracle_1 − MV` por base |
| `figures/fig5_diversity_vs_gap.png` | redundância de erros × folga não explorada |
| `figures/fig6_curves_per_dataset.png` | uma curva Oracle_N por base (29 painéis) |

---

## Achado 1 — o melhor pool produz a pior votação

O GA tem o **maior Oracle_1 de todos** (0.9979) e a **menor votação majoritária de
todas** (0.7824). A folga `Oracle_1 − MV` é de **0.2154**, contra 0.1484 do Bagging
e 0.1421 do Random Forest — 52% maior que a do RF (Friedman p=0.0063; GA vs RF
Δrank 0.83 > CD, Wilcoxon p=0.00067).

Casos extremos, todos com `Oracle_1 = 1.0000` no GA:

| base | Oracle_1 (GA) | votação maj. (GA) | folga | DF/e² (GA) |
|---|---|---|---|---|
| P2 | 1.0000 | 0.5325 | **0.4675** | 1.02 |
| Vehicle | 1.0000 | 0.5959 | 0.4041 | 1.07 |
| Wine | 1.0000 | 0.6294 | 0.3706 | 1.29 |

Em P2 o pool contém um classificador correto para **toda** amostra de teste, os
erros são praticamente independentes (`DF/e² = 1.02`), e a votação majoritária
entrega 53%. Não é falta de informação no pool — é o fusor que não a extrai.

**Leitura para o subprojeto**: é a evidência empírica direta de que MVR é o fusor
errado para pools otimizados por diversidade, e a justificativa quantitativa para
avaliar DCS/DES (Fase 5). O parágrafo do resumo do subprojeto sobre "acertos
distribuídos de forma pouco aproveitável por métodos reais" tem aqui sua medida.

---

## Achado 2 — a folga é previsível pela redundância de erros (objetivo 8)

O Double Fault bruto não compara pools de força diferente: classificadores mais
fracos erram mais, logo erram juntos mais. O GA-Perceptron tem o **maior** DF bruto
(0.1564) e a **menor** acurácia individual (0.7052). Normalizando pelo valor
esperado sob erros independentes (`e²`, com `e = 1 − acurácia individual`):

| modo | acc. individual | DF | DF esperado (e²) | **DF/e²** |
|---|---|---|---|---|
| GA | 0.7052 | 0.1564 | 0.0869 | **2.11** |
| Random Forest | 0.7806 | 0.1069 | 0.0482 | 2.92 |
| Bagging | 0.7931 | 0.1094 | 0.0428 | 3.90 |

Com o índice normalizado o GA é o **mais diverso** (Friedman p=0.00009; GA e RF
empatados em rank, ambos significativamente à frente do Bagging). A conclusão se
inverte em relação ao número bruto — o DF absoluto sozinho leva ao diagnóstico
errado.

E o índice **prevê a folga**:

```
Spearman(DF/e², Oracle_1 − MV) = −0.860   p = 1.6e-26   n = 87 (29 bases × 3 modos)
```

Por modo: GA −0.923, Bagging −0.909, RF −0.785. Quanto mais independentes os erros,
maior a parcela do potencial do pool que a votação majoritária desperdiça.

**Recomendação prática (objetivo 8)**: `DF/e²` é calculável a partir da matriz de
acertos, sem treinar nenhum método de seleção. Um pool com `DF/e²` alto (Thyroid:
6.96–8.58, folga 0.035) tem pouca folga a recuperar — DCS/DES dificilmente compensa
o custo. Um pool com `DF/e²` próximo de 1 (P2: 1.02, folga 0.47) é candidato forte.
É exatamente a "análise prévia do potencial de desempenho" pedida no objetivo 8.

---

## Achado 3 — onde o Oracle deixa de ser otimista (objetivo 7)

O objetivo 7 pergunta se algum nível intermediário de Oracle é um limite superior
mais realista. A resposta direta é `N*`, o menor N com `Oracle_N < votação
majoritária`:

| modo | mediana | IQR | média | folds com N* > 50 |
|---|---|---|---|---|
| GA | 52 | [51, 54] | 51.6 | 79.0% |
| Bagging | 53 | [51, 56] | 54.4 | 81.4% |
| Random Forest | 53 | [51, 56] | 53.0 | 79.3% |

Friedman p=0.37 — **os três modos têm o mesmo N***, e ele fica em `N ≈ M/2`.

Não é coincidência empírica, é quase uma identidade. Em problema binário com M=100,
"a votação majoritária acerta" ⟺ "pelo menos 51 classificadores acertam", ou seja
`Oracle_51 ≡ MV` por construção. 22 das 29 bases são binárias.

**Consequência metodológica**: `Oracle_N` só carrega informação além do MVR em
`N ≪ M/2`. Acima disso a curva é uma reescrita da votação majoritária. Isso
*fundamenta* o recorte N=1..5 adotado em `docs/protocol.md` §8, que até aqui estava
apenas declarado.

**Mas N=1..5 é uma faixa saturada.** Os três modos ficam entre 0.988 e 0.998, e
`Oracle_5 ≥ MV` em **100% dos 870 folds**. As diferenças entre modos são
estatisticamente reais (Friedman p<0.001 em Oracle_1, Oracle_2 e Oracle_5) mas de
magnitude 0.002–0.004. Ou seja: nesta faixa, Oracle_N é **mais conservador que
Oracle_1, porém não o suficiente** para chegar perto de um método real.

Onde o poder discriminante realmente está:

- `Oracle_M` (unanimidade): GA 0.1136 vs Bagging 0.3052 — Δ=0.19, Wilcoxon p=0.00064.
- a folga `Oracle_1 − MV`: GA 0.2154 vs RF 0.1421.
- a **taxa de decaimento** da curva. Em N=75: GA 0.544, RF 0.622, Bagging 0.636.

Um detalhe visível na fig. 2: o GA lidera em N=1, mas o Random Forest o ultrapassa
a partir de **N=3**. A vantagem do GA existe só na ponta mais otimista da curva.

---

## Achado 4 — a combinação suave não ajuda pools de margem (objetivo 6)

Com as 290 linhas do GA preenchidas, a comparação pedida pelo objetivo 6 entre
votação majoritária e combinação suave fica completa:

| modo | votação maj. | combinação suave | Δ | Wilcoxon (N=29) | regra |
|---|---|---|---|---|---|
| GA (Perceptron) | 0.7824 | 0.7743 | **−0.0081** | **p=0.014** | média de margens normalizadas |
| Bagging | 0.8467 | 0.8466 | −0.0001 | p=1.00 | média de probabilidades |
| Random Forest | 0.8549 | 0.8546 | −0.0003 | p=0.59 | média de probabilidades |

Nos pools de árvores as duas regras empatam — a diferença é da ordem de 1e-4 e não
é significativa. No pool de Perceptrons a combinação suave é **significativamente
pior** que a votação majoritária: perde em 222 dos 290 folds, e chega a −0.067 na
Segmentation, −0.046 no Wine, −0.041 no Vehicle.

**Leitura**: a normalização por `||w||` do ADR 0016 remove o artefato de escala —
um classificador com pesos grandes não domina mais a média por acidente de
magnitude — mas não remove o artefato de confiança. A distância ao hiperplano
continua não sendo uma probabilidade: um Perceptron que separou seu bag com folga
larga produz distâncias grandes para *toda* amostra, inclusive as que ele erra, e
essa magnitude entra na média com o mesmo peso de uma competência real. A
probabilidade de uma folha de árvore, por pior que seja calibrada, ao menos é uma
frequência observada no treino.

Consequência para o subprojeto: para pools de classificadores lineares, "média das
probabilidades preditas" não é o método real de combinação a bater — o MVR é. É
mais um argumento para a seleção dinâmica, onde a competência é estimada dos dados
e não do escore do classificador.

---

## Pendências

**DCS/DES (objetivo 6, parte final).** Implementado no ADR 0017 — OLA, LCA,
KNORA-E, KNORA-U e META-DES via DESlib, com a partição de validação como DSEL, e a
métrica `recovered` = parcela da folga `Oracle_1 − MVR` que a seleção dinâmica
alcança. O run correspondente
(`results/experiments/2026-08-23T10-09-30_376203d/`) está em execução; esta seção
será substituída pelos resultados. Ele foi lançado antes do commit do ADR 0017,
então seu manifest traz `git_sha: 376203d` com `git_dirty: true` — o código que
ele executou é o do commit `25f4c14`.

**Viés de DSEL a favor do GA.** O DSEL da seleção dinâmica é a partição de
validação, que o GA já usa na função de fitness. Bagging e RF não a usam para nada,
então a comparação de `recovered` entre modos é otimista para o GA. Corrigir exige
dividir a validação em duas metades, o que muda os pools do GA — registrado no
ADR 0017 como candidato ao próximo run.

**Bases de imagens (objetivo 5).** O protocolo usa as 28 bases da tese de referência
mais Magic. O objetivo 5 menciona preferência por bases de imagens públicas; a
escolha atual privilegia comparabilidade direta com Monteiro et al. (2022).
