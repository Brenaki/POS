# Resultados — Oracle_N em diferentes níveis

- **Run**: `results/experiments/2026-08-23T14-19-59_2286fcc/` (ADR 0018; split de
  três vias, 12 métodos de seleção dinâmica, suíte de métricas por fusor)
- **Commit**: `2286fcc` (branch `master`)
- **Protocolo**: 29 bases × 10-fold estratificado × 3 modos, M=100, GA com 20 gerações,
  `random_state=42`. 870 folds, 0 erros, 4h08 de relógio. Ecoli e Glass excluídas pelo
  portão de viabilidade (menor classe com 2 e 9 instâncias, < 10 folds).
- **Reprodução**: `python scripts/run_experiment.py --full` →
  `python scripts/analyze_results.py results/experiments/<run>` →
  `python scripts/compare_runs.py <run_anterior> <run>`

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

| modo | Oracle_1 | Oracle_2 | Oracle_5 | Oracle_M | votação maj. | média probs | acc. individual | DF | DF/e² |
|---|---|---|---|---|---|---|---|---|---|
| GA (PGDCS, Perceptron) | **0.9976** | 0.9963 | 0.9905 | 0.1179 | 0.7784 | 0.7738 | 0.7052 | 0.1566 | **2.13** |
| Bagging | 0.9951 | 0.9935 | 0.9884 | **0.3052** | 0.8467 | 0.8466 | 0.7931 | 0.1094 | 3.98 |
| Random Forest | 0.9970 | **0.9958** | **0.9920** | 0.1725 | **0.8549** | 0.8546 | 0.7806 | 0.1069 | 2.92 |

Teste de Friedman com pós-teste de Nemenyi (N=29 bases, k=3, CD=0.615) e Wilcoxon
pareado — ver `analysis.txt` do run para o relatório completo.

A coluna `média probs` não usa a mesma regra nos três modos, e a tabela registra
qual foi usada (`soft_fusion_rule`): Bagging e RF usam a média das probabilidades
preditas; o GA usa a média das distâncias normalizadas ao hiperplano, porque o
Perceptron linear da tese não expõe `predict_proba` (ADR 0016). O que é comparável
entre as três colunas é o desempenho da combinação suave, não a escala do escore.

### Figuras

| arquivo | conteúdo |
|---|---|
| `figures/fig1_oracle_curve.png` | curva Oracle_1..M média, com a linha da votação majoritária de cada modo |
| `figures/fig2_oracle_curve_zoom.png` | recorte N=1..10 — a faixa de foco do subprojeto |
| `figures/fig3_nstar.png` | distribuição de N*, o nível em que o Oracle deixa de ser otimista |
| `figures/fig4_gap_per_dataset.png` | folga `Oracle_1 − MV` por base |
| `figures/fig5_diversity_vs_gap.png` | redundância de erros × folga não explorada |
| `figures/fig6_curves_per_dataset.png` | uma curva Oracle_N por base (29 painéis) |
| `figures/fig7_fuser_accuracy.png` | os 14 fusores por modo, contra Oracle_1 e a média individual |
| `figures/fig8_recovered_gap.png` | parcela da folga recuperada pela seleção dinâmica, e sua (não-)relação com `DF/e²` |

---

## Achado 0 — o viés de DSEL era real, mas é pequeno (ADR 0018)

O run anterior (`2026-08-23T10-09-30_376203d`) usava a partição de validação como
DSEL da seleção dinâmica. Essa é a **mesma partição que o GA usa na função de
fitness**, então o número do GA era otimista por construção: o DSEL continha
exatamente os pontos sobre os quais o pool tinha sido escolhido. Bagging e RF não
usam a validação para nada, logo não sofriam o viés.

O ADR 0018 partiu a validação ao meio — `X_val` para o fitness do GA, `X_dsel`
disjunto para a seleção dinâmica — mantendo `X_tr` intacto. Isolar o viés exige
separar dois efeitos que a mudança introduz ao mesmo tempo:

1. o **viés** que se queria remover (só afeta o GA);
2. o **DSEL pela metade**, que degrada a estimativa de competência local de todo
   mundo (afeta os três modos).

Bagging e RF servem de grupo de controle porque sofrem apenas o efeito (2). A
diferença-em-diferenças separa os dois.

**Pré-condição verificada.** Manter `X_tr` intacto tinha de deixar os pools de
árvore bit-idênticos. Verificado nos 870 folds pareados: para Bagging e RF o
`max |novo − antigo|` é **exatamente 0.000000** em `oracle_1..5`, `oracle_M`,
`majority_vote`, `mean_individual_acc`, `double_fault_mean` e `M`, e a curva
Oracle completa sai idêntica em **290/290 folds** de cada modo. Os pools do GA
mudaram nos 290/290, como se esperava — o GA passou a otimizar sobre metade dos
pontos de validação.

**Decomposição** (restrita aos 5 métodos presentes nos dois runs — OLA, LCA,
KNORA-E, KNORA-U, META-DES — porque um máximo sobre 10 métodos é maior que um
máximo sobre 5 por construção):

| | Δ folga recuperada | p (Wilcoxon, por base) |
|---|---|---|
| GA | −0.0003 | 0.468 |
| Bagging | −0.0082 | 0.080 |
| Random Forest | −0.0032 | 0.665 |
| **controle (Bagging+RF)** | **−0.0057** | |
| **viés de DSEL = GA − controle** | **+0.0055** | |

Leitura: o DSEL pela metade custou −0.0057 de folga recuperada aos modos que não
tinham viés nenhum. O GA perdeu quase nada (−0.0003). O viés estimado é
**+0.0055** em unidades de folga recuperada — contra uma recuperação do GA da
ordem de 0.20, isto é, cerca de 3% relativos, e **nenhum dos três deltas é
significativo**.

Conclusão honesta: **o desenho antigo não estava inflando materialmente a vantagem
do GA.** A preocupação que motivou o rerun era metodologicamente correta, e a
correção agora está feita e é a que vale daqui em diante, mas o efeito medido é
pequeno o bastante para não mudar nenhuma conclusão do run anterior. O
enfraquecimento do pool do GA também foi mínimo (`Δ Oracle_1 = −0.0003`,
`Δ MVR = −0.0043`, `Δ acurácia individual = 0.0000`).

Saída completa em `did_vs_previous.txt` do run.

---

## Achado 1 — o melhor pool produz a pior votação

O GA tem o **maior Oracle_1 de todos** (0.9976) e a **menor votação majoritária de
todas** (0.7784). A folga `Oracle_1 − MV` é de **0.2192**, contra 0.1484 do Bagging
e 0.1421 do Random Forest — 54% maior que a do RF (Friedman p=0.0045; GA vs RF
Δrank 0.86 > CD, Wilcoxon p=0.00030).

Casos extremos, todos com `Oracle_1 = 1.0000` no GA:

| base | Oracle_1 (GA) | votação maj. (GA) | folga | DF/e² (GA) |
|---|---|---|---|---|
| P2 | 1.0000 | 0.5420 | **0.4580** | 1.02 |
| Vehicle | 1.0000 | 0.5993 | 0.4007 | 1.08 |
| Wine | 1.0000 | 0.6127 | 0.3873 | 1.24 |
| Diabetes | 1.0000 | 0.6606 | 0.3394 | 1.17 |

Em P2 o pool contém um classificador correto para **toda** amostra de teste, os
erros são praticamente independentes (`DF/e² = 1.02`), e a votação majoritária
entrega 54%. Não é falta de informação no pool — é o fusor que não a extrai.

**Leitura para o subprojeto**: é a evidência empírica direta de que MVR é o fusor
errado para pools otimizados por diversidade, e a justificativa quantitativa para
avaliar DCS/DES. O parágrafo do resumo do subprojeto sobre "acertos distribuídos de
forma pouco aproveitável por métodos reais" tem aqui sua medida.

---

## Achado 2 — a folga é previsível pela redundância de erros (objetivo 8)

O Double Fault bruto não compara pools de força diferente: classificadores mais
fracos erram mais, logo erram juntos mais. O GA-Perceptron tem o **maior** DF bruto
(0.1566) e a **menor** acurácia individual (0.7052). Normalizando pelo valor
esperado sob erros independentes (`e²`, com `e = 1 − acurácia individual`):

| modo | acc. individual | DF | DF esperado (e²) | **DF/e²** |
|---|---|---|---|---|
| GA | 0.7052 | 0.1566 | 0.0869 | **2.13** |
| Random Forest | 0.7806 | 0.1069 | 0.0482 | 2.92 |
| Bagging | 0.7931 | 0.1094 | 0.0428 | 3.98 |

(a coluna `DF/e²` é a média das razões por fold, não a razão das médias)

Com o índice normalizado o GA é o **mais diverso** (Friedman p=0.00009; GA e RF
empatados em rank, ambos significativamente à frente do Bagging). A conclusão se
inverte em relação ao número bruto — o DF absoluto sozinho leva ao diagnóstico
errado.

E o índice **prevê a folga**:

```
Spearman(DF/e², Oracle_1 − MV) = −0.860   p = 8e-27   n = 87 (29 bases × 3 modos)
```

Por modo: GA −0.933, Bagging −0.909, RF −0.785. Quanto mais independentes os erros,
maior a parcela do potencial do pool que a votação majoritária desperdiça.

**Recomendação prática (objetivo 8)**: `DF/e²` é calculável a partir da matriz de
acertos, sem treinar nenhum método de seleção. Um pool com `DF/e²` alto (Thyroid:
6.96–8.58, folga 0.035) tem pouca folga a recuperar — DCS/DES dificilmente compensa
o custo. Um pool com `DF/e²` próximo de 1 (P2: 1.02, folga 0.46) é candidato forte.
É exatamente a "análise prévia do potencial de desempenho" pedida no objetivo 8.

---

## Achado 3 — onde o Oracle deixa de ser otimista (objetivo 7)

O objetivo 7 pergunta se algum nível intermediário de Oracle é um limite superior
mais realista. A resposta direta é `N*`, o menor N com `Oracle_N < votação
majoritária`:

| modo | mediana | IQR | média | folds com N* > 50 |
|---|---|---|---|---|
| GA | 52 | [51, 55] | 51.9 | 79.3% |
| Bagging | 53 | [51, 56] | 54.4 | 81.4% |
| Random Forest | 53 | [51, 56] | 53.0 | 79.3% |

Friedman p=0.40 — **os três modos têm o mesmo N***, e ele fica em `N ≈ M/2`.

Não é coincidência empírica. **Correção (ADR 0019):** versões anteriores deste
documento afirmavam a identidade `Oracle_51 ≡ MVR` em problema binário. Isso está
errado, e a versão certa é mais informativa. O que vale é um *sanduíche*:

```
Oracle_{M/2+1}  ≤  MVR  ≤  Oracle_{M/2}
```

A votação majoritária acerta sempre que mais da metade do pool acerta, logo
`MVR ≥ Oracle_51`; além disso ela só pode ganhar empates 50–50, logo nunca passa de
`Oracle_50`. Verificado: o sanduíche vale em **660/660** folds binários, e a
igualdade `MVR = Oracle_51` vale em apenas **508** deles (77%) — nos outros 23% os
empates são desfeitos a favor, e a diferença média é +0.0035 (máx +0.0968).

É justamente a diferença entre identidade e sanduíche que explica o valor de `N*`.
Como `N*` é o primeiro nível **estritamente abaixo** do MVR, e `Oracle_51 = MVR` não
é "<", os empates empurram `N*` de 51 para 52+ — que é exatamente onde as medianas
caem. 22 das 29 bases são binárias.

**Nas 7 multiclasse o limite não vale**, porque a votação é por pluralidade e um
classificador correto não precisa de 51 votos para vencer. Medido, `N*` é
significativamente menor lá:

| grupo | bases | GA | Bagging | RF |
|---|---|---|---|---|
| binárias | 22 | 53.4 | 53.7 | 53.5 |
| multiclasse | 7 | 44.4 | 48.1 | 47.3 |

Mediana 53.7 contra 47.6, Mann-Whitney **p=2.7e-06**. Dentro de cada grupo os modos
seguem empatados (Friedman p=0.71 nas binárias, p=0.37 nas multiclasse).

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

- `Oracle_M` (unanimidade): GA 0.1179 vs Bagging 0.3052 — Δ=0.19, Wilcoxon p=0.0011.
- a folga `Oracle_1 − MV`: GA 0.2192 vs RF 0.1421.
- a **taxa de decaimento** da curva. Em N=75: GA 0.526, RF 0.661, Bagging 0.679.

Um detalhe visível na fig. 2: o GA lidera em N=1, mas o Random Forest o ultrapassa
a partir de **N=3**. A vantagem do GA existe só na ponta mais otimista da curva.

**Resposta ao objetivo 7**: **nos problemas binários avaliados**, nenhum nível
intermediário de `Oracle_N` serve como limite superior realista. Ou o nível está em
`N ≪ M/2` e continua saturado perto de 1.0, ou está em `N ≈ M/2` e já *é* a votação
majoritária. Não há faixa intermediária útil. A restrição ao caso binário é
deliberada: o sanduíche que fecha o argumento depende de maioria absoluta, e sob
pluralidade ele não vale — com 7 bases multiclasse não há poder para afirmar o
mesmo lá, só para mostrar que `N*` desce. O que o objetivo 7 buscava — uma estimativa conservadora e alcançável — não é
`Oracle_N` para algum N, é a folga `Oracle_1 − MVR` ponderada pela fração dela que
métodos reais recuperam (Achado 5), e `N*` é o diagnóstico que mostra *por que*
nenhum N intermediário funciona.

---

## Achado 4 — a combinação suave não ajuda pools de margem (objetivo 6)

| modo | votação maj. | combinação suave | Δ | Wilcoxon (N=29) | regra |
|---|---|---|---|---|---|
| GA (Perceptron) | 0.7784 | 0.7738 | −0.0046 | p=0.51 | média de margens normalizadas |
| Bagging | 0.8467 | 0.8466 | −0.0001 | p=1.00 | média de probabilidades |
| Random Forest | 0.8549 | 0.8546 | −0.0003 | p=0.59 | média de probabilidades |

Nos pools de árvores as duas regras empatam — a diferença é da ordem de 1e-4, e nos
290 folds de cada modo elas dão o mesmo resultado em 278 (Bagging) e 275 (RF).

No pool de Perceptrons a combinação suave continua **pior em média** (−0.0046, perde
em 87 folds e ganha em 80), mas com o split de três vias essa diferença **deixou de
ser significativa** (p=0.51 contra p=0.014 no run anterior). O sinal se manteve, a
magnitude caiu pela metade e o teste não sustenta mais a afirmação forte. A leitura
que sobrevive é a negativa: para pools de classificadores lineares, a média de
margens **não supera** o MVR; ela não é o método real de combinação a bater.

A explicação mecanicista continua válida: a normalização por `||w||` do ADR 0016
remove o artefato de escala, mas não o de confiança. A distância ao hiperplano não é
uma probabilidade — um Perceptron que separou seu bag com folga larga produz
distâncias grandes para *toda* amostra, inclusive as que ele erra, e essa magnitude
entra na média com o mesmo peso de uma competência real.

---

## Achado 5 — a seleção dinâmica recupera menos de 25% da folga (objetivo 6)

O ADR 0018 levou a comparação de 5 para 12 métodos de seleção (10 dinâmicos + 2
estáticos), medidos nos mesmos 870 folds. DSEL = metade da validação, disjunta do
fitness do GA, k=7 vizinhos.

| fusor | GA | Bagging | RF |
|---|---|---|---|
| votação majoritária (MVR) | 0.7784 | 0.8467 | 0.8549 |
| combinação suave | 0.7738 | 0.8466 | 0.8546 |
| OLA | 0.7829 | 0.8028 | 0.7915 |
| LCA | 0.7271 | 0.7878 | 0.7863 |
| MCB | 0.7828 | 0.7943 | 0.7900 |
| Rank | 0.7760 | 0.8022 | 0.7908 |
| KNORA-E | 0.7897 | 0.8261 | 0.8274 |
| KNORA-U | **0.8056** | 0.8468 | **0.8564** |
| DES-P | 0.8021 | 0.8481 | 0.8566 |
| DES-KNN | 0.7995 | 0.8477 | 0.8492 |
| META-DES | — | **0.8482** | 0.8545 |
| KNOP | — | **0.8486** | 0.8556 |
| melhor individual (estático) | 0.7630 | 0.7999 | 0.7958 |
| seleção estática | 0.7797 | 0.8475 | 0.8546 |
| **melhor por fold (oracle sobre métodos)** | 0.8340 | 0.8688 | 0.8733 |
| Oracle_1 | 0.9976 | 0.9951 | 0.9970 |
| folga `Oracle_1 − MVR` | 0.2192 | 0.1484 | 0.1421 |
| **folga recuperada** | **24.04%** | **15.03%** | **14.24%** |

META-DES e KNOP não rodam sobre o pool do GA: exigem `predict_proba`, que o
Perceptron linear não tem. As 290 linhas correspondentes ficam vazias, com o motivo
registrado em cada manifest de fold.

`melhor por fold` é o máximo entre os **10 métodos dinâmicos** — é um *oracle sobre
métodos*, escolhido no teste, e portanto otimista. Como agora são 10 e não 5, esse
máximo é maior por construção que o do run anterior; a comparação entre runs é feita
só sobre os 5 métodos comuns (Achado 0), nunca sobre esta coluna.

Mesmo com 10 métodos e escolha a posteriori, **nenhum modo passa de 25%**: a folga
que o Oracle_1 anuncia continua, em três quartos ou mais, fora do alcance da seleção
dinâmica. As medianas são bem menores (GA 0.167, Bagging 0.108, RF 0.091) e no
primeiro quartil de Bagging e RF a recuperação é **exatamente zero**.

### Nenhum fusor bate o MVR nos pools de árvores

Wilcoxon pareado de cada fusor contra o MVR, por base (N=29), com correção de Holm
para a família de 13 testes:

**Random Forest** — nenhum ganho sobrevive:

| fusor | média | Δ vs MVR | vitórias/derrotas | p (Holm) |
|---|---|---|---|---|
| LCA | 0.7863 | −0.0686 | 0/29 | **<0.001** |
| MCB | 0.7900 | −0.0649 | 0/29 | **<0.001** |
| Rank | 0.7908 | −0.0641 | 0/29 | **<0.001** |
| OLA | 0.7915 | −0.0634 | 0/29 | **<0.001** |
| melhor individual | 0.7958 | −0.0592 | 1/28 | **<0.001** |
| KNORA-E | 0.8274 | −0.0275 | 3/24 | **0.0004** |
| DES-KNN | 0.8492 | −0.0057 | 10/18 | 0.097 |
| KNORA-U | 0.8564 | +0.0015 | 19/6 | 0.31 |
| DES-P | 0.8566 | +0.0017 | 13/9 | 0.84 |
| KNOP | 0.8556 | +0.0006 | 13/12 | 1.00 |
| META-DES | 0.8545 | −0.0005 | 13/15 | 1.00 |
| seleção estática | 0.8546 | −0.0003 | 14/12 | 1.00 |
| combinação suave | 0.8546 | −0.0003 | 1/2 | 1.00 |

**Bagging** — mesmo padrão: OLA/LCA/MCB/Rank perdem em 27–28 das 28 bases
(Δ entre −0.044 e −0.059, todos p_Holm ≤ 0.0001); KNOP (+0.0019) e DES-P (+0.0014)
são os melhores, e nem eles sobrevivem a Holm (p=0.16).

### E ajuda pools de margem

No GA a ordem se inverte parcialmente:

| fusor | média | Δ vs MVR | vitórias/derrotas | p (Holm) |
|---|---|---|---|---|
| KNORA-U | 0.8056 | **+0.0272** | 22/7 | **0.0045** |
| DES-P | 0.8021 | +0.0238 | 16/11 | 0.37 |
| DES-KNN | 0.7995 | +0.0211 | 13/16 | 1.00 |
| KNORA-E | 0.7897 | +0.0113 | 12/16 | 1.00 |
| OLA | 0.7829 | +0.0046 | 9/20 | 1.00 |
| MCB | 0.7828 | +0.0044 | 9/20 | 1.00 |
| seleção estática | 0.7797 | +0.0013 | 15/12 | 1.00 |
| Rank | 0.7760 | −0.0023 | 9/20 | 1.00 |
| combinação suave | 0.7738 | −0.0046 | 15/13 | 1.00 |
| melhor individual | 0.7630 | −0.0153 | 6/23 | **0.032** |
| LCA | 0.7271 | −0.0512 | 3/26 | **0.0081** |

**KNORA-U é o único fusor que bate o MVR de forma estatisticamente sustentada em
qualquer modo** (+0.0272 no GA, p_Holm=0.0045, vencendo em 147 dos 290 folds contra
49 derrotas e 94 empates). Nos pools de árvores o mesmo método fica em +0.0001 e
+0.0015, sem significância.

O eixo que organiza tudo é **quanto o método poda o pool**:

- podar até 1 classificador (OLA, LCA, MCB, Rank, melhor individual) → perda de
  0.044 a 0.069 nos pools de árvores, sempre significativa;
- podar parcialmente (KNORA-E) → perda intermediária, ainda significativa;
- não podar, só reponderar (KNORA-U, DES-P, META-DES, KNOP, seleção estática) →
  empate técnico com o MVR.

Os dois estáticos acrescentados pelo ADR 0018 confirmam o eixo pelos extremos: a
`seleção estática` (que mantém o pool inteiro, escolhido no DSEL) empata com o MVR
nos três modos, e o `melhor individual` (poda máxima, sem nenhuma localidade) é dos
piores em todos.

### Onde a seleção dinâmica paga

A vantagem do GA se concentra nas bases sintéticas 2D de fronteira não-linear. Em
P2, Lithuanian e Banana (d=2, 2 classes) a recuperação média do GA é **0.766**; nos
mesmos dados, Bagging recupera 0.157 e RF 0.155:

| base | GA | Bagging | RF |
|---|---|---|---|
| Lithuanian | **0.811** | 0.151 | 0.134 |
| Banana | **0.780** | 0.178 | 0.221 |
| P2 | **0.706** | 0.143 | 0.112 |
| Segmentation | **0.516** | 0.201 | 0.247 |
| Wine | 0.401 | 0.083 | 0.500 |
| ... | | | |
| Mammo | 0.063 | 0.172 | 0.157 |
| Blood | 0.055 | 0.122 | 0.153 |
| WDVG | 0.017 | 0.023 | 0.033 |
| Thyroid | −0.028 | 0.230 | 0.167 |

A seleção dinâmica paga quando o classificador-base é **fraco demais para a fronteira
global mas adequado localmente**. Um Perceptron não separa a espiral do P2, mas separa
qualquer vizinhança dela; escolher o Perceptron certo por região é exatamente o que
falta. Uma árvore já resolve a fronteira sozinha, então o voto majoritário já está
perto do que o pool consegue, e não sobra folga *local* para a seleção extrair — só
sobra a folga que o Oracle_1 mede, que é de outra natureza (Achado 3).

Com o split de três vias essa leitura deixa de ter a ressalva do run anterior: o
DSEL do GA já não é a partição do fitness, e o Achado 0 mostra que o viés removido
valia +0.0055.

### A folga recuperada não é prevista pela redundância

O Achado 2 mostrou que `DF/e²` prevê o **tamanho** da folga `Oracle_1 − MVR`. Ele não
prevê quanto dessa folga é alcançável: agregando por base, a correlação de Spearman
entre `recovered` e `df_ratio` não é significativa (pooled rho=+0.194, p=0.074, n=86;
por modo: GA −0.001 p=0.99, Bagging +0.309 p=0.11, RF +0.434 p=0.019 — e o único
sinal não sobrevive à correção para três testes). Por fold ela desaparece de vez
(rho=+0.019, p=0.59, n=829). São duas perguntas distintas — *quanta* folga existe, e
*que fração* dela um método real converte — e o índice de redundância só responde a
primeira. `fig8_recovered_gap.png` mostra as duas metades.

---

## Achado 6 — a leitura em acurácia balanceada não muda o ranking (objetivo 7)

A metodologia do subprojeto (item 7) pede precisão, revocação, F1 e acurácia
balanceada além da acurácia. O ADR 0018 passou a gravar as quatro para **todo** fusor,
mais `oracle_1_balanced`. A pergunta é se a acurácia bruta, que ignora a classe
minoritária, esconde algum ganho da seleção dinâmica — as bases do catálogo chegam a
IMB 71.5.

Não esconde. Em acurácia balanceada tudo cai (a minoria é mais difícil), mas cai de
forma quase uniforme:

| modo | Oracle_1 | Oracle_1 balanc. | MVR | MVR balanc. | folga | folga balanc. |
|---|---|---|---|---|---|---|
| GA | 0.9976 | 0.9967 | 0.7784 | 0.7368 | 0.2192 | 0.2599 |
| Bagging | 0.9951 | 0.9923 | 0.8467 | 0.8163 | 0.1484 | 0.1760 |
| Random Forest | 0.9970 | 0.9949 | 0.8549 | 0.8214 | 0.1421 | 0.1735 |

A folga **cresce** em acurácia balanceada nos três modos (+0.04 no GA, +0.03 nos
outros dois): o Oracle_1 é quase insensível ao desbalanceamento — basta um
classificador do pool acertar a instância minoritária — enquanto o MVR precisa que
51 acertem, e é aí que a minoria se perde. A ordenação entre modos é a mesma
(Friedman p=0.022, GA vs RF acima da CD).

Nos ranks de fusor a única mudança sistemática é o **DCS puro melhorar um pouco**:

| modo | OLA (acurácia) | OLA (balanceada) | LCA (acurácia) | LCA (balanceada) |
|---|---|---|---|---|
| GA | rank 3.84 | 3.52 | 5.62 | 5.28 |
| Bagging | rank 6.12 | 5.69 | 6.45 | 6.52 |
| Random Forest | rank 6.38 | 6.00 | 6.55 | 6.66 |

Faz sentido: escolher o classificador competente na vizinhança da instância é
justamente o que ajuda numa região minoritária, onde o voto global é dominado pela
classe majoritária. Mas o ganho é de 0.3–0.4 posições num ranking de 7, e em nenhum
modo chega perto de mudar o topo: **KNORA-U / META-DES / MVR / combinação suave
continuam empatados na frente, e OLA/LCA continuam no fundo**, com o mesmo Friedman
a p<1e-5.

**Conclusão do objetivo 7 pela via das métricas**: o ranking de fusores é robusto à
escolha da métrica. Nenhuma conclusão deste documento depende de a acurácia bruta
esconder a minoria.

---

## Recomendações consolidadas (objetivo 8)

Estas são as recomendações que o objetivo 8 pede — quando usar Oracle_N como
ferramenta de análise prévia, e quando o custo de DCS/DES se justifica.

**1. Use `Oracle_1 − MVR` como medida de potencial, não `Oracle_1` como meta.**
`Oracle_1` fica entre 0.995 e 0.998 nos três modos e em 29 bases: ele praticamente
não discrimina. A folga contra o MVR discrimina (0.142 a 0.219 entre modos, 0.017 a
0.458 entre bases) e é o que se pode tentar recuperar.

**2. Não procure um `Oracle_N` intermediário como limite realista.** Medido: `N*`
fica em `N ≈ M/2` nos três modos (mediana 52–53 de M=100, Friedman p=0.40), porque
em problema binário vale o sanduíche `Oracle_{M/2+1} ≤ MVR ≤ Oracle_{M/2}` (660/660
folds). Abaixo disso a curva está saturada acima de 0.98; acima disso ela é o MVR sob
outro nome. **Não existe faixa intermediária informativa nos problemas binários** —
nas 7 bases multiclasse `N*` cai para ~47 e a afirmação não foi testada com poder.
Reporte `Oracle_1`, a folga, e `N*` como diagnóstico.

**3. Calcule `DF/e²` antes de treinar qualquer método de seleção.** É `O(M² · n)`
sobre a matriz de acertos que já se tem, e prevê o tamanho da folga com Spearman
−0.86 (n=87). Regra operacional medida neste run:

| `DF/e²` | folga típica | decisão |
|---|---|---|
| ≥ 4 | < 0.05 | não vale DCS/DES — o custo não se paga |
| 2–4 | 0.05–0.15 | marginal; só KNORA-U/DES-P, e sem esperar ganho |
| ≈ 1 | > 0.30 | candidato forte, especialmente se o base for fraco |

Exemplos dos extremos no run: Thyroid (`DF/e²` 6.96–8.58, folga 0.035) contra P2
(`DF/e²` 1.02, folga 0.458).

**Os cortes foram validados fora da amostra** (ADR 0019), porque descobri-los e
avaliá-los nas mesmas 29 bases seria circular. Reajustando os dois limiares em 28
bases e prevendo a 29ª, 29 vezes: acerto **0.805** contra **0.736** da regra trivial
(responder sempre a faixa mais comum), e os cortes reajustados caem na mediana em
**1.14 / 4.82** — praticamente os mesmos ≈1 e ≥4 da tabela. A regra sobrevive, com a
ressalva de que a margem sobre o baseline é de 7 pontos, não de uma ordem de
grandeza: a faixa do meio domina o catálogo (64 dos 87 pares base×modo).

**Sobre o denominador.** `e²` com `e` médio só é o valor esperado sob independência
se todos os classificadores errarem na mesma taxa; o exato é
`2/(M(M−1)) · Σ_{i<j} e_i e_j`. Recalculado a partir das acurácias individuais já
gravadas em cada `fold_manifest`, o índice exato **não muda a conclusão**: Spearman
agrupado −0.8603 → −0.8607 (por modo: GA −0.9325, Bagging −0.9142, RF −0.7877). A
razão exato/médio tem mediana 1.0005, porque o desvio das acurácias individuais
dentro de um pool é pequeno (média 0.054). Vale a correção por rigor — e o índice
exato dá LODO 0.839 contra 0.805 —, não por mudar o resultado.

**4. `DF/e²` não prevê o que você vai *recuperar*.** Ele prevê a folga, não a fração
alcançável (rho=+0.19, p=0.074 por base; +0.02, p=0.59 por fold). Para prever
recuperação o preditor que funcionou foi qualitativo: **base de fronteira não-linear
em baixa dimensão + classificador-base fraco** (GA-Perceptron em P2/Lithuanian/Banana
recupera 0.77; nas mesmas bases o RF recupera 0.16).

**5. Se for usar seleção dinâmica, use um método que não pode o pool.** O eixo
poda→perda é o achado mais robusto do run: OLA, LCA, MCB, Rank e o melhor individual
perdem de 0.044 a 0.069 contra o MVR nos pools de árvores, em 27–29 das 29 bases,
com p_Holm < 0.001. KNORA-U é o único que bate o MVR de forma sustentada em algum
modo (+0.0272 no GA, p_Holm=0.0045) e nunca perde em nenhum.

**6. Em pool homogêneo forte (Bagging, RF), a decisão padrão é não usar DCS/DES.**
Nenhum dos 13 fusores testados supera a votação majoritária de forma significativa em
nenhum dos dois modos. O melhor caso é +0.0019 (KNOP em Bagging), que não sobrevive à
correção de Holm. O MVR é grátis; os outros não.

**7. O custo é real e foi medido.** Passar de 5 para 12 métodos de seleção levou o
run de 2h21 para 4h08 de relógio nas mesmas 29 bases × 10 folds × 3 modos — fator
**1.76x**, ou +1h47. Combinada com a recomendação 6, a leitura é: o custo marginal de
DCS/DES é da ordem do custo de gerar os pools, e num pool de árvores ele compra zero.

---

## Pendências

**Proveniência do run.** O run `2026-08-23T14-19-59_2286fcc` foi lançado de árvore
suja (`git_dirty: true`); o código que ele executou corresponde ao commit `2286fcc`
mais as mudanças do ADR 0018 já commitadas em seguida.

**Bases de imagens (objetivo 5).** O protocolo usa as 28 bases da tese de referência
mais Magic. O objetivo 5 menciona preferência por bases de imagens públicas; a
escolha atual privilegia comparabilidade direta com Monteiro et al. (2022).

**Validação de `T1_fast` contra ECoL/R.** Pendente de instalação do R (Fase 0).

**`GA-F1/T1` vs PGDCS completo.** `get_best_types()` segue desativado; decisão formal
pendente.
