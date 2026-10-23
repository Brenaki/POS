# Síntese da revisão sistemática (PRISMA 2020)

Protocolo em `protocolo.md`; decisões por registro em `triagem.csv`; estudos
incluídos e itens extraídos em `incluidos.csv`; contagens do fluxograma em
`fluxo_prisma.json`, verificadas por script.

## Fluxo

| Etapa | n |
|---|---|
| Identificados (Scopus 319 · IEEE Xplore 108 · OpenAlex 250) | 677 |
| Duplicatas removidas | 329 |
| Triados por título/resumo | 348 |
| Excluídos na triagem: E2 aplicação 148 · E1 fora do tema/"oracle" em outro sentido 42 · E3 não-classificação 31 · E4 não revisado por pares 10 · E5 sem informação 5 | 236 |
| Buscados em texto completo | 112 |
| Não recuperados (sem DOI/sem acesso; R0110 atrás de Cloudflare) | 10 |
| Avaliados em texto completo | 102 |
| Excluídos: F1 não usa o Oracle nem relaciona diversidade e acurácia 74 · F2 "oracle" em outro sentido 2 · F4 retratado/outro relato 2 | 78 |
| **Incluídos** | **24 relatos, 22 estudos** |

Por código: I-a (generaliza o Oracle) 1 · I-b (Oracle como referência
experimental) 10 · I-c (diversidade × acurácia) 13. Dois pares são o mesmo
estudo em dois relatos (R0126/R0128, R0146/R0065).

O Web of Science estava planejado e não foi consultado: a busca não atravessa o
EZproxy da CAPES. Snowballing para frente feito (F1, F2); para trás, não
executado nesta rodada (ver Limitações).

## RQ1 — Como o Oracle é usado

O Oracle aparece em 10 estudos como **referência experimental de teto**, quase
sempre como uma coluna a mais na tabela de resultados: R0146/R0065 (FH-DES), R0246
(MLS), R0277 (Monteiro et al.), R0309, R0321, R0340 (GNN-DES), R0118 e R0179.
Três padrões se repetem:

1. **O Oracle é reportado, mas não é analisado.** R0309 e R0321 o tiram dos
   testes estatísticos por ser "uma abstração"; R0146 o coloca como cota
   superior ao lado de MV, Single Best e GFMM como cotas inferiores. Nenhum
   estudo mede a distância Oracle → melhor método de forma sistemática. Só
   R0246 deixa a folga visível por base (Balance-Scale 0.910 contra 0.987; Car
   0.936 contra 0.998).
2. **O Oracle vira rótulo de meta-aprendizado.** R0340 e R0179 treinam o seletor
   para imitar a informação do Oracle no treino, que é a linha do META-DES (Cruz
   et al., 2017, fora do período).
3. **Uso indevido.** R0113 apresenta o Oracle como "meta-classificador estático"
   implantável e o reporta como o mais acurado. O Oracle consulta o rótulo
   verdadeiro, então isso é vazamento de rótulo. É o exemplo mais claro de por que
   a cota precisa ser lida com cuidado.

Com pools de 100 classificadores, o Oracle fica perto de 1 (R0277: ≈99% para
quase todo gerador de pool). Isso confirma, fora deste projeto, o que o objetivo
1 do relatório mediu: o Oracle_1 discrimina pouco entre pools.

## RQ2 — Limites mais realistas ou generalizações do Oracle

**Um estudo só.** R0001 propõe o *Hardness-aware Oracle* (HaO), que pondera o
acerto pela dificuldade da instância para avaliar a cooperação do ensemble
selecionado e escolher, a posteriori, entre métodos de DES (o R0119 é a
dissertação do mesmo trabalho). O R0120 deriva *oracle bounds* PAC-Bayes de
primeira e segunda ordem para combinação, mas para risco de justiça. Foi
excluído (F2), e mesmo assim a cota de segunda ordem (perda *tandem*) é parente
conceitual do *double fault*.

**Nenhum estudo** de 2021–2026 propõe níveis "pelo menos N de M corretos", uma
curva de Oracles, ou a ligação Oracle_{M/2+1} ≤ MVR ≤ Oracle_{M/2}. **O Oracle_N
deste projeto preenche uma lacuna documentada.** O HaO pondera pela dificuldade
da instância; o Oracle_N conta a redundância de acertos. São eixos diferentes e
complementares.

## RQ3 — Diversidade, redundância e a folga até o Oracle

Doze estudos (13 relatos) relacionam diversidade e acurácia. A maioria
**propõe uma medida nova**:

| Estudo | Medida |
|---|---|
| R0034 | Double Fault Disagreement |
| R0036 | diversidade de interpretabilidade |
| R0125 | distância de ranking por classe |
| R0127 | volume de simplex em embeddings aprendidos |
| R0131 | cobertura/similaridade |
| R0297 | vetor de probabilidades, em CNNs |
| R0126/R0128 | ambiguidade por entropia cruzada e perda hinge |

O resultado empírico é **fraco e dependente do contexto**:

- R0303 encontra correlação positiva fraca;
- em R0126 o sinal da correlação erro–diversidade muda com a taxa de amostragem;
- R0304 (27 bases, ensembles heterogêneos) mostra que o efeito depende do método
  de combinação.

**Nenhum estudo relaciona diversidade ou redundância com a folga Oracle → combinador
real.** Os estudos medem diversidade contra a acurácia do ensemble, nunca contra
o quanto do teto é recuperável. O índice `DF/e²` do relatório (Spearman −0.89 com
a folga, validado fora da amostra) e a medida de recuperação são contribuição
nova em relação a esse corpus. A crítica do R0297 às medidas construídas sobre a
saída 0/1 ("Oracle output") vale também para o Oracle_N, que é uma função da
mesma matriz de acertos.

## RQ4 — Protocolos e bases

- **Bases.** O padrão é 20–35 bases UCI/KEEL (R0001: 30; R0179: 24; R0277: 28;
  R0340: 35; R0309: 30; R0130: 24). Imagem só aparece via atributos profundos:
  CNNs em CIFAR (R0297) e ViT/CNN em texturas (R0118). Isso reforça a escolha do
  ADR 0022 (embeddings congelados + MCS).
- **Pools.** 100 Perceptrons é o pool de referência da linha DS (R0246, R0340,
  R0277), o mesmo do PGDCS.
- **Protocolo.** Holdout replicado (R0277: 20 replicações) ou k-fold. Testes de
  Wilcoxon, Friedman e Quade, com o Oracle quase sempre fora do teste.
- **Risco de viés.** Os dois vieses que este projeto mediu (ADR 0020 §11.7) não
  são discutidos em nenhum estudo incluído: escolher o melhor método no teste e
  medir o desempenho no mesmo conjunto que seleciona.

## Implicações para o subprojeto

1. A literatura recente usa o Oracle como **teto descritivo e não o analisa**. A
   curva Oracle_N e a medida de recuperação respondem a uma pergunta que ninguém
   quantificou: quanto do teto é atingível.
2. A única generalização publicada (HaO) tem eixo diferente: dificuldade da
   instância, não redundância. A seção de trabalhos relacionados de um artigo
   deve contrastar os dois.
3. Diversidade × acurácia continua inconclusiva quando medida contra a acurácia;
   medida contra a folga recuperável, este projeto encontrou sinal forte (`DF/e²`).
   É um ângulo novo.

## Limitações da revisão

- **Um revisor, assistido por IA.** A triagem foi feita por agentes com critérios
  idênticos e conferida pelo autor; não houve segundo revisor independente nem
  kappa. A recomendação é que o orientador revise uma amostra das 236 exclusões
  da triagem e das 74 F1.
- **Cobertura.** Web of Science não consultado; literatura em chinês só a
  indexada no Scopus, IEEE e OpenAlex (CNKI e Wanfang sem acesso); snowballing
  para trás não executado.
- **Elegibilidade baseada em trechos.** No texto completo, o uso do Oracle foi
  julgado pelos trechos ao redor de cada ocorrência de "oracle" e por contagens
  de termos, não por leitura integral. Seis artigos com muitas ocorrências e
  excluídos (R0052, R0063, R0138, R0145, R0148, R0151) foram rechecados com todas
  as ocorrências; a exclusão se manteve (o R0063 usa o "oracle local" do KNORA,
  não o Oracle como cota).
- 10 relatos não foram recuperados. Um deles (R0110) tem resumo que reporta a
  folga Oracle–DCS e seria I-b.
