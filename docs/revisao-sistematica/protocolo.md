# Protocolo da revisão sistemática — PRISMA 2020

Atividade 1 do cronograma (revisão bibliográfica sobre MCS, DCS/DES, diversidade
e Oracle). Protocolo fixado em 2026-10-23, antes da triagem; os desvios estão
na última seção. Segue o checklist PRISMA 2020 (Page et al., 2021).

## 1. Perguntas de pesquisa

- **RQ1.** Como o Oracle é definido e usado como limite superior em sistemas de
  múltiplos classificadores (MCS) e na seleção dinâmica (DCS/DES)?
- **RQ2.** Que trabalhos propõem limites mais realistas que o Oracle, ou
  generalizações dele (k-de-M, níveis, cotas de votação)?
- **RQ3.** Como diversidade e redundância de acertos se relacionam com a folga
  entre o Oracle e os combinadores reais?
- **RQ4.** Que protocolos experimentais e bases são usados nesses estudos?

PICOC: *Population* — pools de classificadores (MCS, DCS, DES); *Intervention* —
Oracle ou outro limite superior, medidas de diversidade; *Comparison* —
combinadores reais (votação, DCS/DES); *Outcome* — acurácia e folga até o limite;
*Context* — classificação supervisionada.

## 2. Critérios de elegibilidade

Definidos pelo autor do subprojeto em 2026-10-23.

| | Critério |
|---|---|
| **Inclusão** | publicado entre 2021 e 2026 |
| | em inglês, português ou chinês |
| | trata do Oracle ou de limite superior em MCS/DCS/DES, **ou** mede empiricamente diversidade × desempenho do ensemble |
| **Exclusão** | preprints (sem revisão por pares); o revisado por pares entra |
| | "oracle" em outro sentido (blockchain, banco de dados, oráculo de teste de software, escrita oracular) |
| | sem texto acessível |
| | duplicatas |

Códigos usados na triagem: `I-a` estuda/generaliza o Oracle ou um limite
superior; `I-b` usa o Oracle como referência relatada de DCS/DES; `I-c` relaciona
diversidade e acurácia empiricamente; `M` método de DCS/DES ou geração de pool
cujo resumo não decide (vai para texto completo); `E1` fora do tema ou "oracle"
em outro sentido; `E2` só aplicação; `E3` não é classificação; `E4` não é
registro revisado por pares; `E5` informação insuficiente.

**Fora do corpus por construção.** Os trabalhos fundadores anteriores a 2021 —
Kuncheva & Whitaker (2003), Kuncheva & Rodríguez (2007), Souza & Cavalcanti
(2017), Cruz et al. (2017, 2018) — ficam fora do corpus sistemático. Eles aparecem
no relatório como referencial teórico, numa subseção marcada como não
sistemática. Monteiro et al. (2023) é semente do snowballing (F1), mas cai no
período e foi recuperado pela S4; por isso é julgado pelos mesmos critérios
que os demais.

## 3. Fontes e estratégia de busca

| Fonte | Acesso | Situação |
|---|---|---|
| Scopus | Periódicos CAPES (CAFe, EZproxy) | consultada |
| IEEE Xplore | aberto (endpoint REST da página de busca) | consultada |
| OpenAlex | API aberta | consultada |
| Web of Science | Periódicos CAPES | **planejada, não consultada**: toda busca devolve `Server.unexpectedError` pelo EZproxy (o websocket da busca não atravessa o proxy); tentado pela busca básica, avançada e pela lista de bases |
| ScienceDirect, SpringerLink | CAPES | cobertas pela indexação do Scopus; usadas para acesso ao texto completo |
| CNKI, Wanfang (chinês) | — | sem acesso; limitação de cobertura para literatura em chinês fora do Scopus/IEEE |

Strings (Scopus, campo `TITLE-ABS-KEY`; nas demais bases, o mesmo booleano sobre
título+resumo/metadados). As strings exatas estão em `queries/`; os filtros
comuns são `PUBYEAR > 2020 AND PUBYEAR < 2027` e idioma inglês/português/chinês.

- **S1 — Oracle e limites:** `("oracle" OR "upper bound" OR "theoretical limit" OR "performance bound" OR "accuracy bound") AND ("multiple classifier system*" OR "classifier ensemble*" OR "ensemble of classifiers" OR "pool of classifiers" OR "dynamic classifier selection" OR "dynamic ensemble selection" OR "dynamic selection" OR "ensemble selection" OR "ensemble pruning" OR "classifier fusion")`
- **S2 — diversidade × acurácia:** `("ensemble diversity" OR "classifier diversity" OR "diversity measure*" OR "double fault" OR "disagreement measure*") AND ("classifier ensemble*" OR "multiple classifier system*" OR "ensemble of classifiers" OR "pool of classifiers") AND ("accuracy" OR "performance")`
- **S3 — seleção dinâmica:** `"dynamic classifier selection" OR "dynamic ensemble selection" OR "dynamic selection of classifiers"`
- **S4 — geração de pool:** `("pool generation" OR "classifier pool" OR "pool of classifiers") AND ("diversity" OR "data complexity")`
- **F1, F2 — snowballing para frente (Scopus `REF`):** quem cita Monteiro et
  al. (2023) e quem cita Souza & Cavalcanti (2017).

Snowballing para trás: das referências dos incluídos, só as de 2021–2026, que
passam pelos mesmos critérios.

## 4. Coleta, deduplicação e triagem

- Os resultados de cada busca são gravados brutos em `raw/<base>_<busca>.json`
  com a string, a data e o total. Os scripts são `scopus_scrape.js` (lista de
  resultados do Scopus, pelo Playwright), `ieee_scrape.js` e `openalex_search.py`.
- `prisma_merge.py` gera `buscas.csv` (log de buscas), `registros.csv` (todos os
  registros identificados) e `unicos.csv` (deduplicado por DOI e, na falta dele,
  por título normalizado: a lista do Scopus não traz DOI).
- **Triagem 1, título e resumo:** decisão e código por registro, em
  `triagem.csv`. **Triagem 2, texto completo:** os `INCLUDE` e `MAYBE` são
  verificados no texto (acesso CAPES ou versão aberta), com a extração dos itens
  de dados da seção 5.
- **Revisores.** Um revisor só, assistido por IA. Os lotes de título/resumo
  foram triados por agentes com instruções e códigos idênticos, e as decisões
  foram conferidas pelo autor da revisão. Não há segundo revisor independente nem
  kappa: é uma limitação declarada. A recomendação é que o orientador revise uma
  amostra aleatória de exclusões.

## 5. Itens de dados extraídos

Por estudo incluído: o papel do Oracle (definido, usado como referência ou
generalizado), se propõe outro limite, o tipo de pool e de classificador-base, os
combinadores comparados, as medidas de diversidade, as bases e o protocolo
(holdout ou k-fold), e a folga Oracle × melhor método quando relatada.

## 6. Síntese

Narrativa por RQ, com tabela de estudos (não há metanálise: as métricas e os
protocolos são heterogêneos). Risco de viés: o item "melhor método escolhido no
teste" e o item "Oracle no mesmo conjunto de seleção" são anotados por estudo,
porque são as duas fontes de otimismo que este projeto mediu (ADR 0020 §11.7).

## 7. Desvios do protocolo

- **Strings ampliadas antes da coleta.** A S1 original, sem `performance
  bound`, `accuracy bound`, `pool of classifiers`, `dynamic selection`, `ensemble
  pruning` e `classifier fusion`, devolvia 21 registros no Scopus; a S2 original
  (`diversity measure*` AND `majority vot*` AND …) devolvia 3. Ampliadas para as
  versões acima, que são as registradas. A S3 e a S4 foram acrescentadas para
  cobrir os estudos de DCS/DES que tratam o Oracle só no texto completo.
- **Exportação do Scopus.** A exportação exige conta pessoal Elsevier. Os
  registros foram lidos da lista de resultados (200 por página, com os resumos
  abertos). A S3 (224 > 200) foi recuperada em duas partes por ano (2021–2023,
  2024–2026), que somam o total.
- **OpenAlex.** Não aceita curinga dentro de frase entre aspas. O `*` foi
  removido, já que o OpenAlex aplica stemming.
- **Web of Science** não consultado, pelo motivo técnico da seção 3.
