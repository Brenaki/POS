# ADR 0024 — Perfil Monteiro v2: votação e GA com as medidas do ECoL 0.3.0

- **Status**: Accepted
- **Date**: 2026-10-23
- **Resolve**: ADR 0023 §6 (qual T1/N1/LSC/F4 o GA e a votação usam)
- **Related**: ADR 0019 (votação do PGDCS), 0020 (perfis Monteiro), 0021 (portão
  por perfil), 0023 (portas exatas validadas contra o R)

## Contexto

O ADR 0023 mostrou que quatro medidas do `fast_adapter` não são as do ECoL 0.3.0,
a versão que Monteiro et al. (2023) usaram (Kendall médio por base contra o R):
T1 −0,36, N1 0,62, F4 0,45, LSC 0,94. As portas exatas existem (`t1_ecol`,
`ecol_exact`), mas nenhum run as tinha usado. A hipótese a testar: parte da
distância até o artigo (MVR do PGDCS 6,2 pp abaixo da Tabela A.10, e só 7/27
acertos do vencedor neighborhood da Tabela 4) vem dessas definições.

## Decisão

1. **Backend por perfil.** `pos/complexity/backends.py` define `fast` (padrão) e
   `ecol-0.3.0`. Este troca T1, N1, LSC e F4 pelas portas exatas; as outras sete
   medidas já concordam com o R (ADR 0023). N4 fica kDN: o N4 do ECoL interpola
   pontos ao acaso e não tem porta exata.
2. **Perfil novo, v1 intacto.** `monteiro-2023-dt-compatible-v2` = v1 +
   `"complexity_backend": "ecol-0.3.0"`. O v1 e o `pos-oracle-v2` mantêm o
   fingerprint (teste fixa `f318d8ef…` e `3a3a6727…`), então os runs já
   publicados continuam reproduzíveis. O v2 usa o mesmo portão (Tabela 2), os
   mesmos splits e as mesmas seeds.
3. **Onde o backend entra.** Na votação Monteiro (10 × 100 subamostras) e no
   fitness do GA (`complexity_data3` em `fitness_evaluator`). Um teste espia as
   duas chamadas e só vê `ecol-0.3.0`.
4. **Custo aceito.** O N1 exato usa matriz de distâncias densa (4,6 s num bag de
   3800 pontos). Nas bases do perfil (a maior é Phoneme, bags de ~1350) isso
   cabe; não foi escrita uma MST esparsa. O run completo levou 2h57 (v1: 3h14,
   com outra carga na máquina).

## Resultado (run `2026-10-23T11-10-40_1741f5c`, contra `2026-08-27T19-47-09_7c5ec6d`)

Run completo: 27 bases × 20 replicações × 3 tratamentos, 1620 linhas, sem erro,
`comparable: true`, `git_dirty: false`. Análise em `scripts/compare_backend_runs.py`
(`backend_comparison.json`, `table4_agreement.csv` no diretório do run).

| | v1 (`fast`) | v2 (`ecol-0.3.0`) |
|---|---|---|
| Acertos do vencedor overlapping da Tabela 4 | 12/27 | 13/27 |
| Acertos do vencedor neighborhood da Tabela 4 | 7/27 | 8/27 |
| Par votado igual ao do v1 | — | 9/27 |
| MVR PGDCS_A − Tabela A.10 (média) | −6,19 pp | −6,39 pp |
| MVR PGDCS_D − Tabela A.10 (média) | −5,25 pp | −5,29 pp |
| MVR Bagging − Tabela A.10 (média) | −0,16 pp | −0,16 pp |
| Oracle PGDCS_A, MAE contra a Tabela 7 | 0,34 | 0,31 |
| PGDCS_A − Bagging (MVR, por base) | +0,39 pp, 22/27, p = 0,003 | +0,20 pp, 18/27, p = 0,058 |

v2 − v1 por base (MVR, Wilcoxon): PGDCS_A −0,20 pp (11 melhores, 12 piores,
p = 0,33); PGDCS_D −0,04 pp (p = 0,56); Bagging idêntico, como esperado.

Leitura:

- **A votação muda muito, a concordância com a Tabela 4 não.** 18 das 27 bases
  trocam de par, mas os acertos sobem só de 12 para 13 e de 7 para 8. Os pares
  discordantes se compensam (overlapping: 4 bases perdem o acerto, 5 ganham;
  neighborhood: 2 perdem, 3 ganham). Não é diferença detectável.
- **A distância até a Tabela A.10 não vem das definições das medidas.** Com as
  medidas do ECoL 0.3.0, o MVR do PGDCS continua ~6 pp abaixo do artigo.
- **A vantagem do PGDCS sobre o Bagging encolhe** de +0,39 pp (p = 0,003) para
  +0,20 pp (p = 0,058). A diferença v2 − v1 não é significativa; o sinal do
  efeito não muda, mas a evidência de que o PGDCS supera o Bagging fica
  dependente da definição das medidas.

## Consequências

- **+** A pendência do ADR 0023 §6 está fechada com um run completo, sem tocar
  nos runs publicados.
- **+** Fica descartada uma explicação para a distância até o artigo; as que
  restam são o próprio GA, o pool e o protocolo (ver `11-limitacoes.tex`).
- **−** O v1 continua sendo o perfil de referência do relatório; o v2 entra como
  análise de sensibilidade. Escolher um como "o" Monteiro exigiria a votação do
  artigo (Tabela 4), que nenhum dos dois reproduz.
- **−** N4 continua não sendo o do ECoL nos dois perfis.
