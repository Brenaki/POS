# Relatório de resultados

PDF que responde, um a um, aos oito objetivos específicos do subprojeto, a partir do
run `2026-08-23T14-19-59_2286fcc`.

```
make          # gera relatorio.pdf via tectonic
make figuras  # recopia as figuras do run de referência
make clean
```

## Organização

| arquivo | conteúdo |
|---|---|
| `relatorio.tex` | documento mestre: só título e a ordem dos `\input` |
| `preambulo.tex` | pacotes, geometria e macros (`\On{N}`, `\dfe`, `\mvr`) |
| `secoes/00-resumo.tex` | resumo |
| `secoes/01-escopo.tex` | protocolo experimental e definições |
| `secoes/02-obj1-conceito.tex` | objetivo 1 — o Oracle como limite superior |
| `secoes/03-obj2-oracle-tradicional.tex` | objetivo 2 — implementação e verificações |
| `secoes/04-obj3-generalizacao.tex` | objetivo 3 — a curva `Oracle_N` |
| `secoes/05-obj4-pools.tex` | objetivo 4 — os três modos de pool e a diversidade |
| `secoes/06-obj5-bases.tex` | objetivo 5 — as 29 bases (e a ressalva das bases de imagens) |
| `secoes/07-obj6-metodos-reais.tex` | objetivo 6 — os 14 fusores, o eixo da poda, o viés de DSEL |
| `secoes/08-obj7-niveis-intermediarios.tex` | objetivo 7 — `N*`, e por que não há nível intermediário útil |
| `secoes/09-obj8-recomendacoes.tex` | objetivo 8 — as sete recomendações |
| `secoes/10-resultados-esperados.tex` | confronto com os resultados esperados no projeto |
| `secoes/11-limitacoes.tex` | limitações |
| `secoes/12-referencias.tex` | referências |
| `figuras/` | cópias das figuras geradas por `scripts/analyze_results.py` |

Para acrescentar uma seção: crie o arquivo em `secoes/` e insira um `\input` na
posição desejada em `relatorio.tex`.

Os números vêm de `analysis.txt`, `analysis.json` e `did_vs_previous.txt` do run, e a
análise textual completa está em `docs/resultados-oracle-n.md`.
