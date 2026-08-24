# ADR 0018 — Split de três vias e suíte de métricas por fusor

- **Status**: Accepted
- **Date**: 2026-08-23
- **Contexto**: objetivo 6 e 7 do subprojeto, marcos 6 e 7 do cronograma, ADR 0017

## Contexto

Fechar o marco 6 (ADR 0017, run `2026-08-23T10-09-30_376203d`) expôs três
problemas que o próprio run não podia resolver.

### 1. O marco 7 pede quatro métricas que não medimos

A metodologia 7 do SubProjeto é explícita:

> Os resultados serão avaliados por métricas como acurácia, **precisão,
> revocação, F1-score e acurácia balanceada**.

O run mede **só acurácia**. Isso não é formalidade: das 29 bases, várias são
fortemente desbalanceadas (Blood, Haberman, ILPD, Thyroid, Faults). Em Blood,
prever sempre a classe majoritária já rende ~0.76 de acurácia — a mesma ordem
de grandeza dos fusores que estamos comparando. Uma conclusão como "KNORA-U
empata com o MVR" não sobrevive necessariamente à acurácia balanceada, e é
exatamente onde a comparação é frágil.

### 2. `recovered` é otimista para o GA

O DSEL do DCS/DES é a partição de validação `X_val`. Bagging e RF não a usam
para nada, mas o GA usa: `voting_classifier(pool, X_val, y_val)` entra na
fitness (`pos/pool/fitness_evaluator.py:135`) e a diversidade é medida sobre
`y_val` (`:128`). A seleção dinâmica do GA estima competência em dados que o
GA já otimizou. A comparação de `recovered` entre modos está confundida.

O relatório hoje defende o Achado 5 com um argumento — o viés seria uniforme
entre bases, mas a vantagem do GA se concentra nas bases 2D de fronteira
não-linear. É um argumento razoável, não uma medida.

### 3. Não gravamos as predições do DCS/DES

Cada fold grava `predictions_<mode>.npz` com `preds` (M × n_test), `probs` e
`y_test` dos classificadores individuais — de onde qualquer métrica nova sai
offline. Do DCS/DES gravamos **apenas o escalar de acurácia**. Consequência
prática: acrescentar precisão/revocação/F1 ao DCS/DES custa um run inteiro.
É uma falha de projeto do ADR 0017, e é a que deve ser corrigida em caráter
permanente — as três acima cabem num run só justamente porque a terceira
obriga a reexecutar de qualquer forma.

## Decisão

### Split de três vias, mantendo `X_tr` intacto

```
X_train (90% da base, via StratifiedKFold)
  ├── X_tr    80%  treino dos pools          ← inalterado
  ├── X_val   10%  fitness do GA
  └── X_dsel  10%  DSEL do DCS/DES
```

A alternativa era tirar o DSEL de `X_tr`, preservando o tamanho de `X_val`.
Rejeitada: mudaria os pools dos **três** modos, e o relatório inteiro perderia
a base de comparação. Com `X_tr` inalterado, **os pools de Bagging e RF saem
bit-idênticos ao run anterior** — os Achados 1, 2 e 3 nesses modos não precisam
de reverificação, e a mudança fica isolada no que se quer testar.

O preço é um confundidor: metade do sinal de fitness enfraquece o pool do GA,
então uma queda no `recovered` do GA pode ser viés removido *ou* GA
enfraquecido. E o DSEL menor piora o DCS/DES de todos os modos. Os dois são
mensuráveis, e o desenho vira **diferença-em-diferenças**:

| quantidade | como se mede |
|---|---|
| efeito do DSEL menor | Δ`recovered` em Bagging/RF (sem viés de fitness) |
| enfraquecimento do GA | Δ`oracle_1`, Δ`mean_individual_acc`, Δ`majority_vote` do GA |
| **viés de DSEL** | Δ`recovered`(GA) − os dois acima |

O viés passa a ser um número, não um argumento.

### Suíte de métricas para todo fusor

Precisão, revocação e F1 em **macro** (média não ponderada sobre classes — a
que expõe o custo em classe minoritária, que é o ponto), mais acurácia
balanceada, para: MVR, fusão suave, cada método de seleção dinâmica, o melhor
classificador individual e a média dos individuais.

`zero_division=0`: em bases desbalanceadas um fusor pode nunca prever uma
classe, e a precisão dessa classe é indefinida. Contar como 0 é a leitura
honesta — o fusor não acertou nenhuma daquela classe — e é o que mantém a
macro comparável entre métodos.

Também se calcula **`oracle_1_balanced`**: a revocação macro do evento "pelo
menos um classificador acertou". Sai da matriz de acertos, sem predição de
rótulo, e responde se o próprio teto do Oracle é inflado pela classe
majoritária — uma pergunta que o run atual não consegue responder.

### Gravar as predições do DCS/DES

`predictions_<mode>.npz` passa a incluir um array por método (`des_ola`,
`des_lca`, …). Qualquer métrica futura sobre seleção dinâmica sai offline.

### Métodos: critério antes da lista

O ADR 0017 mediu cinco métodos e produziu um achado sobre o **eixo da poda**:
quanto mais um método descarta classificadores, pior ele fica. A lista cresce
só com métodos que testam esse eixo ou o ligam à trilha de diversidade.

Benchmark em Phoneme (n=5404, M=100, split de três vias), pool de árvores:

| método | família | custo | por que entra |
|---|---|---|---|
| OLA | DCS | 0.06s | canônico; comparabilidade com o ADR 0017 |
| LCA | DCS | 0.06s | canônico; comparabilidade |
| MCB | DCS | 0.13s | a penalidade do DCS é geral, ou é de OLA/LCA? |
| Rank | DCS | 0.06s | terceira forma de medir competência local |
| KNORA-E | DES | 0.09s | canônico |
| KNORA-U | DES | 0.05s | canônico; o vencedor do run anterior |
| DESP | DES | 0.09s | poda mínima (só descarta o pior que o acaso) |
| DESKNN | DES | **11.3s** | seleção por **diversidade** — liga ao `DF/e²` |
| META-DES | DES | 0.18s | meta-aprendizado |
| KNOP | DES | 0.06s | ordenação por comportamento de saída |
| SingleBest | estático | 0.04s | o extremo do eixo: M → 1 |
| StaticSelection | estático | 0.09s | poda estática, sem competência local |

MLA foi medido e descartado: devolveu acurácia idêntica à do LCA nos dois tipos
de pool, então não acrescenta ponto ao eixo. APriori e APosteriori foram
descartados por serem DCS que exigem `predict_proba` — cobririam só dois dos
três modos, e MCB e Rank já respondem a mesma pergunta nos três.

**Custo do DESKNN.** É o único item de orçamento: 20.9 ms por amostra de teste,
contra menos de 0.4 ms de todos os outros somados. `n_jobs` não ajuda (11.3s com
1, 3 ou todos os núcleos) — o gargalo é o cálculo de diversidade par-a-par entre
os 100 classificadores, serial dentro do DESlib. Somando as 51.783 amostras de
teste das 29 bases nos 3 modos, dá **+0.9 h de CPU**, ~0.3 h de relógio com
`jobs=3`. Aceito: é o único método que mede diversidade explicitamente, que é a
trilha do Achado 2.

**Custo medido (pós-run).** A estimativa acima errou por cerca de 6x. O run do
ADR 0018 (`2026-08-23T14-19-59_2286fcc`) levou **4h08** de relógio contra **2h21**
do run do ADR 0017 (`2026-08-23T10-09-30_376203d`), nas mesmas 29 bases × 10 folds
× 3 modos e no mesmo hardware com `jobs=3` — fator **1.76x**, ou **+1h47** em vez
dos +0.3 h previstos. O erro foi de método: o custo por amostra de teste foi medido
num fold só e extrapolado linearmente em `n_test`, mas as razões por base ficaram
entre 1.27x e 3.56x sem crescer com o tamanho da base (Magic, a maior, deu 2.04x).
O sobrecusto é **por fold e por método**, não por amostra — construir e ajustar
sete estimadores DESlib extras sobre um pool de 100 classificadores domina o
tempo de inferência. Fica registrado para a próxima estimativa: contar
`n_folds × n_modos × n_métodos`, não `n_test`.

**Cinco métodos não rodam sobre o pool do GA** por exigirem `predict_proba`, que
o Perceptron linear não tem: META-DES (já sabido do ADR 0017), KNOP, APriori,
APosteriori e StackedClassifier. Os quatro primeiros ficam com a coluna vazia no
modo `ga` e o motivo gravado no manifest do fold; StackedClassifier não entra.

### Estatística: conjunto primário e conjunto descritivo

Com 12 métodos mais MVR e fusão suave, a diferença crítica de Nemenyi cresce a
ponto de nada ser significativo — o teste perde poder por número de colunas, não
por falta de efeito. Então:

- **Primário** (Friedman + Nemenyi): MVR, fusão suave, OLA, LCA, KNORA-E,
  KNORA-U, META-DES. É exatamente o conjunto do ADR 0017, o que mantém o
  resultado anterior comparável linha a linha.
- **Descritivo**: os demais entram em tabela, com Wilcoxon pareado contra o MVR
  e correção de Holm para a família de comparações.

## Consequências

- **Um run novo**, que substitui `2026-08-23T10-09-30_376203d` como referência
  do relatório. O anterior é preservado: é ele que dá o braço "DSEL enviesado"
  da diferença-em-diferenças.
- **O portão de viabilidade do GA não muda.** `min_instances_per_class_in_bag`
  encadeia k-fold → validação → bag, e o bag sai de `X_tr`, que é o que o ADR
  mantém intacto. As mesmas 29 bases passam, Ecoli e Glass continuam fora pelo
  mesmo motivo. O que é novo é o DSEL, que não alimenta bag nenhum.
- **DSEL magro nas bases pequenas.** Wine (n=178) fica com ~16 instâncias de
  DSEL para k=7 vizinhos. O `k` efetivo por fold e o tamanho do DSEL passam a
  ser gravados no manifest, para que isso apareça na análise em vez de virar
  ruído silencioso.
- **O relatório ganha um eixo.** Toda comparação passa a ter versão em
  acurácia e em acurácia balanceada; onde as duas discordarem, a discordância
  é o resultado.
- `recovered` continua definido sobre acurácia, para não quebrar a comparação
  com o run anterior. A versão balanceada entra como coluna separada.
