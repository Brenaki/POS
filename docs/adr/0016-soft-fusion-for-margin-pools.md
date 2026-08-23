# ADR 0016 — Combinação suave para pools sem `predict_proba`

- **Status**: Accepted
- **Date**: 2026-08-23
- **Contexto**: ADR 0015 (Perceptron como base do GA), objetivo 6 do subprojeto

## Contexto

O objetivo 6 do subprojeto pede comparar o Oracle_N com "métodos reais de
combinação, como votação majoritária e **média das probabilidades preditas**".

O ADR 0015 adotou o Perceptron linear como base do GA, por fidelidade à tese de
referência (Monteiro et al. 2022, sec. 5) e ao objetivo 4 do subprojeto. O
Perceptron do scikit-learn **não expõe `predict_proba`** — só `decision_function`.
Consequência medida no run `2026-08-22T23-33-56_2a8a0f5`: as 290 linhas do modo
`ga` em `summary.csv` saíram com `mean_probs` vazio, deixando o objetivo 6 sem
cobertura justamente no pool que mais interessa.

O ADR 0015 registrou três saídas possíveis, nenhuma adotada:

1. usar `tree` como base do GA — abandona o Perceptron da tese;
2. calibrar o Perceptron (`CalibratedClassifierCV`) — muda o modelo avaliado,
   introduz um split interno extra e deixa de ser o classificador da tese;
3. definir formalmente uma "média de `decision_function`".

## Decisão

Adotada a opção 3, com normalização.

A média das margens brutas é inválida como regra de fusão: `decision_function`
do Perceptron é `w·x + b`, sem escala fixa, então um classificador que
convergiu com pesos grandes domina a média por acidente de magnitude, não por
competência. Dividindo por `||w||`, a margem vira a **distância euclidiana
sinalizada** da amostra ao hiperplano daquele classificador — uma grandeza
geométrica comparável entre membros do pool:

```
d_i(x) = (w_i·x + b_i) / ||w_i||
fusão  = argmax_c  Σ_i d_i(x)[c]
```

Para o caso binário, `decision_function` retorna um vetor e a orientação segue a
convenção do scikit-learn (`> 0` → `classes_[1]`), montada como duas colunas
`[-d, +d]` antes do `argmax`.

Implementação em `pos/oracle/comparison.py`:

- `mean_decision_accuracy(pool, X, y)` — a regra acima.
- `soft_fusion_accuracy(pool, X, y) -> (acc, rule)` — tenta `predict_proba`
  primeiro, cai para a margem normalizada, e devolve `(None, "none")` se o pool
  não suportar nenhuma das duas.

`evaluate_pool` passa a usar `soft_fusion_accuracy`. A coluna `mean_probs`
continua existindo com o mesmo nome, e uma coluna nova **`soft_fusion_rule`**
registra qual regra produziu cada valor (`mean_probs`, `mean_decision_norm` ou
`none`), no `summary.csv` e no `fold_manifest_<mode>.json`.

## Consequências

**Positivas**

- Objetivo 6 passa a ter valor nas 290 linhas do GA, sem trocar o classificador
  da tese.
- A grandeza comparada entre modos é a **acurácia do ensemble sob fusão suave**,
  que é comparável mesmo quando a regra interna difere — é o desempenho do
  método de combinação que interessa, não a escala do escore.
- `soft_fusion_rule` deixa a diferença explícita na própria tabela, em vez de
  escondida numa nota de rodapé.

**Negativas / limitações**

- A regra não é a mesma nos três modos. `rf` e `bagging` usam média de
  probabilidades calibradas por frequência de folhas; `ga` usa média de
  distâncias ao hiperplano. Qualquer comparação direta entre as colunas
  `mean_probs` dos três modos precisa mencionar isso — daí a coluna de regra.
- A distância ao hiperplano não é uma probabilidade: não está em [0,1] e não
  soma 1. Não deve ser reportada como "probabilidade média", e sim como
  "combinação suave".
- Invariante verificado por teste: escalar `coef_` e `intercept_` de um membro
  por uma constante não altera o resultado da fusão
  (`test_normalisation_makes_the_fusion_scale_invariant`).

## Reexecução

Os pools não são persistidos (só predições e matriz de acertos), então preencher
as 290 linhas exige reconstruir os pools. Como `random_state=42` é propagado, o
GA é determinístico e o run novo reproduz os mesmos pools — as colunas Oracle_N,
`majority_vote` e `double_fault_mean` do modo `ga` devem sair idênticas às do run
`2026-08-22T23-33-56_2a8a0f5`. Isso é usado como verificação: se divergirem, algo
além do `mean_probs` mudou.
