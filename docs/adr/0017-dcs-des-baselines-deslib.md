# ADR 0017 — Seleção dinâmica (DCS/DES) como método real de combinação

- **Status**: Accepted
- **Date**: 2026-08-23
- **Contexto**: objetivo 6 do subprojeto, marco 6 do cronograma, ADR 0016

## Contexto

O objetivo 6 do subprojeto pede comparar o Oracle_N com "métodos reais de
combinação". Até aqui existiam dois: votação majoritária (`majority_vote`) e
combinação suave (`mean_probs`, ADR 0016). Falta a família de **seleção
dinâmica**, que é justamente a que o primeiro run indicou como necessária:

```
run 2026-08-22T23-33-56_2a8a0f5, modo ga:
  Oracle_1 = 0.9979    votação majoritária = 0.7824    folga = 0.2154
```

A folga não é falta de informação no pool — é o fusor estático que não a
extrai. Sem medir DCS/DES, o relatório afirma que "existe folga" mas não diz
**quanto dela é alcançável**, que é o que decide se a folga é uma promessa ou
um artefato da métrica.

## Decisão

Cinco métodos, via DESlib, avaliados em todo fold de todo modo:

| método | família | referência |
|---|---|---|
| OLA | DCS | Woods et al. (1997) |
| LCA | DCS | Woods et al. (1997) |
| KNORA-E | DES | Ko et al. (2008) |
| KNORA-U | DES | Ko et al. (2008) |
| META-DES | DES (meta-aprendizado) | Cruz et al. (2015) |

São os dois pares canônicos de DCS e DES mais a referência de
meta-aprendizado; cobrem as três formas de decidir competência (acurácia
local global, acurácia local por classe, e concordância do vizinho) sem
inflar o custo do run.

Implementação em `pos/oracle/des_comparison.py`:

- `evaluate_des(pool, X_dsel, y_dsel, X_test, y_test)` → `(acurácias, notas)`.
  **Nunca levanta exceção**: um método inaplicável devolve `None` e a razão
  fica registrada em `notas`, para não perder a comparação inteira por causa
  de um método.
- `best_des(accs)` → o melhor método daquele fold.
- Colunas novas em `summary.csv`: `des_ola`, `des_lca`, `des_knorae`,
  `des_knorau`, `des_metades`. O `fold_manifest_<mode>.json` guarda também as
  notas.

**META-DES exige `predict_proba`** e por isso fica vazio nos pools de
Perceptron do GA — a mesma limitação que o ADR 0016 tratou para a fusão suave.
Aqui ela não foi contornada: a distância normalizada ao hiperplano serve como
escore de fusão, mas os meta-atributos do META-DES são probabilidades
posteriores, e substituí-las por distâncias mudaria o método, não o adaptaria.

### DSEL = a partição de validação

A região de competência é estimada sobre `X_val`, o split estratificado que
`fold_splitter.stratified_val_split` já separa do treino do pool. Nenhum dado
de teste entra na estimativa, e nenhum classificador do pool viu esses dados
no treino — é o arranjo padrão do DESlib (Cruz et al. 2018) e não exige
nenhum split novo.

### Métrica derivada: folga recuperada

```
recovered = (melhor DCS/DES − votação majoritária) / (Oracle_1 − votação majoritária)
```

0 = a seleção dinâmica não supera o MVR; 1 = ela atinge o Oracle. É a
tradução direta da pergunta do objetivo 8 ("análise prévia do potencial") em
um número comparável entre bases de dificuldade diferente.

## Compatibilidade: DESlib 0.3.5 vs NumPy 1.24

DESlib 0.3.5 (último release, 2021) ainda chama `np.float` e `np.int`, aliases
removidos no NumPy 1.24. Com numpy 1.24.1 o efeito é traiçoeiro: OLA e LCA
funcionam, e KNORA-E, KNORA-U e META-DES morrem com

```
AttributeError: module 'numpy' has no attribute 'float'
```

`pos/oracle/deslib_compat.py` reinstala os dois aliases antes do import. Eles
eram referências aos builtins do Python, então restaurá-los reproduz
exatamente o comportamento contra o qual o DESlib foi escrito — nenhum
resultado numérico muda. Fica isolado num módulo só, com teste de regressão
(`test_knora_survives_the_removed_numpy_aliases`), para que o dia em que o
DESlib for atualizado a correção seja uma deleção.

## Consequências

**Positivas**

- Objetivo 6 completo: MVR, combinação suave e DCS/DES na mesma tabela.
- `recovered` transforma a folga medida no run 1 em uma quantidade acionável.
- Custo desprezível: medido em escala Adult (DSEL 8.9k, teste 4.9k, M=100),
  os cinco métodos somam 0,5–3 s por fold, contra minutos de construção do pool.
- O `Oracle` do próprio DESlib passa a validar o nosso `Oracle_1` por um
  caminho independente (`test_deslib_oracle_agrees_with_our_oracle_1`).

**Negativas / limitações**

- **`X_val` não é neutro entre os modos.** O GA usa `X_val` na função de
  fitness (`voting_classifier(pool, X_val, y_val)` e a diversidade sobre
  `y_val`), então para o modo `ga` o DSEL é um conjunto que já influenciou
  *quais* bags entraram no pool — os classificadores não foram treinados nele,
  mas a composição do pool foi escolhida contra ele. Bagging e RF não usam
  `X_val` para nada. A comparação de `recovered` entre modos é portanto
  **otimista para o GA**. Corrigir exige dividir a validação em duas metades
  (fitness e DSEL), o que muda os pools do GA e quebra a comparabilidade com
  os runs 1 e 2; fica registrado como candidato para o próximo run, não
  aplicado agora.
- `best_des` é o máximo por fold, logo é otimista como estimativa de *um*
  método. Só é usado para responder "quanto da folga é alcançável", nunca
  para eleger o melhor método — para isso valem os testes pareados por fusor.
- META-DES fica ausente do modo `ga`, então a comparação de fusores tem k=6
  no GA e k=7 nos outros. Os testes pareados são feitos por modo, nunca
  cruzando modos com conjuntos de fusores diferentes.
