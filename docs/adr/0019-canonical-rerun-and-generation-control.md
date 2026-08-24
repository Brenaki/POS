# ADR 0019 — Run canônico limpo, PGDCS completo e controle de geração

- **Status**: Accepted
- **Date**: 2026-08-24
- **Contexto**: revisão externa do relatório (`2c33634`), item aberto da Fase 6,
  ADR 0012 (reprodutibilidade), ADR 0015 (backend rápido), ADR 0018 (run vigente)

## Contexto

A revisão externa do relatório aprovou a resposta científica ao subprojeto e
levantou nove pontos a corrigir antes de o trabalho virar artigo. Dois deles não
são de redação: são de experimento, e por isso caem num ADR.

### 1. O run canônico foi lançado de árvore suja

`results/experiments/2026-08-23T14-19-59_2286fcc/run_manifest.json` traz:

```json
"git_sha": "2286fcc7860e20e3ea037e6aee04fb53cc99ea3f",
"git_dirty": true
```

O relatório afirma que os números vêm do commit `2286fcc`. Vêm de
`2286fcc` **mais modificações não commitadas** — que não existem mais em lugar
nenhum. Ninguém consegue reproduzir o que está publicado, e isso vale para o run
do ADR 0017 também (`376203d` sujo, código real `25f4c14`). É a terceira vez que
a mesma disciplina falha; o portão desta vez fica no procedimento (Consequências).

### 2. Geração e classificador-base estão confundidos

Hoje `ga` ⇒ Perceptron e `bagging`/`rf` ⇒ árvore. Toda leitura do tipo "o pool do
GA tem a maior folga `Oracle_1 − MVR`" é inseparável de "o Perceptron é o mais
fraco individualmente" (0.7052 contra 0.8467/0.8549). Sem um quarto ponto no
desenho, a ponte com a tese do coorientador — geração orientada por diversidade
produz especialistas que a votação desperdiça — não se sustenta: pode ser o
método de geração, pode ser só o base learner.

### 3. O que rodamos não é o PGDCS

`pos/oracle/experiment.py:40` e `pos/oracle/run_helpers.py:77` passam
`types=["F1", "T1"]`, o que desliga `get_best_types` — a etapa em que o PGDCS
**escolhe as medidas de complexidade adequadas a cada base**. O item ficou aberto
na Fase 6 ("decidir formalmente `GA-F1/T1` vs PGDCS completo") desde então.

### 4. A poda é inferida, não medida

A conclusão mais forte do run do ADR 0018 — o eixo que organiza os fusores é
quanto cada um poda o pool — foi lida da *natureza* dos algoritmos (OLA fica com
um, KNORA-U fica com todos os localmente corretos). Nunca medimos `E[S]/M`.

## Decisão

Um único run canônico, de árvore limpa, com **cinco modos**:

| modo | geração | base learner | isola |
|---|---|---|---|
| `pgdcs` | GA com seleção de medidas | Perceptron | o PGDCS completo |
| `ga` | GA com `F1`/`T1` fixos | Perceptron | o efeito de fixar as medidas (vs `pgdcs`) |
| `randbag` | bags aleatórios, sem busca | Perceptron | o efeito da busca do GA (vs `ga`) |
| `bagging` | bootstrap | árvore | o efeito do base learner (vs `randbag`) |
| `rf` | bootstrap + subespaço | árvore | o efeito do subespaço de atributos |

Os três contrastes que interessam ficam com **uma variável cada**. `ga`,
`bagging` e `rf` são mantidos exatamente como estão, o que preserva a
comparabilidade com tudo que já foi publicado e dá o critério de aceitação do
run (abaixo).

Roda com DCS/DES instrumentado: além da acurácia por fusor, grava-se a **fração
média do pool selecionada por consulta**.

### Custo

Pela regra que o ADR 0018 aprendeu errando por 6x — contar
`n_folds × n_modos × n_métodos`, nunca extrapolar por `n_test`:

| parcela | horas |
|---|---|
| os 3 modos atuais (medido, ADR 0018) | 4.1 |
| `randbag` + DCS/DES | ~0.5 |
| `pgdcs`: o GA em si | ~3 |
| `pgdcs`: votação de medidas | 10.7 |
| **total** | **~18–19** |

1450 folds contra os 870 do run anterior. O custo do PGDCS é dominado pela
votação, não pelo GA, e dentro dela por duas bases: Magic (5.5 h) e WDVG (2.1 h)
respondem por 71% das 10.7 h. O usuário optou explicitamente por pagar as horas
em vez de aceitar uma variante enfraquecida do método.

## O bloqueador que a decisão 3 expôs

Reativar `get_best_types` sem mais nada teria produzido um resultado sem sentido.
Dois defeitos, ambos medidos nesta sessão:

**Backend errado.** `pos/pool/complexity_voter.py` importa `complexity_data3` de
`pos.complexity`, que resolve para o adapter do **pyhard**. O pyhard não tem
mapeamento para `F1v`, `N3` e `T1`, e devolve `0.0` fixo para as três. Medida
constante tem desvio-padrão zero e nunca vence o `argmax` de `vote_complexity`:
**a votação, como estava ligada, jamais poderia escolher T1** — justamente uma
das duas medidas hardcoded hoje. Pior, o fitness do GA
(`pos/pool/fitness_evaluator.py`) já usa o outro backend, o `fast_adapter`, então
seleção e fitness discordavam numericamente sobre a mesma medida (em Wine, F3
vale 0.153 num e 0.469 no outro; F4, 0.032 contra 0.607).

O `fast_adapter` calcula as onze e é o que os ADRs 0010/0011/0013 validaram
contra o ECoL. O voter passa a usar ele. Um ponto de verdade por medida.

**Não reprodutível.** `complexities()` chamava `train_test_split` sem
`random_state`. A escolha de medidas variaria entre execuções idênticas — o que
sozinho já desqualificaria qualquer run canônico, e contraria o ADR 0012.

O custo da votação também depende disso: com pyhard, um fold de Phoneme leva
229 min e o run seria inviável; com o backend rápido, 2.6 min.

## Alternativas consideradas

- **Dois runs separados** (um limpo reproduzindo o desenho atual, outro depois
  com os modos novos). Custaria mais de 9 h a mais e deixaria dois runs
  candidatos a canônico ao mesmo tempo. Rejeitada.
- **Votar as medidas uma vez por base** em vez de por fold: 10.7 h → 1.1 h. A
  escolha de medidas passaria a enxergar dados de teste dos outros folds.
  Rejeitada: economiza tempo comprando um vazamento.
- **Só renomear para `PGDCS-F1/T1` no texto**, que era a sugestão da revisão.
  Resolve a honestidade da nomenclatura, mas deixa sem resposta se o atalho custa
  resultado. Rejeitada em favor de medir.
- **Reimplementar a seleção do DESlib** para contar os selecionados. Rejeitada:
  toda classe expõe `select(competences)`, então envolver esse método mede o que
  o método de fato fez, sem duplicar lógica que pode divergir.

## Consequências

**Critério de aceitação do run** — antes de qualquer análise, `compare_runs.py`
contra `2026-08-23T14-19-59_2286fcc`, restrito a `ga`/`bagging`/`rf`. Se as
colunas de pool saírem bit-idênticas, o run sujo fica **validado**: os números já
publicados sobrevivem e ganham uma procedência reproduzível. Qualquer divergência
é achado de primeira ordem e o relatório muda antes de tudo o mais.

**Procedimento, para não haver uma quarta vez**: árvore limpa é pré-condição de
lançar `--full`, e o manifesto com `git_dirty: true` deixa de ser aceitável como
fonte de número publicado.

Para o portão poder ser cumprido, `git_dirty` foi corrigido junto: ele marcava
sujo por *saída de experimento* não versionada (`results/**`, cujos `.npy`/`.npz`
são ignorados por design). Uma saída de run nunca é entrada de run, então não
pode comprometer procedência — mas um `.py` não versionado pode, e continua
contando. Um arquivo **versionado** sob `results/` que foi editado também
continua contando; a isenção é só para untracked. Sem essa correção o portão
seria impossível de satisfazer e voltaria a ser ignorado, que é como chegamos
aqui três vezes.

**Limite da instrumentação de poda**: o DESlib curto-circuita as consultas em que
a vizinhança é unânime, e essas nunca chegam ao seletor. O `E[S]/M` medido vale
sobre as consultas **roteadas para seleção dinâmica**, e a fração roteada é
gravada junto — sem esse denominador o número seria sobre um subconjunto não
declarado. Os estáticos não têm região de competência: `single_best` é `1/M` e
`static_sel` é o `pct_classifiers` (0.5), gravados como constantes.

**O que muda de significado**: `df_ratio` continua existindo, mas passa a conviver
com `df_ratio_exact`, que usa o denominador de independência correto quando os
classificadores têm taxas de erro diferentes.
