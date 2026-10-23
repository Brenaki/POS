# ADR 0021 — Portão de conformance por perfil

- **Status**: Accepted
- **Date**: 2026-10-23
- **Contexto**: ADR 0020 §8.2 (portão), §11.11 (P2 no `pos-oracle-v2`), ponto em
  aberto do ROADMAP Fase 9; pré-requisito da bateria de imagens (ADR 0022)

## Contexto

O portão de conformance do ADR 0020 compara cada snapshot com a Tabela 2 de
Monteiro et al. (2023) e pula o que não bate, a menos que o run receba
`--allow-unverified-data` — flag que marca o run **inteiro** `comparable=false`.

Isso está certo para `monteiro-2023-dt-compatible-v1`, cujos números são
confrontados com o artigo. Está errado para `pos-oracle-v2`, que não replica
ninguém: seu escopo de referência é ele mesmo. Duas consequências concretas:

1. P2 (snapshot de 2000 instâncias contra 5000 geradas no artigo) só entrou no
   run canônico `2026-08-27T23-01-32_7c5ec6d` por `--resume` com a flag, e o run
   todo ficou `comparable=false` por causa de uma base.
2. Nenhuma base nova — em particular as bases de imagem do objetivo 5 (ADR 0022)
   — poderia entrar no `pos-oracle-v2` sem derrubar a comparabilidade do run.

## Decisão

Cada perfil declara a sua **referência de dados**, e o portão consulta essa
referência (`pos/oracle/dataset_gate.py`):

| Perfil | Referência | Passa quando |
|---|---|---|
| `monteiro-2023-dt-compatible-v1` | `monteiro-2023-table-2` | veredito da Tabela 2 após reconciliação é `conformant` ou `approved_with_note` (comportamento do ADR 0020, inalterado) |
| `pos-oracle-v2` | `pos-catalog` | o SHA-256 do snapshot bate com `Dataset/pos_catalog.json` |
| qualquer outro | `monteiro-2023-table-2` | fallback para o portão mais estrito |

- `Dataset/pos_catalog.json` é o registro versionado dos snapshots do POS:
  arquivo, SHA-256, instâncias, atributos, classes e, para as bases de imagem, o
  JSON de proveniência ao lado. É gerado por `scripts/build_pos_catalog.py` e
  **commitado**: trocar um snapshot sem regenerar o catálogo faz o portão barrá-lo
  (`mismatch` de hash), que é o objetivo.
- O mapeamento perfil → referência fica no código, **não** como campo novo do
  dicionário do perfil. Um campo novo mudaria o `_fingerprint` dos perfis
  existentes e invalidaria a retomada dos runs já gravados.
- O manifesto ganha, por base, `gate_reference`, `gate_status`, `gate_reason` e
  `gate_passes`, ao lado dos campos de conformance da Tabela 2, que continuam
  sendo gravados para todo perfil (são informação, não decisão).
- `--allow-unverified-data` continua existindo e continua derrubando
  `comparable`. Deixa de ser necessário para P2 e para bases registradas no
  catálogo do POS.
- A comparação publicada (`comparator.py`, `comparison_report.py`,
  `scripts/compare_reference.py`) continua presa à Tabela 2 — é a comparação com
  o artigo, e P2 continua fora dela.

## Verificação

- Os 28 hashes do catálogo batem com os gravados no manifesto do run canônico
  `2026-08-27T23-01-32_7c5ec6d` (0 divergências): o catálogo descreve exatamente
  os dados que produziram os números publicados no relatório.
- Testes: `tests/unit/oracle/test_dataset_gate.py` (referência por perfil, hash
  igual/diferente/ausente, catálogo real coerente com os snapshots, P2 passa no
  `pos-oracle-v2` sem a flag) e `TestPerProfileGate` em
  `test_run_recorder_v2.py` (P2 roda no `pos-oracle-v2`, `comparable` intacto,
  sem `skipped_datasets`; no perfil Monteiro continua pulado).

## Consequências

- O run `2026-08-27T23-01-32_7c5ec6d` mantém `comparable=false` no manifesto —
  não reescrevemos manifestos antigos. Sob este ADR, um rerun do mesmo perfil
  com as 28 bases sairia `comparable=true`.
- Toda base nova do POS passa a exigir uma linha no catálogo; isso é deliberado:
  é o registro de procedência que o ADR 0009 pede.
