# ADR 0023 — Backend de complexidade validado contra o ECoL; R só para desenvolvimento

- **Status**: Accepted (§6 resolvido pelo ADR 0024)
- **Date**: 2026-10-23
- **Supersedes**: ADR 0003 (remoção do R via pyhard)
- **Related**: ADR 0010 (`fast_adapter`), 0011 (KD-tree), 0013 (T1 = N5),
  0014 (pendência "validar `T1_fast` contra ECoL/R"), 0019 (votação do PGDCS)

## Contexto

O ADR 0003 propunha trocar o R/ECoL pelo pyhard. Na prática o pipeline já roda
sobre o `fast_adapter` (ADR 0010/0011) desde o fitness do GA, e o pyhard ficou
inviável como padrão: 229 min de votação em Phoneme contra 2,6 min (Fase 9, B0),
e ele devolve `0.0` para F1v, N3 e T1. Faltava o que o 0003 e o 0014 pediam:
comparar o backend rápido com o ECoL de verdade. Isso dependia do R, instalado em
2026-10-23 (R 4.6.1, `cluster`, `e1071`, `MASS`, `FNN`, `igraph` 2.3.4).

## Decisão

### 1. Duas versões do ECoL, e a do legado é a 0.3.0

O `Cpx.py` chama `ecol.overlapping` e `ecol.neighborhood` e o `base.HEADER` lista
`T1` e `LSC` em neighborhood. Esses nomes só existem no **ECoL 0.3.0** (arquivo do
CRAN, 2019). No **0.4.4** (CRAN atual) o grupo virou `featurebased`, e T1 e LSC
viraram N5 e N6. As versões também discordam no próprio T1:

- 0.3.0: `mean(h)/n`, com `h` o tamanho de cada hiperesfera que sobra no guloso de
  adesão (≈ 1/#esferas);
- 0.4.4: `#esferas/n`, como no artigo de Lorena et al.

As duas usam raio recursivo (`r_i = d_i/2` se i e seu inimigo mais próximo são
mútuos, senão `d_i − r_j`) e distância de Gower = Manhattan nos atributos
normalizados pela amplitude.

### 2. Ponte por subprocesso, com o fonte fixado por hash

`pos/complexity/ecol_bridge.py` baixa o tarball da versão pedida, confere o
SHA-256 (0.3.0: `8e5837c1…`, 0.4.4: `4d5d0785…`), extrai em `~/.cache/pos/ecol`
e roda `scripts/r/ecol_measures.R` por `Rscript`, CSV de entrada e de saída. O
pacote ECoL não precisa estar instalado e o rpy2 não entra no caminho. Os nomes
da 0.4.4 são traduzidos para os da 0.3.0 (`canonical_names`).

### 3. O `fast_adapter` é o padrão; o R é só de desenvolvimento

`pos.complexity.complexity_data3` aponta para o `fast_adapter`. O R entra apenas
para regenerar os golden values e para os testes `requires_r`, que são pulados se
faltar `Rscript` ou os pacotes `cluster`/`e1071`/`MASS`. O pyhard continua
importável (`pyhard_adapter`, extra `[pyhard]`), mas não é padrão. Balance (C1, C2)
e dimensionality (T2, T3, T4) passaram a ser calculados (`simple_measures.py`,
exatos contra o ECoL) em vez de `0.0`.

### 4. Golden values e classificação das 22 medidas

`scripts/gen_ecol_golden.py` grava `tests/_ecol_legacy_golden.json` (ECoL 0.3.0,
split `wine` do `sample.ipynb`, com versão do R, hash do tarball e dos dados).
`tests/characterization/test_ecol_golden.py` exige que cada uma das 22 medidas
esteja numa de duas listas:

| Classe | Medidas |
|---|---|
| exatas no `fast_adapter` | F1, F1v, F2, F3, N2, N3, T2, T3, T4, C1, C2 |
| desvio conhecido, com porta exata | T1, N1, LSC, F4 |
| desvio conhecido, sem porta | N4 (o rápido calcula kDN), L1–L3 e Density/ClsCoef/Hubs (`0.0`) |

As portas exatas (`t1_ecol.py`, `ecol_exact.py`) reproduzem o R com tolerância
1e-12, inclusive em dados com empates, e escalam sem matriz densa (exceto N1):

- **T1**: inimigo mais próximo com desempate pelo menor índice, raio recursivo,
  pertinência estrita `d < r`, adesão gulosa (CELF, mesmo desempate do `which.max`);
- **N1**: MST sobre a matriz completa; fração de pontos com alguma aresta entre classes;
- **LSC**: `1 − Σ|LS(i)|/n²` (o rápido divide cada `|LS(i)|` pelo tamanho da classe);
- **F4**: as amplitudes das classes são recalculadas sobre o que sobra a cada rodada.

### 5. Validação em bags congelados

Protocolo: fold 0 de `StratifiedKFold(10, shuffle, 42)`, 20 bags por base com
`generate_bags(…, random_state=0)`, 35 bases (Ecoli pulada pelo mesmo motivo do
portão do ADR 0014). O que importa para o GA é a ordem dos bags dentro de uma
base, então a métrica principal é o τ de Kendall por base (média sobre as bases),
tanto no valor cru quanto na visão `dispersion_linear` que o fitness usa.

- `results/validation/ecol-r-2026-10-23_a898e80`: rápido contra o R, bags com até
  500 instâncias (o ECoL monta matrizes n×n).
- `results/validation/ecol-port-2026-10-23_a898e80`: rápido contra as portas
  exatas, bags do tamanho real do GA (`tam_bags = 0.5`, sem teto).

| Medida | τ cru | τ dispersão | MAE cru | argmin igual |
|---|---|---|---|---|
| F1 | 0.998 | 0.995 | 0.0009 | 97% |
| F1v | 0.987 | 0.976 | 0.0006 | 97% |
| F2 | 0.964 | 0.951 | <0.0001 | 94% |
| F3 | 1.000 | 0.993 | 0 | 100% |
| N2 | 0.990 | 0.982 | 0.0018 | 100% |
| N3 | 0.974 | 0.944 | 0.0013 | 100% |
| LSC | 0.941 | 0.880 | 0.066 | 91% |
| N1 | 0.616 | 0.390 | 0.126 | 51% |
| F4 | 0.446 | 0.339 | 0.124 | 49% |
| N4 | 0.178 | 0.024 | 0.208 | 34% |
| **T1** vs 0.3.0 | **−0.359** | 0.134 | 0.744 | 0% |
| T1 vs 0.4.4 | 0.368 | 0.148 | 0.174 | 43% |

(Referência ECoL 0.3.0, bags ≤ 500. Nos bags de tamanho real, contra as portas,
os números quase não mudam: T1 −0.348/0.121 contra a 0.3.0 e 0.358/0.128 contra a
0.4.4; N1 0.587/0.364; LSC 0.939/0.881; F4 0.466/0.375.)

**Achados.**

1. **`T1_fast` não equivale a nenhuma das versões.** Contra a 0.3.0, a que o
   legado usava, a ordem dos bags é invertida (τ −0.36, Pearson −0.58): uma é
   ≈ #esferas/n, a outra ≈ 1/#esferas. Contra a 0.4.4, a escala bate (Pearson 0.91
   entre bases), mas dentro de uma base a ordem concorda pouco (τ 0.37), por causa
   do raio `d/2` fixo. Na visão do fitness a concordância cai para ≈ 0.13.
2. As medidas exatas por construção batem. Os resíduos aparecem em poucas bases:
   em N2/N3 só nas que têm pontos duplicados (Blood, Mammo, Monk, Haberman), o
   que indica desempate de vizinhos; em F1/F1v/F2, em Segmentation, CTG,
   Ionosphere, WBC, Faults, Vertebral e Glass (não investigado; suspeita de
   atributos constantes dentro de uma classe).
3. LSC tem outra escala, mas preserva a ordem. N1 e F4 preservam a ordem só em
   parte. N4 é outra medida.
4. Custo: o grupo neighborhood do ECoL somou ≈ 470 s nos 700 bags de até 500
   instâncias; o `fast_adapter` somou 3,8 s nos mesmos bags.

### 6. Qual T1/N1/LSC/F4 o GA e a votação usam (resolvido no ADR 0024)

Os runs publicados no relatório usam as definições do `fast_adapter`. O GA-F1/T1
otimiza a dispersão de T1, e o PGDCS vota sobre as 11 medidas. Trocar pelas
portas exatas aproxima o perfil `monteiro-2023-dt-compatible-v1` do código que
Monteiro et al. rodaram, mas muda os resultados dos próximos runs em relação aos
já relatados. A troca só deve entrar como opção do perfil (uma chave nova muda
o `_fingerprint` só dos perfis que a usarem), e depende de decisão do autor.

**Resolução (ADR 0024):** o autor decidiu pela opção de perfil. O
`monteiro-2023-dt-compatible-v2` usa as portas exatas na votação e no GA; o v1
fica como estava. O run completo do v2 não aproxima os resultados do artigo.

## Consequências

- **+** O R deixa de ser dependência de execução. Em desenvolvimento ele entra
  por `Rscript` com fonte fixado por hash, sem rpy2 nem `R_HOME`.
- **+** As 22 medidas estão classificadas e testadas contra o ECoL; C1, C2, T2,
  T3 e T4 deixam de ser `0.0`.
- **+** A pendência do ADR 0014 está respondida com números.
- **−** O `T1_fast` em uso não é o T1 do ECoL. Os resultados já relatados com
  GA-F1/T1 devem ser lidos como "T1 aproximado (raio `d/2`, #esferas/n)".
- **−** Linearidade e network continuam `0.0` no backend padrão.
