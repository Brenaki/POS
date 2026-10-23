# ADR 0022 — Segunda bateria: bases de imagem via embeddings de CNN

- **Status**: Accepted
- **Date**: 2026-10-23
- **Contexto**: objetivo específico 5 do subprojeto (preferência por bases de
  imagem públicas), pendência da Fase 8 do ROADMAP, ADR 0021 (portão por perfil,
  pré-requisito), ADR 0001 (pins do ambiente principal)

## Contexto

O objetivo 5 pede avaliar os níveis de Oracle "preferencialmente em bases de
imagens públicas e reconhecidas". As 29 bases tabulares foram escolhidas para
serem comparáveis com Monteiro et al. (2023), e o relatório registrava o desvio
como a pendência mais direta. O pipeline do POS é tabular (ARFF → pools → Oracle_N);
classificar pixels crus com árvores e Perceptrons não seria um experimento
sobre imagens, e sim sobre a falta de representação.

## Decisão

Cada imagem vira um vetor de atributos extraído por uma CNN pré-treinada e
congelada, e o vetor entra no pipeline existente como uma base tabular a mais,
sem mudar nada a jusante.

### Bases

| Base | Fonte (torchvision) | Classes | Papel |
|---|---|---|---|
| MNIST | `MNIST` | 10 | multiclasse, fácil |
| FashionMNIST | `FashionMNIST` | 10 | multiclasse, médio |
| CIFAR10 | `CIFAR10` | 10 | multiclasse, difícil (domínio natural) |
| CIFAR10CatDog | `CIFAR10`, classes 3 e 5 | 2 | binária: testa o sanduíche `Oracle_{M/2+1} ≤ MVR ≤ Oracle_{M/2}` em imagem |
| FashionTeeShirt | `FashionMNIST`, classes 0 e 6 | 2 | binária, o par mais confundido |

### Extração (`scripts/extract_image_embeddings.py`)

- **Backbone**: ResNet-18 do torchvision com pesos `IMAGENET1K_V1`, `fc`
  substituída por identidade, `eval()`. O vetor da penúltima camada tem 512
  dimensões. As imagens são redimensionadas para 224×224 e normalizadas com a
  média e o desvio do ImageNet; MNIST e Fashion têm o cinza replicado em 3 canais.
- **Amostra do experimento**: 2000 imagens estratificadas do split oficial de
  **teste** (seed 42); nas binárias, as 2000 imagens das duas classes.
- **PCA para 32 dimensões**, ajustado num conjunto **disjunto** de 2000 imagens
  do split oficial de **treino** (seed 43). Nenhum fold do 10-fold participa do
  ajuste da projeção, então não há vazamento. Com 512 dimensões, o custo do
  pipeline (medidas de complexidade, DES por vizinhança) explodiria: a base mais
  larga do catálogo tabular tem 34 atributos.
- **Proveniência**: `Dataset/<base>.json` traz os índices exatos das imagens das
  duas amostras, os pesos, as versões de torch/torchvision/Python, a variância
  explicada pelo PCA e o SHA-256 do ARFF. `scripts/build_pos_catalog.py` liga o
  JSON ao catálogo do ADR 0021.

### Ambiente isolado

A extração roda num venv separado, **fora do repositório**
(`~/.local/share/pos/venv-embed`, Python 3.12, torch CPU). Dentro do repo, ele
seria varrido pelo teste de tamanho de arquivo, e torch não convive com os pins
do ADR 0001 (numpy 1.24, sklearn 1.2). O ambiente principal só lê o ARFF. A
metade do código sem torch (amostragem, PCA, escrita do ARFF) fica em
`pos/images/embedding_dataset.py` e é testada no ambiente principal; o passo
torch é um script.

### Execução

O perfil é `pos-oracle-v2` sem nenhuma alteração: 10-fold, M=100, tratamentos
`pgdcs_a`, `pgdcs_d` e `bagging`, os seis métodos DES. Desde o ADR 0021 as bases
passam pelo portão por hash do catálogo e o run continua `comparable=true`.
A análise fica em `pos/analysis/v2_battery.py` e
`scripts/analyze_image_battery.py`, contra o run tabular canônico do mesmo perfil
(`2026-08-27T23-01-32_7c5ec6d`).

## Consequências

- O objetivo 5 deixa de ser desvio declarado. A bateria de imagens mede se os
  mecanismos do relatório (N* ≈ M/2 nas binárias, recuperação baixa da folga,
  `DF/e²` como preditor) valem fora das bases tabulares clássicas.
- **Limitações**: a CNN não é ajustada às bases (embeddings congelados do
  ImageNet), são 2000 imagens por base e não o conjunto inteiro, e o PCA
  descarta parte da variância (registrada por base). As conclusões valem para
  "imagens representadas por um extrator genérico", que é o uso comum de MCS
  sobre imagens.

## Resultados (2026-10-23)

Dois runs de árvore limpa, `comparable: true`, 0 erros, 150 folds no total:
`2026-10-23T01-02-10_8a9d9bd` (MNIST, FashionMNIST, FashionTeeShirt; 2h56 com
`--jobs 4`) e `2026-10-23T03-58-14_1e2c140` (CIFAR10, CIFAR10CatDog; 1h30). O
smoke projetava menos de 10 h, então a amostra de 2000 imagens e as 32
dimensões foram mantidas. Variância retida pelo PCA: MNIST 0.796,
FashionMNIST 0.732, FashionTeeShirt 0.737, CIFAR10 0.597, CIFAR10CatDog 0.590.

- Invariantes em 150/150 folds. Sanduíche binário em 60/60 (igualdade em 29).
- N* nas multiclasse: 0.33–0.37·M, abaixo da mediana tabular de 0.50·M (p=0.017).
- O melhor DES recupera 3.4% da folga (mediana), contra 10.4% nas tabulares
  (p=0.002).
- `DF/e²` transfere: a curva isotônica das 84 base×tratamento tabulares prevê
  a folga das 15 de imagem com MAE 0.026 (mediana tabular: 0.057).

Detalhes no Achado 7 de `docs/resultados-oracle-n.md` e na Seção do objetivo 5
do relatório.
