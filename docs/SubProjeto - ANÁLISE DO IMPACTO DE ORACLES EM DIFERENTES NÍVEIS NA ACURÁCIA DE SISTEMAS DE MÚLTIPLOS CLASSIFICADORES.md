**ANÁLISE DO IMPACTO DE ORACLES EM DIFERENTES NÍVEIS NA ACURÁCIA DE SISTEMAS DE MÚLTIPLOS CLASSIFICADORES**

1. **PALAVRAS-CHAVE**

Oracle; Sistemas de Múltiplos Classificadores, Seleção Dinâmica de Classificadores, Limite Superior, Ensemble Learning, Acurácia.

2. # **RESUMO**

Sistemas de Múltiplos Classificadores (*Multiple Classifier Systems* – MCS) combinam modelos distintos para aprimorar a generalização em tarefas reais de classificação, explorando a complementaridade entre seus acertos e erros. Nesse contexto, o Oracle é utilizado como modelo abstrato de referência para estimar o desempenho potencial de um conjunto de classificadores, considerando uma amostra corretamente classificada quando ao menos um classificador do *pool* prediz adequadamente sua classe. Embora seja frequentemente interpretado como limite superior de desempenho, o Oracle tradicional pode produzir uma estimativa excessivamente otimista, pois métodos reais de seleção dinâmica ou combinação nem sempre conseguem identificar o classificador correto para cada instância, especialmente quando o acerto ocorre de forma isolada. Assim, esta pesquisa propõe investigar uma generalização do Oracle tradicional por meio de Oracles em diferentes níveis de exigência. Nessa abordagem, uma amostra será considerada correta apenas quando pelo menos N classificadores acertarem simultaneamente sua classe, permitindo construir uma curva Oracle\_N com limites progressivamente mais conservadores. A pesquisa envolverá a implementação reprodutível dessas medidas, a geração da matriz de acertos dos classificadores, a avaliação experimental em bases de classificação e a comparação com métodos reais de combinação, como votação majoritária e estratégias de seleção dinâmica. Espera-se que a análise forneça uma caracterização mais informativa da relação entre diversidade, redundância de acertos e desempenho do *ensemble*. Com isso, o estudo poderá contribuir para estimar de forma mais realista o potencial de desempenho de *pools* de classificadores e subsidiar decisões sobre o uso de métodos mais complexos em Sistemas de Múltiplos Classificadores.

3. # **CARACTERIZAÇÃO E JUSTIFICATIVA**

O uso de Sistemas de Múltiplos Classificadores, ou *Multiple Classifier Systems* (MCS), tem sido amplamente investigado como estratégia para melhorar o desempenho em tarefas de reconhecimento de padrões e aprendizagem de máquina. Essa abordagem parte do princípio de que classificadores distintos podem apresentar comportamentos complementares, de modo que os erros de um modelo possam ser compensados pelos acertos de outro. Nesse contexto, destacam-se os métodos de Seleção Dinâmica (*Dynamic Selection* — DS), nos quais classificadores-base são selecionados sob demanda para cada nova amostra, considerando a competência estimada em regiões locais do espaço de características (Cruz; Sabourin; Cavalcanti, 2018).

Um conceito central nessa área é o Oracle, modelo abstrato que, para cada instância de teste, seleciona um classificador correto caso exista ao menos um no conjunto. Por representar uma seleção ideal, o Oracle é frequentemente utilizado como limite superior teórico para avaliar métodos de fusão, Seleção Dinâmica de Classificadores (DCS) e Seleção Dinâmica de Ensembles (DES) (Souza; Cavalcanti, 2017). Entretanto, estudos indicam que esse limite pode ser excessivamente otimista: mesmo quando há um classificador correto no pool, métodos reais podem não conseguir identificá-lo a partir de informações locais. Souza e Cavalcanti (2017) mostram que, em alguns cenários, a acurácia do Oracle pode superar técnicas DCS em cerca de 20 pontos percentuais, e que, mesmo com Oracle de 100% no treinamento, técnicas DCS selecionam classificadores competentes para, no máximo, aproximadamente 85% das instâncias em média.

O problema focalizado neste subprojeto é, portanto, a distância entre o limite superior estimado pelo Oracle tradicional e o desempenho efetivamente alcançável por métodos práticos de combinação ou seleção dinâmica. O Oracle convencional considera uma amostra corretamente coberta quando apenas um classificador acerta, sem distinguir situações de acerto isolado daquelas em que há redundância ou consenso entre vários classificadores. Essa limitação dificulta a interpretação do potencial real de um pool, especialmente em cenários nos quais a diversidade existe, mas os acertos estão distribuídos de forma pouco aproveitável por métodos reais.

A originalidade da proposta está em estender o conceito de Oracle para múltiplos níveis de exigência. Em vez de considerar apenas o caso tradicional, no qual basta que pelo menos um classificador acerte a amostra, serão analisados níveis em que pelo menos **N** classificadores devem acertá-la simultaneamente. Formalmente, para um pool com **M** classificadores, define-se **Oracle\_N(x) \= 1** quando ao menos **N** classificadores predizem corretamente a classe da amostra **x**, e **Oracle\_N(x) \= 0** caso contrário. Assim, **Oracle\_1** corresponde ao Oracle tradicional, enquanto **Oracle\_2, Oracle\_3, ..., Oracle\_M** representam limites progressivamente mais conservadores, obedecendo à relação **Oracle\_1 ≥ Oracle\_2 ≥ ... ≥ Oracle\_M**.

A relevância científica dessa análise está em permitir uma caracterização mais detalhada da relação entre diversidade, redundância de acertos e desempenho do ensemble. Embora a diversidade seja reconhecida como aspecto importante em MCS, sua mensuração ainda não possui definição formal universalmente aceita (Kuncheva; Whitaker, 2003). A curva **Oracle\_N** pode contribuir para essa discussão ao indicar não apenas se existe algum classificador correto, mas quantos classificadores acertam simultaneamente cada amostra. Dessa forma, níveis intermediários de Oracle podem fornecer estimativas mais realistas do desempenho potencial de métodos práticos, como votação majoritária e seleção dinâmica.

Espera-se que o subprojeto contribua para o avanço do conhecimento ao propor uma medida complementar ao Oracle tradicional, capaz de avaliar de forma mais informativa o limite superior de desempenho em MCS. Além disso, a implementação reprodutível das medidas **Oracle\_1, Oracle\_2, ..., Oracle\_M** permitirá comparar diferentes pools, bases de dados e estratégias de combinação, auxiliando na identificação de cenários em que métodos complexos de seleção dinâmica são promissores ou, ao contrário, pouco justificáveis devido à baixa redundância de acertos.

4. # **OBJETIVOS**

Investigar sistematicamente o impacto de Oracles em diferentes níveis na estimativa da acurácia máxima de Sistemas de Múltiplos Classificadores, considerando a relação entre diversidade dos classificadores, redundância de acertos, métodos reais de combinação e a distância entre o limite superior teórico e o desempenho efetivamente alcançável.

# **OBJETIVOS ESPECÍFICOS**

1. Estudar o conceito de Oracle em Sistemas de Múltiplos Classificadores, com ênfase em sua utilização como medida de limite superior em técnicas de Seleção Dinâmica de Classificadores e Seleção Dinâmica de Ensembles.

2. Implementar a medida tradicional de Oracle, na qual uma amostra é considerada corretamente classificada quando pelo menos um classificador do conjunto prediz sua classe corretamente.

3. Propor e implementar uma generalização do Oracle em diferentes níveis, considerando cenários em que pelo menos N classificadores devem acertar simultaneamente a mesma amostra.

4. Construir pools de classificadores utilizando algoritmos simples e amplamente conhecidos, como Random Forest e Perceptron, bem como variações de treinamento, hiperparâmetros ou subconjuntos de dados, a fim de induzir diversidade entre os modelos.

5. Avaliar experimentalmente o comportamento dos diferentes níveis de Oracle em bases de dados de classificação, preferencialmente bases de imagens públicas e reconhecidas, provenientes de repositórios como Kaggle, OpenML, UCI Machine Learning Repository ou bases equivalentes.

6. Comparar a acurácia estimada pelos diferentes níveis de Oracle com o desempenho obtido por classificadores individuais e por métodos reais de combinação, como votação majoritária e média de probabilidades.

7. Analisar se níveis intermediários de Oracle podem representar estimativas de limite superior mais conservadoras e mais próximas do desempenho obtido por métodos reais de seleção ou combinação de classificadores.

8. Gerar recomendações sobre o uso de Oracles em diferentes níveis como ferramenta de análise prévia do potencial de desempenho de ensembles, considerando sua possível contribuição para a redução de custo computacional em experimentos com múltiplos classificadores.

   5. # **METODOLOGIA E ESTRATÉGIA DE AÇÃO**

1. **Revisão bibliográfica sobre Sistemas de Múltiplos Classificadores e Oracle:** Inicialmente, será realizada revisão bibliográfica sobre Sistemas de Múltiplos Classificadores, diversidade, métodos de combinação, Seleção Dinâmica e o conceito de Oracle. Essa etapa permitirá compreender o Oracle como limite superior teórico e fundamentar a proposta de Oracles em diferentes níveis.

2. **Seleção de bases de dados de classificação:** Serão selecionadas bases públicas de classificação, preferencialmente relacionadas a imagens ou problemas clássicos de reconhecimento de padrões, provenientes de repositórios como Kaggle, OpenML e UCI Machine Learning Repository. Serão priorizadas bases com disponibilidade pública, classes bem definidas, número adequado de amostras e viabilidade experimental.

3. **Construção do pool de classificadores:** Para cada base selecionada, será construído um pool de classificadores. Inicialmente, serão utilizados modelos simples e interpretáveis, como Random Forest e Perceptron. A diversidade do conjunto poderá ser ampliada por meio de variações de hiperparâmetros, subconjuntos de dados, subconjuntos de atributos ou estratégias de amostragem.

4. **Treinamento, validação e geração da matriz de acertos:** Os classificadores serão treinados e avaliados por protocolos adequados, como validação cruzada ou divisão em treino, validação e teste. A partir das predições, será gerada uma matriz de acertos, registrando 1 para predições corretas e 0 para incorretas. Essa matriz servirá de base para o cálculo do Oracle tradicional e dos Oracles em diferentes níveis.

5. **Implementação e cálculo dos Oracles em diferentes níveis:** O Oracle tradicional será calculado considerando correta toda amostra classificada adequadamente por pelo menos um classificador. Em seguida, será implementado o Oracle\_N, no qual uma amostra será considerada correta apenas quando pelo menos N classificadores acertarem simultaneamente sua classe. Para um pool com M classificadores, serão avaliados os níveis N \= 1, 2, 3, ..., M.

6. **Comparação com métodos reais de classificação e combinação:** Os resultados dos diferentes níveis de Oracle serão comparados com a acurácia dos classificadores individuais e de métodos reais de combinação, como votação majoritária e média das probabilidades preditas. Quando viável, também serão avaliadas técnicas de Seleção Dinâmica de Classificadores ou de Ensembles, utilizando bibliotecas como scikit-learn e DESlib.

7. **Análise quantitativa dos resultados:** Os resultados serão avaliados por métricas como acurácia, precisão, revocação, F1-score e acurácia balanceada. Também serão produzidas curvas Oracle\_N e tabelas comparativas, com o objetivo de analisar a distância entre os limites teóricos propostos e o desempenho obtido por métodos práticos.

8. **Organização dos resultados e elaboração do relatório técnico-científico:** Por fim, os resultados serão sistematizados em relatório técnico-científico, contendo fundamentação teórica, descrição das bases utilizadas, protocolo experimental, implementação, análise dos resultados e discussão sobre a relevância dos Oracles em diferentes níveis para avaliação de Sistemas de Múltiplos Classificadores.

   6. # **RESULTADOS ESPERADOS**

Espera-se caracterizar quantitativamente o comportamento dos Oracles em múltiplos níveis de exigência, analisando a variação da acurácia estimada à medida que aumenta o número mínimo de classificadores corretos para cada amostra. A partir dessa análise, pretende-se construir a curva **Oracle\_N**, capaz de evidenciar a transição entre o Oracle tradicional, mais otimista, e limites superiores progressivamente mais restritivos e conservadores.

Espera-se verificar se níveis intermediários de Oracle apresentam maior proximidade com o desempenho de métodos práticos de combinação e seleção, como votação majoritária e técnicas de Seleção Dinâmica de Classificadores. Caso essa relação seja observada, os Oracles intermediários poderão ser utilizados como medidas complementares para estimar limites superiores mais realistas em Sistemas de Múltiplos Classificadores.

Como resultado técnico, pretende-se desenvolver uma implementação reprodutível das medidas **Oracle\_1, Oracle\_2, ..., Oracle\_M**, contemplando scripts para treinamento dos classificadores-base, geração da matriz de acertos, cálculo das métricas propostas e construção de gráficos comparativos. Essa implementação deverá permitir a aplicação da metodologia em diferentes bases de classificação e facilitar a replicação dos experimentos em trabalhos futuros.

Do ponto de vista científico, espera-se contribuir para a análise do limite superior de desempenho em ensembles, relacionando diversidade, redundância de acertos e desempenho final do conjunto. A pesquisa deverá indicar em quais cenários o Oracle tradicional tende a superestimar o desempenho alcançável e em quais condições os níveis intermediários oferecem estimativas mais compatíveis com métodos práticos.

Por fim, espera-se que a análise da curva **Oracle\_N** auxilie na avaliação do potencial real de melhoria de um pool de classificadores. Essa informação poderá apoiar a decisão sobre o uso de métodos mais complexos de seleção dinâmica ou combinação, especialmente em situações nas quais a baixa redundância de acertos limite os ganhos práticos esperados. Os resultados obtidos deverão subsidiar a apresentação em evento de iniciação científica e servir como base para a elaboração futura de artigo acadêmico.

7. # **REFERÊNCIAS**

KUNCHEVA, Ludmila I.; WHITAKER, Christopher J. Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy. Machine Learning, v. 51, p. 181-207, 2003\. DOI: 10.1023/A:1022859003006. Disponível em: https://link.springer.com/article/10.1023/A:1022859003006. Acesso em: 02 jun. 2026\.

KUNCHEVA, Ludmila I.; RODRÍGUEZ, Juan J. Classifier Ensembles with a Random Linear Oracle. IEEE Transactions on Knowledge and Data Engineering, v. 19, n. 4, p. 500-508, 2007\. DOI: 10.1109/TKDE.2007.1016.

CRUZ, Rafael M. O.; SABOURIN, Robert; CAVALCANTI, George D. C. META-DES.Oracle: Meta-learning and feature selection for dynamic ensemble selection. Information Fusion, v. 38, p. 84-103, 2017\. DOI: 10.1016/j.inffus.2017.02.010. Disponível em: https://ieeexplore.ieee.org/abstract/document/7965873. Acesso em: 02 jun. 2026\.

CRUZ, Rafael M. O.; SABOURIN, Robert; CAVALCANTI, George D. C. Dynamic classifier selection: Recent advances and perspectives. Information Fusion, v. 41, p. 195-216, 2018\. DOI: 10.1016/j.inffus.2017.09.010. Disponível em: https://www.sciencedirect.com/science/article/abs/pii/S1566253517304074. Acesso em: 02 jun. 2026\.

SOUZA, Mariana A.; CAVALCANTI, George D. C. On the Characterization of the Oracle for Dynamic Classifier Selection. In: International Joint Conference on Neural Networks (IJCNN), 2017, Anchorage. Proceedings. IEEE, 2017\. p. 332-339. DOI: 10.1109/IJCNN.2017.7965873.

1. # **CRONOGRAMA DE EXECUÇÃO**

|  Descrição de Atividades |  |  |  |  |  |  |  |  |  |  |  |  |
| :---- | ----- | ----- | ----- | ----- | ----- | ----- | :---: | ----- | ----- | ----- | ----- | ----- |
|  | **Set / 2026** | **Out** | **Nov** | **Dez** | **Jan** | **Fev** | **Mar** | **Abr** | **Mai** | **Jun** | **Jul** | **Ago / 2027** |
| Revisão bibliográfica sobre MCS, DCS/DES, diversidade e Oracle | X | X | X |  |  |  |  |  |  |  |  |  |
| Seleção das bases de dados e definição do protocolo experimental |  | X | X |  |  |  |  |  |  |  |  |  |
| Implementação do pool de classificadores e avaliação individual dos modelos |  |  | X | X | X |  |  |  |  |  |  |  |
| Implementação do Oracle tradicional e dos Oracles em diferentes níveis |  |  |  | X | X | X |  |  |  |  |  |  |
| Relatório Semestral conforme modelo PROPESP, até 15 de março de 2027. |  |  |  |  |  |  | X |  |  |  |  |  |
| Experimentos comparativos com votação majoritária, combinação de probabilidades e métodos de seleção dinâmica |  |  |  |  |  | X | X | X | X | X |  |  |
| Análise dos resultados, geração de gráficos e elaboração de recomendações |  |  |  |  |  |  |  | X | X | X | X |  |
| Relatório final, conforme modelo PROPESP, até o dia 11 de setembro de 2027 |  |  |  |  |  |  |  |  |  |  | X |  X |

