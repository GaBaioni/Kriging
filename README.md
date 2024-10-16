# KrigingModel

**KrigingModel** é uma implementação em Python para interpolação de dados usando o método de krigagem (kriging). A classe suporta tanto modelos unidimensionais quanto multidimensionais, fornecendo uma forma flexível e prática para a construção de metamodelos que permitem prever valores desconhecidos com base em dados de amostragem.

## Funcionalidades

- **Ajuste dos pesos via kriging**: A classe ajusta automaticamente os pesos dos pontos de amostragem, utilizando uma função de variograma para modelar a relação espacial entre os dados.
- **Previsão de novos valores**: A partir dos pesos ajustados, a classe é capaz de prever valores em novos pontos, utilizando os dados já conhecidos.
- **Cálculo do erro quadrático médio (MSE)**: A classe permite calcular o erro quadrático médio (MSE) para avaliar a precisão do modelo, comparando as previsões feitas com os valores reais de um conjunto de validação.
- **Suporte para múltiplas dimensões**: A classe é capaz de lidar com dados em múltiplas dimensões (1D, 2D, 3D, etc.), possibilitando a aplicação da técnica de krigagem em cenários mais complexos.
- **Variograma personalizável**: O usuário pode definir sua própria função de variograma, o que permite flexibilidade na modelagem da dependência espacial dos dados.

## Estrutura da Classe

A classe **KrigingModel** é composta por funções que facilitam o ajuste dos pesos dos pontos de amostragem e a previsão de novos valores. A estrutura básica da classe envolve as seguintes etapas:

1. **Inicialização dos Dados de Treinamento**: A classe é inicializada com um conjunto de pontos de amostragem (conjunto de treinamento), bem como seus valores correspondentes. Também é fornecida uma função de variograma que define a relação espacial entre os pontos.

2. **Matriz de Distâncias**: Para modelar a relação espacial entre os pontos, a classe calcula a matriz de distâncias entre os pontos de amostragem e, posteriormente, entre os pontos de amostragem e os novos pontos para os quais deseja-se realizar previsões.

3. **Matriz de Covariância**: Utilizando as distâncias calculadas e a função de variograma, a classe constrói a matriz de covariância. Essa matriz captura a dependência espacial entre os pontos de amostragem, o que é crucial para ajustar os pesos.

4. **Ajuste dos Pesos**: Os pesos (ou coeficientes) dos pontos de amostragem são ajustados de maneira a minimizar o erro entre os valores preditos e os valores observados nos pontos de amostragem. Isso é feito resolvendo um sistema de equações que envolve a matriz de covariância e a função de variograma.

5. **Previsão de Novos Valores**: Após o ajuste dos pesos, é possível prever o valor de uma nova variável, dado um novo ponto de entrada. Essa previsão é obtida como uma combinação ponderada dos valores conhecidos nos pontos de amostragem.

6. **Erro Quadrático Médio (MSE)**: A precisão do modelo pode ser avaliada usando um conjunto de validação, calculando-se o erro quadrático médio (MSE) entre os valores reais e os valores previstos.

## Uso

A classe **KrigingModel** é ideal para construir metamodelos em cenários onde há uma relação espacial (ou de proximidade) entre os dados, permitindo a previsão de valores desconhecidos. A técnica de krigagem é amplamente utilizada em áreas como geostatística, engenharia e ciências ambientais, e a implementação flexível desta classe permite seu uso em diversas outras áreas, incluindo otimização e simulação.

## Conjunto de Dados de Validação

Ao utilizar a classe **KrigingModel**, recomenda-se a aplicação do modelo a um conjunto de dados conhecidos (conjunto de treinamento), seguido da validação do modelo com um conjunto de pontos não utilizados no treinamento. A validação é importante para garantir que o modelo não esteja superajustado aos dados de amostragem e que seja capaz de generalizar bem para novos dados. O cálculo do erro quadrático médio (MSE) fornece uma métrica direta da precisão do modelo, ajudando a identificar possíveis ajustes no variograma ou nos dados de entrada.

