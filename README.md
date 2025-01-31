# Kriging

## Introdução
A biblioteca **Kriging** é uma implementação em Python do método de krigagem, uma técnica de interpolação geoestatística usada para construir modelos substitutos baseados em amostras discretas de um domínio. O objetivo da krigagem é estimar valores em novos pontos com base em uma função de variograma que descreve a estrutura espacial da variável de interesse.

Esta biblioteca permite:
- Ajustar um variograma com os melhores parâmetros.
- Construir uma matriz de covariância.
- Prever valores em novos pontos com base na interpolação geoestatística.

## Fundamentos Teóricos
A krigagem assume que a variável de interesse $Z(x)$ pode ser modelada como um processo estocástico com média e estrutura de covariância bem definidas. O processo se baseia na minimização do erro quadrático médio entre os valores estimados e os valores reais.

### Variograma
O **variograma** é uma função que descreve a dependência espacial entre pontos amostrados. Ele é definido como:  

$$\gamma(h) = \frac{1}{2} E[(Z(x) - Z(x+h))^2]$$  

onde:
- $h$ é a distância entre os pontos.
- $Z(x)$ e $Z(x+h)$ são os valores da variável de interesse em diferentes posições.
- $\gamma(h)$ representa a variação esperada entre dois pontos separados pela distância $ h $.

A biblioteca implementa três modelos comuns de variogramas:
1. **Exponencial**:
   $$\gamma(h) = C_0 + C (1 - e^{-3h/\alpha})$$
      
3. **Esférico**:
   $$\gamma(h) = C_0 + C \left( \frac{3}{2} \frac{h}{\alpha} - \frac{1}{2} \left(\frac{h}{\alpha}\right)^3 \right), \quad h < \alpha$$  
   
5. **Gaussiano**:
   $\gamma(h) = C_0 + C (1 - e^{-3(h/\alpha)^2})$$  
   
Onde:
- $C_0$ é o efeito pepita (variabilidade independente da distância).
- $C $é o patamar (máximo valor do variograma).
- $\alpha$ é o alcance (distância onde o variograma atinge seu patamar).

### Construção da Matriz de Covariância
A matriz de covariância é construída a partir do variograma ajustado. Para um conjunto de $ n $ pontos, a matriz é definida como:  

$$\mathbf{C} = \begin{bmatrix}
\gamma(h_{11}) & \gamma(h_{12}) & \dots & 1 \\
\gamma(h_{21}) & \gamma(h_{22}) & \dots & 1 \\
\vdots & \vdots & \ddots & \vdots \\
1 & 1 & \dots & 0
\end{bmatrix}$$

Essa matriz é usada para resolver o sistema linear dos pesos $ \lambda $, permitindo a estimativa de novos valores.

### Predição de Novos Pontos
A interpolação em um novo ponto $ x^* $ usa a equação:  

$$Z^*(x^*) = \sum_{i=1}^{n} \lambda_i Z(x_i)$$  

Onde os pesos $ \lambda_i $ são obtidos resolvendo:  

$$\mathbf{C} \lambda = \mathbf{r}$$  

com $$mathbf{r}$$ sendo o vetor das covariâncias entre os novos pontos e os pontos amostrados.

## Estrutura da Biblioteca
A biblioteca possui a seguinte estrutura:

- **`Kriging`**: Classe principal que contém os métodos para ajuste do variograma, construção da matriz de covariância e predição.
  - `__init__(self, X, Y, model='exponential')`: Inicializa o modelo e ajusta os parâmetros do variograma.
  - `variogram_function(self, h, C0, C, alpha)`: Define o modelo de variograma escolhido.
  - `fit_variogram(self)`: Ajusta os parâmetros do variograma otimizando $ C_0 $, $ C $ e $ \alpha $.
  - `build_covariance_matrix(self)`: Constrói a matriz de covariância para os pontos amostrados.
  - `predict(self, X_new)`: Realiza a interpolação nos novos pontos.

## Uso da Biblioteca
Após importar a biblioteca, um usuário pode fornecer dados experimentais e escolher um modelo de variograma para treinar e fazer previsões:

```python
from kriging_library import Kriging
import numpy as np

# Dados de entrada
X_train = np.array([[0], [1], [2], [3], [4]])
Y_train = np.array([10, 12, 18, 25, 30])

# Criando o modelo de Kriging
model = Kriging(X_train, Y_train, model='exponential')

# Prevendo um novo ponto
X_new = np.array([[2.5]])
pred = model.predict(X_new)
print(f'Predição para X_new: {pred}')
```


