import numpy as np
import matplotlib.pyplot as plt

class KrigingModel:
    def __init__(self, x_train, z_train, variogram_func):
        """
        Inicializa o modelo de Kriging com pontos de amostragem e a função de variograma.

        :param x_train: Array de pontos de entrada de treinamento (dimensões N x d, onde N é o número de pontos e d a dimensão).
        :param z_train: Array de valores correspondentes a cada ponto (dimensão N).
        :param variogram_func: Função que define o semivariograma.
        """
        self.x_train = np.array(x_train)
        self.z_train = np.array(z_train)
        self.variogram_func = variogram_func

    def distance_matrix(self, x1, x2):
        """
        Calcula a matriz de distâncias entre dois conjuntos de pontos.
        
        :param x1: Array de pontos x (dimensões N x d).
        :param x2: Array de pontos y (dimensões M x d).
        :return: Matriz de distâncias (dimensões N x M).
        """
        return np.linalg.norm(x1[:, np.newaxis] - x2, axis=2)

    def covariance_matrix(self):
        """
        Cria a matriz de covariância com base nas distâncias entre os pontos de treinamento.
        
        :return: Matriz de covariância (dimensões N x N).
        """
        distances = self.distance_matrix(self.x_train, self.x_train)
        return self.variogram_func(distances)

    def fit(self):
        """
        Ajusta os pesos lambda usando o modelo de Kriging.
        
        :return: Vetor de pesos (lambda) e multiplicador Lagrange (mu).
        """
        n = self.x_train.shape[0]
        cov_matrix = self.covariance_matrix()

        # Adiciona uma coluna de uns para a restrição dos pesos somando 1
        C_ext = np.hstack((cov_matrix, np.ones((n, 1))))
        C_ext = np.vstack((C_ext, np.ones((1, n + 1))))
        C_ext[-1, -1] = 0

        # Vetor c para o ponto a ser estimado
        c = np.zeros((n + 1, 1))
        c[-1] = 1  # Restrição de soma dos pesos

        # Solução do sistema de equações
        rhs = np.hstack([self.z_train, np.array([0])])
        solution = np.linalg.solve(C_ext, rhs)

        self.lambdas = solution[:-1]  # Pesos ajustados (lambda)
        self.mu = solution[-1]  # Multiplicador Lagrange (mu)
        return self.lambdas

    def predict(self, x_new):
        """
        Prediz o valor z no novo ponto x_new com base nos pesos ajustados.

        :param x_new: Novo ponto (dimensão d).
        :return: Valor previsto de z.
        """
        distances = self.distance_matrix(self.x_train, np.array([x_new]))
        cov_vector = self.variogram_func(distances).flatten()
        return np.dot(self.lambdas, cov_vector)

    def mean_squared_error(self, x_valid, z_valid):
        """
        Calcula o erro quadrático médio para pontos de validação.

        :param x_valid: Pontos de validação (dimensão N_valid x d).
        :param z_valid: Valores correspondentes para os pontos de validação (dimensão N_valid).
        :return: Erro quadrático médio (MSE).
        """
        predictions = np.array([self.predict(x) for x in x_valid])
        return np.mean((predictions - z_valid) ** 2)


# Exemplo de uso
if __name__ == "__main__":
    # Função de variograma exponencial
    def exponential_variogram(h, sill=1, range_=1):
        return sill * (1 - np.exp(-h / range_))

    # Gerando dados de treinamento (20 pontos em um grid 2D)
    np.random.seed(0)
    x_train = np.random.rand(20, 2) * 10  # Pontos de entrada (bidimensionais) entre 0 e 10
    z_train = np.sin(x_train[:, 0]) + np.cos(x_train[:, 1]) + np.random.normal(0, 0.1, 20)  # Valores z com um pouco de ruído

    # Criando e ajustando o modelo de Kriging
    model = KrigingModel(x_train, z_train, exponential_variogram)
    model.fit()

    # Gerando novos pontos para validação
    x_valid = np.random.rand(10, 2) * 10  # Novos pontos de entrada para validação
    z_valid = np.sin(x_valid[:, 0]) + np.cos(x_valid[:, 1])  # Valores verdadeiros de z sem ruído

    # Fazendo previsões
    predictions = [model.predict(x) for x in x_valid]
    
    # Calculando o MSE
    mse = model.mean_squared_error(x_valid, z_valid)
    print(f"Erro quadrático médio (MSE): {mse}")

    # Plotando os resultados
    plt.figure(figsize=(10, 6))

    # Dados de treinamento
    plt.scatter(x_train[:, 0], x_train[:, 1], c=z_train, cmap='viridis', label="Pontos de treinamento", marker='o')

    # Dados reais para validação (comparação)
    plt.scatter(x_valid[:, 0], x_valid[:, 1], c=z_valid, cmap='coolwarm', edgecolor='black', label="Valores reais", marker='s')

    # Previsões (dados de validação)
    plt.scatter(x_valid[:, 0], x_valid[:, 1], c=predictions, cmap='coolwarm', label="Previsões", marker='x')

    plt.colorbar(label='Valor de z')
    plt.legend()
    plt.title('Previsões com Kriging em dados bidimensionais')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()