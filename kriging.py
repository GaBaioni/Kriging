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