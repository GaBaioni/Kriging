import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist

class Kriging:
    def __init__(self, X, Y, model='exponential'):
        """
        Inicializa a classe Kriging com os pontos amostrais e ajusta o variograma.
        
        Parâmetros:
        X: array-like, coordenadas dos pontos amostrais.
        Y: array-like, valores observados nos pontos amostrais.
        model: string, tipo de variograma a ser utilizado ('exponential', 'spherical' ou 'gaussian').
        """
        self.X = np.array(X)  # Converte os pontos amostrais para um array numpy
        self.Y = np.array(Y)  # Converte os valores observados para um array numpy
        self.model = model    # Define o modelo de variograma a ser utilizado
        
        # Ajusta os parâmetros do variograma baseado nos dados amostrais
        self.C0, self.C, self.alpha = self.fit_variogram()
        
        # Constrói a matriz de covariância para a interpolação
        self.cov_matrix = self.build_covariance_matrix()
    
    def variogram_function(self, h, C0, C, alpha):
        """
        Define a função do variograma com base no modelo escolhido.
        
        Parâmetros:
        h: array-like, distância entre os pontos amostrais.
        C0: efeito pepita.
        C: patamar do variograma.
        alpha: alcance do variograma.
        """
        if self.model == 'exponential':
            return C0 + C * (1 - np.exp(-3 * h / alpha))
        elif self.model == 'spherical':
            return C0 + C * ((3/2) * (h/alpha) - (1/2) * (h/alpha)**3) * (h < alpha) + C * (h >= alpha)
        elif self.model == 'gaussian':
            return C0 + C * (1 - np.exp(-3 * (h/alpha)**2))
        else:
            raise ValueError("Modelo não suportado. Escolha 'exponential', 'spherical' ou 'gaussian'")
    
    def fit_variogram(self):
        """
        Ajusta os parâmetros do variograma usando o método dos mínimos quadrados.
        """
        # cdist(XA, XB, metric='euclidean') -> Calcula a distância euclidiana entre os pontos de XA e XB
        h = cdist(self.X, self.X, metric='euclidean').flatten()
        # Calcula a semivariância experimental para cada par de pontos
        gamma_exp = 0.5/len(self.Y) * ((self.Y[:, None] - self.Y[None, :])**2).flatten()
        
        # Ignora distâncias zero (pontos idênticos)
        valid = h > 0  
        
        # Ajusta os parâmetros do variograma minimizando o erro quadrático
        #curve_fit(função, x, y, p0) -> popt = parâmetros ótimos
        popt, _ = curve_fit(self.variogram_function, h[valid], gamma_exp[valid], p0=[0.1, 1.0, 1.0])
        return popt
    
    def build_covariance_matrix(self):
        """
        Constrói a matriz de covariância baseada na função do variograma ajustada.
        """
        n = len(self.X)
        # Matriz de covariância expandida para incluir o termo de Lagrange.
        # Termo de Lagrange para garantir que a soma dos pesos seja igual a 1
        cov_matrix = np.zeros((n+1, n+1))  
        
        distances = cdist(self.X, self.X, metric='euclidean')  # Calcula a matriz de distâncias entre os pontos
        
        # Preenche a matriz de covariância com os valores do variograma
        for i in range(n):
            for j in range(n):
                cov_matrix[i, j] = self.variogram_function(distances[i, j], self.C0, self.C, self.alpha)
        
        cov_matrix[:n, n] = 1  # Adiciona o termo de regularização (termo de Lagrange)
        cov_matrix[n, :n] = 1  
        return cov_matrix
    
    def predict(self, X_new):
        """
        Realiza a predição para novos pontos com base nos dados amostrais.
        
        Parâmetros:
        X_new: array-like, coordenadas dos pontos onde a predição será feita.
        """
        X_new = np.array(X_new)
        n = len(self.X)
        predictions = []
        
        for x in X_new:
            # Calcula a covariância entre os novos pontos e os pontos amostrais
            r = np.array([self.variogram_function(np.linalg.norm(x - xi), self.C0, self.C, self.alpha) for xi in self.X] + [1])
            
            # Resolve o sistema linear para obter os pesos da interpolação
            weights = np.linalg.solve(self.cov_matrix, r)
            
            # Calcula o valor predito usando os pesos obtidos
            # np.dot(a, b) -> Produto escalar entre a e b
            pred = np.dot(weights[:-1], self.Y)
            predictions.append(pred)
        
        return np.array(predictions)