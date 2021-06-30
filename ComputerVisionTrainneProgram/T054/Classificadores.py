import numpy as np

class KMeans:

    def __init__(self, K=3, tolerancia=0.001, max_iter=10000):
        self.K = K
        self.tolerancia = tolerancia
        self.max_iter = max_iter
        self.centroides = {}

    def fit(self, dados):
        # Inicializa o centroide
        for i in range(self.K):
            self.centroides[i] = dados[i]

        # iterações
        for i in range(self.max_iter):
            self.classes = {}
            for i in range(self.K):
                self.classes[i] = []

            # Distância entre o ponto e cluster para escolha do centroide mais proximo
            for atributos in dados:
                distancias = [np.linalg.norm(np.array(atributos) - np.array(self.centroides[centroide])) for centroide
                              in self.centroides]
                classificacao = distancias.index(min(distancias))
                self.classes[classificacao].append(atributos)

            anterior = dict(self.centroides)

            #Recalcula o centroide com a média dos dados
            for classificacao in self.classes:
                self.centroides[classificacao] = np.average(self.classes[classificacao], axis=0)

            vlrOtimo = True

            for centroide in self.centroides:
                centroide_orig = anterior[centroide]
                atual = self.centroides[centroide]
                if np.sum((atual - centroide_orig)/centroide_orig * 100) > self.tolerancia:
                    vlrOtimo = False

            # Encerra quando o valor é ótimo ou os centroidens não mudam
            if vlrOtimo:
                break

    def predict(self, dados):
        distancias = [np.linalg.norm(dados - self.centroides[centroide]) for centroide in self.centroides]
        classificacao = distancias.index(min(distancias))
        return classificacao



