import numpy as np
import pandas as pd
import math
import operator
import csv
import matplotlib.pyplot as plt
from matplotlib import style
from Classificadores import KMeans

def hold_out(df, tam_treino, shuffle=True):
    # Mistura base de dados
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    dados = []
    for linha in df.iterrows():
        id, valor = linha
        dados.append(valor.tolist())

    # Separa em treino e teste
    X_train = dados[:int(tam_treino*len(dados))]
    X_test = dados[int(tam_treino*len(dados)):]

    # Pega o label para os atributos
    y_train = [int(x[-1]) for x in X_train]
    y_test = [int(x[-1]) for x in X_test]

    # Remover os labels do conjunto de treino e teste
    X_train = [x[:-1] for x in X_train]
    X_test = [x[:-1] for x in X_test]

    return X_train, X_test, y_train, y_test

def leave_one_out(df, shuffle=True):
    # Mistura base de dados
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    dados = []
    for linha in df.iterrows():
        id, valor = linha
        dados.append(valor.tolist())

    # Lista para armazenamento
    X_treino = []
    X_teste = []
    y_treino = []
    y_teste = []

    for i in range(len(dados)):
        treino = dados.copy()
        treino.remove(dados[i])
        teste = dados[i]

        # Pega o label para os atributos
        y_treino.append([int(x[-1]) for x in treino])
        y_teste.append(int(teste[-1]))

        # Remover os labels do conjunto de treino e teste
        X_treino.append([x[:-1] for x in treino])
        X_teste.append(teste[:-1])

    return X_treino, X_teste, y_treino, y_teste

def dist_euclidiana(x1, x2):
    dist = 0.0
    for x, y in zip(x1, x2):
        dist += pow(float(x) - float(y), 2)
    dist = math.sqrt(dist)
    return dist


if __name__ == '__main__':
    arquivo = 'Data/ionosphere.data'
    dataset = pd.read_csv(arquivo, sep=',', header=None)

    # Aplica Knn para K = 1, 3, 7.
    K = [1, 3, 7]
    for i, k in enumerate(K):
        # Usando método hold out
        X_treino, X_teste, y_treino, y_teste = hold_out(dataset, tam_treino=0.8)
        kmeans = KMeans(K=k, max_iter=10000)
        kmeans.fit(X_treino)

        predicoes = []
        for teste in X_teste:
            predicoes.append(kmeans.predict(teste))

        cont = 0
        for x, y in zip(y_teste, predicoes):
            if x == y:
                cont += 1
        acuracia = cont/len(y_teste)
        print('Método HOLD OUT - Acurácia: {:.4f}'.format(acuracia), 'com K = ', k)
        # Salva os labels verdadeiros e preditos
        with open('Data/HO_labels_orig_e_preditos_K_' + str(k) + '_q54.csv', 'w', newline='') as arquivo:
            linhas = [y_teste, predicoes]
            gravar = csv.writer(arquivo, delimiter=',')
            gravar.writerows(linhas)

        # Usando método leave one out
        X_treino, X_teste, y_treino, y_teste = leave_one_out(dataset)
        cont = 0

        predicoes = []

        for treino, teste, l_treino, l_teste in zip(X_treino, X_teste, y_treino, y_teste):
            lista_teste = []
            kmeans = KMeans(K=k, max_iter=10000)
            kmeans.fit(treino)
            predicao = kmeans.predict(teste)
            predicoes.append(predicao)
            if predicao == l_teste:
                cont += 1
        acuracia = cont / len(y_teste)
        print('Método LEAVE ONE OUT - Acurácia: {:.4f}'.format(acuracia), 'com K = ', k)
        # Salva os labels verdadeiros e preditos
        with open('Data/LOO_labels_orig_e_preditos_K_' + str(k) + '_q54.csv', 'w', newline='') as arquivo:
            linhas = [y_teste, predicoes]
            gravar = csv.writer(arquivo, delimiter=',')
            gravar.writerows(linhas)