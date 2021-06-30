import pandas as pd
import csv
from sklearn.svm import SVC



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

if __name__ == '__main__':
    arquivo = 'Data/ionosphere.data'
    dataset = pd.read_csv(arquivo, sep=',', header=None)

    # Aplica SVM com os kernels 'linear', 'poly', 'rbf', 'sigmoid'
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for i, kernel in enumerate(kernels):
        # Usando método hold out
        X_treino, X_teste, y_treino, y_teste = hold_out(dataset, tam_treino=0.8)

        # SVM
        svm = SVC(kernel=kernel, gamma='auto')
        svm.fit(X_treino, y_treino)
        predicoes = svm.predict(X_teste)
        predicoes = list(predicoes) #gera lista com resultados

        #acurácia
        cont = 0
        for x, y in zip(y_teste, predicoes):
            if x == y:
                cont += 1
        acuracia = cont/len(y_teste)
        print('Método HOLD OUT - Acurácia: {:.4f}'.format(acuracia), 'com kernel = ', kernel)

        # Salva os labels verdadeiros e preditos
        with open('Data/HO_labels_orig_e_preditos_kernel_' + str(kernel) + '_q57.csv', 'w', newline='') as arquivo:
            linhas = [y_teste, predicoes]
            gravar = csv.writer(arquivo, delimiter=',')
            gravar.writerows(linhas)

        # Usando método leave one out
        X_treino, X_teste, y_treino, y_teste = leave_one_out(dataset)

        cont = 0
        # cont_amostra = 0
        predicoes = []
        for treino, teste, l_treino, l_teste in zip(X_treino, X_teste, y_treino, y_teste):
            # print('Treinando amostra {} de {}'.format(cont_amostra+1, len(y_treino)))
            # cont_amostra += 1

            # SVM
            svm = SVC(kernel=kernel, gamma='auto')
            svm.fit(treino, l_treino)
            lista_teste = []
            lista_teste.append(teste)

            predicao = svm.predict(lista_teste)
            predicoes.append(predicao[0])

            if predicao == l_teste:
                cont += 1
        acuracia = cont / len(y_teste)
        print('Método LEAVE ONE OUT - Acurácia: {:.4f}'.format(acuracia), 'com kernel = ', kernel)
        # Salva os labels verdadeiros e preditos
        with open('Data/LOO_labels_orig_e_preditos_kernel_' + str(kernel) + '_q57.csv', 'w', newline='') as arquivo:
            linhas = [y_teste, predicoes]
            gravar = csv.writer(arquivo, delimiter=',')
            gravar.writerows(linhas)
