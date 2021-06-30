import os, csv, glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


if __name__ == '__main__':
    dataset = 'Data/'
    lista_csv = glob.glob(os.path.join(dataset, '*.csv'))

    for tabela in lista_csv:
        print(tabela)
        with open(tabela, 'r') as arquivo:
            leitura = csv.reader(arquivo, delimiter=',')
            dados = []
            for linha in leitura:
                dados.append(linha)

            nome_classes = ['0', '1']

            c_verdadeiras = [int(x) for x in dados[0]]
            c_preditas = [int(x) for x in dados[1]]

            mc = confusion_matrix(c_verdadeiras, c_preditas)

            print('Acurácia: {}'.format(accuracy_score(c_verdadeiras,c_preditas)))
            print('Relatório de classificação: \n', classification_report(c_verdadeiras,c_preditas))
