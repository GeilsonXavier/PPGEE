import os, csv, glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def matriz_confusao(y_true, y_pred, classes,
                    normalize=False,
                    title=None,
                    cmap=plt.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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

            matriz_confusao(c_verdadeiras, c_preditas, classes=nome_classes, title='Matriz de confusão \n' + tabela)
            plt.show()
