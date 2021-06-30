import cv2 as cv
import os, csv, glob
import numpy as no
import numpy as np
from skimage import feature

def extrair_lbp(imgs, n_pontos, raio, eps=1e-7):
    atributos_lbp = []
    for i, img in enumerate(imgs):
        print('Extraindo atributos da imagem {} de {}'.format(i+1, len(imgs)))
        arq = cv.imread(img)
        arq = cv.cvtColor(arq, cv.COLOR_BGR2GRAY)

        lbp = feature.local_binary_pattern(arq, n_pontos, raio, method='uniform')

        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_pontos+3), range=(0, n_pontos+2))
        hist = hist.astype('float')
        hist /= (hist.sum() + eps)

        img_lbp = [item for item in list(hist)]

        atributos_lbp.append(img_lbp)

    print('\n')

    return atributos_lbp

def salvar(nome_extrator, atributos):
    for atributo in atributos:
        print(atributo)
    # Salva em *.csv
    with open(nome_extrator + '.csv', 'w', newline='') as arquivo:
        gravar = csv.writer(arquivo)
        gravar.writerows(atributos)

if __name__ == '__main__':
    dataset = 'Imagens/'
    lista_img = glob.glob(os.path.join(dataset, '*.jpg'))
    atributos = extrair_lbp(lista_img, n_pontos=36, raio=10)
    salvar(dataset + 'lbp', atributos)