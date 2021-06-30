import cv2 as cv
import os, csv, glob
import numpy as np
from skimage import feature

def extrair_glcm(imgs, distancias, angulos):
    atributos_glcm = []
    for i, img in enumerate(imgs):
        print('Extraindo atributos da imagem {} de {}'.format(i+1, len(imgs)))
        arq = cv.imread(img)
        arq = cv.cvtColor(arq, cv.COLOR_BGR2GRAY)

        glcm = feature.greycomatrix(arq, distancias, angulos, 256, symmetric=False, normed=True)

        propriedades_glcm = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        atributos = [feature.greycoprops(glcm, propriedade_glcm)[0,0] for propriedade_glcm in propriedades_glcm]

        atributos_glcm.append(atributos)

    print('\n')

    return atributos_glcm

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
    atributos = extrair_glcm(lista_img, distancias=[3], angulos=[0])
    salvar(dataset + 'glcm', atributos)