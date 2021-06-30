import cv2 as cv
import numpy as np

# ler arquivo txt
with open('Imagens/pixels_matrix_limiarizacao.txt', 'r') as arquivo:
    for cont, valores in enumerate(arquivo):
        linha = [int(valor) for valor in valores.split()]
        if cont == 0:
            img = np.hstack(linha) # primeira linha da imagem
        else:
            img = np.vstack([img, linha]) # adiciona novas linhas na imagem

# Converte para uint8
img_lida = np.array(img, np.uint8)

# mostra imagem reconstruída
cv.imshow('Imagem lida', img_lida)
cv.waitKey(0)